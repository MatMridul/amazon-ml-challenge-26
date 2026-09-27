"""
ML Challenge 2026 — Phase 2: Supervised Training Pair Construction

Generates labeled (S1, Candidate) pairs using the frozen Blocking V1.1 pipeline:
- Positives (label = 1): candidates present in train_ground_truth.tsv
- Hard Negatives (label = 0): candidates retrieved by the blocker that are NOT in ground truth
- Excludes held-out validation S1 entities to guarantee zero data leakage
- Exports dataset to Parquet for Phase 3 feature engineering & Phase 4 LightGBM training
"""

import os
import sys
import argparse
from collections import defaultdict
import numpy as np
import polars as pl
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.blocking import LexicalBlocker

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VAL_DIR = os.path.join(BASE_DIR, "dataset", "validation")
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "student_resource", "dataset", "train")
PAIRS_DIR = os.path.join(BASE_DIR, "dataset", "training_pairs")


def load_candidate_pool(country: str):
    """Loads S2 and S3 candidate pool for the country."""
    print(f"Loading candidate pool for country '{country}'...")
    s2_path = os.path.join(TRAIN_DIR, "train_source2.tsv")
    s3_path = os.path.join(TRAIN_DIR, "train_source3.tsv")

    s2_df = pl.read_csv(s2_path, separator="\t").filter(pl.col("country") == country)
    s3_df = pl.read_csv(s3_path, separator="\t").filter(pl.col("country") == country)

    pool = pl.concat([s2_df, s3_df])
    print(f"[{country}] Pool loaded: {len(pool):,} records ({len(s2_df):,} S2 + {len(s3_df):,} S3)")
    return pool


def load_training_queries(country: str, sample_size: int = 5000, seed: int = 101):
    """
    Loads training S1 queries, strictly excluding validation S1 entities.
    """
    val_s1_path = os.path.join(VAL_DIR, "val_source1.tsv")
    val_eids = set(pl.read_csv(val_s1_path, separator="\t", columns=["entity_id"])["entity_id"])

    s1_path = os.path.join(TRAIN_DIR, "train_source1.tsv")
    gt_path = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

    print(f"Loading train queries (excluding {len(val_eids):,} validation entities)...")
    s1_df = pl.read_csv(s1_path, separator="\t").filter(pl.col("country") == country)
    gt_df = pl.read_csv(gt_path, separator="\t")

    # Filter out validation queries
    s1_train = s1_df.filter(~pl.col("entity_id").is_in(list(val_eids)))
    
    # Sample queries
    if sample_size and sample_size < len(s1_train):
        s1_train = s1_train.sample(n=sample_size, seed=seed)

    merged = s1_train.join(gt_df, left_on="entity_id", right_on="source1_entity_id", how="inner")
    print(f"Sampled {len(merged):,} training S1 entities for '{country}'.")
    return merged


def build_training_pairs(country: str = "India", sample_size: int = 5000):
    os.makedirs(PAIRS_DIR, exist_ok=True)
    out_parquet = os.path.join(PAIRS_DIR, f"train_pairs_{country.lower()}_{sample_size}.parquet")

    # 1. Load data
    train_df = load_training_queries(country=country, sample_size=sample_size)
    pool_df = load_candidate_pool(country=country)

    # 2. Build index
    blocker = LexicalBlocker(country=country)
    blocker.index_records(pool_df)

    # 3. Generate candidate pairs & label against ground truth
    print(f"\nGenerating candidate pairs for {len(train_df):,} training queries...")

    pair_rows = []
    
    # Statistics tracking
    total_candidates_generated = 0
    total_positives_captured = 0
    total_negatives_generated = 0
    total_ground_truth_links = 0
    
    positives_per_s1 = []
    candidates_per_s1 = []
    s1_zero_positives_count = 0

    for row in tqdm(train_df.iter_rows(named=True), total=len(train_df), desc="Building pairs"):
        s1_id = row["entity_id"]
        s1_name = row["business_name"] or ""
        s1_addr = row["business_address"] or ""
        mids_str = row["matched_entity_ids"]
        true_set = set(mids_str.split(",")) if mids_str else set()

        total_ground_truth_links += len(true_set)

        # Retrieve candidates from frozen V1.1 blocker
        candidates, attribution = blocker.get_candidates(
            name=s1_name,
            addr=s1_addr,
            max_total_candidates=100
        )

        n_cands = len(candidates)
        candidates_per_s1.append(n_cands)
        total_candidates_generated += n_cands

        # Match evaluation
        captured_pos = len(candidates & true_set)
        positives_per_s1.append(captured_pos)
        total_positives_captured += captured_pos
        total_negatives_generated += (n_cands - captured_pos)

        if len(true_set) > 0 and captured_pos == 0:
            s1_zero_positives_count += 1

        # Build pair records
        for cid in candidates:
            label = 1 if cid in true_set else 0
            cand_name, cand_addr = blocker.raw_store.get(cid, ("", ""))
            
            # Blocker attribution signals
            exact_core = 1 if cid in attribution.get("exact_core", set()) else 0
            compact_name = 1 if cid in attribution.get("compact_name", set()) else 0
            rare_name = 1 if cid in attribution.get("rare_name_tokens", set()) else 0
            exact_addr = 1 if cid in attribution.get("exact_address", set()) else 0
            rare_addr = 1 if cid in attribution.get("rare_address_tokens", set()) else 0
            n_agree = sum(1 for b in attribution if cid in attribution[b])

            pair_rows.append({
                "s1_id": s1_id,
                "s1_name": s1_name,
                "s1_addr": s1_addr,
                "cand_id": cid,
                "cand_name": cand_name,
                "cand_addr": cand_addr,
                "country": country,
                "is_exact_core": exact_core,
                "is_compact_name": compact_name,
                "is_rare_name": rare_name,
                "is_exact_addr": exact_addr,
                "is_rare_addr": rare_addr,
                "blocker_agreement_count": n_agree,
                "label": label
            })

    # Save to Parquet
    print(f"\nConverting {len(pair_rows):,} pairs to Polars DataFrame and saving...")
    pairs_df = pl.DataFrame(pair_rows)
    pairs_df.write_parquet(out_parquet)
    print(f"Saved: {out_parquet} ({os.path.getsize(out_parquet)/(1024*1024):.2f} MB)")

    # 4. Phase 2 Report Generation
    pos_rate = (total_positives_captured / total_candidates_generated * 100) if total_candidates_generated else 0.0
    recall_rate = (total_positives_captured / total_ground_truth_links * 100) if total_ground_truth_links else 0.0
    missed_gt = total_ground_truth_links - total_positives_captured

    non_singleton_queries = [cnt for cnt in positives_per_s1 if total_ground_truth_links > 0]
    
    print("\n" + "=" * 80)
    print(f"PHASE 2 REPORT: SUPERVISED TRAINING-PAIR DATASET ({country})")
    print("=" * 80)
    print(f"  Total S1 Entities Processed:      {len(train_df):,}")
    print(f"  Total Candidate Pairs Generated:  {total_candidates_generated:,}")
    print(f"  Positive Pairs (label = 1):       {total_positives_captured:,} ({pos_rate:.2f}%)")
    print(f"  Hard Negative Pairs (label = 0):  {total_negatives_generated:,} ({100 - pos_rate:.2f}%)")
    print(f"  Average Candidates / S1:          {np.mean(candidates_per_s1):.2f}")
    print(f"  Median (P50) Candidates / S1:     {np.median(candidates_per_s1):.1f}")
    print(f"  Average Positives / S1:           {np.mean(positives_per_s1):.2f}")
    print(f"  Total Ground Truth Links:         {total_ground_truth_links:,}")
    print(f"  Ground Truth Links Captured:      {total_positives_captured:,} ({recall_rate:.2f}%)")
    print(f"  Ground Truth Links Missed:        {missed_gt:,} ({100 - recall_rate:.2f}%)")
    print(f"  S1 Entities with 0 Positives:     {s1_zero_positives_count:,} ({s1_zero_positives_count/len(train_df)*100:.2f}%)")

    print("\nDistribution of Positives Captured per S1 Entity:")
    pos_counts = defaultdict(int)
    for p in positives_per_s1:
        pos_counts[p] += 1
    for k in sorted(pos_counts.keys())[:10]:
        print(f"    {k} positives captured: {pos_counts[k]:,} queries ({pos_counts[k]/len(train_df)*100:.2f}%)")

    print("=" * 80)
    print("PHASE 2 COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build training pairs from frozen blocker")
    parser.add_argument("--country", type=str, default="India", choices=["India", "US"])
    parser.add_argument("--sample-size", type=int, default=5000, help="Number of S1 training queries")
    args = parser.parse_args()

    build_training_pairs(country=args.country, sample_size=args.sample_size)
