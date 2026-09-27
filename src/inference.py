"""
ML Challenge 2026 — Phase 11: End-to-End Test Inference Pipeline

Generates official submission files:
1. output/matching_results.tsv (scored on leaderboard)
2. output/candidate_pairs.tsv (candidate generation set for audit)

Features:
- Partitioned country execution (India, US, France)
- Memory-efficient streaming batch processing
- RapidFuzz C++ SIMD feature extraction
- LightGBM calibrated inference
- Preserves exact Source 1 test ordering
"""

import os
import sys
import time
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import polars as pl
import lightgbm as lgb
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.blocking import LexicalBlocker
from src.normalization import normalize_name, normalize_address
from src.feature_engineering import compute_pairwise_features, FEATURE_COLUMNS

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_inference(
    test_dir: str,
    output_dir: str,
    model_path: str,
    threshold: float = 0.60,
    batch_size: int = 25000,
    max_queries_per_country: int = None
):
    os.makedirs(output_dir, exist_ok=True)
    matching_out_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_out_path = os.path.join(output_dir, "candidate_pairs.tsv")

    print("=" * 80)
    print("PHASE 11: FULL TEST SET INFERENCE PIPELINE")
    print(f"Test Directory:     {test_dir}")
    print(f"Output Directory:   {output_dir}")
    print(f"Model Path:         {model_path}")
    print(f"Decision Threshold: {threshold:.2f}")
    print("=" * 80)

    # 1. Load LightGBM model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    print(f"Loading LightGBM model from {model_path}...")
    model = lgb.Booster(model_file=model_path)
    print(f"Model loaded successfully ({model.num_trees()} trees).")

    # 2. Read test_source1 to know required queries and ordering
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    print(f"Loading test queries from {s1_path}...")
    s1_df = pl.read_csv(s1_path, separator="\t")
    print(f"Total Source 1 test queries: {len(s1_df):,}")

    unique_countries = s1_df["country"].unique().to_list()
    print(f"Countries present in test set: {unique_countries}")

    # Load S2 and S3 candidate records
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")
    print(f"Loading test candidate records from S2 and S3...")
    t0 = time.time()
    s2_df = pl.read_csv(s2_path, separator="\t")
    s3_df = pl.read_csv(s3_path, separator="\t")
    print(f"Loaded {len(s2_df):,} S2 records and {len(s3_df):,} S3 records in {time.time() - t0:.1f}s.")

    # Result stores
    final_candidates: Dict[str, List[str]] = {}
    final_matches: Dict[str, List[str]] = {}

    for country in unique_countries:
        country_s1 = s1_df.filter(pl.col("country") == country)
        if max_queries_per_country:
            country_s1 = country_s1.head(max_queries_per_country)
        print("\n" + "-" * 60)
        print(f"PROCESSING COUNTRY: '{country}' ({len(country_s1):,} S1 queries)")
        print("-" * 60)

        # Filter candidate pool for this country
        t_c = time.time()
        c_s2 = s2_df.filter(pl.col("country") == country)
        c_s3 = s3_df.filter(pl.col("country") == country)
        country_pool = pl.concat([c_s2, c_s3])
        print(f"[{country}] Candidate pool size: {len(country_pool):,} records ({len(c_s2):,} S2 + {len(c_s3):,} S3)")

        # Build Inverted Index Blocker
        print(f"[{country}] Building Lexical Blocker inverted indices...")
        blocker = LexicalBlocker(country)
        blocker.index_records(country_pool)
        print(f"[{country}] Inverted index built in {time.time() - t_c:.1f}s.")

        # Process S1 queries in streaming batches
        n_queries = len(country_s1)
        for start_idx in range(0, n_queries, batch_size):
            end_idx = min(start_idx + batch_size, n_queries)
            batch_s1 = country_s1.slice(start_idx, end_idx - start_idx)
            print(f"\n[{country}] Processing batch {start_idx:,} to {end_idx:,} ({len(batch_s1):,} queries)...")

            # A. Generate candidates for batch
            batch_pair_rows = []

            t_block = time.time()
            for row in batch_s1.iter_rows(named=True):
                s1_id = row["entity_id"]
                s1_name = row["business_name"] or ""
                s1_addr = row["business_address"] or ""
                candidates, attribution = blocker.get_candidates(
                    name=s1_name,
                    addr=s1_addr,
                    max_total_candidates=100
                )
                cand_list = list(candidates)
                final_candidates[s1_id] = cand_list

                if not cand_list:
                    final_matches[s1_id] = []
                    continue

                for cid in cand_list:
                    cand_name, cand_addr = blocker.raw_store.get(cid, ("", ""))
                    exact_core = 1 if cid in attribution.get("exact_core", set()) else 0
                    compact_name = 1 if cid in attribution.get("compact_name", set()) else 0
                    rare_name = 1 if cid in attribution.get("rare_name_tokens", set()) else 0
                    exact_addr = 1 if cid in attribution.get("exact_address", set()) else 0
                    rare_addr = 1 if cid in attribution.get("rare_address_tokens", set()) else 0
                    n_agree = sum(1 for b in attribution if cid in attribution[b])

                    batch_pair_rows.append({
                        "s1_id": s1_id,
                        "s1_name": s1_name,
                        "s1_addr": s1_addr,
                        "cand_id": cid,
                        "cand_name": cand_name,
                        "cand_addr": cand_addr,
                        "is_exact_core": exact_core,
                        "is_compact_name": compact_name,
                        "is_rare_name": rare_name,
                        "is_exact_addr": exact_addr,
                        "is_rare_addr": rare_addr,
                        "blocker_agreement_count": n_agree,
                        "label": 0  # Dummy for inference
                    })

            print(f"[{country}] Blocking generated {len(batch_pair_rows):,} candidate pairs in {time.time() - t_block:.1f}s.")

            if not batch_pair_rows:
                continue

            # B. Compute features
            t_feat = time.time()
            pair_df = pl.DataFrame(batch_pair_rows)
            feat_df = compute_pairwise_features(pair_df)
            X = feat_df.select(FEATURE_COLUMNS).to_numpy()
            print(f"[{country}] Extracted features for {len(X):,} pairs in {time.time() - t_feat:.1f}s.")

            # C. Score with LightGBM
            t_pred = time.time()
            preds = model.predict(X)
            print(f"[{country}] Scored pairs in {time.time() - t_pred:.1f}s.")

            # D. Apply Decision Threshold
            s1_preds = defaultdict(list)
            for row_dict, prob in zip(batch_pair_rows, preds):
                s1_id = row_dict["s1_id"]
                cand_id = row_dict["cand_id"]
                if prob >= threshold:
                    s1_preds[s1_id].append(cand_id)

            for row in batch_s1.iter_rows(named=True):
                s1_id = row["entity_id"]
                if s1_id not in final_matches:
                    final_matches[s1_id] = s1_preds.get(s1_id, [])

    # 3. Write output files in exact order of test_source1.tsv
    print("\n" + "=" * 80)
    print("WRITING SUBMISSION FILES IN REQUIRED FORMAT")
    print("=" * 80)

    total_test_rows = len(s1_df) if not max_queries_per_country else len(final_matches)
    s1_all_ids = s1_df["entity_id"].to_list()
    if max_queries_per_country:
        s1_all_ids = [eid for eid in s1_all_ids if eid in final_matches]

    print(f"Writing {len(s1_all_ids):,} rows to {matching_out_path}...")
    with open(matching_out_path, "w", encoding="utf-8") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_all_ids:
            matches = final_matches.get(s1_id, [])
            match_str = ",".join(matches)
            f_match.write(f"{s1_id}\t{match_str}\n")

    print(f"Writing {len(s1_all_ids):,} rows to {candidate_out_path}...")
    with open(candidate_out_path, "w", encoding="utf-8") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_all_ids:
            cands = final_candidates.get(s1_id, [])
            cand_str = ",".join(cands)
            f_cand.write(f"{s1_id}\tcand_str\n" if False else f"{s1_id}\t{cand_str}\n")

    print(f"\nFiles generated successfully:")
    print(f"  - {matching_out_path} ({os.path.getsize(matching_out_path)/(1024*1024):.2f} MB)")
    print(f"  - {candidate_out_path} ({os.path.getsize(candidate_out_path)/(1024*1024):.2f} MB)")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Test Inference Pipeline")
    parser.add_argument("--test-dir", type=str, default=os.path.join(BASE_DIR, "dataset", "student_resource", "dataset", "test"))
    parser.add_argument("--output-dir", type=str, default=os.path.join(BASE_DIR, "output"))
    parser.add_argument("--model-path", type=str, default=os.path.join(BASE_DIR, "models", "lightgbm_matcher_v1.txt"))
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--batch-size", type=int, default=25000)
    parser.add_argument("--max-queries-per-country", type=int, default=None, help="Limit for testing")
    args = parser.parse_args()

    run_inference(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        threshold=args.threshold,
        batch_size=args.batch_size,
        max_queries_per_country=args.max_queries_per_country
    )
