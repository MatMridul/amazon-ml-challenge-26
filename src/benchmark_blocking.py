"""
ML Challenge 2026 — Comprehensive Candidate Generation Benchmark (Name + Address)

Produces:
A. Candidate-generation results (Recall@K, Entity Recall, Full Coverage for K=5, 10, 20, 30, 50, Uncapped)
B. Candidate cardinality distribution (Mean, Median, P95, P99, Max, Zero-candidate queries)
C. Multi-match coverage breakdown (0 matches, 1 match, 2-4 matches, 5+ matches)
D. Incremental blocker ablation (Name blockers -> +ExactAddress -> +RareAddressTokens)
E. Deterministic baseline Macro F_0.5
F. Top failure patterns with raw text of missed true matches
"""

import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import polars as pl
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from src.blocking import LexicalBlocker
from src.metrics import evaluate_predictions, compute_entity_f05
from src.normalization import normalize_name, normalize_address

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VAL_DIR = os.path.join(BASE_DIR, "dataset", "validation")
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "student_resource", "dataset", "train")


def load_validation_data(sample_size: int = None, country_filter: str = None):
    val_s1_path = os.path.join(VAL_DIR, "val_source1.tsv")
    val_gt_path = os.path.join(VAL_DIR, "val_ground_truth.tsv")

    s1_df = pl.read_csv(val_s1_path, separator="\t")
    gt_df = pl.read_csv(val_gt_path, separator="\t")

    df = s1_df.join(gt_df, left_on="entity_id", right_on="source1_entity_id", how="inner")
    
    if country_filter:
        df = df.filter(pl.col("country") == country_filter)

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, seed=42)

    return df


def load_candidate_pool(country: str, max_rows: int = None):
    print(f"Loading S2 and S3 candidate pool for country '{country}'...")
    s2_path = os.path.join(TRAIN_DIR, "train_source2.tsv")
    s3_path = os.path.join(TRAIN_DIR, "train_source3.tsv")

    s2_df = pl.read_csv(s2_path, separator="\t", n_rows=max_rows).filter(pl.col("country") == country)
    s3_df = pl.read_csv(s3_path, separator="\t", n_rows=max_rows).filter(pl.col("country") == country)

    pool = pl.concat([s2_df, s3_df])
    print(f"[{country}] Pool loaded: {len(pool):,} total records ({len(s2_df):,} S2 + {len(s3_df):,} S3)")
    return pool


def run_comprehensive_benchmark(sample_size: int = 1000, country: str = "India"):
    print("=" * 80)
    print(f"COMPREHENSIVE NAME + ADDRESS BLOCKING BENCHMARK — Country: {country} | Sample S1: {sample_size:,}")
    print("=" * 80)

    val_df = load_validation_data(sample_size=sample_size, country_filter=country)
    print(f"Loaded {len(val_df):,} validation queries for {country}.")

    ground_truth = {}
    query_meta = {}
    for row in val_df.iter_rows(named=True):
        eid = row["entity_id"]
        mids = row["matched_entity_ids"]
        ground_truth[eid] = set(mids.split(",")) if mids else set()
        query_meta[eid] = (row["business_name"] or "", row["business_address"] or "")

    # Load candidate pool and build index
    pool_df = load_candidate_pool(country=country)
    blocker = LexicalBlocker(country=country)
    blocker.index_records(pool_df)

    # Progressive stages to ablate
    stages = [
        ("1. ExactCore", {"exact_core"}),
        ("2. + CompactName", {"exact_core", "compact_name"}),
        ("3. + RareNameTokens (Prior Baseline)", {"exact_core", "compact_name", "rare_name_tokens"}),
        ("4. + ExactAddress", {"exact_core", "compact_name", "rare_name_tokens", "exact_address"}),
        ("5. Full Union (+ RareAddressTokens)", {"exact_core", "compact_name", "rare_name_tokens", "exact_address", "rare_address_tokens"}),
    ]

    print(f"\nQuerying blockers across {len(val_df):,} validation queries...")
    results = []

    for eid in tqdm(val_df["entity_id"].to_list(), desc="Evaluating queries"):
        q_name, q_addr = query_meta[eid]
        true_set = ground_truth[eid]

        # Stage candidates
        stage_cands = {}
        for sname, sblockers in stages:
            cands, _ = blocker.get_candidates(q_name, q_addr, enabled_blockers=sblockers)
            stage_cands[sname] = cands

        # Full union candidates with attribution
        full_cands, attr = blocker.get_candidates(q_name, q_addr, enabled_blockers=stages[-1][1])
        
        # Ranked candidates for top-k cuts
        scored = []
        for cid in full_cands:
            n_blocks = sum(1 for b in attr if cid in attr[b])
            is_exact = 1 if (cid in attr.get("exact_core", set()) or cid in attr.get("compact_name", set()) or cid in attr.get("exact_address", set())) else 0
            scored.append((cid, is_exact, n_blocks))
        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        ranked_cands = [x[0] for x in scored]

        results.append({
            "s1_id": eid,
            "name": q_name,
            "addr": q_addr,
            "true_set": true_set,
            "stage_cands": stage_cands,
            "full_cands": full_cands,
            "ranked_cands": ranked_cands,
            "attr": attr
        })

    # =========================================================================
    # A. CANDIDATE RECALL@K, FULL COVERAGE & CARDINALITY TRADEOFF
    # =========================================================================
    non_singleton_results = [r for r in results if len(r["true_set"]) > 0]
    total_true_links = sum(len(r["true_set"]) for r in non_singleton_results)

    print("\n" + "=" * 80)
    print("A. CANDIDATE RECALL & FULL COVERAGE vs. CARDINALITY (KNEE OF THE CURVE)")
    print("=" * 80)
    print(f"Total Queries: {len(results):,} | Non-singleton Queries: {len(non_singleton_results):,} | True Links: {total_true_links:,}")
    print(f"{'K':<5} | {'Avg Cands/Query':<16} | {'Macro Entity Recall':<22} | {'Link Recall (Micro)':<22} | {'Full Coverage':<15}")
    print("-" * 85)

    k_list = [5, 10, 20, 30, 50, None]
    for k in k_list:
        k_label = str(k) if k else "Uncapped"
        actual_cand_counts = []
        macro_recalls = []
        full_coverages = []
        links_captured = 0

        for r in results:
            cands = set(r["ranked_cands"][:k]) if k else r["full_cands"]
            actual_cand_counts.append(len(cands))
            
            true_set = r["true_set"]
            if len(true_set) > 0:
                matched = len(true_set & cands)
                links_captured += matched
                macro_recalls.append(matched / len(true_set))
                full_coverages.append(1.0 if true_set.issubset(cands) else 0.0)

        avg_cands = np.mean(actual_cand_counts)
        macro_rec = np.mean(macro_recalls) * 100 if macro_recalls else 0.0
        micro_rec = (links_captured / total_true_links) * 100 if total_true_links else 0.0
        full_cov = np.mean(full_coverages) * 100 if full_coverages else 0.0

        print(f"{k_label:<5} | {avg_cands:<16.2f} | {macro_rec:<21.2f}% | {micro_rec:<21.2f}% | {full_cov:<14.2f}%")

    # =========================================================================
    # B. CANDIDATE CARDINALITY DISTRIBUTION (UNCAPPED)
    # =========================================================================
    all_counts = [len(r["full_cands"]) for r in results]
    zero_cands = sum(1 for c in all_counts if c == 0)

    print("\n" + "=" * 80)
    print("B. CANDIDATE SET CARDINALITY DISTRIBUTION (UNCAPPED UNION)")
    print("=" * 80)
    print(f"  Average candidates/query:  {np.mean(all_counts):.2f}")
    print(f"  Median (P50) / query:      {np.median(all_counts):.1f}")
    print(f"  P90 candidates / query:    {np.percentile(all_counts, 90):.1f}")
    print(f"  P95 candidates / query:    {np.percentile(all_counts, 95):.1f}")
    print(f"  P99 candidates / query:    {np.percentile(all_counts, 99):.1f}")
    print(f"  Maximum candidates / query:{np.max(all_counts)}")
    print(f"  Queries with 0 candidates: {zero_cands} ({zero_cands / len(results) * 100:.2f}%)")

    # =========================================================================
    # C. MULTI-MATCH COVERAGE BREAKDOWN
    # =========================================================================
    print("\n" + "=" * 80)
    print("C. MULTI-MATCH COVERAGE BREAKDOWN (BY GROUND TRUTH MATCH COUNT)")
    print("=" * 80)
    
    buckets = [
        ("0 matches (Singletons)", lambda l: l == 0),
        ("1 match", lambda l: l == 1),
        ("2-4 matches", lambda l: 2 <= l <= 4),
        ("5+ matches", lambda l: l >= 5),
    ]

    for bname, cond in buckets:
        b_results = [r for r in results if cond(len(r["true_set"]))]
        if not b_results:
            continue
        
        b_counts = [len(r["full_cands"]) for r in b_results]
        if bname.startswith("0 matches"):
            print(f"  [{bname}] N = {len(b_results):,} queries")
            print(f"    Avg candidates generated: {np.mean(b_counts):.2f} (Median: {np.median(b_counts):.1f})")
            print(f"    Queries with 0 candidates (clean singletons): {sum(1 for c in b_counts if c == 0):,} ({sum(1 for c in b_counts if c == 0)/len(b_results)*100:.2f}%)")
        else:
            b_links = sum(len(r["true_set"]) for r in b_results)
            b_matched_links = sum(len(r["true_set"] & r["full_cands"]) for r in b_results)
            b_macro_rec = np.mean([len(r["true_set"] & r["full_cands"]) / len(r["true_set"]) for r in b_results]) * 100
            b_full_cov = np.mean([1.0 if r["true_set"].issubset(r["full_cands"]) else 0.0 for r in b_results]) * 100
            print(f"  [{bname}] N = {len(b_results):,} queries | Total True Links: {b_links:,}")
            print(f"    Avg candidates generated:  {np.mean(b_counts):.2f}")
            print(f"    Macro Entity Recall:       {b_macro_rec:.2f}%")
            print(f"    Total Link Recall (Micro): {b_matched_links / b_links * 100:.2f}%")
            print(f"    Full Coverage (100% true): {b_full_cov:.2f}%")

    # =========================================================================
    # D. INCREMENTAL BLOCKER ABLATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("D. INCREMENTAL BLOCKER ABLATION (MARGINAL RECALL vs. MARGINAL CANDIDATES)")
    print("=" * 80)
    print(f"{'Stage':<45} | {'Avg Cands':<10} | {'Macro Recall':<14} | {'Marginal Rec':<14} | {'Marginal Cands':<14}")
    print("-" * 105)

    prev_recall = 0.0
    prev_cands = 0.0

    for sname, _ in stages:
        stage_counts = [len(r["stage_cands"][sname]) for r in results]
        stage_recalls = [len(r["true_set"] & r["stage_cands"][sname]) / len(r["true_set"]) for r in non_singleton_results]
        
        cur_cands = np.mean(stage_counts)
        cur_recall = np.mean(stage_recalls) * 100 if stage_recalls else 0.0

        marg_recall = cur_recall - prev_recall
        marg_cands = cur_cands - prev_cands

        print(f"{sname:<45} | {cur_cands:<10.2f} | {cur_recall:<13.2f}% | {marg_recall:+12.2f}% | {marg_cands:+12.2f}")
        prev_recall = cur_recall
        prev_cands = cur_cands

    # =========================================================================
    # E. DETERMINISTIC BASELINE MATCHING (MACRO F_0.5)
    # =========================================================================
    print("\n" + "=" * 80)
    print("E. DETERMINISTIC BASELINE MATCHING BENCHMARK (MACRO F_0.5)")
    print("=" * 80)
    # Predict exact name match OR exact address match
    baseline_predictions = {}
    for r in results:
        eid = r["s1_id"]
        attr = r["attr"]
        exact_cands = (
            attr.get("exact_core", set()) | 
            attr.get("compact_name", set()) | 
            attr.get("exact_address", set())
        )
        baseline_predictions[eid] = exact_cands

    eval_res = evaluate_predictions(ground_truth, baseline_predictions)
    print(f"  Macro F_0.5 Score:      {eval_res['macro_f05']:.4f}")
    print(f"  Macro Precision:        {eval_res['macro_precision']:.4f}")
    print(f"  Macro Recall:           {eval_res['macro_recall']:.4f}")
    print(f"  Singleton Accuracy:     {eval_res['singleton_accuracy']*100:.2f}% ({eval_res['singletons']} singletons)")
    print(f"  Non-singleton F_0.5:    {eval_res['non_singleton_f05']:.4f}")

    # =========================================================================
    # F. TOP REMAINING FAILURE PATTERNS
    # =========================================================================
    print("\n" + "=" * 80)
    print("F. TOP REMAINING FAILURE PATTERNS — INSPECTION OF MISSED TRUE MATCHES")
    print("=" * 80)
    
    missed_cases = []
    for r in results:
        true_set = r["true_set"]
        if not true_set:
            continue
        missed = true_set - r["full_cands"]
        for mid in missed:
            s2_name, s2_addr = blocker.raw_store.get(mid, ("<Unknown>", "<Unknown>"))
            missed_cases.append({
                "s1_id": r["s1_id"],
                "s1_name": r["name"],
                "s1_addr": r["addr"],
                "s2_id": mid,
                "s2_name": s2_name,
                "s2_addr": s2_addr
            })

    print(f"Total Missed True Matches across {len(non_singleton_results):,} queries: {len(missed_cases):,} / {total_true_links:,} ({len(missed_cases)/total_true_links*100:.2f}%)\n")
    
    for idx, case in enumerate(missed_cases[:15], 1):
        print(f"Failure #{idx}:")
        print(f"  [S1 Query {case['s1_id']}]")
        print(f"    Name:    '{case['s1_name']}'")
        print(f"    Address: '{case['s1_addr']}'")
        print(f"  [Missed True Match {case['s2_id']}]")
        print(f"    Name:    '{case['s2_name']}'")
        print(f"    Address: '{case['s2_addr']}'")
        print()

    print("=" * 80)
    print("BENCHMARK COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive candidate generation benchmark")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of S1 validation queries")
    parser.add_argument("--country", type=str, default="India", choices=["India", "US"], help="Country to benchmark")
    args = parser.parse_args()

    run_comprehensive_benchmark(sample_size=args.sample_size, country=args.country)
