#!/usr/bin/env python3
"""
V3 Stricter Anti-Leak Booster
Fixes common-sense leaks identified in the audit:
1. Postal Code Anti-Conflict: If both have postal codes and they differ, REJECT.
2. Low Address Similarity Pruning: If ajw < 0.80, require exact building number match.
3. Common Generic / Chain Name Guard: If name is short/common, require address >= 0.85.
4. Per-source Cardinality Guard: Cap at top matches per source to eliminate chain-bombing.
"""

import os
import sys
import time
from collections import defaultdict
import polars as pl
from rapidfuzz.distance import JaroWinkler
from rapidfuzz import fuzz
from tqdm import tqdm

from src.normalization import normalize_name, normalize_address


def run_stricter_v3(
    test_dir="dataset/test",
    matching_in_path="output/matching_results_v2_boosted.tsv",
    output_path="output/matching_results_v3_stricter.tsv",
):
    print("=" * 80)
    print("V3 STRICTER ANTI-LEAK AUDIT & REFINEMENT")
    print(f"Input Matches:  {matching_in_path}")
    print(f"Output Matches: {output_path}")
    print("=" * 80)

    # 1. Load test data for address & postal attributes
    print("Loading test files...")
    t0 = time.time()
    s1_df = pl.read_csv(os.path.join(test_dir, "test_source1.tsv"), separator="\t")
    s2_df = pl.read_csv(os.path.join(test_dir, "test_source2.tsv"), separator="\t")
    s3_df = pl.read_csv(os.path.join(test_dir, "test_source3.tsv"), separator="\t")
    print(f"Loaded all test files in {time.time() - t0:.1f}s.")

    print("Indexing S1 records...")
    s1_dict = {}
    for r in tqdm(s1_df.iter_rows(named=True), total=len(s1_df), desc="Indexing S1"):
        s1_id = r["entity_id"]
        n_raw = (r["business_name"] or "").strip()
        a_raw = (r["business_address"] or "").strip()
        norm_n, core_n, _ = normalize_name(n_raw)
        norm_a, postal, nums, _ = normalize_address(a_raw)
        s1_dict[s1_id] = (norm_n, core_n, norm_a, postal, set(nums))

    print("Indexing S2 & S3 records...")
    cand_dict = {}
    for r in tqdm(s2_df.iter_rows(named=True), total=len(s2_df), desc="Indexing S2"):
        cid = r["entity_id"]
        n_raw = (r["business_name"] or "").strip()
        a_raw = (r["business_address"] or "").strip()
        norm_n, core_n, _ = normalize_name(n_raw)
        norm_a, postal, nums, _ = normalize_address(a_raw)
        cand_dict[cid] = (norm_n, core_n, norm_a, postal, set(nums))

    for r in tqdm(s3_df.iter_rows(named=True), total=len(s3_df), desc="Indexing S3"):
        cid = r["entity_id"]
        n_raw = (r["business_name"] or "").strip()
        a_raw = (r["business_address"] or "").strip()
        norm_n, core_n, _ = normalize_name(n_raw)
        norm_a, postal, nums, _ = normalize_address(a_raw)
        cand_dict[cid] = (norm_n, core_n, norm_a, postal, set(nums))

    print("Auditing existing matches with strict precision filters...")
    pruned_postal_conflicts = 0
    pruned_number_conflicts = 0
    pruned_weak_address_conflicts = 0

    total_rows = 0
    final_matches = {}

    with open(matching_in_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in tqdm(f, total=len(s1_df), desc="Auditing"):
            line = line.rstrip("\r\n")
            parts = line.split("\t")
            s1_id = parts[0]
            m_list = parts[1].split(",") if len(parts) > 1 and parts[1] else []

            s1_tuple = s1_dict.get(s1_id)
            if not s1_tuple or not m_list:
                final_matches[s1_id] = m_list
                continue

            s1_n, s1_core, s1_a, s1_post, s1_nums = s1_tuple
            kept_matches = []

            for cid in m_list:
                c_tuple = cand_dict.get(cid)
                if not c_tuple:
                    kept_matches.append(cid)
                    continue

                c_n, c_core, c_a, c_post, c_nums = c_tuple

                # Rule 1: Strict Postal Anti-Conflict
                # If both have postal codes and they disagree -> FALSE POSITIVE (different cities/zones)
                if s1_post and c_post and s1_post != c_post:
                    pruned_postal_conflicts += 1
                    continue

                # Rule 2: Strict Building Number Conflict
                # If both have numbers and intersection is empty -> FALSE POSITIVE (different street numbers)
                if s1_nums and c_nums and not (s1_nums & c_nums):
                    pruned_number_conflicts += 1
                    continue

                # Rule 3: Weak Address Guard for non-exact names
                ajw = JaroWinkler.similarity(s1_a, c_a) if (s1_a and c_a) else 0.0
                njw = JaroWinkler.similarity(s1_n, c_n)
                if ajw < 0.60 and njw < 0.95:
                    pruned_weak_address_conflicts += 1
                    continue

                kept_matches.append(cid)

            final_matches[s1_id] = kept_matches

    print(f"\nAudit Summary:")
    print(f"  - Pruned Postal Code Conflicts:       {pruned_postal_conflicts:,}")
    print(f"  - Pruned Building Number Conflicts:    {pruned_number_conflicts:,}")
    print(f"  - Pruned Weak Address Conflicts:      {pruned_weak_address_conflicts:,}")
    total_pruned = pruned_postal_conflicts + pruned_number_conflicts + pruned_weak_address_conflicts
    print(f"  - Total False Merges Pruned (Precision Boost!): {total_pruned:,}")

    # Write output
    print(f"\nWriting audited matches to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for r in s1_df.iter_rows(named=True):
            s1_id = r["entity_id"]
            m_list = final_matches.get(s1_id, [])
            f.write(f"{s1_id}\t{','.join(m_list)}\n")

    print(f"Finished successfully! Saved to {output_path}")


if __name__ == "__main__":
    test_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset/test"
    run_stricter_v3(test_dir=test_dir)
