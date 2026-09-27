"""
ML Challenge 2026 — Phase 12: High-Precision Candidate Recovery & F0.5 Booster (V2)

Leverages the audited candidate_pairs.tsv to recover high-confidence true matches
that were dropped by conservative relative-margin thresholding:
1. Exact Core Name + Address Locality recovery
2. Exact Address + Core Name Similarity recovery
3. High Dual-Similarity recovery (Name JW >= 0.88 AND Addr JW >= 0.80)
4. False-Singleton Recovery for high-confidence candidates
"""

import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Set
import polars as pl
from rapidfuzz.distance import JaroWinkler
from rapidfuzz import fuzz
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.normalization import normalize_name, normalize_address


def run_boosted_recovery(
    test_dir: str,
    matching_in_path: str,
    candidate_in_path: str,
    output_path: str
):
    print("=" * 80)
    print("PHASE 12: HIGH-PRECISION CANDIDATE RECOVERY (V2 BOOSTER)")
    print(f"Test Directory:      {test_dir}")
    print(f"Input Matches:       {matching_in_path}")
    print(f"Input Candidates:    {candidate_in_path}")
    print(f"Output Matches:      {output_path}")
    print("=" * 80)

    # 1. Load test Source 1, 2, 3
    print("Loading test data files...")
    t0 = time.time()
    s1_df = pl.read_csv(os.path.join(test_dir, "test_source1.tsv"), separator="\t")
    s2_df = pl.read_csv(os.path.join(test_dir, "test_source2.tsv"), separator="\t")
    s3_df = pl.read_csv(os.path.join(test_dir, "test_source3.tsv"), separator="\t")
    print(f"Loaded all test files in {time.time() - t0:.1f}s.")

    # Build lookup dictionaries for fast normalization
    print("Indexing S1 text...")
    s1_data = {}
    for r in tqdm(s1_df.iter_rows(named=True), total=len(s1_df), desc="S1 text"):
        s1_id = r["entity_id"]
        n_raw = (r["business_name"] or "").strip()
        a_raw = (r["business_address"] or "").strip()
        norm_n, core_n, comp_n = normalize_name(n_raw)
        norm_a, postal, nums, toks = normalize_address(a_raw)
        s1_data[s1_id] = (norm_n, core_n, norm_a, postal, set(nums))

    print("Indexing S2 and S3 candidate text...")
    cand_data = {}
    for r in tqdm(s2_df.iter_rows(named=True), total=len(s2_df), desc="S2 text"):
        cid = r["entity_id"]
        n_raw = (r["business_name"] or "").strip()
        a_raw = (r["business_address"] or "").strip()
        norm_n, core_n, comp_n = normalize_name(n_raw)
        norm_a, postal, nums, toks = normalize_address(a_raw)
        cand_data[cid] = (norm_n, core_n, norm_a, postal, set(nums))

    for r in tqdm(s3_df.iter_rows(named=True), total=len(s3_df), desc="S3 text"):
        cid = r["entity_id"]
        n_raw = (r["business_name"] or "").strip()
        a_raw = (r["business_address"] or "").strip()
        norm_n, core_n, comp_n = normalize_name(n_raw)
        norm_a, postal, nums, toks = normalize_address(a_raw)
        cand_data[cid] = (norm_n, core_n, norm_a, postal, set(nums))

    # 2. Read current matches
    print(f"Reading existing matches from {matching_in_path}...")
    existing_matches = {}
    with open(matching_in_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.rstrip("\r\n")
            parts = line.split("\t")
            s1_id = parts[0]
            m_list = parts[1].split(",") if len(parts) > 1 and parts[1] else []
            existing_matches[s1_id] = m_list

    # 3. Read candidates and apply recovery
    print(f"Reading candidates from {candidate_in_path} and applying recovery...")
    recovered_matches = defaultdict(list)
    recovered_count = 0
    false_singleton_count = 0

    with open(candidate_in_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in tqdm(f, total=len(s1_df), desc="Recovering"):
            line = line.rstrip("\r\n")
            parts = line.split("\t")
            s1_id = parts[0]
            cand_list = parts[1].split(",") if len(parts) > 1 and parts[1] else []

            current = existing_matches.get(s1_id, [])
            current_set = set(current)
            s1_tuple = s1_data.get(s1_id)
            if not s1_tuple or not cand_list:
                recovered_matches[s1_id] = current
                continue

            s1_n, s1_core, s1_a, s1_post, s1_nums = s1_tuple
            expanded = list(current)

            for cid in cand_list:
                if cid in current_set:
                    continue
                c_tuple = cand_data.get(cid)
                if not c_tuple:
                    continue
                c_n, c_core, c_a, c_post, c_nums = c_tuple

                # Postal/Number conflict check
                num_conflict = 0
                if s1_nums and c_nums and not (s1_nums & c_nums):
                    # Conflicting building numbers
                    num_conflict = 1

                if num_conflict:
                    continue

                # Similarities
                njw = JaroWinkler.similarity(s1_n, c_n)
                cjw = JaroWinkler.similarity(s1_core, c_core) if (s1_core and c_core) else 0.0
                ts = fuzz.token_set_ratio(s1_n, c_n) / 100.0
                ajw = JaroWinkler.similarity(s1_a, c_a) if (s1_a and c_a) else 0.0
                exact_core = 1 if (s1_core and s1_core == c_core) else 0
                exact_addr = 1 if (s1_a and s1_a == c_a) else 0

                # High-confidence recovery criteria
                is_match = False

                # Case A: Exact Address match + decent name similarity
                if exact_addr and (njw >= 0.75 or ts >= 0.75):
                    is_match = True

                # Case B: Exact Core Name + strong address locality (or exact postal)
                elif exact_core and (ajw >= 0.70 or (s1_post and s1_post == c_post)):
                    is_match = True

                # Case C: High Dual-Similarity (Name >= 0.88 AND Address >= 0.80)
                elif njw >= 0.88 and ajw >= 0.80:
                    is_match = True

                # Case D: Near-identical Name (JW >= 0.95) with reasonable address support
                elif njw >= 0.95 and ajw >= 0.65:
                    is_match = True

                if is_match:
                    expanded.append(cid)
                    current_set.add(cid)
                    recovered_count += 1
                    if not current:
                        false_singleton_count += 1

            recovered_matches[s1_id] = expanded

    print(f"\nRecovery Summary:")
    print(f"  - Total additional true matches recovered: {recovered_count:,}")
    print(f"  - False singletons converted to matches:    {false_singleton_count:,}")

    # 4. Write new matching results file preserving exact S1 order
    print(f"Writing updated matches to {output_path}...")
    s1_all_ids = s1_df["entity_id"].to_list()
    with open(output_path, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_all_ids:
            matches = recovered_matches.get(s1_id, [])
            f_out.write(f"{s1_id}\t{','.join(matches)}\n")

    print(f"V2 Boosted file generated: {output_path} ({os.path.getsize(output_path)/(1024*1024):.2f} MB)")
    print("=" * 80)


if __name__ == "__main__":
    test_d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, "dataset", "student_resource", "dataset", "test")
    m_in = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE_DIR, "output", "matching_results_v1_baseline_0.816.tsv")
    c_in = sys.argv[3] if len(sys.argv) > 3 else os.path.join(BASE_DIR, "output", "candidate_pairs.tsv")
    m_out = sys.argv[4] if len(sys.argv) > 4 else os.path.join(BASE_DIR, "output", "matching_results_v2_boosted.tsv")

    run_boosted_recovery(test_d, m_in, c_in, m_out)
