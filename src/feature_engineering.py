"""
ML Challenge 2026 — Phase 3: Pairwise Feature Engineering

Extracts high-signal similarity features using RapidFuzz (C++ accelerated):
- Multi-field Name similarities (Jaro-Winkler, Levenshtein, Token Sort, Token Set)
- Address similarities (Jaro-Winkler, Token Sort, Token Jaccard)
- Numeric token overlap and conflicts (house/office/street numbers)
- Blocker attribution and agreement count
- Missingness indicators (e.g. empty address)
"""

import os
import sys
import re
from typing import List, Dict, Any, Tuple
import numpy as np
import polars as pl
from tqdm import tqdm
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.normalization import normalize_name, normalize_address

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def compute_pairwise_features(pairs_df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes pairwise feature vectors for all pairs in pairs_df.
    Expected columns: s1_name, s1_addr, cand_id, cand_name, cand_addr,
                      is_exact_core, is_compact_name, is_rare_name,
                      is_exact_addr, is_rare_addr, blocker_agreement_count,
                      (and optionally label)
    """
    print(f"Extracting pairwise features for {len(pairs_df):,} pairs...")

    s1_names = pairs_df["s1_name"].to_list()
    s1_addrs = pairs_df["s1_addr"].to_list()
    c_ids = pairs_df["cand_id"].to_list()
    c_names = pairs_df["cand_name"].to_list()
    c_addrs = pairs_df["cand_addr"].to_list()

    # Pre-normalized fields for fast similarity
    print("Normalizing text fields in batches...")
    s1_norm_names, s1_cores, s1_compacts = [], [], []
    for n in tqdm(s1_names, desc="S1 Names", mininterval=2.0):
        norm, core, comp = normalize_name(n)
        s1_norm_names.append(norm)
        s1_cores.append(core)
        s1_compacts.append(comp)

    c_norm_names, c_cores, c_compacts = [], [], []
    for n in tqdm(c_names, desc="Cand Names", mininterval=2.0):
        norm, core, comp = normalize_name(n)
        c_norm_names.append(norm)
        c_cores.append(core)
        c_compacts.append(comp)

    s1_norm_addrs, s1_postals, s1_nums, s1_toks = [], [], [], []
    for a in tqdm(s1_addrs, desc="S1 Addresses", mininterval=2.0):
        norm, postal, nums, toks = normalize_address(a)
        s1_norm_addrs.append(norm)
        s1_postals.append(postal)
        s1_nums.append(set(nums))
        s1_toks.append(set(toks))

    c_norm_addrs, c_postals, c_nums, c_toks = [], [], [], []
    for a in tqdm(c_addrs, desc="Cand Addresses", mininterval=2.0):
        norm, postal, nums, toks = normalize_address(a)
        c_norm_addrs.append(norm)
        c_postals.append(postal)
        c_nums.append(set(nums))
        c_toks.append(set(toks))

    # Feature lists
    f_name_jw = []
    f_name_lev = []
    f_name_token_sort = []
    f_name_token_set = []
    f_core_jw = []
    f_core_exact = []
    f_compact_exact = []
    f_name_len_diff = []

    f_addr_jw = []
    f_addr_token_sort = []
    f_addr_token_jaccard = []
    f_addr_exact = []
    f_cand_addr_empty = []
    f_num_overlap_count = []
    f_num_jaccard = []
    f_num_conflict = []

    f_is_s2 = []

    print("Computing string similarity metrics with RapidFuzz...")
    n_pairs = len(pairs_df)
    for i in tqdm(range(n_pairs), desc="Pairwise metrics", mininterval=2.0):
        s1_n = s1_norm_names[i]
        c_n = c_norm_names[i]
        s1_c = s1_cores[i]
        c_c = c_cores[i]
        s1_comp = s1_compacts[i]
        c_comp = c_compacts[i]

        s1_a = s1_norm_addrs[i]
        c_a = c_norm_addrs[i]
        s1_nm = s1_nums[i]
        c_nm = c_nums[i]
        s1_tk = s1_toks[i]
        c_tk = c_toks[i]

        cid = c_ids[i]

        # Name metrics
        f_name_jw.append(JaroWinkler.similarity(s1_n, c_n))
        f_name_lev.append(Levenshtein.normalized_similarity(s1_n, c_n))
        f_name_token_sort.append(fuzz.token_sort_ratio(s1_n, c_n) / 100.0)
        f_name_token_set.append(fuzz.token_set_ratio(s1_n, c_n) / 100.0)
        f_core_jw.append(JaroWinkler.similarity(s1_c, c_c))
        f_core_exact.append(1 if (s1_c and s1_c == c_c) else 0)
        f_compact_exact.append(1 if (s1_comp and s1_comp == c_comp) else 0)
        
        max_nlen = max(len(s1_n), len(c_n), 1)
        f_name_len_diff.append(abs(len(s1_n) - len(c_n)) / max_nlen)

        # Address metrics
        is_empty = 1 if not c_a.strip() else 0
        f_cand_addr_empty.append(is_empty)

        if not is_empty and s1_a:
            f_addr_jw.append(JaroWinkler.similarity(s1_a, c_a))
            f_addr_token_sort.append(fuzz.token_sort_ratio(s1_a, c_a) / 100.0)
            
            # Token Jaccard
            u = s1_tk | c_tk
            f_addr_token_jaccard.append(len(s1_tk & c_tk) / len(u) if u else 0.0)
            f_addr_exact.append(1 if (len(s1_a) >= 15 and s1_a == c_a) else 0)
            
            # Numeric overlap
            overlap_nm = s1_nm & c_nm
            union_nm = s1_nm | c_nm
            f_num_overlap_count.append(len(overlap_nm))
            f_num_jaccard.append(len(overlap_nm) / len(union_nm) if union_nm else 0.0)
            # Conflict: both have distinct numbers, but zero overlap
            f_num_conflict.append(1 if (s1_nm and c_nm and not overlap_nm) else 0)
        else:
            f_addr_jw.append(0.0)
            f_addr_token_sort.append(0.0)
            f_addr_token_jaccard.append(0.0)
            f_addr_exact.append(0)
            f_num_overlap_count.append(0)
            f_num_jaccard.append(0.0)
            f_num_conflict.append(0)

        # Metadata
        f_is_s2.append(1 if cid.startswith("S2-") else 0)

    # Attach computed features
    features_dict = {
        "name_jw": f_name_jw,
        "name_lev": f_name_lev,
        "name_token_sort": f_name_token_sort,
        "name_token_set": f_name_token_set,
        "core_jw": f_core_jw,
        "core_exact": f_core_exact,
        "compact_exact": f_compact_exact,
        "name_len_diff": f_name_len_diff,
        "addr_jw": f_addr_jw,
        "addr_token_sort": f_addr_token_sort,
        "addr_token_jaccard": f_addr_token_jaccard,
        "addr_exact": f_addr_exact,
        "cand_addr_empty": f_cand_addr_empty,
        "num_overlap_count": f_num_overlap_count,
        "num_jaccard": f_num_jaccard,
        "num_conflict": f_num_conflict,
        "is_s2": f_is_s2,
    }

    feat_df = pairs_df.with_columns([
        pl.Series(k, v) for k, v in features_dict.items()
    ])
    
    print(f"Feature engineering complete. Total feature columns: {len(features_dict) + 6}")
    return feat_df


FEATURE_COLUMNS = [
    "name_jw", "name_lev", "name_token_sort", "name_token_set",
    "core_jw", "core_exact", "compact_exact", "name_len_diff",
    "addr_jw", "addr_token_sort", "addr_token_jaccard", "addr_exact",
    "cand_addr_empty", "num_overlap_count", "num_jaccard", "num_conflict",
    "is_exact_core", "is_compact_name", "is_rare_name",
    "is_exact_addr", "is_rare_addr", "blocker_agreement_count", "is_s2"
]


if __name__ == "__main__":
    # Unit test on sample pair
    test_df = pl.DataFrame([{
        "s1_name": "Apex India Private Limited",
        "s1_addr": "Morcha Road, Patna City, Patna, Bihar",
        "cand_id": "S2-12345",
        "cand_name": "Apex India Private Ltd",
        "cand_addr": "MORCHA ROAD, PATNA CITY, Bihar",
        "is_exact_core": 1,
        "is_compact_name": 0,
        "is_rare_name": 1,
        "is_exact_addr": 0,
        "is_rare_addr": 1,
        "blocker_agreement_count": 3,
        "label": 1
    }])
    res = compute_pairwise_features(test_df)
    print("Test features extracted:")
    for col in FEATURE_COLUMNS:
        print(f"  {col:25s}: {res[col][0]}")
