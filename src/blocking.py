"""
ML Challenge 2026 — Modular Lexical & Address Blocking System

Implements independent, composable candidate generators:
1. Exact Core Name Blocker
2. Compact Alphanumeric Name Blocker
3. Rare Name Token Inverted Index Blocker
4. Exact Normalized Address Blocker
5. Rare Address Token + Numeric Overlap Blocker
6. Postal Code + First Name Token Blocker

All blockers operate strictly within country partitions with strict cardinality safeguards.
"""

import os
import sys
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import polars as pl
from tqdm import tqdm

from src.normalization import (
    normalize_name,
    normalize_address,
    get_character_ngrams,
    clean_text,
    extract_identifiers,
    extract_name_2grams
)

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class LexicalBlocker:
    """
    In-memory modular lexical & address blocking index for a single country.
    """
    def __init__(self, country: str):
        self.country = country
        
        # Name index tables
        self.exact_core_index: Dict[str, List[str]] = defaultdict(list)
        self.compact_name_index: Dict[str, List[str]] = defaultdict(list)
        self.rare_name_token_index: Dict[str, List[str]] = defaultdict(list)
        self.postal_first_token_index: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.name_2gram_index: Dict[str, List[str]] = defaultdict(list)
        
        # Address index tables
        self.exact_address_index: Dict[str, List[str]] = defaultdict(list)
        self.rare_addr_token_index: Dict[str, List[str]] = defaultdict(list)
        self.phone_index: Dict[str, List[str]] = defaultdict(list)
        self.corp_id_index: Dict[str, List[str]] = defaultdict(list)
        
        # Token frequency maps for rare-token filtering
        self.name_token_freq: Dict[str, int] = defaultdict(int)
        self.addr_token_freq: Dict[str, int] = defaultdict(int)
        
        # Record metadata store for fast scoring
        # entity_id -> (norm_name, core_name, norm_addr, postal_code, num_tokens_set)
        self.record_meta: Dict[str, Tuple[str, str, str, Optional[str], Set[str]]] = {}
        
        # Raw text store for failure analysis
        # entity_id -> (raw_name, raw_address)
        self.raw_store: Dict[str, Tuple[str, str]] = {}
        
        self.total_records = 0

    def index_records(
        self,
        df: pl.DataFrame,
        max_name_token_freq: int = 500,
        max_addr_token_freq: int = 300
    ):
        """
        Populate index from a Polars DataFrame of S2/S3 records for this country.
        """
        print(f"[{self.country}] Indexing {len(df):,} records...")
        
        entity_ids = df["entity_id"].to_list()
        names = df["business_name"].to_list()
        addresses = df["business_address"].to_list()
        
        # Pass 1: compute token frequencies for rare-token filtering
        print(f"[{self.country}] Pass 1: Computing name & address token frequencies...")
        for name, addr in tqdm(zip(names, addresses), total=len(names), desc="Frequencies", mininterval=2.0):
            if name:
                _, core, _ = normalize_name(name)
                for t in set(core.split()):
                    if len(t) >= 3:
                        self.name_token_freq[t] += 1
            if addr:
                _, _, _, addr_tokens = normalize_address(addr, self.country)
                for t in set(addr_tokens):
                    if len(t) >= 4:
                        self.addr_token_freq[t] += 1

        # Pass 2: build inverted indexes
        print(f"[{self.country}] Pass 2: Building inverted index tables...")
        for eid, name, addr in tqdm(zip(entity_ids, names, addresses), total=len(entity_ids), desc="Indexing", mininterval=2.0):
            norm_name, core_name, compact_name = normalize_name(name)
            norm_addr, postal, nums, addr_tokens = normalize_address(addr, self.country)
            
            # Store metadata
            nums_set = set(nums)
            self.record_meta[eid] = (norm_name, core_name, norm_addr, postal, nums_set)
            self.raw_store[eid] = (name or "", addr or "")
            
            # 1. Exact core name index
            if core_name and len(core_name) >= 3:
                self.exact_core_index[core_name].append(eid)
                
            # 2. Compact alphanumeric name index
            if compact_name and len(compact_name) >= 4:
                self.compact_name_index[compact_name].append(eid)
                
            # 3. Rare name token index
            name_tokens = core_name.split() if core_name else []
            for tok in name_tokens:
                if len(tok) >= 4 and self.name_token_freq.get(tok, 0) <= max_name_token_freq:
                    self.rare_name_token_index[tok].append(eid)

            # 4. Postal + First Token index
            first_tok = name_tokens[0] if name_tokens else None
            if postal and first_tok and len(first_tok) >= 3:
                self.postal_first_token_index[(postal, first_tok)].append(eid)

            # 5. Exact normalized address index
            if norm_addr and len(norm_addr) >= 15:
                self.exact_address_index[norm_addr].append(eid)

            # 6. Rare address token index
            for atok in set(addr_tokens):
                if len(atok) >= 4 and self.addr_token_freq.get(atok, 0) <= max_addr_token_freq:
                    self.rare_addr_token_index[atok].append(eid)

            # 7. Name 2-grams index
            for g in extract_name_2grams(core_name):
                self.name_2gram_index[g].append(eid)

            # 8. Phones & Corporate IDs
            if addr:
                phones, corp_ids = extract_identifiers(addr, self.country)
                for p in phones:
                    self.phone_index[p].append(eid)
                for cid in corp_ids:
                    self.corp_id_index[cid].append(eid)

            self.total_records += 1

        print(f"[{self.country}] Indexed {self.total_records:,} records successfully.")
        print(f"  Exact core names:    {len(self.exact_core_index):,} unique keys")
        print(f"  Compact names:       {len(self.compact_name_index):,} unique keys")
        print(f"  Rare name tokens:    {len(self.rare_name_token_index):,} unique keys")
        print(f"  Name 2-grams:        {len(self.name_2gram_index):,} unique keys")
        print(f"  Exact addresses:     {len(self.exact_address_index):,} unique keys")
        print(f"  Rare address tokens: {len(self.rare_addr_token_index):,} unique keys")
        print(f"  Phone numbers:       {len(self.phone_index):,} unique keys")
        print(f"  Corporate IDs:       {len(self.corp_id_index):,} unique keys")

    def get_candidates(
        self,
        name: str,
        addr: str,
        enabled_blockers: Optional[Set[str]] = None,
        max_rare_token_matches: int = 15,
        max_address_matches: int = 15,
        max_total_candidates: Optional[int] = None
    ) -> Tuple[Set[str], Dict[str, Set[str]]]:
        """
        Retrieves candidate entity_ids for an S1 query.
        
        Args:
            enabled_blockers: Optional set of blocker names to enable.
                Options: {"exact_core", "compact_name", "rare_name_tokens", "name_2grams", "exact_address", "rare_address_tokens", "postal_first_token", "structured_ids"}
                If None, all blockers are enabled.
        Returns:
            candidates: set of candidate IDs
            attribution: {blocker_name: set_of_ids_found}
        """
        if enabled_blockers is None:
            enabled_blockers = {
                "exact_core",
                "compact_name",
                "rare_name_tokens",
                "name_2grams",
                "exact_address",
                "rare_address_tokens",
                "postal_first_token",
                "structured_ids"
            }

        norm_name, core_name, compact_name = normalize_name(name)
        norm_addr, postal, nums, addr_tokens = normalize_address(addr, self.country)
        name_tokens = core_name.split() if core_name else []
        nums_set = set(nums)

        candidates = set()
        attribution = defaultdict(set)

        # 1. Exact core name
        if "exact_core" in enabled_blockers and core_name:
            matches = self.exact_core_index.get(core_name, [])
            if 0 < len(matches) <= 200:
                candidates.update(matches)
                attribution["exact_core"].update(matches)

        # 2. Compact name
        if "compact_name" in enabled_blockers and compact_name:
            matches = self.compact_name_index.get(compact_name, [])
            if 0 < len(matches) <= 200:
                candidates.update(matches)
                attribution["compact_name"].update(matches)

        # 3. Rare name token matching
        if "rare_name_tokens" in enabled_blockers and name_tokens:
            rare_counts: Dict[str, int] = defaultdict(int)
            for tok in name_tokens:
                if len(tok) >= 4 and tok in self.rare_name_token_index:
                    tok_matches = self.rare_name_token_index[tok]
                    for cid in tok_matches:
                        rare_counts[cid] += 1
            
            if rare_counts:
                sorted_rare = sorted(rare_counts.items(), key=lambda x: x[1], reverse=True)
                top_rare = [cid for cid, cnt in sorted_rare[:max_rare_token_matches]]
                candidates.update(top_rare)
                attribution["rare_name_tokens"].update(top_rare)

        # 4. Exact normalized address
        if "exact_address" in enabled_blockers and norm_addr and len(norm_addr) >= 15:
            addr_matches = self.exact_address_index.get(norm_addr, [])
            if 0 < len(addr_matches) <= 100:
                candidates.update(addr_matches)
                attribution["exact_address"].update(addr_matches)

        # 5. Rare address tokens + numeric overlap
        if "rare_address_tokens" in enabled_blockers and addr_tokens:
            addr_cand_counts: Dict[str, int] = defaultdict(int)
            for atok in set(addr_tokens):
                if len(atok) >= 4 and atok in self.rare_addr_token_index:
                    for cid in self.rare_addr_token_index[atok]:
                        addr_cand_counts[cid] += 1

            if addr_cand_counts:
                # Require >= 2 rare address tokens, OR (1 rare token + matching numeric token)
                valid_addr_cands = []
                for cid, token_count in addr_cand_counts.items():
                    if token_count >= 2:
                        valid_addr_cands.append((cid, token_count))
                    elif token_count == 1 and nums_set:
                        # Check numeric overlap
                        cand_meta = self.record_meta.get(cid)
                        if cand_meta and (nums_set & cand_meta[4]):
                            valid_addr_cands.append((cid, 1.5))

                if valid_addr_cands:
                    valid_addr_cands.sort(key=lambda x: x[1], reverse=True)
                    top_addr = [cid for cid, _ in valid_addr_cands[:max_address_matches]]
                    candidates.update(top_addr)
                    attribution["rare_address_tokens"].update(top_addr)

        # 6. Postal + First Token
        if "postal_first_token" in enabled_blockers and postal and name_tokens:
            first_tok = name_tokens[0]
            p_matches = self.postal_first_token_index.get((postal, first_tok), [])
            if 0 < len(p_matches) <= 100:
                candidates.update(p_matches)
                attribution["postal_first_token"].update(p_matches)

        # 7. Name 2-Grams
        if "name_2grams" in enabled_blockers and core_name:
            for g in extract_name_2grams(core_name):
                g_matches = self.name_2gram_index.get(g, [])
                if 0 < len(g_matches) <= 60:
                    top_g = g_matches[:20]
                    candidates.update(top_g)
                    attribution["name_2grams"].update(top_g)

        # 8. Structured Identifiers (Phone & Corporate ID)
        if "structured_ids" in enabled_blockers and addr:
            q_phones, q_corps = extract_identifiers(addr, self.country)
            for p in q_phones:
                p_matches = self.phone_index.get(p, [])
                if 0 < len(p_matches) <= 20:
                    candidates.update(p_matches)
                    attribution["exact_phone"].update(p_matches)
            for cid in q_corps:
                c_matches = self.corp_id_index.get(cid, [])
                if 0 < len(c_matches) <= 10:
                    candidates.update(c_matches)
                    attribution["exact_corp_id"].update(c_matches)

        # Rank/truncate candidates if max_total_candidates is set
        if max_total_candidates and len(candidates) > max_total_candidates:
            scored_cands = []
            for cid in candidates:
                n_blocks = sum(1 for b in attribution if cid in attribution[b])
                is_exact_name = 1 if (cid in attribution.get("exact_core", set()) or cid in attribution.get("compact_name", set())) else 0
                is_exact_addr = 1 if cid in attribution.get("exact_address", set()) else 0
                scored_cands.append((cid, (is_exact_name + is_exact_addr, n_blocks)))
            scored_cands.sort(key=lambda x: x[1], reverse=True)
            candidates = {cid for cid, _ in scored_cands[:max_total_candidates]}

        return candidates, dict(attribution)
