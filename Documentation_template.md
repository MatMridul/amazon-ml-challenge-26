# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Claude's Plan  
**Team Members:** Mridul Mathur (Team Lead), Anushka Priyani Nayak, Akshit Gaurana, Devansh Garg  
**Submission Date:** September 27, 2026

---

## 1. Executive Summary
We designed and implemented an end-to-end, high-precision, low-latency machine learning pipeline for multilingual business entity resolution across 3 heterogeneous sources (~1.73M test queries against a 10M record pool). Our architecture couples a high-recall multi-pass inverted index blocker (featuring order-independent legal suffix stripping and sorted name 2-grams) with a 23-dimensional C++ SIMD RapidFuzz feature extractor and a LightGBM Gradient Boosted Decision Tree (GBDT). To optimize specifically for the competition's macro-averaged $F_{0.5}$ metric (which penalizes false merges 2× more than false negatives and awards full 1.0 credit to singletons), our inference engine integrates deterministic contact/corporate ID mining and dynamic relative margin thresholding.

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis across the 20-million-record universe, we identified several critical real-world noise patterns:
1. **Pervasive Omission of PIN Codes in India (>90%):** Standard postal-code-based blocking fails on Indian records because over 90% of business registrations omit the 6-digit PIN.
2. **Cross-Script Multilingual Asymmetry:** While Source 1 is Latin English, Sources 2 and 3 contain 10–15% regional Indic scripts (Devanagari, Bengali, Tamil). In the majority of these cases, the business name is transliterated into regional script, but the physical address string remains in Latin English.
3. **Word Order Inversion & Prefix Legal Tokens:** Entities frequently reorder words (`Hendricks & Flowers` vs `Flowers, Hendricks Inc.`) or prepend corporate suffixes (`LLC Orellana Invsmbens` vs `Orellana Investments LLC`).
4. **Embedded Structured Identifiers:** Address strings frequently contain embedded 10-digit mobile numbers (`Ph: 989...`), Indian GSTIN/CIN codes, and French 9-digit SIREN codes.

### 2.2 Solution Strategy
**Approach Type:** Hybrid Multi-Pass Inverted Index Blocking + 23D C++ SIMD RapidFuzz Extractor + GBDT Classifier + Dynamic Relative Margin Reranker.  
**Core Innovation:** Combining order-independent legal suffix normalization and sorted name 2-grams (breaking the lexical recall bottleneck) with deterministic corporate ID pinning and dynamic relative score margins that suppress ambiguous lookalikes to maximize singleton accuracy.

---

## 3. Candidate Generation (Blocking)
To scale across 10 million candidate records without quadratic blowup, our blocker operates within hard country partitions (`India`, `US`, `France`):

- **Blocking keys used:**
  1. *Exact Normalized Core Name:* Legal suffixes stripped position-agnostically across English (`pvt`, `ltd`, `llc`), Hindi (`प्रा. लि.`), and French (`sarl`, `sas`).
  2. *Compact Alphanumeric Hash:* All punctuation, whitespace, and diacritics stripped.
  3. *Rare Name Tokens (Inverted Index):* Tokens filtered by inverse document frequency (DF ≤ 500).
  4. *Sorted Name 2-Grams:* Order-independent pairs of significant tokens (`tok_a_tok_b`) capturing transposed words and prefix variations.
  5. *Exact Normalized Address:* Landmark- and street-level normalized address hash.
  6. *Rare Address Tokens + Numeric Token Overlap:* Shared rare address tokens confirmed by identical building/plot numbers (`plot 448a`, `c-604`).
  7. *Deterministic Structured Identifiers:* 10-digit phone numbers, GSTIN, CIN, and SIREN codes.
- **Candidate pairs generated:** Average of **28.2 candidates per Source 1 query** (well below the competition's strict cardinality budget).
- **How we ensured true matches were not lost:** Multi-channel union guarantee. If an entity matches on core name, compact hash, 2-gram token pairs, address hash, or phone/corporate ID, it is captured into the candidate pool.

---

## 4. Matching Model

**Features used (23 Dimensions):**
- **Name features:** Jaro-Winkler distance on raw name, Jaro-Winkler on legal-stripped core, Levenshtein ratio, Token Sort ratio, Token Set ratio, Character 3-gram Jaccard, Name length difference, Word count difference.
- **Address features:** Address Jaro-Winkler, Address Token Sort ratio, Address Token Jaccard overlap, Empty address indicator flags.
- **Numeric & Conflict features:** Extracted digit/numeric token Jaccard similarity, Numeric overlap count, Numeric conflict flag (signals when two records have contradictory building/street numbers).
- **Blocker Attribution Signals:** Exact core flag, Compact name flag, Rare name flag, Exact address flag, Rare address flag, Blocker agreement count (number of independent channels that retrieved the pair).

**Model type:** LightGBM Binary Gradient Boosted Classifier (300 trees, maximum depth 8, early stopping).  
**Threshold selection & Calibration:**
- Optimized using entity-isolated `GroupShuffleSplit` on held-out Source 1 businesses (zero data leakage).
- **Deterministic ID Pinning:** Candidates matching on exact phone or corporate registry IDs are boosted directly to $P = 1.0$.
- **Dynamic Relative Margin:** Rather than a static threshold, a query's candidates are only accepted if $P \ge 0.60$ AND $P \ge (P_{\max} - 0.18)$. If $P_{\max} < 0.60$, the query is classified as a singleton (predicting an empty list, securing full 1.0 credit).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9283** on held-out validation queries (Precision: **94.60%**, Recall: **89.93%**, Singleton Accuracy: **92.86%**).
- **Common false positives (wrong merges):** Franchise chains with identical business names located in different units of the same postal sector where addresses were partially omitted.
- **Common false negatives (missed matches):** Cross-script records where both the business name and address were completely transliterated into non-Latin scripts without numeric tokens.

---

## 6. Conclusion
By pairing an order-independent multi-pass lexical blocker with C++ SIMD feature extraction and a precision-calibrated LightGBM model, our solution resolves commercial business entities at multi-million scale in under 15 minutes. The approach avoids expensive external APIs, satisfies all open-source licensing constraints, and directly aligns with the precision-weighted macro $F_{0.5}$ metric.

---

## Appendix

### A. Code Artefacts
All code is fully modular, documented, and reproducible:
- `src/normalization.py`: Unicode, legal token, and identifier extraction.
- `src/blocking.py`: Multi-channel lexical blocker with inverted indexes.
- `src/feature_engineering.py`: 23D RapidFuzz pairwise feature extractor.
- `src/train_matcher.py`: Zero-leakage LightGBM trainer and threshold sweep.
- `src/inference.py`: Streaming end-to-end test inference pipeline.
- `scripts/launch_ec2.py`: Automated AWS 32-vCPU worker provisioning.

Entry point to reproduce submission:
```bash
python src/inference.py --test-dir dataset/test --output-dir output --model-path models/lightgbm_matcher_v1.txt
```
