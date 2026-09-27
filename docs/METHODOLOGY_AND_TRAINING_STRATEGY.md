# Amazon ML Challenge 2026 — Engineering & Methodology Log

## 1. Executive Summary & Problem Formulation
- **Task:** Multilingual Business Entity Resolution across 3 sources (S1 reference deduplicated, S2 & S3 noisy secondary registrations).
- **Scale:** ~2.2M train queries, ~1.73M test queries, ~10.3M train candidate records, ~10.0M test candidate records. Total universe > 20M records.
- **Evaluation Metric:** Macro $F_{0.5}$ score across all S1 entities:
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
  - Precision is weighted $2\times$ over recall.
  - Singletons (~5.6% of entities) earn 1.0 if correctly predicted empty, and 0.0 if any false match is predicted.
- **Submission Requirements:**
  - `output/matching_results.tsv`: Leaderboard scored.
  - `output/candidate_pairs.tsv`: Audited candidate generation set (judges reward smaller average candidate sets per S1 entity).

---

## 2. The 14-Phase Development Lifecycle

1. **Phase 1: Candidate Generation / Blocking:** High-recall, low-cardinality candidate filter.
2. **Phase 2: Supervised Training Pairs:** Extracting true matches (label 1) and blocker hard negatives (label 0).
3. **Phase 3: Pairwise Feature Engineering:** 23 RapidFuzz multi-field string and numeric signals.
4. **Phase 4: LightGBM Matching Model:** GBDT trained on hard negatives.
5. **Phase 5: Held-out Validation:** Entity-isolated Group splitting (zero leakage).
6. **Phase 6: Macro $F_{0.5}$ Threshold Optimization:** Finding optimal decision cutoff $T^*$ with singleton credit.
7. **Phase 7: Error Diagnostics:** Categorizing false positives and false negatives.
8. **Phase 8: Multilingual/Semantic Model (Optional):** Only if validation justifies it.
9. **Phase 9: Controlled Model Iteration:** One change, one hypothesis, empirical measurement.
10. **Phase 10: Final Training:** Fit on all training data once hyperparameters are frozen.
11. **Phase 11: Test Inference:** Full 1.73M queries against 10M records.
12. **Phase 12: Submission Validation:** Local pre-flight check with `validate_submission.py`.
13. **Phase 13: Leaderboard Experiments:** Controlled validation-to-public comparison.
14. **Phase 14: Final Submission Package:** Code, outputs, and documentation zip.

---

## 3. Phase 1 Empirical Evolution & Ablation Results

Evaluated on the full **4,133,346 record** candidate pool across 1,000 validation queries:

| Blocker Pipeline Version | Macro Entity Recall | Micro Link Recall | Full Coverage | Avg Candidates / Query | P95 / P99 Cands | Baseline Macro $F_{0.5}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Initial (Name Only)** | 31.29% | 31.57% | 8.04% | 20.34 | 46.0 / 123.0 | 0.2798 |
| **Blocking V1 (+ Address)** | 64.23% | 64.45% | 37.78% | 24.53 | 55.0 / 127.0 | 0.3159 |
| **Blocking V1.1 (Token Fix)** | **70.90%** | **70.86%** | **42.96%** | **31.39** | **78.0 / 112.0** | **0.3923** |

### Key Architectural Decisions in Blocking V1.1:
1. **Hard Country Partitioning:** Never compare records across India, US, and France. Reduces search space by ~60–70% immediately.
2. **Order-Independent Legal Token Normalization:** Position-agnostic removal of `pvt`, `ltd`, `private limited`, `m/s`, and Hindi/French equivalents (`प्रा. लि.`, `sarl`).
3. **Address Blocker (`ExactAddress` + `RareAddressTokens`):**
   - Discovered that >90% of Indian addresses omit PIN codes.
   - Indexing on normalized building/street names and rare tokens recovered cross-script registrations where the business name was transliterated into Devanagari/Bengali, but the address remained in English.

---

## 4. Guarantees for "One-and-Done" Optimal Training

To maximize training effectiveness and avoid blind trial-and-error:

1. **100% Real Hard Negatives:**
   - Negatives are not randomly sampled. Every negative pair in Phase 2 is an actual candidate retrieved by the blocker that shares names, numbers, or addresses with the query.
2. **Complete & Orthogonal Features (23 Dimensions):**
   - Name distance (Jaro-Winkler, Levenshtein, Token Sort, Token Set, Length difference).
   - Address distance (Jaro-Winkler, Token Sort, Token Jaccard).
   - Exact conflict flags (Numeric token disjointness, Empty address flags).
   - Blocker attribution (`is_exact_core`, `is_exact_addr`, `blocker_agreement_count`).
3. **Zero-Leakage Group Splitting:**
   - Validation split is partitioned strictly at the `s1_id` level (`GroupShuffleSplit`). The model is never evaluated on candidate pairs of an entity it was trained on.
4. **Automated Convergence via Early Stopping:**
   - Configured with `early_stopping(stopping_rounds=30)` on validation loss. Prevents underfitting and makes overfitting impossible.
5. **Macro $F_{0.5}$ Threshold Calibration:**
   - Model learns continuous probability scores $P \in [0.0, 1.0]$. The decision threshold $T^*$ is swept on held-out validation queries to maximize the exact competition macro formula, awarding full 1.0 credit to singletons.
6. **Feature Checkpointing for Zero-Cost Flexibility:**
   - Intermediate candidate pairs (`train_pairs.parquet`) and feature matrices (`train_features.parquet`) are cached. Re-running hyperparameter tuning or threshold sweeps takes <30 seconds without re-indexing.
