# Proposed Solution — Hybrid Multi-View Entity Resolution

## 1. Executive Summary

This proposal treats the ML Challenge as a large-scale **entity resolution / record linkage** problem.

We have:

- **S1** — deduplicated reference source; every S1 record is a query.
- **S2 + S3** — noisy secondary sources.
- Every S1 entity may match **0, 1, or multiple** S2/S3 entities.
- Records have unique IDs, business names, addresses, and country.
- Matching is constrained to the **same country**.
- External enrichment, web lookup, geocoding APIs, scraping, and commercial ER APIs are prohibited.
- The test set introduces **France**, which is absent from training, so the solution must generalize across countries.

The proposed architecture is:

> **Multi-view normalization → multiple high-recall candidate generators → candidate pruning → rich pairwise feature generation → learned match classifier → conservative entity-level decision.**

The user's original idea is incorporated as a central part of candidate generation:

1. A **semantic + character vector space / ANN graph** over S1/S2/S3.
2. A separate **address / geographic ("world") representation / graph**, where supported by the supplied data.
3. Graph-derived evidence such as semantic similarity, character similarity, rank, mutual-nearest-neighbor behavior, shared neighbors, and geographic/address proximity.

The key design principle is that these graphs should primarily **find and describe candidate relationships**, while a learned classifier makes the final "same entity / different entity" decision.

---

# 2. Problem Definition

For every S1 entity, find all entities in S2 and S3 that refer to the same real-world business.

Formally:

`q ∈ S1`

and the desired output is:

`M(q) ⊆ S2 ∪ S3`

where `M(q)` may be empty, contain one entity, or contain multiple entities.

Example:

```text
S1-001    S2-014,S3-882
S1-002
S1-003    S2-991,S2-992,S3-104
```

An empty value means that the system believes the S1 entity has no corresponding S2/S3 entity.

---

# 3. Challenge-Specific Constraints

## Dataset scale

Training:

- S1: ~2.2M
- S2: ~5.0M
- S3: ~5.3M

Test:

- S1: ~1.73M
- S2: ~4.89M
- S3: ~5.08M

Total data is 20M+ records.

Therefore:

- No all-pairs comparison.
- Candidate generation must be indexed and scalable.
- Memory and I/O must be designed carefully.
- Any expensive model must operate only on a relatively small candidate set.

## Countries

Training:

- US
- India

Test:

- US
- India
- France

Therefore:

- Country should be a hard search-space constraint.
- Matching must never cross countries.
- Learned features should be largely country-agnostic.
- Avoid overfitting to India/US-specific address parsing.
- French records must be handled zero-shot.

## Evaluation

The main metric is macro-averaged F0.5 over S1 entities:

`F0.5 = 1.25 * Precision * Recall / (0.25 * Precision + Recall)`

Precision is weighted more heavily than recall.

Therefore:

- False merges are expensive.
- The system must be comfortable predicting no match.
- Thresholding is important.
- Pair-level accuracy alone is not sufficient.
- Validation should reproduce the competition's entity-level scoring behavior.

## Candidate submission

`candidate_pairs.tsv` is itself important.

The candidate set must:

- contain every predicted match;
- have high true-match recall;
- remain as small as practical.

Therefore candidate generation is a first-class optimization problem, not merely an implementation detail.

---

# 4. Core Architecture

```text
                         RAW S1 / S2 / S3
                                |
                                v
                    +-----------------------+
                    | Multi-view normalization |
                    +-----------+-----------+
                                |
            +-------------------+-------------------+
            |                   |                   |
            v                   v                   v
       Lexical view       Semantic view       Address/world view
            |                   |                   |
            v                   v                   v
       Character /         Vector / ANN       Address / spatial
       token retrieval        graph              graph
            |                   |                   |
            +-------------------+-------------------+
                                |
                                v
                       Candidate union
                                |
                                v
                       Candidate pruning
                                |
                                v
                       Candidate pairs
                                |
                                v
                  +---------------------------+
                  | Pairwise feature engine   |
                  +-------------+-------------+
                                |
                                v
                       Match classifier
                                |
                                v
                     P(same entity | pair)
                                |
                                v
                     Threshold + margin
                                |
                    +-----------+-----------+
                    |                       |
                    v                       v
                  MATCH                  REJECT
                    |
                    v
              Entity-level resolver
                    |
                    v
           matching_results.tsv
```

---

# 5. Stage 1 — Multi-View Normalization

Do not collapse every field into one normalized string.

Maintain multiple representations because different representations solve different types of corruption.

## Business name representations

For every record retain:

- raw name
- Unicode-normalized name
- lower/case-normalized name where appropriate
- punctuation-normalized name
- whitespace-normalized name
- canonicalized legal-form representation
- core business name with legal form separated
- token list
- compact alphanumeric representation
- character n-gram representation

Example:

```text
Raw:
"ACME Technologies Pvt. Ltd."

Core name:
"acme technologies"

Legal form:
"private limited"

Tokens:
["acme", "technologies", "private", "limited"]
```

Do not assume legal forms should simply be deleted. Preserve them as a separate feature.

## Legal-form normalization

Build a language/country-extensible vocabulary for common legal forms, e.g.:

- Pvt Ltd
- Private Limited
- Ltd
- Limited
- Inc
- Incorporated
- Corp
- Corporation
- LLC
- LLP
- SARL
- SA

The representation should support equivalent forms while retaining the original information.

## Address representations

Retain:

- raw address
- normalized address
- address tokens
- character n-grams
- numeric tokens
- postal-code-like tokens
- house-number-like tokens
- city/locality/region when reliably extractable
- country

Do not depend on a single country-specific parser.

The system must degrade gracefully when address components are missing or unstructured.

---

# 6. Stage 2 — Country Partitioning

Country is a hard constraint.

For every S1 query:

```text
S1 country = India
```

only search:

```text
S2 India
S3 India
```

Never search US or France records.

Maintain separate indexes or partitions by country where practical.

This provides:

- huge reduction in search space;
- better semantic retrieval;
- fewer false positives;
- guaranteed compliance with the same-country matching rule.

The system must still support France in test even though France has no training labels.

---

# 7. Stage 3 — Candidate Generation

Candidate generation should optimize:

> **Very high true-match recall with as few candidates as possible.**

Do not rely on a single blocker.

Use multiple independent retrieval mechanisms and union their results.

## 7.1 Exact/canonical name retrieval

Build inverted indexes over:

- canonical core name;
- normalized full name;
- useful exact token signatures.

This catches obvious matches at extremely low cost.

---

## 7.2 Character retrieval

Use character n-gram based retrieval/similarity to handle:

- typos;
- punctuation differences;
- abbreviations;
- transliteration noise;
- small spelling changes.

Possible implementation:

- character TF-IDF + sparse retrieval;
- BM25-like character/token retrieval;
- another scalable character ANN/index if justified.

Do not assume a single fuzzy metric is sufficient at 10M+ scale.

---

## 7.3 Token/name retrieval

Use token-based representations to handle:

- word reordering;
- inserted legal forms;
- common business words;
- partial name overlap.

Potential signals:

- token Jaccard;
- token containment;
- TF-IDF cosine;
- BM25;
- informative-token overlap.

---

## 7.4 Address retrieval

Use address information independently of name.

Useful retrieval keys/features:

- postal-code-like token;
- city/locality;
- house number;
- street tokens;
- address character n-grams;
- address token similarity.

This can recover matches when business names are noisy.

---

## 7.5 Semantic ANN retrieval — USER'S CORE IDEA

Create embeddings for each record using an allowed open-source model.

Embedding input should combine relevant information, for example:

```text
business_name + normalized_address
```

but retain separate name/address representations for later features.

Build ANN indexes over S2/S3, partitioned by country.

For each S1:

```text
query embedding
      |
      v
ANN search
      |
      v
top-K candidate records
```

Important:

- Start with K values such as 5/10/20/50 and benchmark.
- Do not blindly use "top 5%" because dataset scale varies and candidate sets become huge.
- Use both rank and similarity threshold.
- Measure candidate recall vs candidate-set size.

The semantic graph is primarily a **high-recall retrieval mechanism**, not the final matcher.

---

# 8. Stage 4 — Address / Geographic ("World") Graph

This is the second major component from the original idea.

If the supplied challenge data contains usable coordinates or geographic information, construct a spatial representation/index over all records.

Possible signals:

- geographic distance;
- same locality;
- same city;
- same region;
- same postal area;
- nearby address.

If only textual addresses are available, construct an address-similarity graph/index instead of using prohibited external geocoding.

## Important rule

Do not make geographic proximity an absolute identity rule unless validation proves that it is safe.

A company can have:

- multiple branches;
- headquarters and operating locations;
- stale addresses;
- mailing addresses;
- inconsistent source addresses.

Therefore geography should normally be treated as **evidence**, not truth.

---

# 9. Unified Multi-Source Vector/Graph Representation

All S1/S2/S3 records can be represented in a common embedding space.

Every node contains:

```text
entity_id
source
country
embedding
normalized_name
normalized_address
metadata
```

Edges can represent different evidence types:

- semantic similarity;
- character similarity;
- address similarity;
- geographic proximity.

Conceptually:

```text
                    S1-A
                  /      \
          semantic        address
              /              \
           S2-X              S3-Y
             \              /
              \  semantic  /
                 S3-Y
```

The graph should initially be used for:

1. candidate discovery;
2. similarity/rank features;
3. mutual-nearest-neighbor features;
4. shared-neighbor features;
5. cross-source agreement.

Do NOT immediately perform unrestricted graph transitive closure.

Avoid assuming:

```text
A ~ B
B ~ C
therefore A ~ C
```

because this can create false entity clusters.

---

# 10. Candidate Union

For each S1 entity:

```text
C_exact
C_character
C_token
C_address
C_semantic
C_world
```

are generated independently.

Then:

```text
C = union(C_exact,
          C_character,
          C_token,
          C_address,
          C_semantic,
          C_world)
```

Deduplicate candidate IDs.

The candidate generation stage should record which retrieval mechanisms produced each candidate.

Example:

```text
S1-001 -> S2-100
found_by_exact_name = 1
found_by_character = 1
found_by_semantic = 1
found_by_address = 0
```

This becomes useful later as a feature.

---

# 11. Candidate Pruning

Because candidate_pairs.tsv is audited/rewarded, we should not dump every retrieval result into the final candidate set.

Benchmark candidate generation on training data.

For each retrieval method measure:

- true-match recall;
- average candidates per S1;
- percentile candidate size;
- contribution of unique true matches;
- overlap with other retrieval methods.

Example experiment:

```text
Semantic K=5:
  recall = X
  avg candidates = 5

Semantic K=10:
  recall = Y
  avg candidates = 10

Semantic K=20:
  recall = Z
  avg candidates = 20
```

Do this for every retrieval mechanism and for their unions.

Goal:

> **Maximize candidate recall while minimizing average candidates.**

Candidate generation should be treated as its own optimization problem.

---

# 12. Stage 5 — Pairwise Feature Generation

Once we have candidate pairs, compute detailed features.

For each pair `(S1_i, candidate_j)`:

## Name features

- exact normalized match;
- core-name exact match;
- character similarity;
- edit similarity;
- token Jaccard;
- token containment;
- TF-IDF similarity;
- semantic name similarity;
- length difference;
- informative-token agreement.

## Address features

- exact normalized address;
- character similarity;
- token similarity;
- numeric-token overlap;
- house-number agreement;
- postal agreement;
- city/locality agreement;
- region agreement;
- address semantic similarity.

## Cross-field features

- name + address combined similarity;
- name agreement with address agreement;
- name/address disagreement;
- number of independently matching fields.

## Semantic/graph features

- combined embedding cosine;
- semantic rank;
- semantic similarity margin;
- character rank;
- mutual-nearest-neighbor flag;
- shared-neighbor count;
- cross-source graph agreement;
- number of retrieval mechanisms that found the candidate.

## Geographic/world features

Where available:

- distance;
- distance bucket;
- same locality;
- same city;
- same postal area;
- spatial-neighbor rank.

## Metadata

- country;
- source pair (S1-S2 or S1-S3);
- missing-field indicators;
- name/address lengths;
- source-specific characteristics if discovered through EDA.

---

# 13. "Agreement Across Independent Views" Feature

A particularly useful feature:

```text
block_agreement_count
```

Example:

```text
exact name        YES
character         YES
token             YES
address           YES
semantic          YES
world             NO

block_agreement_count = 5
```

This represents independent evidence that the candidate is worth trusting.

Another candidate might be:

```text
exact name        NO
character         NO
token             NO
address           NO
semantic          YES
world             NO

block_agreement_count = 1
```

The classifier can learn whether this distinction is useful.

---

# 14. Stage 6 — Pairwise ML Classifier

Start simple.

## Model 1: Logistic Regression

Use it as:

- baseline;
- interpretable model;
- probability-producing model;
- feature sanity check.

Output:

```text
P(same_entity | pair_features)
```

Then test a stronger tree-based model such as an allowed gradient-boosted tree implementation.

The tree model can learn interactions such as:

```text
IF name similarity is high
AND address similarity is high
AND postal matches
THEN strong match

BUT

IF name similarity is high
AND address strongly conflicts
AND postal conflicts
THEN likely false merge
```

---

# 15. Hard Negative Mining

This is critical.

Random negatives are too easy.

Example easy negative:

```text
ABC Technologies
vs
Rajesh Textiles
```

More valuable negative:

```text
ABC Technologies Bangalore
vs
ABC Technologies Mumbai
```

Hard negatives should be mined from high-scoring candidate pairs that are known to be wrong.

Training should contain:

- positive pairs;
- ordinary negatives;
- hard negatives;
- near-duplicate-looking false matches.

This should improve precision, which is especially valuable because F0.5 weights precision more heavily.

---

# 16. Stage 7 — Entity-Level Decision

The classifier scores every candidate independently.

Example:

```text
S1-001 -> S2-183 = 0.998
S1-001 -> S2-992 = 0.082
S1-001 -> S3-011 = 0.971
S1-001 -> S3-551 = 0.421
```

Do not automatically choose only the highest-scoring candidate.

The task is 1-to-many.

Potential output:

```text
S1-001    S2-183,S3-011
```

If all candidates are weak:

```text
S1-002
```

with an empty match list.

---

# 17. Singleton / Abstention Logic

A key principle:

> **The system must be willing to say "no match."**

Do not use:

```text
always select best candidate
```

Instead:

```text
keep candidate if probability >= threshold
```

Potentially also use a confidence margin:

```text
P1 - P2 >= margin
```

where appropriate.

This should be evaluated empirically rather than assumed.

Singletons are particularly important because an incorrect false match can reduce that entity's score to zero.

---

# 18. Threshold Optimization

Do not assume `0.5` is the correct probability threshold.

Sweep thresholds using training/validation data:

```text
0.50
0.55
0.60
...
0.95
0.96
...
```

For each threshold calculate the actual entity-level competition metric.

Optimize **macro F0.5**, not generic accuracy or pair-level F1.

Also evaluate:

- precision;
- recall;
- singleton precision;
- multi-match recall;
- candidate recall;
- average candidates.

---

# 19. Validation Strategy

Do not evaluate only on the entire training set.

Create a validation split at the S1 entity level.

Important:

- All candidate pairs associated with an S1 should remain in the same split.
- Avoid leakage between train and validation.
- Evaluate the complete pipeline, not only the classifier.

Track:

```text
Candidate Recall
Average Candidates / S1
P50 Candidate Size
P95 Candidate Size
Final Precision
Final Recall
Final F0.5
Singleton Precision
Multi-match Recall
```

Also perform country-specific diagnostics for US and India.

France cannot be directly scored because it has no labels, but the model should remain country-agnostic.

---

# 20. Ablation Plan

Do not assume every sophisticated component helps.

Build progressively:

### Baseline A
Exact normalized matching.

### Baseline B
+ character retrieval.

### Baseline C
+ address retrieval.

### Baseline D
+ semantic ANN.

### Baseline E
+ pairwise classifier.

### Baseline F
+ hard negative mining.

### Baseline G
+ graph-derived features.

### Baseline H
+ world/geographic evidence where available.

For every version measure:

```text
candidate recall
average candidate count
precision
recall
F0.5
```

Keep a component only if it provides measurable value.

---

# 21. Important Design Principle

Do NOT start with an 8B LLM.

This is fundamentally a large-scale structured retrieval + classification problem.

A smaller embedding model plus:

- inverted indexes;
- ANN;
- character retrieval;
- address retrieval;
- classical ML;
- efficient candidate pruning

is likely to be much more computationally appropriate.

The 8B parameter limit is a maximum constraint, not a target.

---

# 22. Expected Final Architecture

```text
                        +-----------------------+
                        |    S1 / S2 / S3       |
                        |       20M+            |
                        +-----------+-----------+
                                    |
                                    v
                     +-------------------------+
                     |  MULTI-VIEW NORMALIZE  |
                     |                         |
                     | name / tokens / chars  |
                     | address / numbers      |
                     | legal form / country   |
                     +------------+------------+
                                  |
                 +----------------+----------------+
                 |                |                |
                 v                v                v
          Lexical Retrieval  Semantic ANN     World/Address
          char + token       Vector Graph       Graph
                 |                |                |
                 +----------------+----------------+
                                  |
                                  v
                         Candidate Union
                                  |
                                  v
                         Candidate Pruning
                                  |
                                  v
                       candidate_pairs.tsv
                                  |
                                  v
                    Pairwise Feature Generation
                                  |
                 +----------------+----------------+
                 |                |                |
                 v                v                v
              Lexical         Semantic          World
              evidence        evidence          evidence
                 |                |                |
                 +----------------+----------------+
                                  |
                                  v
                         Pair Classifier
                                  |
                                  v
                     P(SAME ENTITY | PAIR)
                                  |
                                  v
                       Threshold + Margin
                                  |
                      +-----------+-----------+
                      |                       |
                      v                       v
                    MATCH                   REJECT
                      |
                      v
                  1-to-many
                entity resolution
                      |
                      v
              matching_results.tsv
```

---

# 23. Key Hypotheses to Validate

The following should be treated as hypotheses, not assumptions:

1. Semantic embeddings improve candidate recall beyond lexical retrieval.
2. Character retrieval catches cases semantic retrieval misses.
3. Address retrieval catches cases name retrieval misses.
4. Combining retrieval mechanisms produces much higher recall than any single mechanism.
5. Independent blocker agreement is predictive of true matches.
6. Semantic similarity should be used as evidence rather than the final decision.
7. Geographic/address proximity improves precision when available.
8. Graph neighborhood features improve pair classification.
9. Hard-negative mining materially improves precision.
10. A conservative threshold improves macro F0.5 because false merges are expensive.
11. The system can generalize to unseen France without country-specific supervised rules.
12. Candidate pruning can reduce candidate count without materially reducing true-match recall.

---

# 24. What We Should NOT Decide Before Experiments

The following should remain configurable:

- embedding model;
- embedding dimensions;
- ANN implementation;
- K for semantic retrieval;
- K for character retrieval;
- K for address retrieval;
- exact blocking keys;
- similarity thresholds;
- classifier choice;
- probability threshold;
- margin threshold;
- geographic distance threshold;
- whether graph propagation is useful;
- whether geographic evidence is strong enough to be a blocker or only a feature.

The training data should decide these.

---

# 25. First Implementation Milestone

Before implementing the full architecture, build an offline experimental framework that can answer:

```text
1. How many true matches does each blocker recover?
2. How many candidates does each blocker produce?
3. What is the recall of their union?
4. Which features distinguish positive/negative pairs?
5. How well does Logistic Regression perform?
6. How much does a tree model improve?
7. What threshold maximizes validation F0.5?
8. How much do semantic embeddings help?
9. How much do graph features help?
10. What is the smallest candidate set that preserves high recall?
```

Only after these questions have measurements should the final production pipeline be locked.

---

# 26. Guiding Principle

The architecture should not be optimized for "maximum sophistication."

It should be optimized for:

> **high candidate recall + very small candidate sets + strong false-merge resistance + robust zero-shot generalization.**

The semantic/character graph and world/address graph are therefore not competing with classical entity-resolution techniques.

They are additional views of the same identity problem.

The intended hybrid is:

> **Lexical retrieval finds textual twins.  
> Semantic retrieval finds meaning-level twins.  
> Address/world retrieval finds contextual/geographic twins.  
> The graph connects these pieces of evidence.  
> The classifier decides whether the evidence is sufficient to call two records the same entity.**
