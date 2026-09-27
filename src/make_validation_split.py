"""
ML Challenge 2026 — Validation Split Generator

Creates a representative, stratified 50,000 S1 validation set from train data:
- Stratified by country (US ~60%, India ~40%)
- Preserving true singleton proportion (~5.6%)
- Saves to:
    dataset/validation/val_source1.tsv
    dataset/validation/val_ground_truth.tsv
"""

import os
import sys
import polars as pl

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "student_resource", "dataset", "train")
VAL_DIR = os.path.join(BASE_DIR, "dataset", "validation")


def create_validation_split(sample_size: int = 50000, seed: int = 42):
    os.makedirs(VAL_DIR, exist_ok=True)
    val_s1_path = os.path.join(VAL_DIR, "val_source1.tsv")
    val_gt_path = os.path.join(VAL_DIR, "val_ground_truth.tsv")

    if os.path.exists(val_s1_path) and os.path.exists(val_gt_path):
        print(f"Validation split already exists at {VAL_DIR}")
        return

    print("Loading train_source1.tsv and train_ground_truth.tsv...")
    s1_path = os.path.join(TRAIN_DIR, "train_source1.tsv")
    gt_path = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

    s1_df = pl.read_csv(s1_path, separator="\t")
    gt_df = pl.read_csv(gt_path, separator="\t")

    # Join on entity_id == source1_entity_id
    print("Joining S1 metadata with Ground Truth...")
    merged = s1_df.join(
        gt_df,
        left_on="entity_id",
        right_on="source1_entity_id",
        how="inner"
    )

    # Add singleton flag for stratification
    merged = merged.with_columns(
        (pl.col("matched_entity_ids").is_null() | (pl.col("matched_entity_ids") == "")).alias("is_singleton")
    )

    print("Stratified sampling...")
    # Calculate group sample sizes based on (country, is_singleton)
    group_counts = merged.group_by(["country", "is_singleton"]).len()
    total_len = len(merged)
    
    samples = []
    for row in group_counts.iter_rows(named=True):
        country = row["country"]
        is_singleton = row["is_singleton"]
        n_group = row["len"]
        target_n = max(1, int(round((n_group / total_len) * sample_size)))
        
        group_df = merged.filter(
            (pl.col("country") == country) & (pl.col("is_singleton") == is_singleton)
        ).sample(n=target_n, seed=seed)
        samples.append(group_df)

    val_df = pl.concat(samples).sample(fraction=1.0, shuffle=True, seed=seed)
    # Trim or ensure exact size
    val_df = val_df.head(sample_size)

    print(f"Sampled {len(val_df):,} validation entities:")
    print(val_df.group_by(["country", "is_singleton"]).len())

    # Export val_source1.tsv
    val_s1 = val_df.select(["entity_id", "business_name", "business_address", "country"])
    val_s1.write_csv(val_s1_path, separator="\t")
    print(f"Saved: {val_s1_path} ({len(val_s1):,} rows)")

    # Export val_ground_truth.tsv
    val_gt = val_df.select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("matched_entity_ids").fill_null("")
    ])
    val_gt.write_csv(val_gt_path, separator="\t")
    print(f"Saved: {val_gt_path} ({len(val_gt):,} rows)")


if __name__ == "__main__":
    create_validation_split(sample_size=50000)
