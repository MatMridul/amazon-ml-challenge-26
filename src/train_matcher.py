"""
ML Challenge 2026 — Phases 4, 5, 6: LightGBM Training & F_0.5 Threshold Calibration

1. Loads labeled pairs from Phase 2
2. Computes RapidFuzz pairwise features from Phase 3
3. Splits by S1 entity into Train (80%) and Held-out Validation (20%)
4. Fits LightGBM binary classifier (GBDT)
5. Performs threshold sweep on held-out validation to maximize Macro F_0.5 (with singleton credit)
6. Outputs feature importances & saves model to models/lightgbm_matcher_v1.txt
"""

import os
import sys
import argparse
from typing import Dict, List, Set, Tuple
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.feature_engineering import compute_pairwise_features, FEATURE_COLUMNS
from src.metrics import evaluate_predictions, compute_entity_f05

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODELS_DIR = os.path.join(BASE_DIR, "models")
PAIRS_DIR = os.path.join(BASE_DIR, "dataset", "training_pairs")


def train_and_evaluate_matcher(
    pairs_parquet_path: str,
    output_model_name: str = "lightgbm_matcher_v1.txt",
    test_size: float = 0.2,
    seed: int = 42
):
    os.makedirs(MODELS_DIR, exist_ok=True)
    out_model_path = os.path.join(MODELS_DIR, output_model_name)

    print("=" * 80)
    print("PHASES 4, 5, 6: LIGHTGBM MATCHER TRAINING & THRESHOLD CALIBRATION")
    print("=" * 80)
    
    # 1. Load pairs dataset
    print(f"Loading pairs from: {pairs_parquet_path}")
    raw_pairs_df = pl.read_parquet(pairs_parquet_path)
    print(f"Loaded {len(raw_pairs_df):,} total pairs across {raw_pairs_df['s1_id'].n_unique():,} unique S1 entities.")

    # 2. Compute pairwise features
    feat_df = compute_pairwise_features(raw_pairs_df)

    # Convert to pandas/numpy for LightGBM
    pdf = feat_df.to_pandas()
    X = pdf[FEATURE_COLUMNS]
    y = pdf["label"].values
    groups = pdf["s1_id"].values

    # 3. Entity-level Group Split (zero leakage across S1 entities)
    print(f"\nSplitting data at S1 entity level ({int((1-test_size)*100)}% Train / {int(test_size*100)}% Val)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(gss.split(X, y, groups=groups))

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    val_groups = groups[val_idx]
    val_cand_ids = pdf["cand_id"].iloc[val_idx].values

    print(f"Train set: {len(X_train):,} pairs ({np.sum(y_train):,} positive, {len(y_train)-np.sum(y_train):,} negative)")
    print(f"Val set:   {len(X_val):,} pairs ({np.sum(y_val):,} positive, {len(y_val)-np.sum(y_val):,} negative)")

    # 4. Train LightGBM Model
    print("\nTraining LightGBM Classifier...")
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "random_state": seed
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )

    print(f"Trained {model.best_iteration} boosting trees successfully.")
    model.save_model(out_model_path)
    print(f"Saved model to: {out_model_path}")

    # 5. Print Feature Importances
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE RANKING (Split Gain)")
    print("=" * 80)
    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(FEATURE_COLUMNS, importance), key=lambda x: x[1], reverse=True)
    for rank, (fname, imp) in enumerate(feat_imp[:12], 1):
        print(f"  {rank:2d}. {fname:25s}: {imp:10.2f}")

    # 6. Held-out Validation & Macro F_0.5 Threshold Sweep
    print("\n" + "=" * 80)
    print("PHASE 6: MACRO F_0.5 THRESHOLD CALIBRATION ON HELD-OUT ENTITIES")
    print("=" * 80)
    
    val_preds_prob = model.predict(X_val, num_iteration=model.best_iteration)

    # Reconstruct true ground truth for held-out validation entities
    unique_val_s1 = np.unique(val_groups)
    val_ground_truth = defaultdict(set)
    for s1, cid, lbl in zip(val_groups, val_cand_ids, y_val):
        if lbl == 1:
            val_ground_truth[s1].add(cid)
    # Ensure every val S1 has an entry (including singletons)
    for s1 in unique_val_s1:
        if s1 not in val_ground_truth:
            val_ground_truth[s1] = set()

    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    best_thresh = 0.50
    best_f05 = -1.0
    best_eval = {}

    print(f"{'Threshold':<10} | {'Macro F_0.5':<14} | {'Precision':<12} | {'Recall':<12} | {'Singleton Acc':<15}")
    print("-" * 72)

    for t in thresholds:
        pred_dict = defaultdict(set)
        for s1, cid, prob in zip(val_groups, val_cand_ids, val_preds_prob):
            if prob >= t:
                pred_dict[s1].add(cid)
        for s1 in unique_val_s1:
            if s1 not in pred_dict:
                pred_dict[s1] = set()

        ev = evaluate_predictions(val_ground_truth, pred_dict)
        f05 = ev["macro_f05"]
        print(f"{t:<10.2f} | {f05:<14.4f} | {ev['macro_precision']:<12.4f} | {ev['macro_recall']:<12.4f} | {ev['singleton_accuracy']*100:<14.2f}%")

        if f05 > best_f05:
            best_f05 = f05
            best_thresh = t
            best_eval = ev

    print("\n" + "=" * 80)
    print(f"OPTIMAL DECISION THRESHOLD: T* = {best_thresh:.2f}")
    print(f"  Best Validation Macro F_0.5: {best_eval['macro_f05']:.4f}")
    print(f"  Precision:                   {best_eval['macro_precision']:.4f}")
    print(f"  Recall:                      {best_eval['macro_recall']:.4f}")
    print(f"  Singleton Accuracy:          {best_eval['singleton_accuracy']*100:.2f}%")
    print(f"  Non-singleton F_0.5:         {best_eval['non_singleton_f05']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM entity resolution model")
    parser.add_argument("--pairs-file", type=str, default="dataset/training_pairs/train_pairs_india_5000.parquet")
    parser.add_argument("--model-name", type=str, default="lightgbm_matcher_v1.txt")
    args = parser.parse_args()

    train_and_evaluate_matcher(
        pairs_parquet_path=args.pairs_file,
        output_model_name=args.model_name
    )
