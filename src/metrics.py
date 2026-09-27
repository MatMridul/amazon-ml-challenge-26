"""
ML Challenge 2026 — Exact Metric Implementation: Macro F_0.5

F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    = (1 + 0.5^2) * P * R / (0.5^2 * P + R)

Calculated per Source 1 entity, then macro-averaged across all Source 1 entities.
Singletons:
  - If true is empty and pred is empty: score = 1.0
  - If true is empty and pred is non-empty: score = 0.0
  - If true is non-empty and pred is empty: score = 0.0
"""

from typing import Dict, Set, Iterable


def compute_entity_f05(true_matches: Set[str], pred_matches: Set[str]) -> float:
    """Compute F_0.5 for a single S1 entity."""
    if not true_matches and not pred_matches:
        return 1.0
    if not true_matches or not pred_matches:
        return 0.0

    tp = len(true_matches & pred_matches)
    if tp == 0:
        return 0.0

    precision = tp / len(pred_matches)
    recall = tp / len(true_matches)

    # F_0.5 formula: (1.25 * P * R) / (0.25 * P + R)
    numerator = 1.25 * precision * recall
    denominator = 0.25 * precision + recall
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]]
) -> Dict[str, float]:
    """
    Evaluates predictions against ground truth.
    Returns:
        macro_f05, precision, recall, singleton_acc, non_singleton_f05
    """
    total_f05 = 0.0
    total_prec = 0.0
    total_rec = 0.0
    count = len(ground_truth)
    if count == 0:
        return {"macro_f05": 0.0}

    singleton_count = 0
    singleton_correct = 0

    non_singleton_f05 = 0.0
    non_singleton_count = 0

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())

        if not true_set:
            singleton_count += 1
            if not pred_set:
                singleton_correct += 1
                score = 1.0
                prec = 1.0
                rec = 1.0
            else:
                score = 0.0
                prec = 0.0
                rec = 0.0
        else:
            non_singleton_count += 1
            if not pred_set:
                score = 0.0
                prec = 0.0
                rec = 0.0
            else:
                tp = len(true_set & pred_set)
                prec = tp / len(pred_set)
                rec = tp / len(true_set)
                if tp == 0:
                    score = 0.0
                else:
                    score = (1.25 * prec * rec) / (0.25 * prec + rec)
            non_singleton_f05 += score

        total_f05 += score
        total_prec += prec
        total_rec += rec

    return {
        "macro_f05": total_f05 / count,
        "macro_precision": total_prec / count,
        "macro_recall": total_rec / count,
        "singleton_accuracy": (singleton_correct / singleton_count) if singleton_count else 0.0,
        "non_singleton_f05": (non_singleton_f05 / non_singleton_count) if non_singleton_count else 0.0,
        "total_entities": count,
        "singletons": singleton_count
    }


if __name__ == "__main__":
    # Test example from Problem Statement:
    # S1-00001: true = [S2-00047, S3-00812], pred = [S2-00047, S2-00193, S3-00812]
    # Precision = 2/3, Recall = 1.0, F0.5 = (1.25 * 2/3 * 1.0) / (0.25 * 2/3 + 1.0) = 0.7142857
    t = {"S2-00047", "S3-00812"}
    p = {"S2-00047", "S2-00193", "S3-00812"}
    score = compute_entity_f05(t, p)
    print(f"Sample test score: {score:.4f} (Expected: ~0.7143)")
    assert abs(score - 0.7142857) < 1e-4, f"Mismatch: {score}"
    print("Metric unit test PASSED!")
