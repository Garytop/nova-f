from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import normalize_cve_labels


def labels_from_text(value: object) -> set[str]:
    return set(normalize_cve_labels(value))


def evaluate(truth_path: Path, prediction_path: Path, *, id_column: str = "id") -> dict[str, float]:
    truth = pd.read_csv(truth_path)
    pred = pd.read_csv(prediction_path)

    if id_column not in truth.columns or id_column not in pred.columns:
        raise KeyError(f"Both files must contain {id_column!r}")
    if "cve_labels" not in truth.columns or "cve_labels" not in pred.columns:
        raise KeyError("Both files must contain 'cve_labels'")

    truth[id_column] = truth[id_column].astype(str)
    pred[id_column] = pred[id_column].astype(str)
    merged = truth[[id_column, "cve_labels"]].merge(
        pred[[id_column, "cve_labels"]],
        on=id_column,
        how="left",
        suffixes=("_true", "_pred"),
    )

    true_sets = merged["cve_labels_true"].apply(labels_from_text)
    pred_sets = merged["cve_labels_pred"].fillna("").apply(labels_from_text)

    tp = fp = fn = 0
    exact = answered = 0
    per_label: dict[str, Counter[str]] = {}

    for expected, predicted in zip(true_sets, pred_sets):
        if predicted:
            answered += 1
        if expected == predicted:
            exact += 1

        for label in expected | predicted:
            bucket = per_label.setdefault(label, Counter())
            if label in expected and label in predicted:
                bucket["tp"] += 1
            elif label in predicted:
                bucket["fp"] += 1
            else:
                bucket["fn"] += 1

        tp += len(expected & predicted)
        fp += len(predicted - expected)
        fn += len(expected - predicted)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    f1_scores = []
    for counts in per_label.values():
        label_precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
        label_recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
        f1_scores.append(
            2 * label_precision * label_recall / (label_precision + label_recall)
            if label_precision + label_recall
            else 0.0
        )
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return {
        "rows": float(len(merged)),
        "answer_rate": answered / len(merged) if len(merged) else 0.0,
        "exact_match": exact / len(merged) if len(merged) else 0.0,
        "precision": precision,
        "recall": recall,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate nova-f prediction CSV against labeled CSV.")
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--id-column", default="id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(args.truth, args.pred, id_column=args.id_column)
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}" if isinstance(value, float) else f"{name}: {value}")


if __name__ == "__main__":
    main()
