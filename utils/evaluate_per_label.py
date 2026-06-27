from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import normalize_cve_labels


def labels_from_text(value: object) -> set[str]:
    return set(normalize_cve_labels(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prediction quality per CVE label.")
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    truth = pd.read_csv(args.truth)
    pred = pd.read_csv(args.pred)
    if args.id_column not in truth.columns or args.id_column not in pred.columns:
        raise KeyError(f"both CSVs must contain {args.id_column!r}")
    if "cve_labels" not in truth.columns or "cve_labels" not in pred.columns:
        raise KeyError("both CSVs must contain 'cve_labels'")

    truth[args.id_column] = truth[args.id_column].astype(str)
    pred[args.id_column] = pred[args.id_column].astype(str)
    merged = truth[[args.id_column, "cve_labels"]].merge(
        pred[[args.id_column, "cve_labels"]],
        on=args.id_column,
        how="left",
        suffixes=("_true", "_pred"),
    )

    stats: dict[str, Counter[str]] = defaultdict(Counter)
    for expected, predicted in zip(
        merged["cve_labels_true"].fillna("").apply(labels_from_text),
        merged["cve_labels_pred"].fillna("").apply(labels_from_text),
    ):
        for label in expected & predicted:
            stats[label]["tp"] += 1
        for label in predicted - expected:
            stats[label]["fp"] += 1
        for label in expected - predicted:
            stats[label]["fn"] += 1

    rows: list[dict[str, float | int | str]] = []
    for label, counts in stats.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "label": label,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "support": tp + fn,
                "predicted": tp + fp,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    summary = pd.DataFrame(rows).sort_values(["f1", "support", "label"], ascending=[True, False, True])
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary, index=False)

    zero_f1 = int((summary["f1"] == 0).sum()) if not summary.empty else 0
    low_f1 = int((summary["f1"] < 0.2).sum()) if not summary.empty else 0
    print(f"labels: {len(summary)}")
    print(f"zero_f1_labels: {zero_f1}")
    print(f"low_f1_labels_lt_0.2: {low_f1}")
    print("\nLowest F1 labels:")
    print(summary.head(args.top_n).to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nTop FN labels:")
    print(summary.sort_values(["fn", "support"], ascending=[False, False]).head(args.top_n).to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nWrote {args.output_summary}")


if __name__ == "__main__":
    main()
