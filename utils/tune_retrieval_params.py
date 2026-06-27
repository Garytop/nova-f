from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import normalize_cve_labels
from src.search_faiss import (
    adaptive_predict,
    aggregate_cve_candidates,
    has_cve_label,
    should_suppress_by_empty_neighbors,
)


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def labels_from_text(value: object) -> set[str]:
    return set(normalize_cve_labels(value))


def evaluate(true_sets: list[set[str]], pred_sets: list[set[str]], mask: np.ndarray) -> dict[str, float]:
    tp = fp = fn = exact = answered = 0
    per_label: dict[str, Counter[str]] = {}
    rows = int(mask.sum())

    for expected, predicted, use_row in zip(true_sets, pred_sets, mask):
        if not use_row:
            continue
        if predicted:
            answered += 1
        if expected == predicted:
            exact += 1

        tp += len(expected & predicted)
        fp += len(predicted - expected)
        fn += len(expected - predicted)

        for label in expected | predicted:
            bucket = per_label.setdefault(label, Counter())
            if label in expected and label in predicted:
                bucket["tp"] += 1
            elif label in predicted:
                bucket["fp"] += 1
            else:
                bucket["fn"] += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    f1_scores: list[float] = []
    for counts in per_label.values():
        label_precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
        label_recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
        f1_scores.append(
            2 * label_precision * label_recall / (label_precision + label_recall)
            if label_precision + label_recall
            else 0.0
        )

    return {
        "rows": float(rows),
        "answer_rate": answered / rows if rows else 0.0,
        "exact_match": exact / rows if rows else 0.0,
        "precision": precision,
        "recall": recall,
        "micro_f1": micro_f1,
        "macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
    }


def predict_from_search(
    distances: np.ndarray,
    indices: np.ndarray,
    train_labels: list[list[str]],
    train_has_cve: list[bool],
    *,
    base_threshold: float,
    vote_weight: float,
    empty_margin: float,
    empty_floor: float,
    empty_ratio: float,
    max_candidates: int,
) -> list[set[str]]:
    pred_sets: list[set[str]] = []
    for idxs, sims in zip(indices, distances):
        candidates, _ = aggregate_cve_candidates(
            idxs,
            sims,
            train_labels,
            base_threshold=base_threshold,
            max_candidates=max_candidates,
            min_votes=1,
            vote_weight=vote_weight,
        )
        preds = adaptive_predict(
            candidates,
            base_threshold=base_threshold,
            high_confidence=max(base_threshold + 0.01, 0.87),
            medium_confidence=max(base_threshold - 0.08, 0.78),
            max_diff_second=0.15,
            max_diff_third=0.05,
        )
        if preds and should_suppress_by_empty_neighbors(
            idxs,
            sims,
            train_has_cve,
            base_threshold=base_threshold,
            empty_penalty_margin=empty_margin,
            empty_penalty_floor=empty_floor,
            empty_penalty_ratio=empty_ratio,
        ):
            preds = []
        pred_sets.append(set(preds))
    return pred_sets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune nova-f retrieval thresholds from cached top-k search results.")
    parser.add_argument("--truth", required=True, type=Path, help="Truth CSV containing id/cve_labels")
    parser.add_argument("--search", required=True, type=Path, help="NPZ file containing D and I arrays")
    parser.add_argument("--meta", required=True, type=Path, help="FAISS metadata JSON containing cve_labels")
    parser.add_argument("--bases", default="0.84,0.86,0.88,0.90", help="Comma-separated base thresholds")
    parser.add_argument("--vote-weights", default="0,0.015", help="Comma-separated vote weights")
    parser.add_argument("--empty-margins", default="-1,0.02,0.05,0.08", help="Comma-separated empty-neighbour margins; negative disables suppression")
    parser.add_argument("--empty-floors", default="0.80,0.84,0.88", help="Comma-separated empty-neighbour floors")
    parser.add_argument("--empty-ratio", type=float, default=0.5)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--sort-by", default="micro_f1", choices=["exact_match", "precision", "recall", "micro_f1", "macro_f1"])
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output for all sweep rows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    truth = pd.read_csv(args.truth)
    if "cve_labels" not in truth.columns:
        raise KeyError("truth CSV must contain cve_labels")
    true_sets = [labels_from_text(value) for value in truth["cve_labels"].fillna("")]

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    train_labels = [normalize_cve_labels(value) for value in meta["cve_labels"]]
    train_has_cve = [has_cve_label(labels) for labels in train_labels]

    search = np.load(args.search)
    distances = search["D"]
    indices = search["I"]
    if len(distances) != len(true_sets):
        raise ValueError(f"search rows ({len(distances)}) != truth rows ({len(true_sets)})")

    mask_all = np.ones(len(true_sets), dtype=bool)
    rows: list[dict[str, float | str]] = []
    for base in parse_float_list(args.bases):
        for vote_weight in parse_float_list(args.vote_weights):
            for empty_margin in parse_float_list(args.empty_margins):
                floors = [0.0] if empty_margin < 0 else parse_float_list(args.empty_floors)
                for empty_floor in floors:
                    pred_sets = predict_from_search(
                        distances,
                        indices,
                        train_labels,
                        train_has_cve,
                        base_threshold=base,
                        vote_weight=vote_weight,
                        empty_margin=empty_margin,
                        empty_floor=empty_floor,
                        empty_ratio=args.empty_ratio,
                        max_candidates=args.max_candidates,
                    )
                    metrics = evaluate(true_sets, pred_sets, mask_all)
                    rows.append(
                        {
                            "base_threshold": base,
                            "vote_weight": vote_weight,
                            "empty_margin": empty_margin,
                            "empty_floor": empty_floor,
                            **metrics,
                        }
                    )

    result = pd.DataFrame(rows).sort_values(args.sort_by, ascending=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
    print(result.head(args.top_n).to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
