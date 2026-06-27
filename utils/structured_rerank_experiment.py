from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import clean_payload_text, normalize_cve_labels
from src.search_faiss import adaptive_predict, aggregate_cve_candidates, has_cve_label, should_suppress_by_empty_neighbors
from src.structured_features import feature_bonus, parse_payload
from utils.evaluate_predictions import evaluate


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def load_train_payloads(paths: list[Path], payload_column: str) -> list[str]:
    payloads: list[str] = []
    for path in paths:
        frame = pd.read_csv(path)
        if payload_column not in frame.columns:
            raise KeyError(f"{path} missing {payload_column!r}")
        payloads.extend(frame[payload_column].fillna("").astype(str).tolist())
    return payloads


def labels_from_text(value: object) -> set[str]:
    return set(normalize_cve_labels(value))


def predict_sets(
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


def write_predictions(ids: list[str], pred_sets: list[set[str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id": ids,
            "cve_labels": [" ".join(sorted(labels)) for labels in pred_sets],
        }
    ).to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment with structural reranking over cached FAISS neighbours.")
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--search", required=True, type=Path)
    parser.add_argument("--meta", required=True, type=Path)
    parser.add_argument("--train", action="append", required=True, type=Path)
    parser.add_argument("--train-payload-column", default="payload_clean")
    parser.add_argument("--payload-column", default="payload_decoded")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--alphas", default="0,0.01,0.02,0.03,0.05")
    parser.add_argument("--base-threshold", type=float, default=0.86)
    parser.add_argument("--vote-weight", type=float, default=0.015)
    parser.add_argument("--empty-margin", type=float, default=0.05)
    parser.add_argument("--empty-floor", type=float, default=0.80)
    parser.add_argument("--empty-ratio", type=float, default=0.50)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--best-output", type=Path, default=None)
    args = parser.parse_args()

    truth = pd.read_csv(args.truth)
    if args.payload_column not in truth.columns:
        raise KeyError(f"truth missing {args.payload_column!r}")
    ids = truth[args.id_column].astype(str).tolist()
    true_sets = [labels_from_text(value) for value in truth["cve_labels"].fillna("")]

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    train_labels = [normalize_cve_labels(value) for value in meta["cve_labels"]]
    train_has_cve = [has_cve_label(labels) for labels in train_labels]
    train_payloads = load_train_payloads(args.train, args.train_payload_column)
    if len(train_payloads) != len(train_labels):
        raise ValueError(f"train payload rows ({len(train_payloads)}) != meta labels ({len(train_labels)})")

    search = np.load(args.search)
    base_distances = search["D"].astype(np.float32)
    indices = search["I"]
    if len(base_distances) != len(truth):
        raise ValueError(f"search rows ({len(base_distances)}) != truth rows ({len(truth)})")

    print("Parsing structural features...", flush=True)
    test_features = [parse_payload(payload) for payload in truth[args.payload_column].fillna("")]
    train_features = [parse_payload(payload) for payload in train_payloads]

    print("Computing feature bonuses...", flush=True)
    bonuses = np.zeros_like(base_distances, dtype=np.float32)
    for row_idx, idxs in enumerate(indices):
        query_features = test_features[row_idx]
        for col_idx, train_idx in enumerate(idxs):
            if train_idx < 0 or train_idx >= len(train_features):
                continue
            bonuses[row_idx, col_idx] = feature_bonus(query_features, train_features[int(train_idx)])

    mask = np.ones(len(true_sets), dtype=bool)
    rows: list[dict[str, float]] = []
    best_metric = -1.0
    best_pred_sets: list[set[str]] | None = None
    best_alpha = 0.0
    for alpha in parse_float_list(args.alphas):
        distances = base_distances + (alpha * bonuses)
        pred_sets = predict_sets(
            distances,
            indices,
            train_labels,
            train_has_cve,
            base_threshold=args.base_threshold,
            vote_weight=args.vote_weight,
            empty_margin=args.empty_margin,
            empty_floor=args.empty_floor,
            empty_ratio=args.empty_ratio,
            max_candidates=args.max_candidates,
        )
        temp_output = args.output_summary.with_suffix(f".alpha_{alpha:.3f}.tmp.csv")
        write_predictions(ids, pred_sets, temp_output)
        metrics = evaluate(args.truth, temp_output, id_column=args.id_column)
        temp_output.unlink(missing_ok=True)
        row = {"alpha": alpha, **metrics}
        rows.append(row)
        print(
            f"alpha={alpha:.3f} precision={metrics['precision']:.6f} recall={metrics['recall']:.6f} "
            f"micro_f1={metrics['micro_f1']:.6f} macro_f1={metrics['macro_f1']:.6f}",
            flush=True,
        )
        if metrics["micro_f1"] > best_metric:
            best_metric = metrics["micro_f1"]
            best_pred_sets = pred_sets
            best_alpha = alpha

    summary = pd.DataFrame(rows).sort_values("micro_f1", ascending=False)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary, index=False)
    if args.best_output and best_pred_sets is not None:
        write_predictions(ids, best_pred_sets, args.best_output)
        print(f"Best alpha={best_alpha:.3f}; wrote {args.best_output}")
    print(f"Wrote {args.output_summary}")


if __name__ == "__main__":
    main()
