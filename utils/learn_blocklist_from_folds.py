from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import faiss
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


def evaluate(true_sets: list[set[str]], pred_sets: list[set[str]]) -> dict[str, float]:
    tp = fp = fn = exact = answered = 0
    per_label: dict[str, Counter[str]] = {}
    for expected, predicted in zip(true_sets, pred_sets):
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
        "rows": float(len(true_sets)),
        "answer_rate": answered / len(true_sets) if true_sets else 0.0,
        "exact_match": exact / len(true_sets) if true_sets else 0.0,
        "precision": precision,
        "recall": recall,
        "micro_f1": micro_f1,
        "macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
    }


def per_label_stats(true_sets: list[set[str]], pred_sets: list[set[str]]) -> pd.DataFrame:
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    for expected, predicted in zip(true_sets, pred_sets):
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
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows).sort_values(["fp", "fn", "label"], ascending=[False, False, True])


def predict_fold(
    index_vectors: np.ndarray,
    query_vectors: np.ndarray,
    train_labels: list[list[str]],
    *,
    top_k: int,
    base_threshold: float,
    vote_weight: float,
    max_candidates: int,
    empty_penalty_margin: float,
    empty_penalty_floor: float,
    empty_penalty_ratio: float,
    search_batch_size: int,
) -> list[set[str]]:
    train_has_cve = [has_cve_label(labels) for labels in train_labels]
    vectors = index_vectors.astype("float32", copy=True)
    queries = query_vectors.astype("float32", copy=True)
    faiss.normalize_L2(vectors)
    faiss.normalize_L2(queries)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    pred_sets: list[set[str]] = []
    for start in range(0, len(queries), search_batch_size):
        end = min(start + search_batch_size, len(queries))
        distances, indices = index.search(queries[start:end], top_k)
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
                high_confidence=0.87,
                medium_confidence=0.78,
                max_diff_second=0.15,
                max_diff_third=0.05,
            )
            if preds and should_suppress_by_empty_neighbors(
                idxs,
                sims,
                train_has_cve,
                base_threshold=base_threshold,
                empty_penalty_margin=empty_penalty_margin,
                empty_penalty_floor=empty_penalty_floor,
                empty_penalty_ratio=empty_penalty_ratio,
            ):
                preds = []
            pred_sets.append(set(preds))
    return pred_sets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn a CVE blocklist from out-of-fold predictions on an existing vector store.")
    parser.add_argument("--store-dir", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--base-threshold", type=float, default=0.86)
    parser.add_argument("--vote-weight", type=float, default=0.015)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--empty-penalty-margin", type=float, default=0.05)
    parser.add_argument("--empty-penalty-floor", type=float, default=0.80)
    parser.add_argument("--empty-penalty-ratio", type=float, default=0.50)
    parser.add_argument("--search-batch-size", type=int, default=1024)
    parser.add_argument("--min-fp", type=int, default=10)
    parser.add_argument("--max-precision", type=float, default=0.10)
    parser.add_argument("--min-folds", type=int, default=2, help="Minimum folds where a label must meet block criteria")
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--output-fold-summary", required=True, type=Path)
    parser.add_argument("--output-blocklist", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta_path = args.store_dir / "meta.json"
    vectors_path = args.store_dir / "train_embeddings.npy"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    vectors = np.load(vectors_path).astype("float32")
    labels = [normalize_cve_labels(value) for value in meta["cve_labels"]]
    true_sets = [set(label for label in item if label.startswith("CVE-")) for item in labels]

    if len(vectors) != len(labels):
        raise ValueError(f"vectors rows ({len(vectors)}) != labels rows ({len(labels)})")
    if args.folds < 2:
        raise ValueError("--folds must be >= 2")

    rng = np.random.default_rng(args.seed)
    order = np.arange(len(vectors))
    rng.shuffle(order)
    fold_ids = np.empty(len(vectors), dtype=np.int32)
    for fold_no, fold_indices in enumerate(np.array_split(order, args.folds)):
        fold_ids[fold_indices] = fold_no

    all_true: list[set[str]] = []
    all_pred: list[set[str]] = []
    fold_rows: list[dict[str, float | int]] = []
    fold_block_hits: dict[str, int] = defaultdict(int)

    for fold_no in range(args.folds):
        valid_mask = fold_ids == fold_no
        train_mask = ~valid_mask
        train_labels = [labels[i] for i in np.where(train_mask)[0]]
        fold_pred = predict_fold(
            vectors[train_mask],
            vectors[valid_mask],
            train_labels,
            top_k=args.top_k,
            base_threshold=args.base_threshold,
            vote_weight=args.vote_weight,
            max_candidates=args.max_candidates,
            empty_penalty_margin=args.empty_penalty_margin,
            empty_penalty_floor=args.empty_penalty_floor,
            empty_penalty_ratio=args.empty_penalty_ratio,
            search_batch_size=args.search_batch_size,
        )
        fold_true = [true_sets[i] for i in np.where(valid_mask)[0]]
        all_true.extend(fold_true)
        all_pred.extend(fold_pred)

        fold_metrics = evaluate(fold_true, fold_pred)
        fold_metrics["fold"] = fold_no
        fold_rows.append(fold_metrics)

        fold_stats = per_label_stats(fold_true, fold_pred)
        blocked_in_fold = fold_stats[
            (fold_stats["fp"] >= args.min_fp)
            & (fold_stats["precision"] <= args.max_precision)
        ]
        for label in blocked_in_fold["label"].tolist():
            fold_block_hits[str(label)] += 1
        print(
            f"fold={fold_no} rows={len(fold_true)} micro_f1={fold_metrics['micro_f1']:.6f} "
            f"precision={fold_metrics['precision']:.6f} recall={fold_metrics['recall']:.6f} "
            f"candidate_blocks={len(blocked_in_fold)}",
            flush=True,
        )

    summary = per_label_stats(all_true, all_pred)
    summary["block_folds"] = summary["label"].map(lambda label: fold_block_hits.get(str(label), 0))
    blocklist = sorted(
        summary[
            (summary["block_folds"] >= args.min_folds)
            & (summary["fp"] >= args.min_fp)
            & (summary["precision"] <= args.max_precision)
        ]["label"].tolist()
    )

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary, index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_fold_summary, index=False)
    args.output_blocklist.write_text("\n".join(blocklist) + ("\n" if blocklist else ""), encoding="utf-8")

    overall = evaluate(all_true, all_pred)
    filtered_pred = [pred - set(blocklist) for pred in all_pred]
    filtered = evaluate(all_true, filtered_pred)
    print("\nOOF baseline:")
    for key, value in overall.items():
        print(f"{key}: {value:.6f}")
    print("\nOOF filtered by learned blocklist:")
    for key, value in filtered.items():
        print(f"{key}: {value:.6f}")
    print("\nLearned blocklist:")
    for label in blocklist:
        print(label)


if __name__ == "__main__":
    main()
