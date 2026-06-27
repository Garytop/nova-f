from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import clean_payload_text, normalize_cve_labels


def label_set(value: object) -> set[str]:
    return set(normalize_cve_labels(value))


def load_optional_search(search_path: Path | None, meta_path: Path | None) -> tuple[np.ndarray | None, np.ndarray | None, list[list[str]] | None]:
    if not search_path or not meta_path:
        return None, None, None
    search = np.load(search_path)
    distances = search["D"]
    indices = search["I"]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    train_labels = [normalize_cve_labels(value) for value in meta["cve_labels"]]
    return distances, indices, train_labels


def neighbour_summary(
    row_idx: int,
    distances: np.ndarray | None,
    indices: np.ndarray | None,
    train_labels: list[list[str]] | None,
    *,
    top_n: int,
) -> str:
    if distances is None or indices is None or train_labels is None:
        return ""
    parts: list[str] = []
    for idx, score in zip(indices[row_idx][:top_n], distances[row_idx][:top_n]):
        if idx < 0 or idx >= len(train_labels):
            continue
        labels = " ".join(train_labels[idx]) if train_labels[idx] else "EMPTY"
        parts.append(f"{float(score):.4f}:{labels}")
    return " | ".join(parts)


def summarize_patterns(payloads: list[str], *, top_n: int) -> str:
    methods: Counter[str] = Counter()
    first_paths: Counter[str] = Counter()
    tokens: Counter[str] = Counter()

    suspicious = [
        "../",
        "%2e%2e",
        "union",
        "select",
        "jndi",
        "<script",
        "cmd=",
        "wget",
        "curl",
        "<?php",
        "base64",
        "eval(",
    ]

    for payload in payloads:
        cleaned = clean_payload_text(payload)
        first = cleaned.split(" ", 2)
        if first:
            methods[first[0].upper()] += 1
        if len(first) > 1:
            path = first[1].split("?", 1)[0]
            first_paths[path[:80]] += 1
        lower = cleaned.lower()
        for token in suspicious:
            if token in lower:
                tokens[token] += 1

    return json.dumps(
        {
            "methods": methods.most_common(top_n),
            "paths": first_paths.most_common(top_n),
            "suspicious_tokens": tokens.most_common(top_n),
        },
        ensure_ascii=False,
    )


def build_analysis(
    truth_path: Path,
    pred_path: Path,
    *,
    id_column: str,
    payload_column: str,
    search_path: Path | None,
    meta_path: Path | None,
    examples_per_label: int,
    neighbour_top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = pd.read_csv(truth_path)
    pred = pd.read_csv(pred_path)
    if id_column not in truth.columns or id_column not in pred.columns:
        raise KeyError(f"Both files must contain {id_column!r}")
    if "cve_labels" not in truth.columns or "cve_labels" not in pred.columns:
        raise KeyError("Both files must contain cve_labels")

    truth[id_column] = truth[id_column].astype(str)
    pred[id_column] = pred[id_column].astype(str)
    columns = [id_column, "cve_labels"]
    if payload_column in truth.columns:
        columns.append(payload_column)
    merged = truth[columns].merge(
        pred[[id_column, "cve_labels"]],
        on=id_column,
        how="left",
        suffixes=("_true", "_pred"),
    )

    true_sets = [label_set(value) for value in merged["cve_labels_true"].fillna("")]
    pred_sets = [label_set(value) for value in merged["cve_labels_pred"].fillna("")]
    payloads = merged[payload_column].fillna("").astype(str).tolist() if payload_column in merged.columns else ["" for _ in range(len(merged))]

    distances, indices, train_labels = load_optional_search(search_path, meta_path)
    if distances is not None and len(distances) != len(merged):
        raise ValueError(f"search rows ({len(distances)}) != merged rows ({len(merged)})")

    per_label: dict[str, Counter[str]] = defaultdict(Counter)
    example_rows: list[dict[str, Any]] = []
    fp_payloads: dict[str, list[str]] = defaultdict(list)
    fn_payloads: dict[str, list[str]] = defaultdict(list)

    for row_idx, (row_id, expected, predicted, payload) in enumerate(
        zip(merged[id_column], true_sets, pred_sets, payloads)
    ):
        for label in expected & predicted:
            per_label[label]["tp"] += 1
        for label in predicted - expected:
            per_label[label]["fp"] += 1
            if len(fp_payloads[label]) < examples_per_label:
                fp_payloads[label].append(payload)
            if len([item for item in example_rows if item["label"] == label and item["type"] == "FP"]) < examples_per_label:
                example_rows.append(
                    {
                        "type": "FP",
                        "label": label,
                        "id": row_id,
                        "true": " ".join(sorted(expected)),
                        "pred": " ".join(sorted(predicted)),
                        "payload_clean": clean_payload_text(payload)[:500],
                        "neighbours": neighbour_summary(
                            row_idx,
                            distances,
                            indices,
                            train_labels,
                            top_n=neighbour_top_n,
                        ),
                    }
                )
        for label in expected - predicted:
            per_label[label]["fn"] += 1
            if len(fn_payloads[label]) < examples_per_label:
                fn_payloads[label].append(payload)
            if len([item for item in example_rows if item["label"] == label and item["type"] == "FN"]) < examples_per_label:
                example_rows.append(
                    {
                        "type": "FN",
                        "label": label,
                        "id": row_id,
                        "true": " ".join(sorted(expected)),
                        "pred": " ".join(sorted(predicted)),
                        "payload_clean": clean_payload_text(payload)[:500],
                        "neighbours": neighbour_summary(
                            row_idx,
                            distances,
                            indices,
                            train_labels,
                            top_n=neighbour_top_n,
                        ),
                    }
                )

    summary_rows: list[dict[str, Any]] = []
    for label, counts in per_label.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        summary_rows.append(
            {
                "label": label,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fp_patterns": summarize_patterns(fp_payloads[label], top_n=3) if fp_payloads[label] else "",
                "fn_patterns": summarize_patterns(fn_payloads[label], top_n=3) if fn_payloads[label] else "",
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["fp", "fn", "label"], ascending=[False, False, True])
    examples = pd.DataFrame(example_rows)
    return summary, examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze nova-f false positives/false negatives by CVE label.")
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--payload-column", default="payload_decoded")
    parser.add_argument("--search", type=Path, default=None, help="Optional NPZ top-k search cache with D/I arrays")
    parser.add_argument("--meta", type=Path, default=None, help="Optional FAISS meta.json for neighbour label summaries")
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--output-examples", required=True, type=Path)
    parser.add_argument("--examples-per-label", type=int, default=3)
    parser.add_argument("--neighbour-top-n", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, examples = build_analysis(
        args.truth,
        args.pred,
        id_column=args.id_column,
        payload_column=args.payload_column,
        search_path=args.search,
        meta_path=args.meta,
        examples_per_label=args.examples_per_label,
        neighbour_top_n=args.neighbour_top_n,
    )
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_examples.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary, index=False)
    examples.to_csv(args.output_examples, index=False)

    print("\nTop false positives:")
    print(summary.sort_values("fp", ascending=False).head(args.top_n).to_string(index=False))
    print("\nTop false negatives:")
    print(summary.sort_values("fn", ascending=False).head(args.top_n).to_string(index=False))
    print(f"\nWrote {args.output_summary}")
    print(f"Wrote {args.output_examples}")


if __name__ == "__main__":
    main()
