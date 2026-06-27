from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import clean_payload_text, normalize_cve_labels


def labels_from_text(value: object) -> set[str]:
    return set(normalize_cve_labels(value))


def parse_request(payload: object) -> dict[str, object]:
    cleaned = clean_payload_text("" if payload is None else str(payload))
    parts = cleaned.split(" ", 2)
    method = parts[0].upper() if parts else ""
    target = parts[1] if len(parts) > 1 else ""
    try:
        split = urlsplit(target)
        raw_path = split.path
        raw_query = split.query
    except ValueError:
        raw_path, _, raw_query = target.partition("?")
    path = raw_path.lower()
    query_keys = sorted({key.lower() for key, _ in parse_qsl(raw_query, keep_blank_values=True)})
    lower = cleaned.lower()
    body = parts[2] if len(parts) > 2 else ""
    body_keys: set[str] = set()
    for key, _ in parse_qsl(body, keep_blank_values=True):
        if key:
            body_keys.add(key.lower())
    return {
        "cleaned": cleaned,
        "method": method,
        "path": path,
        "query_keys": query_keys,
        "body_keys": sorted(body_keys),
        "tokens": sorted(token for token in SUSPICIOUS_TOKENS if token in lower),
    }


SUSPICIOUS_TOKENS = [
    "../",
    "%2e%2e",
    "base64",
    "cmd=",
    "curl",
    "eval(",
    "jndi",
    "php",
    "select",
    "union",
    "wget",
    "whoami",
    "xml",
]


def summarize(items: list[object], limit: int) -> str:
    counter = Counter(items)
    return json.dumps(counter.most_common(limit), ensure_ascii=False)


def summarize_sets(items: list[list[str]], limit: int) -> str:
    counter: Counter[str] = Counter()
    for values in items:
        counter.update(values)
    return json.dumps(counter.most_common(limit), ensure_ascii=False)


def training_support(frame: pd.DataFrame, *, payload_column: str, label_column: str) -> dict[str, dict[str, object]]:
    stats: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "train_support": 0,
            "train_methods": [],
            "train_paths": [],
            "train_query_keys": [],
            "train_body_keys": [],
            "train_tokens": [],
        }
    )
    for payload, raw_labels in zip(frame[payload_column].fillna(""), frame[label_column].fillna("")):
        labels = labels_from_text(raw_labels)
        if not labels:
            continue
        parsed = parse_request(payload)
        for label in labels:
            bucket = stats[label]
            bucket["train_support"] = int(bucket["train_support"]) + 1
            bucket["train_methods"].append(parsed["method"])
            bucket["train_paths"].append(parsed["path"])
            bucket["train_query_keys"].append(parsed["query_keys"])
            bucket["train_body_keys"].append(parsed["body_keys"])
            bucket["train_tokens"].append(parsed["tokens"])
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose zero-F1/high-FN CVEs with train/test structural coverage.")
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--train", action="append", required=True, type=Path)
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--payload-column", default="payload_decoded")
    parser.add_argument("--train-payload-column", default="payload_clean")
    parser.add_argument("--label-column", default="cve_labels")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=80)
    args = parser.parse_args()

    truth = pd.read_csv(args.truth)
    pred = pd.read_csv(args.pred)
    truth[args.id_column] = truth[args.id_column].astype(str)
    pred[args.id_column] = pred[args.id_column].astype(str)
    merged = truth[[args.id_column, args.payload_column, args.label_column]].merge(
        pred[[args.id_column, args.label_column]],
        on=args.id_column,
        how="left",
        suffixes=("_true", "_pred"),
    )

    train_frames = []
    for path in args.train:
        frame = pd.read_csv(path)
        if args.train_payload_column not in frame.columns:
            raise KeyError(f"{path} missing {args.train_payload_column!r}")
        train_frames.append(frame[[args.train_payload_column, args.label_column]].copy())
    train = pd.concat(train_frames, ignore_index=True)
    support = training_support(train, payload_column=args.train_payload_column, label_column=args.label_column)

    per_label: dict[str, Counter[str]] = defaultdict(Counter)
    fn_payloads: dict[str, list[dict[str, object]]] = defaultdict(list)
    fp_payloads: dict[str, list[dict[str, object]]] = defaultdict(list)

    for payload, raw_true, raw_pred in zip(
        merged[args.payload_column].fillna(""),
        merged[f"{args.label_column}_true"].fillna(""),
        merged[f"{args.label_column}_pred"].fillna(""),
    ):
        expected = labels_from_text(raw_true)
        predicted = labels_from_text(raw_pred)
        parsed = parse_request(payload)
        for label in expected & predicted:
            per_label[label]["tp"] += 1
        for label in predicted - expected:
            per_label[label]["fp"] += 1
            if len(fp_payloads[label]) < 20:
                fp_payloads[label].append(parsed)
        for label in expected - predicted:
            per_label[label]["fn"] += 1
            if len(fn_payloads[label]) < 20:
                fn_payloads[label].append(parsed)

    rows: list[dict[str, object]] = []
    for label, counts in per_label.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        train_info = support.get(label, {})
        fn_items = fn_payloads.get(label, [])
        fp_items = fp_payloads.get(label, [])
        rows.append(
            {
                "label": label,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "support": tp + fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "train_support": train_info.get("train_support", 0),
                "fn_methods": summarize([item["method"] for item in fn_items], 8),
                "fn_paths": summarize([item["path"] for item in fn_items], 12),
                "fn_query_keys": summarize_sets([item["query_keys"] for item in fn_items], 12),
                "fn_body_keys": summarize_sets([item["body_keys"] for item in fn_items], 12),
                "fn_tokens": summarize_sets([item["tokens"] for item in fn_items], 12),
                "fp_paths": summarize([item["path"] for item in fp_items], 12),
                "train_paths": summarize(train_info.get("train_paths", []), 12),
                "train_query_keys": summarize_sets(train_info.get("train_query_keys", []), 12),
                "train_body_keys": summarize_sets(train_info.get("train_body_keys", []), 12),
                "train_tokens": summarize_sets(train_info.get("train_tokens", []), 12),
            }
        )

    result = pd.DataFrame(rows).sort_values(["f1", "fn", "support"], ascending=[True, False, False])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print(f"labels: {len(result)}")
    print(f"zero_f1: {int((result['f1'] == 0).sum()) if not result.empty else 0}")
    print("\nTop zero/low-F1 high-FN labels:")
    print(
        result.sort_values(["f1", "fn"], ascending=[True, False])
        .head(args.top_n)
        .to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
