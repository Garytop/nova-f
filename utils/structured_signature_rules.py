from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import clean_payload_text, normalize_cve_labels
from src.structured_features import candidate_signatures


def load_frame(paths: list[Path], payload_column: str, label_column: str) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if payload_column not in frame.columns:
            raise KeyError(f"{path} missing {payload_column!r}")
        if label_column not in frame.columns:
            raise KeyError(f"{path} missing {label_column!r}")
        frames.append(frame[[payload_column, label_column]].copy())
    return pd.concat(frames, ignore_index=True)


def mine_rules(
    signatures_list: list[list[str]],
    labels_list: list[set[str]],
    rows: np.ndarray,
    *,
    min_support: int,
    min_precision: float,
    max_labels_per_signature: int,
) -> list[dict[str, object]]:
    signature_counts: Counter[str] = Counter()
    signature_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row_idx in rows:
        labels = labels_list[int(row_idx)]
        if not labels:
            continue
        for signature in signatures_list[int(row_idx)]:
            signature_counts[signature] += 1
            signature_label_counts[signature].update(labels)

    rules: list[dict[str, object]] = []
    for signature, total_hits in signature_counts.items():
        for label, support in signature_label_counts[signature].most_common(max_labels_per_signature):
            precision = support / total_hits if total_hits else 0.0
            if support >= min_support and precision >= min_precision:
                rules.append(
                    {
                        "signature": signature,
                        "label": label,
                        "support": support,
                        "total_hits": total_hits,
                        "precision": precision,
                        "rule_type": signature.split("|", 1)[0],
                    }
                )
    return rules


def evaluate_rule_set(
    signatures_list: list[list[str]],
    labels_list: list[set[str]],
    rows: np.ndarray,
    rules: list[dict[str, object]],
    *,
    max_additions: int,
) -> dict[str, float | int]:
    rule_map = {str(rule["signature"]): str(rule["label"]) for rule in rules}
    target_labels = set(rule_map.values())
    tp = fp = fn = changed = 0
    per_label: dict[str, Counter[str]] = defaultdict(Counter)
    for row_idx in rows:
        expected = labels_list[int(row_idx)] & target_labels
        additions: list[str] = []
        for signature in signatures_list[int(row_idx)]:
            label = rule_map.get(signature)
            if label and label not in additions:
                additions.append(label)
                if len(additions) >= max_additions:
                    break
        predicted = set(additions)
        if predicted:
            changed += 1
        tp += len(expected & predicted)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
        for label in expected | predicted:
            bucket = per_label[label]
            if label in expected and label in predicted:
                bucket["tp"] += 1
            elif label in predicted:
                bucket["fp"] += 1
            else:
                bucket["fn"] += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f1s = []
    for counts in per_label.values():
        label_precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
        label_recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
        f1s.append(2 * label_precision * label_recall / (label_precision + label_recall) if label_precision + label_recall else 0.0)
    return {
        "rule_count": len(rules),
        "changed_rows": changed,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "micro_f1": micro_f1,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
    }


def run_mine(args: argparse.Namespace) -> None:
    modes = set(args.modes.split(","))
    frame = load_frame(args.input, args.payload_column, args.label_column)
    labels_list = [set(label for label in normalize_cve_labels(value) if label.startswith("CVE-")) for value in frame[args.label_column].fillna("")]
    signatures_list = [candidate_signatures(payload, modes=modes) for payload in frame[args.payload_column].fillna("")]
    rules = mine_rules(
        signatures_list,
        labels_list,
        np.arange(len(frame)),
        min_support=args.min_support,
        min_precision=args.min_precision,
        max_labels_per_signature=args.max_labels_per_signature,
    )
    result = pd.DataFrame(rules).sort_values(["support", "precision", "signature"], ascending=[False, False, True])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.head(args.top_n).to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Wrote {args.output}")


def run_oof(args: argparse.Namespace) -> None:
    modes = set(args.modes.split(","))
    frame = load_frame(args.input, args.payload_column, args.label_column)
    labels_list = [set(label for label in normalize_cve_labels(value) if label.startswith("CVE-")) for value in frame[args.label_column].fillna("")]
    signatures_list = [candidate_signatures(payload, modes=modes) for payload in frame[args.payload_column].fillna("")]
    indices = np.arange(len(frame))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)
    fold_ids = np.empty(len(frame), dtype=np.int32)
    for fold_no, fold_rows in enumerate(np.array_split(indices, args.folds)):
        fold_ids[fold_rows] = fold_no

    fold_rows_out: list[dict[str, object]] = []
    rule_rows: list[dict[str, object]] = []
    for fold_no in range(args.folds):
        train_rows = np.where(fold_ids != fold_no)[0]
        valid_rows = np.where(fold_ids == fold_no)[0]
        rules = mine_rules(
            signatures_list,
            labels_list,
            train_rows,
            min_support=args.min_support,
            min_precision=args.min_precision,
            max_labels_per_signature=args.max_labels_per_signature,
        )
        metrics = evaluate_rule_set(signatures_list, labels_list, valid_rows, rules, max_additions=args.max_additions)
        fold_rows_out.append({"fold": fold_no, **metrics})
        for rule in rules:
            rule_rows.append({"fold": fold_no, **rule})
        print(
            f"fold={fold_no} rules={len(rules)} changed={metrics['changed_rows']} "
            f"precision={metrics['precision']:.6f} recall={metrics['recall']:.6f} "
            f"micro_f1={metrics['micro_f1']:.6f}",
            flush=True,
        )

    folds = pd.DataFrame(fold_rows_out)
    rules_df = pd.DataFrame(rule_rows)
    args.output_folds.parent.mkdir(parents=True, exist_ok=True)
    args.output_rules.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.output_folds, index=False)
    rules_df.to_csv(args.output_rules, index=False)
    print("\nMean metrics:")
    print(folds[["precision", "recall", "micro_f1", "macro_f1"]].mean().to_string(float_format=lambda value: f"{value:.6f}"))
    print(f"Wrote {args.output_folds}")
    print(f"Wrote {args.output_rules}")


def run_apply(args: argparse.Namespace) -> None:
    modes = set(args.modes.split(","))
    rules = pd.read_csv(args.rules)
    required = {"signature", "label", "support", "precision"}
    missing = required - set(rules.columns)
    if missing:
        raise KeyError(f"rules missing columns: {', '.join(sorted(missing))}")
    rules = rules[(rules["support"].astype(int) >= args.min_support) & (rules["precision"].astype(float) >= args.min_precision)]
    if args.rule_type:
        rules = rules[rules["rule_type"].isin(args.rule_type)]
    rule_map = {str(row.signature): str(row.label) for row in rules.itertuples(index=False)}

    source = pd.read_csv(args.truth_or_test)
    pred = pd.read_csv(args.pred)
    source[args.id_column] = source[args.id_column].astype(str)
    pred[args.id_column] = pred[args.id_column].astype(str)
    merged = pred.merge(source[[args.id_column, args.payload_column]], on=args.id_column, how="left", validate="one_to_one")

    rows: list[dict[str, str]] = []
    changes: list[dict[str, str]] = []
    for row in merged.itertuples(index=False):
        row_id = str(getattr(row, args.id_column))
        labels = normalize_cve_labels(getattr(row, "cve_labels"))
        payload = getattr(row, args.payload_column)
        additions: list[str] = []
        if not args.only_empty or not labels:
            for signature in candidate_signatures(payload, modes=modes):
                label = rule_map.get(signature)
                if label and label not in labels and label not in additions:
                    additions.append(label)
                    if len(additions) >= args.max_additions:
                        break
        final_labels = sorted(set(labels + additions))
        rows.append({"id": row_id, "cve_labels": " ".join(final_labels)})
        if additions:
            changes.append(
                {
                    "id": row_id,
                    "added": " ".join(additions),
                    "labels": " ".join(final_labels),
                    "payload_clean": clean_payload_text("" if payload is None else str(payload))[:300],
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    if args.changes_output:
        args.changes_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(changes).to_csv(args.changes_output, index=False)
    print(f"Loaded rules: {len(rule_map)}")
    print(f"Changed rows: {len(changes)}")
    print(f"Wrote {args.output}")


def add_common_rule_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--payload-column", default="payload_clean")
    parser.add_argument("--label-column", default="cve_labels")
    parser.add_argument("--modes", default="path,query_keys,body_keys,query_key_value,body_key_value,token")
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--min-precision", type=float, default=1.0)
    parser.add_argument("--max-labels-per-signature", type=int, default=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine, OOF-validate, and apply structured HTTP signature rules.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mine_parser = subparsers.add_parser("mine")
    add_common_rule_args(mine_parser)
    mine_parser.add_argument("--output", required=True, type=Path)
    mine_parser.add_argument("--top-n", type=int, default=50)

    oof_parser = subparsers.add_parser("oof")
    add_common_rule_args(oof_parser)
    oof_parser.add_argument("--folds", type=int, default=5)
    oof_parser.add_argument("--seed", type=int, default=20260626)
    oof_parser.add_argument("--max-additions", type=int, default=1)
    oof_parser.add_argument("--output-folds", required=True, type=Path)
    oof_parser.add_argument("--output-rules", required=True, type=Path)

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--rules", required=True, type=Path)
    apply_parser.add_argument("--truth-or-test", required=True, type=Path)
    apply_parser.add_argument("--pred", required=True, type=Path)
    apply_parser.add_argument("--output", required=True, type=Path)
    apply_parser.add_argument("--changes-output", type=Path, default=None)
    apply_parser.add_argument("--id-column", default="id")
    apply_parser.add_argument("--payload-column", default="payload_decoded")
    apply_parser.add_argument("--modes", default="path,query_keys,body_keys,query_key_value,body_key_value,token")
    apply_parser.add_argument("--min-support", type=int, default=20)
    apply_parser.add_argument("--min-precision", type=float, default=1.0)
    apply_parser.add_argument("--only-empty", action="store_true")
    apply_parser.add_argument("--max-additions", type=int, default=1)
    apply_parser.add_argument("--rule-type", action="append", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "mine":
        run_mine(args)
    elif args.command == "oof":
        run_oof(args)
    elif args.command == "apply":
        run_apply(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
