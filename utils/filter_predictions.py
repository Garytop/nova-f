from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import normalize_cve_labels


def parse_label_list(value: str) -> set[str]:
    return {label.strip().upper() for label in value.split(",") if label.strip()}


def load_blocklist(args: argparse.Namespace) -> set[str]:
    labels = parse_label_list(args.block_labels or "")
    if args.analysis_summary:
        summary = pd.read_csv(args.analysis_summary)
        required = {"label", "fp", "precision"}
        missing = required - set(summary.columns)
        if missing:
            raise KeyError(f"analysis summary missing columns: {sorted(missing)}")
        selected = summary[
            (summary["fp"] >= args.min_fp)
            & (summary["precision"] <= args.max_precision)
            & (summary["tp"] >= args.min_tp if "tp" in summary.columns else True)
        ]
        labels.update(str(label).upper() for label in selected["label"])
    return labels


def filter_predictions(pred_path: Path, output_path: Path, blocklist: set[str], *, id_column: str) -> None:
    pred = pd.read_csv(pred_path)
    if id_column not in pred.columns or "cve_labels" not in pred.columns:
        raise KeyError(f"prediction CSV must contain {id_column!r} and cve_labels")

    rows: list[dict[str, str]] = []
    removed = 0
    changed = 0
    for row_id, raw_labels in zip(pred[id_column].astype(str), pred["cve_labels"].fillna("")):
        labels = normalize_cve_labels(raw_labels)
        kept = [label for label in labels if label not in blocklist]
        removed += len(labels) - len(kept)
        if labels != kept:
            changed += 1
        rows.append({id_column: row_id, "cve_labels": " ".join(kept)})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Blocked labels: {' '.join(sorted(blocklist)) if blocklist else '(none)'}")
    print(f"Changed rows: {changed}")
    print(f"Removed labels: {removed}")
    print(f"Wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter prediction labels by explicit or analysis-derived CVE blocklist.")
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--block-labels", default="", help="Comma-separated CVE labels to remove from final predictions")
    parser.add_argument("--analysis-summary", type=Path, default=None, help="Optional error_summary CSV from analyze_errors.py")
    parser.add_argument("--min-fp", type=int, default=50)
    parser.add_argument("--max-precision", type=float, default=0.05)
    parser.add_argument("--min-tp", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    blocklist = load_blocklist(args)
    filter_predictions(args.pred, args.output, blocklist, id_column=args.id_column)


if __name__ == "__main__":
    main()
