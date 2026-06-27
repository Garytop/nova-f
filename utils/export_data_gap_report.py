from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import normalize_cve_labels


def label_set(value: object) -> set[str]:
    return set(normalize_cve_labels(value))


def count_train_support(paths: list[Path], label_column: str) -> dict[str, int]:
    support: dict[str, int] = {}
    for path in paths:
        frame = pd.read_csv(path)
        if label_column not in frame.columns:
            raise KeyError(f"{path} missing {label_column!r}")
        for raw_labels in frame[label_column].fillna(""):
            for label in label_set(raw_labels):
                if label.startswith("CVE-"):
                    support[label] = support.get(label, 0) + 1
    return support


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CVEs that need external/authorized sample supplementation.")
    parser.add_argument("--per-label", required=True, type=Path)
    parser.add_argument("--train", action="append", required=True, type=Path)
    parser.add_argument("--label-column", default="cve_labels")
    parser.add_argument("--min-fn", type=int, default=20)
    parser.add_argument("--max-train-support", type=int, default=0)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    per_label = pd.read_csv(args.per_label)
    required = {"label", "fn", "support", "f1"}
    missing = required - set(per_label.columns)
    if missing:
        raise KeyError(f"per-label CSV missing columns: {', '.join(sorted(missing))}")
    train_support = count_train_support(args.train, args.label_column)

    rows: list[dict[str, object]] = []
    for row in per_label.itertuples(index=False):
        label = str(row.label)
        support = int(getattr(row, "support"))
        fn = int(getattr(row, "fn"))
        f1 = float(getattr(row, "f1"))
        train_count = train_support.get(label, 0)
        if fn >= args.min_fn and train_count <= args.max_train_support:
            rows.append(
                {
                    "label": label,
                    "test_support": support,
                    "fn": fn,
                    "f1": f1,
                    "train_support": train_count,
                    "priority": "high" if fn >= 50 else "medium",
                    "recommended_action": "add_authorized_or_public_payload_samples",
                }
            )

    result = pd.DataFrame(rows).sort_values(["fn", "test_support", "label"], ascending=[False, False, True])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)

    lines = [
        "# CVE Data Gap Report",
        "",
        "This report lists high-FN CVEs with little or no training support. These should not be fixed by test-set hand-written rules; add authorized/public payload samples instead.",
        "",
        f"- min_fn: {args.min_fn}",
        f"- max_train_support: {args.max_train_support}",
        f"- total_gaps: {len(result)}",
        "",
        "| CVE | FN | Test Support | Train Support | F1 | Priority |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in result.head(80).itertuples(index=False):
        lines.append(
            f"| {row.label} | {row.fn} | {row.test_support} | {row.train_support} | {row.f1:.6f} | {row.priority} |"
        )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Data gaps: {len(result)}")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
