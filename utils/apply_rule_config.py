from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import clean_payload_text, normalize_cve_labels
from src.structured_features import candidate_signatures


def load_rules(path: Path, *, include_disabled: bool, min_support: int, min_precision: float) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rules = payload.get("rules", [])
    rule_map: dict[str, str] = {}
    for rule in rules:
        if not include_disabled and not bool(rule.get("enabled", False)):
            continue
        if int(rule.get("support", 0)) < min_support:
            continue
        if float(rule.get("precision", 0.0)) < min_precision:
            continue
        signature = str(rule.get("signature", ""))
        target = str(rule.get("target_cve", "")).upper()
        if signature and target.startswith("CVE-"):
            rule_map[signature] = target
    return rule_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply auditable structured rule config to prediction CSV.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--truth-or-test", required=True, type=Path)
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--changes-output", type=Path, default=None)
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--payload-column", default="payload_decoded")
    parser.add_argument("--modes", default="path,query_keys,body_keys,query_key_value,body_key_value,token")
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--min-precision", type=float, default=1.0)
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--only-empty", action="store_true")
    parser.add_argument("--max-additions", type=int, default=1)
    args = parser.parse_args()

    modes = set(args.modes.split(","))
    rule_map = load_rules(
        args.config,
        include_disabled=args.include_disabled,
        min_support=args.min_support,
        min_precision=args.min_precision,
    )

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
    print(f"Loaded active rules: {len(rule_map)}")
    print(f"Changed rows: {len(changes)}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
