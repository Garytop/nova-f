from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Export mined structured rules to an auditable JSON config.")
    parser.add_argument("--rules", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", default="train-mined")
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--min-precision", type=float, default=1.0)
    parser.add_argument("--enabled", action="store_true")
    parser.add_argument("--risk-level", default="low")
    args = parser.parse_args()

    rules = pd.read_csv(args.rules)
    required = {"signature", "label", "support", "precision"}
    missing = required - set(rules.columns)
    if missing:
        raise KeyError(f"rules missing columns: {', '.join(sorted(missing))}")
    rules = rules[
        (rules["support"].astype(int) >= args.min_support)
        & (rules["precision"].astype(float) >= args.min_precision)
    ].copy()

    records: list[dict[str, object]] = []
    for idx, row in enumerate(rules.sort_values(["label", "signature"]).itertuples(index=False), start=1):
        signature = str(row.signature)
        rule_type = str(getattr(row, "rule_type", signature.split("|", 1)[0]))
        records.append(
            {
                "rule_id": f"{args.source}-{idx:04d}",
                "enabled": bool(args.enabled),
                "rule_type": rule_type,
                "signature": signature,
                "target_cve": str(row.label),
                "support": int(row.support),
                "precision": float(row.precision),
                "source": args.source,
                "risk_level": args.risk_level,
            }
        )

    payload = {
        "schema_version": 1,
        "description": "Auditable structured signature rules for nova-f post-processing.",
        "rules": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported rules: {len(records)}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
