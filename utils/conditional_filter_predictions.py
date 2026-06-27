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


def load_blocklist(path: Path) -> set[str]:
    labels: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        for token in raw_line.replace(",", " ").split():
            label = token.strip().upper()
            if label.startswith("CVE-"):
                labels.add(label)
    return labels


def load_allow_signatures(path: Path | None, *, include_disabled: bool) -> dict[str, set[str]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, set[str]] = {}
    for rule in payload.get("rules", []):
        if not include_disabled and not bool(rule.get("enabled", False)):
            continue
        target = str(rule.get("target_cve", "")).upper()
        signature = str(rule.get("signature", ""))
        if target.startswith("CVE-") and signature:
            result.setdefault(target, set()).add(signature)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter CVE labels unless payload matches allowlisted structural signatures.")
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--truth-or-test", required=True, type=Path)
    parser.add_argument("--blocklist", required=True, type=Path)
    parser.add_argument("--allow-config", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--changes-output", type=Path, default=None)
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--payload-column", default="payload_decoded")
    parser.add_argument("--modes", default="path,query_keys,body_keys,query_key_value,body_key_value,token")
    parser.add_argument("--include-disabled", action="store_true")
    args = parser.parse_args()

    blocklist = load_blocklist(args.blocklist)
    allow = load_allow_signatures(args.allow_config, include_disabled=args.include_disabled)
    modes = set(args.modes.split(","))

    source = pd.read_csv(args.truth_or_test)
    pred = pd.read_csv(args.pred)
    source[args.id_column] = source[args.id_column].astype(str)
    pred[args.id_column] = pred[args.id_column].astype(str)
    merged = pred.merge(source[[args.id_column, args.payload_column]], on=args.id_column, how="left", validate="one_to_one")

    rows: list[dict[str, str]] = []
    changes: list[dict[str, str]] = []
    removed = 0
    preserved = 0
    for row in merged.itertuples(index=False):
        row_id = str(getattr(row, args.id_column))
        payload = getattr(row, args.payload_column)
        signatures = set(candidate_signatures(payload, modes=modes))
        labels = normalize_cve_labels(getattr(row, "cve_labels"))
        kept: list[str] = []
        removed_labels: list[str] = []
        for label in labels:
            if label not in blocklist:
                kept.append(label)
                continue
            allow_hits = allow.get(label, set()) & signatures
            if allow_hits:
                kept.append(label)
                preserved += 1
            else:
                removed_labels.append(label)
                removed += 1
        rows.append({"id": row_id, "cve_labels": " ".join(sorted(kept))})
        if removed_labels:
            changes.append(
                {
                    "id": row_id,
                    "removed": " ".join(sorted(removed_labels)),
                    "kept": " ".join(sorted(kept)),
                    "payload_clean": clean_payload_text("" if payload is None else str(payload))[:300],
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    if args.changes_output:
        args.changes_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(changes).to_csv(args.changes_output, index=False)
    print(f"Blocked labels: {len(blocklist)}")
    print(f"Removed labels: {removed}")
    print(f"Preserved by allow signatures: {preserved}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
