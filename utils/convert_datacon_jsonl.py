from __future__ import annotations

import argparse
import base64
import gzip
import json
import logging
import zlib
from pathlib import Path

import pandas as pd


def decode_payload(encoded: str) -> str:
    """Decode DataCon payload field: base64(zlib(http_text))."""
    if not encoded:
        return ""
    raw = base64.b64decode(encoded)
    try:
        return zlib.decompress(raw).decode("utf-8", errors="replace")
    except zlib.error:
        return zlib.decompress(raw, -zlib.MAX_WBITS).decode("utf-8", errors="replace")


def normalize_label_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        labels = value
    else:
        labels = str(value).replace(",", " ").split()
    cves = sorted({label.strip().upper() for label in labels if label and str(label).upper().startswith("CVE-")})
    return " ".join(cves)


def convert_jsonl_gz(input_path: Path, output_path: Path) -> None:
    rows: list[dict[str, object]] = []
    with gzip.open(input_path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(
                {
                    "id": str(obj.get("id", line_no)),
                    "payload_decoded": decode_payload(str(obj.get("payload", ""))),
                    "labeled": obj.get("labeled", ""),
                    "cve_labels": normalize_label_text(obj.get("cve_labels", "")),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    logging.info("Wrote %d rows to %s", len(rows), output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DataCon JSONL.GZ payload data to nova-f CSV.")
    parser.add_argument("--input", required=True, type=Path, help="Path to train.json.gz/test.json.gz")
    parser.add_argument("--output", required=True, type=Path, help="CSV output path")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    convert_jsonl_gz(args.input, args.output)


if __name__ == "__main__":
    main()
