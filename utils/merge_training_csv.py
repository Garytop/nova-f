from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge nova-f training CSV files.")
    parser.add_argument("--input", action="append", required=True, type=Path, help="Input CSV path; repeat for multiple files")
    parser.add_argument("--output", required=True, type=Path, help="Merged CSV output path")
    parser.add_argument("--drop-duplicates", action="store_true", help="Drop duplicate payload/label rows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames: list[pd.DataFrame] = []
    for input_path in args.input:
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        frame = pd.read_csv(input_path)
        required = {"id", "payload_decoded", "cve_labels"}
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"{input_path} missing columns: {sorted(missing)}")
        frames.append(frame[["id", "payload_decoded", "cve_labels"]].copy())

    merged = pd.concat(frames, ignore_index=True)
    merged["id"] = range(len(merged))
    if args.drop_duplicates:
        merged = merged.drop_duplicates(subset=["payload_decoded", "cve_labels"]).reset_index(drop=True)
        merged["id"] = range(len(merged))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Wrote {len(merged)} rows to {args.output}")


if __name__ == "__main__":
    main()
