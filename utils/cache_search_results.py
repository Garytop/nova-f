from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_expected_ids(path: Path | None, id_column: str) -> list[str] | None:
    if path is None:
        return None
    import pandas as pd

    frame = pd.read_csv(path)
    if id_column not in frame.columns:
        raise KeyError(f"expected-id CSV must contain {id_column!r}")
    return frame[id_column].astype(str).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache FAISS top-k search results from an existing vector store.")
    parser.add_argument("--store-dir", required=True, type=Path, help="Directory containing faiss.index, meta.json and test_embeddings.npy")
    parser.add_argument("--output", required=True, type=Path, help="Output NPZ path containing D and I arrays")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--search-batch-size", type=int, default=2048)
    parser.add_argument("--test-ids-csv", type=Path, default=None, help="Optional CSV used to validate cached test ids")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists; use --overwrite: {args.output}")

    index_path = args.store_dir / "faiss.index"
    meta_path = args.store_dir / "meta.json"
    test_embeddings_path = args.store_dir / "test_embeddings.npy"
    test_cache_meta_path = args.store_dir / "test_cache_meta.json"

    for path in [index_path, meta_path, test_embeddings_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    index = faiss.read_index(str(index_path))
    vectors = np.load(test_embeddings_path).astype("float32")
    faiss.normalize_L2(vectors)

    expected_ids = load_expected_ids(args.test_ids_csv, args.id_column)
    cached_ids = None
    if test_cache_meta_path.exists():
        cache_meta = json.loads(test_cache_meta_path.read_text(encoding="utf-8"))
        if isinstance(cache_meta, dict):
            cached_ids = [str(item) for item in cache_meta.get("ids", [])]

    if expected_ids is not None:
        if cached_ids is None:
            raise ValueError("test_cache_meta.json does not contain ids; cannot validate expected ids")
        if expected_ids != cached_ids:
            raise ValueError("test id validation failed: CSV ids differ from cached embedding ids")

    if index.d != vectors.shape[1]:
        raise ValueError(f"index dimension ({index.d}) != embedding dimension ({vectors.shape[1]})")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    row_count = vectors.shape[0]
    distances = np.empty((row_count, args.top_k), dtype=np.float32)
    indices = np.empty((row_count, args.top_k), dtype=np.int64)

    for start in tqdm(range(0, row_count, args.search_batch_size), desc="cache FAISS search"):
        end = min(start + args.search_batch_size, row_count)
        D, I = index.search(vectors[start:end], args.top_k)
        distances[start:end] = D
        indices[start:end] = I

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        D=distances,
        I=indices,
        top_k=np.array([args.top_k], dtype=np.int32),
        store_dir=np.array([str(args.store_dir)]),
        train_count=np.array([len(meta.get("ids", []))], dtype=np.int64),
        test_count=np.array([row_count], dtype=np.int64),
    )
    print(f"Wrote {args.output}")
    print(f"rows={row_count} top_k={args.top_k} dim={vectors.shape[1]}")


if __name__ == "__main__":
    main()
