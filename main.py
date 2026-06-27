from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Sequence

try:
    import numpy as np
    import pandas as pd
    import faiss
    from tqdm import tqdm
    from src import build_faiss as build_mod
    from src import preprocess as preprocess_mod
    from src import search_faiss as search_mod
except Exception:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        np = None
        pd = None
        faiss = None
        tqdm = None
        build_mod = None
        preprocess_mod = None
        search_mod = None
    else:
        raise


SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


class ColorFormatter(logging.Formatter):
    RESET = "\033[0m"
    WHITE = "\033[37m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"

    COLORS = {
        logging.DEBUG: CYAN,
        logging.INFO: WHITE,
        SUCCESS_LEVEL: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.WHITE)
        return f"{color}{super().format(record)}{self.RESET}"


def log_success(message: str, *args: object) -> None:
    logging.getLogger().log(SUCCESS_LEVEL, message, *args)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter("%(asctime)s | %(levelname)s | %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(handler)


def load_prediction_blocklist(path: Path | None) -> set[str]:
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(f"prediction blocklist not found: {path}")

    labels: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        for token in raw_line.replace(",", " ").split():
            label = token.strip().upper()
            if label.startswith("CVE-"):
                labels.add(label)
    return labels


def preprocess_training_data(
    train_path: Path,
    output_path: Path,
    *,
    payload_column: str,
    id_column: str,
    overwrite: bool,
) -> Path:
    if output_path.exists() and not overwrite:
        logging.info("Found existing cleaned training file, skip preprocessing: %s", output_path)
        return output_path

    logging.info("Loading training data: %s", train_path)
    df = pd.read_csv(train_path)
    logging.info("Training data loaded: %d rows", len(df))

    processed = preprocess_mod.preprocess_dataframe(
        df,
        payload_column=payload_column,
        id_column=id_column,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)
    log_success("Cleaned training data written: %s", output_path)
    return output_path


def preprocess_test_data(
    test_path: Path,
    output_path: Path,
    *,
    payload_column: str,
    id_column: str,
    overwrite: bool,
) -> Path:
    if output_path.exists() and not overwrite:
        logging.info("Found existing cleaned test file, skip preprocessing: %s", output_path)
        return output_path

    logging.info("Loading test data: %s", test_path)
    df = pd.read_csv(test_path)
    logging.info("Test data loaded: %d rows", len(df))

    processed = preprocess_mod.preprocess_dataframe(
        df,
        payload_column=payload_column,
        id_column=id_column,
    )
    processed = processed.drop(columns=["cve_labels"], errors="ignore")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)
    log_success("Cleaned test data written: %s", output_path)
    return output_path


def build_vector_store(
    clean_train_path: Path,
    store_dir: Path,
    *,
    model_path: str,
    device: str | None,
    batch_size: int,
    train_text_prefix: str,
    overwrite: bool,
) -> tuple[Path, Path]:
    index_path = store_dir / "faiss.index"
    meta_path = store_dir / "meta.json"

    store_dir.mkdir(parents=True, exist_ok=True)

    if index_path.exists() and meta_path.exists() and not overwrite:
        logging.info("Found existing index and metadata, skip index rebuild: %s", store_dir)
        return index_path, meta_path

    if overwrite:
        logging.warning("Overwrite index enabled, clearing files in: %s", store_dir)
        for item in store_dir.iterdir():
            if item.is_file():
                item.unlink()

    payloads, ids, label_lists = build_mod.load_dataset(
        clean_train_path,
        payload_column="payload_clean",
        id_column="id",
    )

    model = build_mod.load_sentence_encoder(model_path, device)
    vectors = build_mod.embed_texts(
        model,
        payloads,
        batch_size=max(1, batch_size),
        desc="Generating training vectors",
        text_prefix=train_text_prefix,
    )

    npy_path = store_dir / "train_embeddings.npy"
    np.save(npy_path, vectors)
    log_success("Training vectors saved: %s", npy_path)

    index_path = build_mod.build_faiss_index(
        vectors,
        store_dir=store_dir,
        index_name="faiss.index",
    )

    meta: Dict[str, Sequence] = {
        "ids": ids,
        "cve_labels": label_lists,
        "vector_file": npy_path.name,
        "index_file": index_path.name,
        "payload_column": "payload_clean",
        "id_column": "id",
        "embedding_model": model_path,
        "train_text_prefix": train_text_prefix,
        "created_at": datetime.now(tz=UTC).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log_success("Index metadata saved: %s", meta_path)

    return index_path, meta_path


def label_test_payloads(
    test_path: Path,
    store_dir: Path,
    output_path: Path,
    *,
    payload_column: str,
    id_column: str,
    model_path: str | None,
    device: str | None,
    batch_size: int,
    top_k: int,
    max_candidates: int,
    base_threshold: float,
    high_confidence: float,
    medium_confidence: float,
    second_gap: float,
    third_gap: float,
    search_batch_size: int,
    min_votes: int,
    vote_weight: float,
    empty_penalty_margin: float,
    empty_penalty_floor: float,
    empty_penalty_ratio: float,
    test_text_prefix: str | None,
    prediction_blocklist: set[str] | None,
    structured_rerank_alpha: float,
    train_feature_path: Path | None,
    reuse_cache: bool,
) -> Path:
    index_path = store_dir / "faiss.index"
    meta_path = store_dir / "meta.json"
    cache_emb_path = store_dir / "test_embeddings.npy"
    cache_meta_path = store_dir / "test_cache_meta.json"

    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError("missing FAISS index or metadata; build the vector store first")

    meta: Dict[str, Sequence] = json.loads(meta_path.read_text(encoding="utf-8"))
    train_ids = list(meta.get("ids", []))
    raw_train_labels = meta.get("cve_labels", [])
    if not train_ids or not raw_train_labels:
        raise ValueError("metadata must contain non-empty ids and cve_labels")

    train_labels: List[List[str]] = [
        preprocess_mod.normalize_cve_labels(labels) for labels in raw_train_labels
    ]
    train_has_cve = [search_mod.has_cve_label(labels) for labels in train_labels]
    train_features = None
    if structured_rerank_alpha > 0:
        feature_source = train_feature_path or test_path
        feature_frame = pd.read_csv(feature_source)
        if "payload_clean" not in feature_frame.columns:
            raise KeyError(f"structured rerank requires payload_clean in {feature_source}")
        if len(feature_frame) != len(train_labels):
            raise ValueError(
                f"structured rerank feature rows ({len(feature_frame)}) != train labels ({len(train_labels)})"
            )
        logging.info("Loading structured rerank features from %s", feature_source)
        train_features = [
            search_mod.parse_structured_features(payload)
            for payload in feature_frame["payload_clean"].fillna("").astype(str)
        ]

    index = faiss.read_index(str(index_path))

    payloads, test_ids = search_mod.load_test_payloads(test_path, payload_column, id_column)
    test_features = None
    if structured_rerank_alpha > 0:
        logging.info("Structured rerank enabled: alpha=%.4f", structured_rerank_alpha)
        test_features = [search_mod.parse_structured_features(payload) for payload in payloads]

    model_name = model_path or meta.get("embedding_model")
    if not model_name:
        raise ValueError("embedding model is not specified by args or metadata")
    prefix_for_test = test_text_prefix
    if prefix_for_test is None:
        prefix_for_test = str(meta.get("test_text_prefix", ""))

    test_vectors: np.ndarray | None = None
    if reuse_cache and cache_emb_path.exists() and cache_meta_path.exists():
        cache_info = json.loads(cache_meta_path.read_text(encoding="utf-8"))
        cached_ids = cache_info.get("ids") if isinstance(cache_info, dict) else None
        cached_model = cache_info.get("embedding_model") if isinstance(cache_info, dict) else None
        cached_prefix = cache_info.get("test_text_prefix", "") if isinstance(cache_info, dict) else ""
        if cached_ids == test_ids and cached_model == model_name and cached_prefix == prefix_for_test:
            logging.info("Found matching cached test vectors, reusing: %s", cache_emb_path)
            test_vectors = np.load(cache_emb_path)
        else:
            logging.info("Test vector cache does not match current task; recomputing")

    if test_vectors is None:
        model = build_mod.load_sentence_encoder(model_name, device)
        test_vectors = search_mod.embed_texts(
            model,
            payloads,
            batch_size=max(1, batch_size),
            desc="Generating test vectors",
            text_prefix=prefix_for_test,
        )
        np.save(cache_emb_path, test_vectors)
        cache_payload = {
            "ids": test_ids,
            "embedding_model": model_name,
            "test_text_prefix": prefix_for_test,
        }
        cache_meta_path.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log_success("Test vectors and cache metadata saved; --reuse-cache can reuse them")

    faiss.normalize_L2(test_vectors)

    results: List[dict[str, str]] = []
    total_candidates = 0
    filtered_out = 0
    score_stats: List[float] = []
    pred_counts: Dict[int, int] = {}
    blocked_prediction_count = 0
    active_blocklist = prediction_blocklist or set()

    search_offsets = range(0, len(test_vectors), search_batch_size)
    progress = tqdm(search_offsets, desc="Searching and predicting", unit="batch")
    for start in progress:
        end = min(start + search_batch_size, len(test_vectors))
        batch_vectors = test_vectors[start:end]
        D, I = index.search(batch_vectors, top_k)

        for offset, row_id in enumerate(test_ids[start:end]):
            sims = D[offset]
            idxs = I[offset]
            if structured_rerank_alpha > 0 and train_features is not None and test_features is not None:
                query_features = test_features[start + offset]
                sims = np.asarray(sims, dtype=np.float32).copy()
                for col_idx, train_idx in enumerate(idxs):
                    if train_idx < 0 or train_idx >= len(train_features):
                        continue
                    sims[col_idx] += structured_rerank_alpha * search_mod.structured_feature_bonus(
                        query_features,
                        train_features[int(train_idx)],
                    )

            for idx, score in zip(idxs, sims):
                total_candidates += 1
                if idx < 0 or idx >= len(train_ids):
                    continue
                score_stats.append(float(score))

            candidates, filtered = search_mod.aggregate_cve_candidates(
                idxs,
                sims,
                train_labels,
                base_threshold=base_threshold,
                max_candidates=max_candidates,
                min_votes=max(1, min_votes),
                vote_weight=vote_weight,
            )
            filtered_out += filtered

            preds = search_mod.adaptive_predict(
                candidates,
                base_threshold=base_threshold,
                high_confidence=high_confidence,
                medium_confidence=medium_confidence,
                max_diff_second=second_gap,
                max_diff_third=third_gap,
            )
            if preds and search_mod.should_suppress_by_empty_neighbors(
                idxs,
                sims,
                train_has_cve,
                base_threshold=base_threshold,
                empty_penalty_margin=empty_penalty_margin,
                empty_penalty_floor=empty_penalty_floor,
                empty_penalty_ratio=empty_penalty_ratio,
            ):
                preds = []
            if preds and active_blocklist:
                before_count = len(preds)
                preds = [label for label in preds if label not in active_blocklist]
                blocked_prediction_count += before_count - len(preds)
            pred_counts[len(preds)] = pred_counts.get(len(preds), 0) + 1
            results.append({"id": row_id, "cve_labels": " ".join(preds)})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    log_success("Prediction results saved: %s", output_path)

    if active_blocklist:
        logging.info(
            "Prediction blocklist active: %d labels, %d predicted labels removed",
            len(active_blocklist),
            blocked_prediction_count,
        )

    if score_stats:
        scores = np.asarray(score_stats)
        logging.info(
            "Candidates: %d | filtered: %d (%.1f%%) | similarity mean: %.3f median: %.3f",
            total_candidates,
            filtered_out,
            filtered_out / max(total_candidates, 1) * 100,
            float(scores.mean()),
            float(np.median(scores)),
        )
    logging.info(
        "Prediction count distribution: %s",
        {k: f"{v} ({v / max(len(results), 1) * 100:.1f}%)" for k, v in sorted(pred_counts.items())},
    )

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NOVA-F local CVE labeling pipeline")
    parser.add_argument("--train-path", required=True, help="Raw training CSV path")
    parser.add_argument("--test-path", required=True, help="Raw test CSV path")
    parser.add_argument("--train-payload-column", default="payload_decoded", help="Training payload column")
    parser.add_argument("--test-payload-column", default="payload_decoded", help="Test payload column")
    parser.add_argument("--train-id-column", default="id", help="Training ID column")
    parser.add_argument("--test-id-column", default="id", help="Test ID column")
    parser.add_argument("--clean-train-path", default=None, help="Cleaned training CSV output path")
    parser.add_argument("--clean-test-path", default=None, help="Cleaned test CSV output path")
    parser.add_argument("--store-dir", default="embeddings/faiss_store", help="FAISS store directory")
    parser.add_argument("--output-path", default="ans/test_label_pipeline.csv", help="Prediction CSV output path")
    parser.add_argument("--model-path", default=None, help="SentenceTransformer path or model name")
    parser.add_argument("--device", default=None, help="Model device, for example cpu or cuda")
    parser.add_argument("--train-batch-size", type=int, default=32, help="Training embedding batch size")
    parser.add_argument("--test-batch-size", type=int, default=32, help="Test embedding batch size")
    parser.add_argument("--train-text-prefix", default="", help="Training text prefix, for example E5 'passage: '")
    parser.add_argument("--test-text-prefix", default=None, help="Test text prefix, for example E5 'query: '")
    parser.add_argument("--search-batch-size", type=int, default=512, help="FAISS search batch size")
    parser.add_argument("--top-k", type=int, default=50, help="Number of FAISS neighbours")
    parser.add_argument("--max-candidates", type=int, default=5, help="Maximum aggregated CVE candidates per sample")
    parser.add_argument("--base-threshold", type=float, default=0.86, help="Base similarity threshold")
    parser.add_argument("--high-confidence", type=float, default=0.87, help="Second-label high confidence threshold")
    parser.add_argument("--medium-confidence", type=float, default=0.78, help="Second-label medium confidence threshold")
    parser.add_argument("--second-gap", type=float, default=0.15, help="Maximum score gap between top-1 and top-2")
    parser.add_argument("--third-gap", type=float, default=0.05, help="Maximum score gap between top-2 and top-3")
    parser.add_argument("--min-votes", type=int, default=1, help="CVE aggregate minimum neighbour votes")
    parser.add_argument("--vote-weight", type=float, default=0.015, help="CVE aggregate vote bonus")
    parser.add_argument("--empty-penalty-margin", type=float, default=0.05, help="Empty-label suppression similarity gap")
    parser.add_argument("--empty-penalty-floor", type=float, default=0.80, help="Minimum similarity for empty-label evidence")
    parser.add_argument("--empty-penalty-ratio", type=float, default=0.50, help="Required empty-neighbour vote ratio")
    parser.add_argument("--prediction-blocklist", default=None, help="Optional newline/comma separated CVE blocklist")
    parser.add_argument("--structured-rerank-alpha", type=float, default=0.0, help="Structured rerank weight; 0 disables rerank")
    parser.add_argument("--train-feature-path", default=None, help="Cleaned training CSV aligned with FAISS metadata")
    parser.add_argument("--overwrite-clean", action="store_true", help="Regenerate cleaned training data")
    parser.add_argument("--overwrite-test-clean", action="store_true", help="Regenerate cleaned test data")
    parser.add_argument("--overwrite-index", action="store_true", help="Rebuild FAISS index")
    parser.add_argument("--reuse-cache", action="store_true", help="Reuse cached test vectors when metadata matches")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    train_path = Path(args.train_path)
    test_path = Path(args.test_path)
    if not train_path.exists():
        raise FileNotFoundError(f"training data not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"test data not found: {test_path}")

    clean_train_path = (
        Path(args.clean_train_path)
        if args.clean_train_path
        else train_path.with_name(f"{train_path.stem}_cleaned.csv")
    )
    clean_test_path = (
        Path(args.clean_test_path)
        if args.clean_test_path
        else test_path.with_name(f"{test_path.stem}_cleaned.csv")
    )

    clean_path = preprocess_training_data(
        train_path,
        clean_train_path,
        payload_column=args.train_payload_column,
        id_column=args.train_id_column,
        overwrite=args.overwrite_clean,
    )
    clean_test = preprocess_test_data(
        test_path,
        clean_test_path,
        payload_column=args.test_payload_column,
        id_column=args.test_id_column,
        overwrite=args.overwrite_test_clean or args.overwrite_clean,
    )

    store_dir = Path(args.store_dir)
    model_for_training = args.model_path or "sentence-transformers/all-MiniLM-L6-v2"
    prediction_blocklist = load_prediction_blocklist(Path(args.prediction_blocklist) if args.prediction_blocklist else None)
    if prediction_blocklist:
        logging.info("Loaded prediction blocklist with %d CVE labels", len(prediction_blocklist))

    build_vector_store(
        clean_path,
        store_dir,
        model_path=model_for_training,
        device=args.device,
        batch_size=args.train_batch_size,
        train_text_prefix=args.train_text_prefix,
        overwrite=args.overwrite_index,
    )

    predictions_path = label_test_payloads(
        clean_test,
        store_dir,
        Path(args.output_path),
        payload_column="payload_clean",
        id_column="id",
        model_path=args.model_path or model_for_training,
        device=args.device,
        batch_size=args.test_batch_size,
        top_k=args.top_k,
        max_candidates=args.max_candidates,
        base_threshold=args.base_threshold,
        high_confidence=args.high_confidence,
        medium_confidence=args.medium_confidence,
        second_gap=args.second_gap,
        third_gap=args.third_gap,
        search_batch_size=args.search_batch_size,
        min_votes=args.min_votes,
        vote_weight=args.vote_weight,
        empty_penalty_margin=args.empty_penalty_margin,
        empty_penalty_floor=args.empty_penalty_floor,
        empty_penalty_ratio=args.empty_penalty_ratio,
        test_text_prefix=args.test_text_prefix,
        prediction_blocklist=prediction_blocklist,
        structured_rerank_alpha=args.structured_rerank_alpha,
        train_feature_path=Path(args.train_feature_path) if args.train_feature_path else None,
        reuse_cache=args.reuse_cache,
    )
    log_success("Pipeline completed. Final prediction file: %s", predictions_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if not logging.getLogger().handlers:
            configure_logging(False)
        logging.exception("Pipeline failed: %s", exc)
        print(
            "\033[31mDebug tips: check input CSV paths, local model path, FAISS store, "
            "Python environment, and structured-rerank train-feature alignment.\033[0m",
            file=sys.stderr,
        )
        raise SystemExit(1)
