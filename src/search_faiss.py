from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    from preprocess import clean_payload_text, normalize_cve_labels
    from structured_features import feature_bonus as structured_feature_bonus
    from structured_features import parse_payload as parse_structured_features
except ImportError:  # pragma: no cover
    from .preprocess import clean_payload_text, normalize_cve_labels
    from .structured_features import feature_bonus as structured_feature_bonus
    from .structured_features import parse_payload as parse_structured_features


def configure_logging(verbose: bool) -> None:
    """初始化日志输出。"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def load_prediction_blocklist(path: Path | None) -> set[str]:
    """Load CVE labels that should be removed from final predictions."""
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


def load_sentence_encoder(model_path: str, device: str | None = None) -> SentenceTransformer:
    """加载本地 SentenceTransformer 模型。"""
    logging.info("加载本地嵌入模型: %s", model_path)
    return SentenceTransformer(model_path, device=device)


def embed_texts(
    model: SentenceTransformer,
    texts: Sequence[str],
    *,
    batch_size: int,
    desc: str,
    normalize_vectors: bool = False,
    text_prefix: str = "",
) -> np.ndarray:
    """批量生成文本向量。"""
    total = len(texts)
    if total == 0:
        raise ValueError("输入文本为空，无法生成嵌入向量")

    chunks: list[np.ndarray] = []
    for start in tqdm(range(0, total, batch_size), desc=desc):
        end = min(start + batch_size, total)
        chunk = list(texts[start:end])
        if not chunk:
            continue
        if text_prefix:
            chunk = [f"{text_prefix}{text}" for text in chunk]
        chunk_vecs = model.encode(
            chunk,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        if not isinstance(chunk_vecs, np.ndarray):
            chunk_vecs = np.asarray(chunk_vecs)
        chunks.append(chunk_vecs.astype(np.float32))

    if not chunks:
        raise RuntimeError("未生成任何嵌入向量，检查输入或模型配置")

    vectors = np.vstack(chunks)
    if normalize_vectors:
        faiss.normalize_L2(vectors)
    return vectors


def load_test_payloads(path: Path, payload_column: str, id_column: str) -> tuple[List[str], List[str]]:
    """读取测试集并返回 payload 与 id 列表。"""
    if not path.exists():
        raise FileNotFoundError(f"未找到测试集: {path}")

    frame = pd.read_csv(path)
    logging.info("已载入测试集，共 %d 行", len(frame))

    if payload_column not in frame.columns:
        raise KeyError(f"找不到字段: {payload_column}")
    if id_column not in frame.columns:
        raise KeyError(f"找不到字段: {id_column}")

    tqdm.pandas(desc="清理测试报文")
    payload_series = frame[payload_column]
    if payload_column != "payload_clean":
        payload_series = payload_series.progress_apply(clean_payload_text)
    else:
        payload_series = payload_series.fillna("")

    ids = frame[id_column].astype(str).tolist()
    payloads = payload_series.astype(str).tolist()
    return payloads, ids


def adaptive_predict(
    candidates: Sequence[tuple[str, float]],
    *,
    base_threshold: float,
    high_confidence: float,
    medium_confidence: float,
    max_diff_second: float,
    max_diff_third: float,
) -> List[str]:
    """按照自适应策略决定最终输出的 CVE 标签。"""
    if not candidates:
        return []

    preds: List[str] = []

    if candidates[0][1] >= base_threshold:
        preds.append(candidates[0][0])
    else:
        return []

    if len(candidates) > 1:
        score1, score2 = candidates[0][1], candidates[1][1]
        if score2 >= high_confidence:
            preds.append(candidates[1][0])
        elif score2 >= medium_confidence and (score1 - score2) < max_diff_second:
            preds.append(candidates[1][0])
        elif score2 >= (medium_confidence + 0.05):
            preds.append(candidates[1][0])

    if len(candidates) > 2 and len(preds) >= 2:
        score1, score2, score3 = candidates[0][1], candidates[1][1], candidates[2][1]
        if (
            score1 >= max(high_confidence, 0.90)
            and score2 >= 0.82
            and score3 >= 0.82
            and (score2 - score3) < max_diff_third
        ):
            preds.append(candidates[2][0])

    return preds


def aggregate_cve_candidates(
    idxs: Sequence[int],
    sims: Sequence[float],
    train_labels: Sequence[Sequence[str]],
    *,
    base_threshold: float,
    max_candidates: int,
    min_votes: int = 1,
    vote_weight: float = 0.015,
) -> tuple[List[tuple[str, float]], int]:
    """Aggregate CVE evidence across nearest neighbours.

    The original pipeline kept the first label seen for each neighbour. That is
    brittle when many near-duplicate payloads point to the same CVE but one
    slightly closer sample has a noisy or broad multi-label annotation. This
    helper scores each CVE by its best similarity plus a small vote bonus from
    additional neighbours.
    """
    stats: dict[str, dict[str, float | int]] = {}
    filtered_out = 0

    for idx, score in zip(idxs, sims):
        if idx < 0 or idx >= len(train_labels):
            continue
        score_float = float(score)
        if score_float < base_threshold:
            filtered_out += 1
            continue

        for cve in train_labels[idx]:
            base_cve = cve.upper()
            if not base_cve.startswith("CVE-"):
                continue
            item = stats.setdefault(base_cve, {"best": score_float, "votes": 0})
            item["best"] = max(float(item["best"]), score_float)
            item["votes"] = int(item["votes"]) + 1

    ranked: list[tuple[str, float]] = []
    for cve, item in stats.items():
        votes = int(item["votes"])
        if votes < min_votes:
            continue
        best_score = float(item["best"])
        ranked.append((cve, best_score + min(votes - 1, 8) * vote_weight))

    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked[:max_candidates], filtered_out


def has_cve_label(labels: Sequence[str]) -> bool:
    return any(str(label).upper().startswith("CVE-") for label in labels)


def should_suppress_by_empty_neighbors(
    idxs: Sequence[int],
    sims: Sequence[float],
    train_has_cve: Sequence[bool],
    *,
    base_threshold: float,
    empty_penalty_margin: float,
    empty_penalty_floor: float,
    empty_penalty_ratio: float,
) -> bool:
    """Use empty-label neighbours as negative evidence.

    Full official training data contains many benign or unlabeled payloads. If
    those samples are almost as close as the best CVE-bearing neighbour, a CVE
    prediction is usually a false positive. The ratio guard prevents one weak
    empty neighbour from suppressing several strong CVE neighbours.
    """
    if empty_penalty_margin < 0:
        return False

    best_cve = -1.0
    best_empty = -1.0
    cve_votes = 0
    empty_votes = 0

    for idx, score in zip(idxs, sims):
        if idx < 0 or idx >= len(train_has_cve):
            continue
        score_float = float(score)
        if train_has_cve[idx]:
            best_cve = max(best_cve, score_float)
            if score_float >= base_threshold:
                cve_votes += 1
        else:
            best_empty = max(best_empty, score_float)
            if score_float >= empty_penalty_floor:
                empty_votes += 1

    if best_cve < base_threshold or best_empty < empty_penalty_floor:
        return False

    required_empty_votes = max(1, int(np.ceil(cve_votes * empty_penalty_ratio)))
    return (best_cve - best_empty) < empty_penalty_margin and empty_votes >= required_empty_votes


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 FAISS 索引进行离线 CVE 检索")
    parser.add_argument("--test-path", required=True, help="测试集 CSV 路径")
    parser.add_argument("--store-dir", default="./embeddings/faiss_store", help="FAISS 索引目录")
    parser.add_argument("--output-path", default="../ans/test_label_faiss.csv", help="预测输出文件")
    parser.add_argument("--payload-column", default="payload_clean", help="测试集文本列名")
    parser.add_argument("--id-column", default="id", help="测试集 ID 列名")
    parser.add_argument("--model-path", default=None, help="SentenceTransformer 模型路径或名称（默认读取索引元数据）")
    parser.add_argument("--device", default=None, help="模型加载设备，如 cpu/cuda")
    parser.add_argument("--batch-size", type=int, default=32, help="嵌入批量大小")
    parser.add_argument("--text-prefix", default=None, help="测试文本编码前缀，例如 E5 使用 'query: '；默认读取索引元数据或空字符串")
    parser.add_argument("--reuse-cache", action="store_true", help="如缓存存在则复用测试向量")
    parser.add_argument("--top-k", type=int, default=50, help="FAISS 检索的候选数量")
    parser.add_argument("--max-candidates", type=int, default=5, help="每条样本保留的候选 CVE 数量")
    parser.add_argument("--base-threshold", type=float, default=0.86, help="基础相似度阈值")
    parser.add_argument("--high-confidence", type=float, default=0.87, help="高置信阈值")
    parser.add_argument("--medium-confidence", type=float, default=0.78, help="中等置信阈值")
    parser.add_argument("--second-gap", type=float, default=0.15, help="第二个候选与第一个的最大差值")
    parser.add_argument("--third-gap", type=float, default=0.05, help="第三个候选与第二个的最大差值")
    parser.add_argument("--min-votes", type=int, default=1, help="CVE aggregate minimum neighbour votes")
    parser.add_argument("--vote-weight", type=float, default=0.015, help="CVE aggregate vote bonus")
    parser.add_argument("--empty-penalty-margin", type=float, default=0.05, help="Suppress prediction when empty-label neighbours are within this similarity gap; set negative to disable")
    parser.add_argument("--empty-penalty-floor", type=float, default=0.80, help="Minimum similarity for empty-label negative evidence")
    parser.add_argument("--empty-penalty-ratio", type=float, default=0.50, help="Required empty-neighbour votes relative to CVE-neighbour votes")
    parser.add_argument("--prediction-blocklist", default=None, help="Optional newline/comma separated CVE labels to remove from final predictions")
    parser.add_argument("--search-batch-size", type=int, default=512, help="FAISS 检索时的批量大小")
    parser.add_argument("--verbose", action="store_true", help="开启详细日志")
    args = parser.parse_args()

    configure_logging(args.verbose)

    store_dir = Path(args.store_dir)
    index_path = store_dir / "faiss.index"
    meta_path = store_dir / "meta.json"
    cache_emb_path = store_dir / "test_embeddings.npy"
    cache_meta_path = store_dir / "test_cache_meta.json"

    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"缺少索引或元数据，请先运行 build_faiss.py 构建索引: {store_dir}")

    meta: Dict[str, Sequence] = json.loads(meta_path.read_text(encoding="utf-8"))
    train_ids = list(meta.get("ids", []))
    train_labels_raw = meta.get("cve_labels", [])
    if not train_ids or not train_labels_raw:
        raise ValueError("元数据缺少 ids 或 cve_labels 信息")

    train_labels: List[List[str]] = [normalize_cve_labels(labels) for labels in train_labels_raw]
    train_has_cve = [has_cve_label(labels) for labels in train_labels]

    logging.info("加载 FAISS 索引: %s", index_path)
    index = faiss.read_index(str(index_path))

    payloads, test_ids = load_test_payloads(Path(args.test_path), args.payload_column, args.id_column)

    model_path = args.model_path or meta.get("embedding_model")
    if not model_path:
        raise ValueError("未在参数或索引元数据中找到 embedding 模型信息，请使用 --model-path 指定。")
    text_prefix = args.text_prefix
    if text_prefix is None:
        text_prefix = str(meta.get("test_text_prefix", ""))
    logging.info("使用嵌入模型: %s", model_path)

    test_vectors: np.ndarray | None = None
    if args.reuse_cache and cache_emb_path.exists() and cache_meta_path.exists():
        cache_info = json.loads(cache_meta_path.read_text(encoding="utf-8"))
        cached_ids = cache_info.get("ids") if isinstance(cache_info, dict) else None
        cached_model = cache_info.get("embedding_model") if isinstance(cache_info, dict) else None
        cached_prefix = cache_info.get("test_text_prefix", "") if isinstance(cache_info, dict) else ""
        if cached_ids == test_ids and cached_model == model_path and cached_prefix == text_prefix:
            logging.info("检测到匹配的测试向量缓存，直接复用: %s", cache_emb_path)
            test_vectors = np.load(cache_emb_path)
        else:
            logging.warning("缓存校验失败（ID 或模型不匹配），将重新计算测试向量。")

    if test_vectors is None:
        model = load_sentence_encoder(model_path, args.device)

        test_vectors = embed_texts(
            model,
            payloads,
            batch_size=max(1, args.batch_size),
            desc="生成测试向量",
            text_prefix=text_prefix,
        )
        np.save(cache_emb_path, test_vectors)
        cache_payload = {
            "ids": test_ids,
            "embedding_model": model_path,
            "test_text_prefix": text_prefix,
        }
        cache_meta_path.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("测试向量与缓存元数据已保存，后续可使用 --reuse-cache 复用。")

    faiss.normalize_L2(test_vectors)

    results: List[dict[str, str]] = []
    total_candidates = 0
    filtered_out = 0
    score_stats: List[float] = []
    pred_counts: Dict[int, int] = {}
    prediction_blocklist = load_prediction_blocklist(Path(args.prediction_blocklist) if args.prediction_blocklist else None)
    blocked_prediction_count = 0

    for start in tqdm(range(0, len(test_vectors), args.search_batch_size), desc="执行向量检索"):
        end = min(start + args.search_batch_size, len(test_vectors))
        batch_vectors = test_vectors[start:end]
        D, I = index.search(batch_vectors, args.top_k)

        for offset, row_id in enumerate(test_ids[start:end]):
            sims = D[offset]
            idxs = I[offset]
            for idx, score in zip(idxs, sims):
                total_candidates += 1
                if idx < 0 or idx >= len(train_ids):
                    continue
                score_stats.append(float(score))

            candidates, filtered = aggregate_cve_candidates(
                idxs,
                sims,
                train_labels,
                base_threshold=args.base_threshold,
                max_candidates=args.max_candidates,
                min_votes=max(1, args.min_votes),
                vote_weight=args.vote_weight,
            )
            filtered_out += filtered

            preds = adaptive_predict(
                candidates,
                base_threshold=args.base_threshold,
                high_confidence=args.high_confidence,
                medium_confidence=args.medium_confidence,
                max_diff_second=args.second_gap,
                max_diff_third=args.third_gap,
            )
            if preds and should_suppress_by_empty_neighbors(
                idxs,
                sims,
                train_has_cve,
                base_threshold=args.base_threshold,
                empty_penalty_margin=args.empty_penalty_margin,
                empty_penalty_floor=args.empty_penalty_floor,
                empty_penalty_ratio=args.empty_penalty_ratio,
            ):
                preds = []
            if preds and prediction_blocklist:
                before_count = len(preds)
                preds = [label for label in preds if label not in prediction_blocklist]
                blocked_prediction_count += before_count - len(preds)
            pred_counts[len(preds)] = pred_counts.get(len(preds), 0) + 1
            results.append({"id": row_id, "cve_labels": " ".join(preds)})

    if prediction_blocklist:
        print(f"Prediction blocklist: {len(prediction_blocklist)} labels, removed {blocked_prediction_count} predicted labels")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"✅ 预测结果已保存: {output_path}")

    if score_stats:
        scores = np.array(score_stats)
        print("\n📊 相似度统计：")
        print(f"  候选总数: {total_candidates}")
        print(f"  被阈值过滤: {filtered_out} ({filtered_out / max(total_candidates, 1) * 100:.1f}%)")
        print(f"  平均值: {scores.mean():.3f}, 中位数: {np.median(scores):.3f}")
        print("  分布区间:")
        print(f"    >0.85 : {(scores > 0.85).sum()} ({(scores > 0.85).sum() / len(scores) * 100:.1f}%)")
        print(f"    0.70-0.85 : {(((scores >= 0.70) & (scores <= 0.85)).sum())} ({(((scores >= 0.70) & (scores <= 0.85)).sum()) / len(scores) * 100:.1f}%)")
        print(f"    0.60-0.70 : {(((scores >= 0.60) & (scores < 0.70)).sum())} ({(((scores >= 0.60) & (scores < 0.70)).sum()) / len(scores) * 100:.1f}%)")
        print(f"    <0.60 : {(scores < 0.60).sum()} ({(scores < 0.60).sum() / len(scores) * 100:.1f}%)")

    print("\n📊 预测数量分布：")
    total_rows = len(results)
    for k in sorted(pred_counts.keys()):
        count = pred_counts[k]
        print(f"  预测{k}个: {count} ({count / max(total_rows, 1) * 100:.1f}%)")


if __name__ == "__main__":
    main()
