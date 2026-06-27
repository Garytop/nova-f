from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    from preprocess import clean_payload_text, normalize_cve_labels
except ImportError:  # pragma: no cover
    from .preprocess import clean_payload_text, normalize_cve_labels

def configure_logging(verbose: bool) -> None:
    """初始化日志输出。"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")

def load_sentence_encoder(model_path: str, device: str | None = None) -> SentenceTransformer:
    """加载本地 SentenceTransformer 模型。"""
    logging.info("加载本地嵌入模型: %s", model_path)
    encoder = SentenceTransformer(model_path, device=device)
    return encoder


def _resolve_column(frame, preferred: str, fallbacks: List[str], *, column_role: str) -> str:
    """返回在 DataFrame 中实际存在的列名。"""
    candidates = [preferred] + [col for col in fallbacks if col != preferred]
    for name in candidates:
        if name in frame.columns:
            if name != preferred:
                logging.info("%s 字段未找到，使用备用列: %s", column_role, name)
            return name
    raise KeyError(f"找不到字段: {preferred} (可选备选列: {', '.join(candidates)})")


def load_dataset(path: Path, payload_column: str, id_column: str) -> tuple[List[str], List[str], List[List[str]]]:
    """读取并清洗数据，返回 payload、id 以及 CVE 标签列表。"""
    if not path.exists():
        raise FileNotFoundError(f"未找到输入文件: {path}")

    logging.info("开始读取数据集: %s", path)
    import pandas as pd  # 延迟导入以减少初始化开销

    frame = pd.read_csv(path)
    logging.info("读取完成，共 %d 行", len(frame))

    payload_column = _resolve_column(
        frame,
        payload_column,
        fallbacks=["payload_decoded", "payload", "request", "text"],
        column_role="payload",
    )

    id_column = _resolve_column(
        frame,
        id_column,
        fallbacks=["sample_id", "request_id"],
        column_role="id",
    )

    payloads: List[str] = []
    ids: List[str] = []
    labels: List[List[str]] = []

    tqdm.pandas(desc="清理HTTP报文")
    cleaned = frame[payload_column]
    if payload_column != "payload_clean":
        cleaned = cleaned.progress_apply(clean_payload_text)
    else:
        cleaned = cleaned.fillna("")

    label_column = None
    for candidate in ["cve_labels", "cve_labels_decoded", "labels", "label", "target"]:
        if candidate in frame.columns:
            label_column = candidate
            break
    if label_column is None:
        logging.warning("未找到CVE标签列，将使用空列表作为默认标签")
        label_source: Iterable = [[] for _ in range(len(frame))]
    else:
        label_source = frame[label_column].tolist()

    for payload, record_id, raw_labels in zip(cleaned, frame[id_column], label_source):
        ids.append(str(record_id))
        payloads.append(str(payload) if payload is not None else "")
        labels.append(normalize_cve_labels(raw_labels))

    return payloads, ids, labels


def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
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
        chunk = texts[start:end]
        if text_prefix:
            chunk = [f"{text_prefix}{text}" for text in chunk]
        if not chunk:
            continue
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

    vectors = np.vstack(chunks)
    if normalize_vectors:
        faiss.normalize_L2(vectors)
    return vectors


def build_faiss_index(
    vectors: np.ndarray,
    *,
    store_dir: Path,
    index_name: str,
) -> Path:
    """创建并保存 FAISS 索引，返回索引路径。"""
    vectors_copy = vectors.copy()
    faiss.normalize_L2(vectors_copy)

    index = faiss.IndexFlatIP(vectors_copy.shape[1])
    index.add(vectors_copy)

    index_path = store_dir / index_name
    faiss.write_index(index, str(index_path))
    logging.info("FAISS 索引已写出: %s", index_path)
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 FAISS 检索索引 (离线向量)")
    parser.add_argument("--input-path", required=True, help="清洗后的训练 CSV 路径")
    parser.add_argument("--store-dir", default="embeddings/faiss_store", help="索引与向量的输出目录")
    parser.add_argument("--payload-column", default="payload_clean", help="清洗后的文本列名")
    parser.add_argument("--id-column", default="id", help="ID 列名")
    parser.add_argument("--model-path", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer 模型路径或名称")
    parser.add_argument("--device", default=None, help="模型加载设备，如 cpu/cuda")
    parser.add_argument("--batch-size", type=int, default=32, help="编码批量大小")
    parser.add_argument("--text-prefix", default="", help="训练文本编码前缀，例如 E5 使用 'passage: '")
    parser.add_argument("--overwrite", action="store_true", help="是否覆盖已有目录")
    parser.add_argument("--verbose", action="store_true", help="开启详细日志")
    args = parser.parse_args()

    configure_logging(args.verbose)

    store_dir = Path(args.store_dir)
    if store_dir.exists() and args.overwrite:
        logging.info("检测到已有目录，执行覆盖: %s", store_dir)
        for item in store_dir.iterdir():
            if item.is_file():
                item.unlink()
    store_dir.mkdir(parents=True, exist_ok=True)

    payloads, ids, label_lists = load_dataset(Path(args.input_path), args.payload_column, args.id_column)

    model = load_sentence_encoder(args.model_path, args.device)

    vectors = embed_texts(
        model,
        payloads,
        batch_size=max(1, args.batch_size),
        desc="生成训练向量",
        text_prefix=args.text_prefix,
    )

    npy_path = store_dir / "train_embeddings.npy"
    np.save(npy_path, vectors)
    logging.info("训练向量已保存: %s", npy_path)

    index_path = build_faiss_index(vectors, store_dir=store_dir, index_name="faiss.index")

    meta = {
        "ids": ids,
        "cve_labels": label_lists,
        "vector_file": npy_path.name,
        "index_file": index_path.name,
        "payload_column": args.payload_column,
        "id_column": args.id_column,
        "embedding_model": args.model_path,
        "train_text_prefix": args.text_prefix,
        "created_at": datetime.now(tz=UTC).isoformat(),
    }
    meta_path = store_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("元数据已保存: %s", meta_path)

    print("✅ FAISS 索引构建完成")
    print(f"✅ 嵌入文件: {npy_path}")
    print(f"✅ 索引文件: {index_path}")
    print(f"✅ 元数据: {meta_path}")


if __name__ == "__main__":
    main()
