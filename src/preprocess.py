from __future__ import annotations

import argparse
import ast
import json
import logging
import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from tqdm import tqdm


HEADERS_TO_REMOVE: List[str] = [
    "Host",
    "User-Agent",
    "Accept",
    "Accept-Encoding",
    "Accept-Language",
    "Connection",
    "Cache-Control",
    "Pragma",
    "Upgrade-Insecure-Requests",
    "Sec-Fetch-Dest",
    "Sec-Fetch-Mode",
    "Sec-Fetch-Site",
    "Sec-Fetch-User",
    "Sec-Ch-Ua",
    "Sec-Ch-Ua-Mobile",
    "Sec-Ch-Ua-Platform",
    "Te",
    "If-Modified-Since",
    "If-None-Match",
    "Dnt",
]
HEADER_PATTERN = re.compile(
    r"^(Host|User-Agent|Accept|Accept-Encoding|Accept-Language|Connection|"
    r"Cache-Control|Pragma|Upgrade-Insecure-Requests|Sec-Fetch-Dest|Sec-Fetch-Mode|"
    r"Sec-Fetch-Site|Sec-Fetch-User|Sec-Ch-Ua|Sec-Ch-Ua-Mobile|Sec-Ch-Ua-Platform|"
    r"Te|If-Modified-Since|If-None-Match|Dnt|Referer|Origin)\s*:",
    re.IGNORECASE
)


def configure_logging(verbose: bool) -> None:
    """初始化日志配置。"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def clean_payload_text(payload: str) -> str:
    """清理HTTP报文，移除冗余头并归一化空白。"""
    if payload is None or (isinstance(payload, float) and pd.isna(payload)):
        return ""

    if not isinstance(payload, str):
        payload = str(payload)

    payload = payload.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    normalized = payload.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")

    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if HEADER_PATTERN.match(stripped):
            continue
        cleaned_lines.append(stripped)

    if not cleaned_lines:
        return ""

    cleaned_text = " ".join(cleaned_lines)
    cleaned_text = cleaned_text.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
    cleaned_text = cleaned_text.replace("\n", " ")
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    return cleaned_text.strip()


def normalize_cve_labels(raw_value: str | Iterable[str]) -> List[str]:
    """解析并标准化CVE标签列表。"""
    if raw_value is None:
        return []
    if isinstance(raw_value, float) and pd.isna(raw_value):
        return []

    if isinstance(raw_value, list):
        labels = raw_value
    else:
        text = str(raw_value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = [token.strip() for token in text.split(" ") if token.strip()]
        labels = parsed if isinstance(parsed, list) else [str(parsed)]

    normalized = []
    for label in labels:
        if not label:
            continue
        label_text = str(label).strip()
        if not label_text or label_text.lower() in {"nan", "none", "null"}:
            continue
        normalized.append(label_text.upper())
    return sorted(set(normalized))


def preprocess_dataframe(
    df: pd.DataFrame,
    *,
    payload_column: str = "payload_decoded",
    id_column: str = "id",
) -> pd.DataFrame:
    """对原始数据集进行清洗并返回新DataFrame。"""
    if payload_column not in df.columns:
        raise KeyError(f"找不到字段: {payload_column}")
    if id_column not in df.columns:
        raise KeyError(f"找不到字段: {id_column}")

    tqdm.pandas(desc="清理HTTP报文")
    df = df.copy()
    df["payload_clean"] = df[payload_column].progress_apply(clean_payload_text)

    if "cve_labels" in df.columns:
        logging.debug("正在解析CVE标签字段")
        df["cve_labels"] = df["cve_labels"].progress_apply(normalize_cve_labels)
    else:
        df["cve_labels"] = [[] for _ in range(len(df))]

    df[id_column] = df[id_column].astype(str)
    df["cve_labels_json"] = df["cve_labels"].apply(json.dumps)
    result = df[[id_column, "payload_clean", "cve_labels_json"]].rename(columns={id_column: "id", "cve_labels_json": "cve_labels"})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="清洗HTTP报文数据集")
    parser.add_argument("--input-path", required=True, help="原始CSV路径")
    parser.add_argument("--output-path", required=True, help="清洗后CSV输出路径")
    parser.add_argument("--payload-column", default="payload_decoded", help="HTTP报文字段名")
    parser.add_argument("--id-column", default="id", help="ID字段名")
    parser.add_argument("--verbose", action="store_true", help="开启详细日志")
    args = parser.parse_args()

    configure_logging(args.verbose)

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"未找到输入文件: {input_path}")

    logging.info("开始读取原始数据: %s", input_path)
    df = pd.read_csv(input_path)
    print(f"✅ 已加载输入数据，共 {len(df)} 行")

    processed = preprocess_dataframe(
        df,
        payload_column=args.payload_column,
        id_column=args.id_column,
    )
    print("✅ 完成HTTP报文清洗")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)
    print(f"✅ 已写出清洗结果: {output_path}")


if __name__ == "__main__":
    main()
