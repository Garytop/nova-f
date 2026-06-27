from __future__ import annotations

import re
from urllib.parse import parse_qsl, unquote, urlsplit

try:
    from preprocess import clean_payload_text
except ImportError:  # pragma: no cover
    from .preprocess import clean_payload_text


NOISY_KEYS = {
    "",
    "_",
    "callback",
    "cb",
    "id",
    "lang",
    "locale",
    "page",
    "q",
    "r",
    "rand",
    "s",
    "search",
    "t",
    "time",
    "timestamp",
    "token",
    "type",
    "v",
}

SUSPICIOUS_TOKENS = [
    "../",
    "%2e%2e",
    "base64",
    "cmd=",
    "curl",
    "etc/passwd",
    "eval(",
    "jndi",
    "php://",
    "select",
    "union",
    "wget",
    "whoami",
]


def safe_urlsplit(target: str) -> tuple[str, str]:
    try:
        split = urlsplit(target)
        return split.path, split.query
    except ValueError:
        path, _, query = target.partition("?")
        return path, query


def normalize_path(path: str) -> str:
    path = unquote(path.lower())
    path = re.sub(r"%[0-9a-fA-F]{2}", "%xx", path)
    path = re.sub(r"[0-9a-fA-F]{16,}", "{hex}", path)
    path = re.sub(r"\d+", "{num}", path)
    path = re.sub(r"/+", "/", path)
    return path or "/"


def normalize_value_token(value: str) -> str:
    value = unquote(value.lower())
    value = re.sub(r"https?://[^&\s]+", "{url}", value)
    value = re.sub(r"[0-9a-fA-F]{16,}", "{hex}", value)
    value = re.sub(r"\d+", "{num}", value)
    return value[:80]


def normalized_tokens(value: str) -> set[str]:
    value = unquote(value.lower())
    return {token for token in re.split(r"[^a-z0-9_]+", value) if len(token) >= 3}


def parse_payload(payload: object) -> dict[str, object]:
    cleaned = clean_payload_text("" if payload is None else str(payload))
    parts = cleaned.split(" ", 2)
    method = parts[0].upper() if parts else ""
    target = parts[1] if len(parts) > 1 else ""
    rest = parts[2] if len(parts) > 2 else ""
    raw_path, raw_query = safe_urlsplit(target)
    path = normalize_path(raw_path)
    path_parts = [part for part in path.split("/") if part]
    query_pairs = [(key.lower(), normalize_value_token(value)) for key, value in parse_qsl(raw_query, keep_blank_values=True)]
    body_pairs = [(key.lower(), normalize_value_token(value)) for key, value in parse_qsl(rest, keep_blank_values=True)]
    return {
        "cleaned": cleaned.lower(),
        "method": method,
        "path": path,
        "path_prefix2": "/" + "/".join(path_parts[:2]) if len(path_parts) >= 2 else path,
        "path_tokens": normalized_tokens(path),
        "query_pairs": query_pairs,
        "body_pairs": body_pairs,
        "query_keys": {key for key, _ in query_pairs if key},
        "body_keys": {key for key, _ in body_pairs if key},
        "tokens": normalized_tokens(cleaned),
    }


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def feature_bonus(query: dict[str, object], train: dict[str, object]) -> float:
    bonus = 0.0
    if query["method"] and query["method"] == train["method"]:
        bonus += 0.20
    if query["path"] and query["path"] == train["path"]:
        bonus += 0.35
    bonus += 0.20 * jaccard(query["path_tokens"], train["path_tokens"])  # type: ignore[arg-type]
    bonus += 0.15 * jaccard(query["query_keys"], train["query_keys"])  # type: ignore[arg-type]
    bonus += 0.10 * jaccard(query["body_keys"], train["body_keys"])  # type: ignore[arg-type]
    bonus += 0.05 * jaccard(query["tokens"], train["tokens"])  # type: ignore[arg-type]
    return bonus


def candidate_signatures(payload: object, *, modes: set[str]) -> list[str]:
    parsed = parse_payload(payload)
    method = str(parsed["method"])
    path = str(parsed["path"])
    prefix2 = str(parsed["path_prefix2"])
    query_pairs: list[tuple[str, str]] = parsed["query_pairs"]  # type: ignore[assignment]
    body_pairs: list[tuple[str, str]] = parsed["body_pairs"]  # type: ignore[assignment]
    cleaned = str(parsed["cleaned"])
    keys: list[str] = []

    if method and path:
        if "path" in modes:
            keys.append(f"path|{method}|{path}")
        if "path_prefix2" in modes and prefix2 and prefix2 != path:
            keys.append(f"path_prefix2|{method}|{prefix2}")

    query_keys = sorted({key for key, _ in query_pairs if key not in NOISY_KEYS})
    body_keys = sorted({key for key, _ in body_pairs if key not in NOISY_KEYS})
    if "query_keys" in modes and method and path and query_keys:
        keys.append(f"query_keys|{method}|{path}|{','.join(query_keys[:8])}")
    if "body_keys" in modes and method and path and body_keys:
        keys.append(f"body_keys|{method}|{path}|{','.join(body_keys[:8])}")

    if "query_key_value" in modes and method and path:
        for key, value in query_pairs:
            if key and key not in NOISY_KEYS and value and len(value) >= 3:
                keys.append(f"query_kv|{method}|{path}|{key}={value}")
    if "body_key_value" in modes and method and path:
        for key, value in body_pairs:
            if key and key not in NOISY_KEYS and value and len(value) >= 3:
                keys.append(f"body_kv|{method}|{path}|{key}={value}")

    if "token" in modes and method and path:
        for token in SUSPICIOUS_TOKENS:
            if token in cleaned:
                keys.append(f"token|{method}|{path}|{token}")

    return sorted(set(keys))
