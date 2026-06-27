#!/usr/bin/env bash
set -Eeuo pipefail

RED=$'\033[31m'
GREEN=$'\033[32m'
YELLOW=$'\033[33m'
WHITE=$'\033[37m'
CYAN=$'\033[36m'
RESET=$'\033[0m'

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

print_logo() {
  cat <<'EOF'
   _  ______ _   _____           ____
  / |/ / __ \ | / / _ |   ____  / __/
 /    / /_/ / |/ / __ |  /___/ / _/
/_/|_/\____/|___/_/ |_|       /_/
No-cost On-site Vulnerability Analyzer - Fast
EOF
}

usage() {
  print_logo
  cat <<'EOF'

NOVA-F local runner

Usage:
  ./run.sh [options]

Modes:
  --mode base       Basic retrieval pipeline.
  --mode precision  Retrieval + OOF blocklist. Default.
  --mode recall     Retrieval + OOF blocklist + structured rerank.

Common options:
  --train PATH              Training CSV. Default: ./data/train_with_ultimate.csv
  --test PATH               Test CSV. Default: ./data/test_payload.csv
  --store-dir PATH          FAISS store. Default: ./embeddings/faiss_store_combined
  --model-path PATH         Local SentenceTransformer model. Default: ./models/all-MiniLM-L6-v2
  --output PATH             Output CSV. Default: ./data/experiments/run_<mode>.csv
  --blocklist PATH          OOF blocklist path.
  --train-feature-path PATH Cleaned train feature CSV aligned with FAISS metadata.

Tuning options:
  --top-k N                 FAISS neighbours. Default: 50
  --base-threshold FLOAT    Base similarity threshold. Default: 0.86
  --alpha FLOAT             Structured rerank alpha for recall mode. Default: 0.03
  --search-batch-size N     FAISS search batch size. Default: 512
  --test-batch-size N       Embedding batch size. Default: 32

Other:
  --no-cache                Do not pass --reuse-cache.
  --dry-run                 Print command only.
  -h, --help                Show this help.

Examples:
  ./run.sh --mode precision
  ./run.sh --mode recall --alpha 0.03
  ./run.sh --mode base --top-k 100 --base-threshold 0.88

Parameter notes:
  Increase --base-threshold to reduce false positives, usually lowering recall.
  Decrease --base-threshold to improve recall, usually increasing false positives.
  Increase --top-k when candidate evidence is too sparse; it costs more search time.
  Increase --alpha only for recall-first experiments; too large may hurt precision.
EOF
}

die() {
  echo -e "${RED}[nova-f] error: $*${RESET}" >&2
  exit 1
}

warn() {
  echo -e "${YELLOW}[nova-f] warn: $*${RESET}" >&2
}

info() {
  echo -e "${WHITE}[nova-f] $*${RESET}"
}

ok() {
  echo -e "${GREEN}[nova-f] $*${RESET}"
}

SPINNER_PID=""

clear_spinner_line() {
  if [[ -t 2 ]]; then
    printf '\r\033[K' >&2
  fi
}

start_spinner() {
  if [[ ! -t 2 || -n "${SPINNER_PID:-}" ]]; then
    return 0
  fi

  (
    frames=("/" "-" "\\" "|")
    i=0
    while true; do
      printf '\r%s[nova-f] running %s%s' "$CYAN" "${frames[$((i % 4))]}" "$RESET" >&2
      i=$((i + 1))
      sleep 0.2
    done
  ) &
  SPINNER_PID=$!
}

stop_spinner() {
  if [[ -n "${SPINNER_PID:-}" ]]; then
    kill "$SPINNER_PID" >/dev/null 2>&1 || true
    wait "$SPINNER_PID" >/dev/null 2>&1 || true
    SPINNER_PID=""
    clear_spinner_line
  fi
}

run_with_spinner() {
  local exit_code

  start_spinner
  set +e
  "${CMD[@]}" \
    > >(while IFS= read -r line; do clear_spinner_line; printf '%s\n' "$line"; done) \
    2> >(while IFS= read -r line; do clear_spinner_line; printf '%s\n' "$line" >&2; done)
  exit_code=$?
  set -e
  stop_spinner

  return "$exit_code"
}

debug_tips() {
  cat >&2 <<EOF
${RED}[nova-f] Debug checklist:
  1. Activate environment or run: source ./setup_install.sh
  2. Check input CSV paths under ./data/
  3. Check local model path: ./models/all-MiniLM-L6-v2
  4. Check FAISS store: ./embeddings/faiss_store_combined/faiss.index
  5. For --mode recall, ensure --train-feature-path is aligned with FAISS metadata
  6. If cache is stale, retry with --no-cache or remove test_embeddings.npy/test_cache_meta.json${RESET}
EOF
}

trap 'code=$?; stop_spinner; echo -e "${RED}[nova-f] command failed with exit code ${code}${RESET}" >&2; debug_tips; exit "$code"' ERR

MODE="precision"
TRAIN="./data/train_with_ultimate.csv"
TEST="./data/test_payload.csv"
STORE_DIR="./embeddings/faiss_store_combined"
MODEL_PATH="./models/all-MiniLM-L6-v2"
OUTPUT=""
BLOCKLIST="./data/experiments/fold_blocklist_fp20_p002_mf2.txt"
TRAIN_FEATURE="./data/experiments/train_combined_cleaned.csv"
TOP_K="50"
BASE_THRESHOLD="0.86"
ALPHA="0.03"
SEARCH_BATCH_SIZE="512"
TEST_BATCH_SIZE="32"
REUSE_CACHE=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --mode) MODE="${2:-}"; shift 2 ;;
    --train) TRAIN="${2:-}"; shift 2 ;;
    --test) TEST="${2:-}"; shift 2 ;;
    --store-dir) STORE_DIR="${2:-}"; shift 2 ;;
    --model-path) MODEL_PATH="${2:-}"; shift 2 ;;
    --output) OUTPUT="${2:-}"; shift 2 ;;
    --blocklist) BLOCKLIST="${2:-}"; shift 2 ;;
    --train-feature-path) TRAIN_FEATURE="${2:-}"; shift 2 ;;
    --top-k) TOP_K="${2:-}"; shift 2 ;;
    --base-threshold) BASE_THRESHOLD="${2:-}"; shift 2 ;;
    --alpha) ALPHA="${2:-}"; shift 2 ;;
    --search-batch-size) SEARCH_BATCH_SIZE="${2:-}"; shift 2 ;;
    --test-batch-size) TEST_BATCH_SIZE="${2:-}"; shift 2 ;;
    --no-cache) REUSE_CACHE=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) die "unknown option: $1. Use -h for help." ;;
  esac
done

case "$MODE" in
  base|precision|recall) ;;
  *) die "--mode must be one of: base, precision, recall" ;;
esac

if [[ -z "$OUTPUT" ]]; then
  OUTPUT="./data/experiments/run_${MODE}.csv"
fi

print_logo
info "mode: $MODE"

deps_ok() {
  "$PYTHON_CMD" - <<'PY' >/dev/null 2>&1
import faiss
import numpy
import pandas
import sentence_transformers
PY
}

try_activate_conda() {
  local conda_sh
  for conda_sh in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"
  do
    if [[ -f "$conda_sh" ]]; then
      # shellcheck disable=SC1090
      source "$conda_sh"
      conda activate nova-f >/dev/null 2>&1 && return 0
    fi
  done
  return 1
}

if [[ -f ".venv-linux/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv-linux/bin/activate"
elif [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
else
  warn "no Linux virtualenv found; using current Python. Run 'source ./setup_install.sh' if imports fail."
fi

PYTHON_CMD="python"
if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
  PYTHON_CMD="python3"
fi
command -v "$PYTHON_CMD" >/dev/null 2>&1 || die "python/python3 not found"

if deps_ok; then
  ok "dependency check ok"
else
  warn "dependency check failed in current Python; trying conda env: nova-f"
  if declare -F deactivate >/dev/null 2>&1; then
    deactivate >/dev/null 2>&1 || true
  fi
  if try_activate_conda; then
    PYTHON_CMD="python"
  fi
  if deps_ok; then
    ok "dependency check ok after activating conda env nova-f"
  elif [[ "$DRY_RUN" -eq 1 ]]; then
    warn "dry-run continues without importable runtime dependencies"
  else
    die "missing Python dependencies. Run 'source ./setup_install.sh' or activate conda env 'nova-f'."
  fi
fi

[[ -f "$TRAIN" ]] || die "training CSV not found: $TRAIN"
[[ -f "$TEST" ]] || die "test CSV not found: $TEST"
[[ -d "$STORE_DIR" ]] || die "FAISS store dir not found: $STORE_DIR"
[[ -f "$STORE_DIR/faiss.index" ]] || die "FAISS index not found: $STORE_DIR/faiss.index"
[[ -f "$STORE_DIR/meta.json" ]] || die "FAISS meta not found: $STORE_DIR/meta.json"

if [[ ! -d "$MODEL_PATH" ]]; then
  warn "model path not found: $MODEL_PATH"
  warn "main.py may try HuggingFace online loading. Offline WSL runs should provide a local model."
fi

if [[ "$MODE" != "base" ]]; then
  [[ -f "$BLOCKLIST" ]] || die "blocklist not found: $BLOCKLIST"
fi

if [[ "$MODE" == "recall" ]]; then
  if [[ ! -f "$TRAIN_FEATURE" ]]; then
    warn "train feature file not found: $TRAIN_FEATURE"
    warn "trying to generate it from known cleaned train files"
    "$PYTHON_CMD" - <<'PY'
from pathlib import Path
import pandas as pd
paths = [Path("data/train_with_ultimate_cleaned.csv"), Path("data/experiments/train_payload_cleaned.csv")]
missing = [str(path) for path in paths if not path.exists()]
if missing:
    raise SystemExit(f"missing cleaned train files: {missing}")
out = Path("data/experiments/train_combined_cleaned.csv")
out.parent.mkdir(parents=True, exist_ok=True)
pd.concat([pd.read_csv(path) for path in paths], ignore_index=True).to_csv(out, index=False)
print(f"[nova-f] wrote {out}")
PY
  fi
  [[ -f "$TRAIN_FEATURE" ]] || die "train feature file not found after generation: $TRAIN_FEATURE"
fi

mkdir -p "$(dirname "$OUTPUT")"

CMD=(
  "$PYTHON_CMD" main.py
  --train-path "$TRAIN"
  --test-path "$TEST"
  --test-payload-column payload_decoded
  --store-dir "$STORE_DIR"
  --output-path "$OUTPUT"
  --model-path "$MODEL_PATH"
  --top-k "$TOP_K"
  --base-threshold "$BASE_THRESHOLD"
  --search-batch-size "$SEARCH_BATCH_SIZE"
  --test-batch-size "$TEST_BATCH_SIZE"
)

if [[ "$REUSE_CACHE" -eq 1 ]]; then
  CMD+=(--reuse-cache)
fi

if [[ "$MODE" != "base" ]]; then
  CMD+=(--prediction-blocklist "$BLOCKLIST")
fi

if [[ "$MODE" == "recall" ]]; then
  CMD+=(--structured-rerank-alpha "$ALPHA" --train-feature-path "$TRAIN_FEATURE")
fi

echo -e "${CYAN}[nova-f] command:${RESET}"
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "$DRY_RUN" -eq 1 ]]; then
  ok "dry-run complete"
  exit 0
fi

run_with_spinner
ok "output written: $OUTPUT"
