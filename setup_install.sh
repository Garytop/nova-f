#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "[nova-f] project: $PROJECT_ROOT"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[nova-f] error: $PYTHON_BIN not found. Install Python 3.10+ first." >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
major, minor = sys.version_info[:2]
if (major, minor) < (3, 10):
    raise SystemExit("Python 3.10+ is required")
print(f"[nova-f] python: {sys.version.split()[0]}")
PY

if [ -d "$VENV_DIR" ] && [ ! -f "$VENV_DIR/bin/activate" ]; then
  if [ "${VENV_DIR:-.venv}" = ".venv" ]; then
    echo "[nova-f] found .venv but it is not a Linux venv; using .venv-linux"
    VENV_DIR=".venv-linux"
  else
    echo "[nova-f] error: $VENV_DIR exists but has no bin/activate" >&2
    exit 1
  fi
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "[nova-f] creating virtual environment: $VENV_DIR"
  if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
    cat >&2 <<'EOF'
[nova-f] failed to create venv.
On Ubuntu/WSL, install venv support first:
  sudo apt update
  sudo apt install -y python3-venv python3-pip
Then rerun:
  source ./setup_install.sh
EOF
    exit 1
  fi
else
  echo "[nova-f] reusing virtual environment: $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[nova-f] upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

echo "[nova-f] installing runtime dependencies"
python -m pip install \
  "numpy>=1.26,<2.0" \
  "pandas>=2.2,<3.0" \
  "tqdm>=4.66,<5.0" \
  "faiss-cpu>=1.8" \
  "sentence-transformers>=5.1,<6.0"

echo "[nova-f] verifying imports"
python - <<'PY'
import faiss
import numpy
import pandas
import sentence_transformers
import tqdm
print("[nova-f] env ok")
PY

if [ ! -d "models/all-MiniLM-L6-v2" ]; then
  cat <<'EOF'
[nova-f] note: models/all-MiniLM-L6-v2 was not found.
The pipeline can still use HuggingFace online model names, but offline WSL runs should place the model at:
  models/all-MiniLM-L6-v2
EOF
fi

cat <<EOF
[nova-f] environment is ready.

Current shell is activated if you ran:
  source ./setup_install.sh

If you executed the script with bash, activate manually:
  source $VENV_DIR/bin/activate

Quick check:
  python -c "import faiss, pandas, numpy; print('env ok')"
EOF
