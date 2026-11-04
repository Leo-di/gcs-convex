#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="${PROJECT_DIR}/.tools"
mkdir -p "${TOOLS_DIR}"

if [[ ! -x "${TOOLS_DIR}/uv" ]]; then
  UV_INSTALL_DIR="${TOOLS_DIR}" UV_NO_MODIFY_PATH=1 \
    /bin/sh -c "$(curl -LsSf https://astral.sh/uv/install.sh)"
fi

# OMPL 2.0.1 has no CPython 3.14 wheel. Keep this runtime separate from the
# existing scene environment and preserve an already-created environment.
if [[ ! -x "${TOOLS_DIR}/refinement-venv/bin/python" ]]; then
  "${TOOLS_DIR}/uv" venv "${TOOLS_DIR}/refinement-venv" --python 3.13
fi
"${TOOLS_DIR}/uv" pip install --python "${TOOLS_DIR}/refinement-venv/bin/python" \
  -e "${PROJECT_DIR}[dev,benchmark]" 'drake==1.57.0'
