#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="${PROJECT_DIR}/.tools"

mkdir -p "${TOOLS_DIR}"

if [[ ! -x "${TOOLS_DIR}/uv" ]]; then
  UV_INSTALL_DIR="${TOOLS_DIR}" \
  UV_NO_MODIFY_PATH=1 \
  /bin/sh -c "$(curl -LsSf https://astral.sh/uv/install.sh)"
fi

"${TOOLS_DIR}/uv" python install \
  3.14 \
  --install-dir "${TOOLS_DIR}/python" \
  --no-bin

UV_PYTHON_INSTALL_DIR="${TOOLS_DIR}/python" \
  "${TOOLS_DIR}/uv" venv "${PROJECT_DIR}/.venv" --python 3.14

"${TOOLS_DIR}/uv" pip install \
  --python "${PROJECT_DIR}/.venv/bin/python" \
  -e "${PROJECT_DIR}[dev]"

echo "Environment ready."
echo "Activate with: source ${PROJECT_DIR}/.venv/bin/activate"
