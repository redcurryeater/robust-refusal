#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d .venv ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
[ -d .venv-vllm ] || uv venv --python 3.12 .venv-vllm
uv pip install --python .venv-vllm/bin/python -r requirements-vllm.txt
