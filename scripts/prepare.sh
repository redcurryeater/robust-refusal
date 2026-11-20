#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-Qwen/Qwen3.5-2B}
TEACHER=data/teacher/$(basename "$MODEL").jsonl
.venv/bin/python -I src/data.py --raw data/raw --out data/splits
[ -s "$TEACHER" ] || .venv-vllm/bin/python src/teacher.py --model "$MODEL" --out "$TEACHER"
