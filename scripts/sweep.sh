#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-Qwen/Qwen3.5-2B}
TEACHER=data/teacher/$(basename "$MODEL").jsonl
SPECS=${SPECS:-"dropout:0.5 randdir:0.5 noise:0.5 dropout_fixed:0.5 dropout_top3:0.5 dropout:0.25"}
SEEDS=${SEEDS:-"0 1 2"}
SHARD=${SHARD:-0}; NSHARD=${NSHARD:-1}
[ -f evals/base/gens_lora_data.jsonl ] || bash scripts/evaluate.sh "$MODEL" base
for seed in $SEEDS; do
  j=0
  for spec in $SPECS; do
    mine=$(( (j + seed) % NSHARD == SHARD )); j=$((j + 1))
    [ "$mine" -eq 1 ] || continue
    c=${spec%%:*}; p=${spec##*:}; name=${c}_p${p}_s${seed}
    [ -f evals/$name/.prune_0.03.done ] && continue
    if [ ! -f evals/$name/.untie.done ] && [ ! -f runs/$name/model/config.json ]; then
      .venv/bin/python src/train.py --model "$MODEL" --teacher_data "$TEACHER" --condition $c --p $p --seed $seed \
        --out runs/$name
    fi
    bash scripts/evaluate.sh runs/$name/model $name
    rm -rf runs/$name/model
  done
done
