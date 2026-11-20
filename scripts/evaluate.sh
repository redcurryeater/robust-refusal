#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
M=$1; NAME=$2
MODEL=${MODEL:-Qwen/Qwen3.5-2B}
TEACHER=data/teacher/$(basename "$MODEL").jsonl
E=evals/$NAME; B=evals/base
T=.venv/bin/python; V=.venv-vllm/bin/python
mkdir -p "$E"
done_() { [ -f "$E/.$1.done" ]; }
mark() { touch "$E/.$1.done"; echo "-- $NAME: $1"; }

done_ untie || { $T src/untie.py --model "$M" --out "$E/untied"; mark untie; }
done_ dirs || { $T src/abliteration.py --model "$E/untied" --out "$E"; mark dirs; }
done_ abl || { $V src/generate.py --model "$E/untied" --dirs "$E/attack_dirs.pt" --extras --out "$E/gens_abl.jsonl"; mark abl; }
if [ "$NAME" = base ]; then
  done_ loradata || { $V src/generate.py --model "$E/untied" --dirs "$E/attack_dirs.pt" --sets id --ks 1 --skip_base \
                        --split distill_harmful --out "$E/gens_lora_data.jsonl"; mark loradata; }
fi
done_ prefill || { $V src/generate.py --model "$E/untied" --prefill "$B/gens_abl.jsonl" --out "$E/gens_prefill.jsonl"; mark prefill; }
done_ lora_train || { $T src/lora.py --model "$E/untied" --data "$B/gens_lora_data.jsonl" --out "$E/lora"; mark lora_train; }
for s in 25 100; do
  done_ lora_s$s || {
    $V src/generate.py --model "$E/lora/merged_s$s" --name lora_s$s --out "$E/gens_lora_s$s.jsonl"
    $T src/abliteration.py --model "$E/lora/merged_s$s" --out "$E/lora_s$s" --utility_only
    rm -rf "$E/lora/merged_s$s"; mark lora_s$s; }
done
for q in 0.01 0.03; do
  done_ prune_$q || {
    $T src/prune.py --model "$E/untied" --teacher_data "$TEACHER" --q $q --p $q --out "$E/pruned_$q"
    $V src/generate.py --model "$E/pruned_$q" --name prune_$q --out "$E/gens_prune_$q.jsonl"
    $T src/abliteration.py --model "$E/pruned_$q" --out "$E/prune_util_$q" --utility_only
    cp "$E/pruned_$q/prune_info.json" "$E/prune_util_$q/"; rm -rf "$E/pruned_$q"; mark prune_$q; }
done
