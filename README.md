# Directional Dropout

Self-distillation under random ablation of the refusal direction, and whether it makes refusal harder to remove.

A frozen copy of the model is the teacher. The student is trained to match the teacher's next-token distribution on the teacher's own responses, while on a random fraction `p` of sequences the refusal direction (difference of means, Arditi et al. 2024) is projected out of every residual-stream write at every layer and position. The direction is recomputed from the student during training.

## Conditions

| `--condition`   | Perturbation on a sequence                                                     |
|-----------------|--------------------------------------------------------------------------------|
| `dropout`       | ablate the refusal direction, recomputed every 50 steps                        |
| `dropout_fixed` | ablate the refusal direction computed once at the start                        |
| `dropout_top3`  | ablate the top 3 iteratively found refusal directions, recomputed              |
| `noise`         | add isotropic noise with the same norm as the refusal-direction projection    |
| `randdir`       | ablate a variance-matched direction nearly orthogonal to the refusal direction |

## Attacks and metrics

Each model is attacked with re-derived abliteration (k = 1..10 directions, two direction-finding distributions), LoRA fine-tuning on harmful completions (25 and 100 steps), SNIP set-difference pruning (1%, 3%), and prefilling (10 and 40 tokens). Harmfulness is scored with the HarmBench classifier (and optionally the StrongREJECT judge); utility with MMLU and WikiText-2 perplexity; over-refusal with XSTest.

## Usage

Requires one 48 GB GPU (L40S). Run everything from the repository root.

```bash
bash scripts/setup.sh
bash scripts/prepare.sh
bash scripts/sweep.sh
.venv-vllm/bin/python src/judge.py --inputs evals/*/gens_*.jsonl
```

`sweep.sh` evaluates the original model first (`evals/base`), then trains and attacks every condition and seed. It resumes after interruption. To split the sweep across machines, copy `data/` and `evals/base/gens_*.jsonl` to each and run with `SHARD=i NSHARD=n`. `SPECS`, `SEEDS` and `MODEL` can be overridden through the environment.

Single steps:

```bash
.venv/bin/python src/train.py --teacher_data data/teacher/Qwen3.5-2B.jsonl --condition dropout --p 0.5 --seed 0 --out runs/dropout_p0.5_s0
bash scripts/evaluate.sh runs/dropout_p0.5_s0/model dropout_p0.5_s0
```

The StrongREJECT judge (`--strongreject`) needs an `HF_TOKEN` with access to `google/gemma-2b`.

## Layout

```
src/
  common.py        model loading, chat formatting, residual-stream perturbation hooks
  directions.py    refusal-direction finding and selection, matched control direction
  data.py          prompt splits
  teacher.py       teacher responses (vLLM)
  train.py         directional-dropout self-distillation
  untie.py         untie embeddings before weight orthogonalization
  abliteration.py  attacker directions, MMLU and perplexity under ablation
  generate.py      generation under abliteration, prefilling, or as-is (vLLM)
  lora.py          LoRA fine-tuning attack
  prune.py         pruning attack
  judge.py         HarmBench / StrongREJECT scoring
scripts/
  setup.sh  prepare.sh  evaluate.sh  sweep.sh
```

Outputs: `data/splits`, `data/teacher`, `runs/<name>/log.jsonl`, `evals/<name>/`.
