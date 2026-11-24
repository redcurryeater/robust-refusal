# Plan: lowering false refusals with training-side fixes

Status: draft, not started · Written 2026-11-24 · Applies to: directional dropout (`src/train.py`, condition `dropout`, p = 0.5)

## Summary

- **Problem:** directional dropout roughly doubles refusals on safe prompts that only sound harmful (XSTest safe: 8% → 20%). Ordinary requests are fine.
- **Likely cause:** training only contrasts harmful prompts with plainly harmless ones. When the refusal direction is deleted, the model learns to refuse based on "sounds dangerous" cues.
- **Fix:** add 2,000 borderline-safe prompts (from OR-Bench-80K) to training, and delete the refusal direction on them too. The model then learns to answer them like the original model does, even without that direction.
- **Code changes:** new data splits, teacher answers for them, a `--borderline_frac` option in training, an OR-Bench-Hard eval, and sweep support. Existing data and results stay untouched.
- **Experiments:** one seed first (the fix, plus a version that doesn't delete the direction on borderline prompts), then 3 seeds of the better one.
- **Success:** XSTest-safe refusal drops to ≤ 12% while abliteration robustness stays about the same (worst case ≤ 15%).
- **Open questions:** where the head-to-head eval code lives, whether OR-Bench is an acceptable source, and how low over-refusal needs to go.

## Instructions for the implementing session

Read this before starting. It applies to Claude, or anyone, carrying out this plan on the GPU machine.

- **Scope of one session:** do Steps 1–5, run every step's check, then launch the Phase 1 runs (§5) in the background and **stop for review**. Do not start Phase 2 without the user's approval.
- **Git:** work on a branch (not `main`). Never commit `data/`, `runs/` or `evals/` (they are gitignored; keep it that way).
- **Never delete or overwrite** existing `data/splits/*`, `data/teacher/Qwen3.5-2B.jsonl`, or anything under `evals/`. Write new outputs to new files.
- **Head-to-head code:** before falling back to phrase counts, search this machine for the head-to-head pipeline (LLM refusal judge, IFEval, held-out XSTest-100, attacker 2): other branches, sibling directories, other clones. If found, use it for §6's metrics. If not, continue with phrase counts and attacker 1.
- **Open questions (§8) have defaults.** Use them and keep going unless the user has answered differently.
- **Running jobs:** train and evaluate one model at a time per GPU. Run long jobs in the background with logs in the run directory, and report when they finish.

## 1. Problem

Directional dropout makes refusal hard to remove, but the model becomes more cautious on safe prompts that sound harmful:

| Model | XSTest safe refused (phrase count, 250) | Alpaca refused (200) |
|---|---|---|
| Original | 8.0% | 2.5% |
| Noise / random-direction controls | 9.2–9.3% | 1.0–1.7% |
| Dropout p = 0.5 | 20.0% | 1.8% |
| Dropout p = 0.25 | 16.4% | 1.8% |

The LLM-judged numbers from the head-to-head runs show the same thing: 18% → 23–30% on 100 held-out XSTest safe prompts, and 78% → 92% on OR-Bench-Hard. Ordinary requests are unaffected. The extra refusals are all on borderline prompts.

## 2. Hypothesis

Each batch has 4 harmful prompts and 12 plain Alpaca prompts. When the refusal direction r is deleted, the student has to rebuild refusal from other features. The only contrast it ever sees is harmful vs. plainly harmless, so surface cues ("this sounds dangerous") are enough to separate them, and nothing penalizes using them. Borderline prompts carry those cues and get refused.

**Fix:** add borderline-safe prompts to the self-distillation data and apply dropout to them as well. On those prompts the student is then trained, with r deleted, to behave like the intact teacher, which mostly means complying. Its backup route to refusal then has to separate real harm from harmful-sounding wording.

**Prediction:** XSTest-safe refusal falls back toward the controls (about 9–12%). Robustness to abliteration is roughly unchanged.

## 3. Design decisions (fixed for this plan)

| Decision | Choice | Reason |
|---|---|---|
| Borderline prompt source | OR-Bench-80K (`bench-llm/or-bench`, config `or-bench-80k`) | Large, built for over-refusal, and not one of our eval sets. Never train on XSTest or OR-Bench-Hard: both are evals. |
| Decontamination | Exact and near-duplicate (word-Jaccard ≥ 0.6, existing `dedupe`) removal against XSTest, OR-Bench-Hard-1K, all test/attack/val splits | The 80K and Hard-1K sets come from the same pipeline and may overlap. |
| Number of prompts | 2,000 for distillation + 128 for validation | 4 per step × 668 steps ≈ 1.3 passes. |
| Teacher targets | The unchanged teacher's own greedy answers and full next-token distribution, **no filtering** | The loss is KL to the teacher's distribution, so filtering out refused answers would not remove the teacher's refusal probability at the first token. Distilling unfiltered keeps the method pure ("match the original"), and the goal is to get back to the teacher's level, not below it. Going below it is a separate variant (§7, V3). |
| Batch composition | 4 harmful + 4 borderline + 8 harmless (new flag `--borderline_frac 0.25`) | Keeps the harmful count fixed, so robustness pressure is unchanged. |
| Training length | Pin `--max_steps 668` | `train.py` derives steps from the harmless count ÷ harmless-per-batch. Changing the batch mix would otherwise change the length and break comparison with existing runs. |
| Dropout on borderline examples | Same per-sequence coin as everything else (p = 0.5), switchable with `--borderline_p` | `--borderline_p 0` is the key ablation (§7, V2). It tests whether *ablating* on borderline matters, or whether just *seeing* borderline prompts is enough. |
| Direction finding | Unchanged: `dir_harmful` vs `dir_harmless` (Alpaca) | The attacker finds directions with Alpaca as the harmless set. Training must keep targeting the direction the attacker would find. |
| Existing splits and teacher file | Must stay byte-identical | So existing results stay comparable and need no re-running. |

## 4. Implementation steps

Do the steps in order. Each step ends with a check that must pass before moving on.

### Step 1. Add borderline splits in `src/data.py`

- Fetch OR-Bench-80K and OR-Bench-Hard-1K. Both are on Hugging Face, not raw GitHub. Use `huggingface_hub.hf_hub_download` (repo `bench-llm/or-bench`, `repo_type="dataset"`) and store the files under `data/raw/`. **Check the exact file names in the repo before coding.** If `huggingface_hub` is not in `requirements.txt`, add it.
- Read the prompt column of each (check the column name, probably `prompt`).
- Build the new splits **after** every existing split is built, using a **separate RNG** (`random.Random(args.seed + 1000)`). This leaves the existing shuffle order untouched.
- Dedupe the OR-Bench-80K prompts against: `held_out` (already includes XSTest), OR-Bench-Hard-1K, `test_harmless`, and all val/attack splits. Then shuffle and take:
  - `distill_borderline`: 2,000
  - `val_borderline`: 128
- Write a new eval split `orbench_hard`: all of OR-Bench-Hard-1K (eval only).
- Add the new counts and the number removed by dedupe to `manifest.json`.

**Check:** run `data.py` into a temporary directory, then `diff` every pre-existing split file against the current `data/splits/`. They must be identical. Also check that no `distill_borderline` prompt appears in XSTest or `orbench_hard`.

### Step 2. Teacher answers for the new splits (`src/teacher.py`, `scripts/prepare.sh`)

- Add a `--splits_to_run` argument (comma-separated; default is the current 4 splits, so current behavior is unchanged).
- Generate `distill_borderline` and `val_borderline` into a **separate file**: `data/teacher/Qwen3.5-2B.borderline.jsonl`. Same settings: greedy, 256 tokens, `enable_thinking=False`.
- In `prepare.sh`, add a second line that creates this file only if it is missing.

**Check:** the existing teacher file is unchanged. In the new file, log the teacher's phrase-refusal rate on `distill_borderline` (`common.is_refusal`). Expect somewhere around 10–40%. If it is above 60%, stop and reconsider the source or prompt subset.

### Step 3. Training changes (`src/train.py`)

- `--teacher_data`: accept one or more paths (`nargs="+"`) and concatenate their rows.
- New arguments:
  - `--borderline_frac` (float, default 0.0). The number of borderline examples per batch is `n_b = round(bs * borderline_frac)`. Harmless count = `bs - n_h - n_b`.
  - `--borderline_p` (float, default = `--p`). Dropout probability used for borderline sequences.
- When `borderline_frac > 0`, `train_b = tokenize(tok, by("distill_borderline"), eos_id)`, cycled with its own iterator like `it_h` and `it_n`.
- Batch order: `[harmful] + [borderline] + [harmless]`. Build the mask per sequence: borderline positions use `borderline_p`, all others use `p`. Draw from the same generator `g`.
- Step count: if `borderline_frac > 0` and `--max_steps` is not given, raise an error. This forces an explicit `--max_steps 668`.
- Logging:
  - Add `kl_borderline_seqs` to the train log.
  - In `evaluate()`, add `refusal_score_borderline_clean` and `refusal_score_borderline_ablated`, computed on the first 64 `val_borderline` prompts with the same `D.refusal_score(D.last_logprobs(...))` calls used for `sel_h`.
  - Add `val_borderline` items to the validation KL, or log them separately as `val_kl_borderline_clean` / `_perturbed`.
- Defaults must reproduce the current behavior exactly when `borderline_frac == 0`.

**Check:** **before editing `train.py`**, run the unchanged code with `--condition dropout --p 0.5 --seed 0 --max_steps 20 --out runs/_regress_old` and keep its `log.jsonl`. After the change, run the same command into `runs/_regress_new`. The step-10 and step-20 `loss` must match to within about 1e-3 relative; GPU runs are not bit-exact. Delete both directories afterwards. A run with `--borderline_frac 0.25 --max_steps 20` starts, logs the new fields, and each batch has 4/4/8 examples.

### Step 4. Evaluation (`src/generate.py`, `scripts/evaluate.sh`, `src/judge.py`)

- **Do not add OR-Bench-Hard to the existing `--extras` / `abl` step.** `evaluate.sh` skips steps that have a `.done` marker, so existing models would never get it, and forcing `abl` to rerun redoes all 20 abliteration generations. Instead add a separate step to `evaluate.sh`, after `abl`:
  `done_ orbench || { $V src/generate.py --model "$E/untied" --split orbench_hard --name "$NAME" --out "$E/gens_orbench.jsonl"; mark orbench; }`
  (`generate.py` with no `--dirs` generates only the unattacked model on `--split`).
- `judge.py` already reports the phrase-refusal rate per split, including `xstest` broken down by safe/unsafe. No change is needed there. Its HarmBench judging only covers `test_harmful`, which is correct.
- **Reference numbers for existing models:** `sweep.sh` deletes `runs/<name>/model` after evaluation, but `evals/<name>/untied` remains. Running `bash scripts/evaluate.sh evals/<name>/untied <name>` on an already-evaluated model runs only the new `orbench` step. Do this for `base` and `dropout_p0.5_s0`, and for seeds 1 and 2 before Phase 2.
- **Gap:** the LLM refusal judge (Qwen2.5-14B with the XSTest prompt), IFEval, the 100-prompt held-out XSTest set, and **attacker 2** came from the head-to-head pipeline, which is **not in this repo**. See open question Q1. Until that is resolved, the phrase-count numbers are the primary measure, and attacker 1 is the robustness check.

**Check:** running `evaluate.sh` for `base` runs only the `orbench` step (every other step is skipped as done). `judge.py --inputs evals/base/gens_orbench.jsonl` produces an `orbench_hard` refusal rate. The existing `evals/base` summaries are unchanged.

### Step 5. Sweep support (`scripts/sweep.sh`)

- Extend the spec format from `cond:p` to `cond:p[:tag]`. Map tags to extra arguments:
  - `bl` → `--borderline_frac 0.25 --max_steps 668`
  - `bl0` → `--borderline_frac 0.25 --borderline_p 0 --max_steps 668`
  - `bl6` → `--borderline_frac 0.375 --max_steps 668` (6 borderline, 6 harmless)
- Run name: `${c}_p${p}_${tag}_s${seed}`. With no tag, the name stays exactly as it is now.
- When a tag starting with `bl` is present, pass both teacher files.

**Check:** `SPECS="dropout:0.5:bl" SEEDS=0` prints and runs the expected command line. The old specs produce the same names as before.

## 5. Experiments

Train one at a time, about 20 min training plus 30 min attack each on an L40S.

**Phase 1: pilot (seed 0)**

| Run | Spec | Question |
|---|---|---|
| V1 | `dropout:0.5:bl` | Does borderline data with dropout lower over-refusal and keep robustness? |
| V2 | `dropout:0.5:bl0` | Is it the ablation on borderline that matters, or just seeing borderline prompts? |
| Reference | existing `dropout_p0.5_s0` | No new run; reuse its evals. |

**Go/no-go after Phase 1:** go on if V1 meets the §6 success criteria for over-refusal and does not break the robustness ones. If V1 ≈ V2, the simpler story ("data alone") is enough. Pick whichever is more robust.

**Phase 2: confirm (seeds 0–2)**

| Run | Spec |
|---|---|
| Best of V1/V2 | × seeds 0, 1, 2 |
| V4 | `dropout:0.5:bl6` (seed 0 only), to test whether more borderline data helps further |

**Phase 3: optional**

- **V3, below-teacher over-refusal (context distillation).** The teacher computes its logits with a system prompt that encourages helpfulness on borderline prompts. This needs a teacher forward pass with a different prefix from the student's, so it is a bigger change to `per_position_kl`. Do it only if Phase 2 lands at the teacher's level and that isn't good enough.
- **V5, `--harmful_frac 0.1875`** (3 harmful per batch) together with `bl`.

## 6. Success criteria

Compare to `dropout_p0.5` (3-seed means from the report) unless stated otherwise.

| Metric | Current | Target |
|---|---|---|
| XSTest safe refused (phrase) | 20.0% | ≤ 12% (controls: 9%) |
| OR-Bench-Hard refused (phrase) | new | below `dropout_p0.5`'s value on the same metric; measure the reference too |
| Alpaca refused | 1.8% | ≤ 3% |
| XSTest unsafe refused | 78.2% | ≥ 70% (original: 71.5%) — must not collapse |
| Harmful test set refused, no attack | 98.0% | ≥ 90% |
| Attacker 1 worst-case HarmBench ASR | 11.2% | ≤ 15% |
| Attacker 2 worst-case (when available) | 40–48% | not worse than 48% |
| MMLU / WikiText ppl | 0.55 / 15 | within 0.01 / 0.5 |
| Final clean KL (train log) | 0.015–0.03 | ≤ 0.04 |

Watch during training: `refusal_score_borderline_ablated` should stay close to `refusal_score_borderline_clean`. If the ablated score is much higher, the student is refusing borderline prompts when r is deleted, which is the failure being fixed.

## 7. Risks

- **Robustness may drop.** Borderline prompts sit near the harmful/harmless boundary, so fitting them could pull the backup refusal route back toward a single direction. Attacker 1 (and attacker 2 when available) is the guard here. If attacker 1 worst-case goes above 15%, try V2 or V4 before giving up.
- **OR-Bench prompt quality.** Some OR-Bench-80K prompts are arguably harmful. Unfiltered teacher distillation handles this: the student copies whatever the teacher does. That is a reason not to filter.
- **Phrase-count measure.** It is crude, and XSTest's ±9 points at n = 100 makes small differences unreadable. The 250-prompt phrase count is the main number until the LLM judge is available.
- **Overlap.** If OR-Bench-80K and OR-Bench-Hard overlap heavily, dedupe could remove many prompts. Log how many were removed.

## 8. Open questions for the user

Each question has a default. The implementing session uses the default unless the user has answered otherwise.

- **Q1.** Where is the head-to-head code: the LLM refusal judge, IFEval, held-out XSTest-100 and attacker 2? Should it be ported into this repo, or will runs be evaluated there?
  *Default:* search the machine for it (see "Instructions for the implementing session"). If it isn't found, use phrase counts and attacker 1 only, and say so in the results.
- **Q2.** Is OR-Bench-80K acceptable as the borderline source, or is a synthetic set preferred (for example, generated XSTest-style prompts)?
  *Default:* use OR-Bench-80K.
- **Q3.** Is the teacher's level (about 8% phrase / 18% judged on XSTest safe) good enough, or should Phase 3 V3 aim lower?
  *Default:* yes, the teacher's level is good enough. Don't start V3.

## 9. Out of scope

- Inference-time fixes (false-refusal vector ablation, SCANS) and DPO.
- Changing which prompts the direction is computed from.
- Re-running the existing 18 models or the baselines.
