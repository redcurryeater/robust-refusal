Robust refusal · Qwen3.5-2B · updated 23 Nov 2026

# Does directional dropout make refusal harder to remove?

All 18 trained models · 6 conditions × 3 seeds Head-to-head vs ReFAT and DeepRefusal · two attackers · 1–2 seeds Harmfulness judged by the HarmBench classifier; refusal phrases shown as secondary

We fine-tuned a small chat model to keep refusing harmful requests while its "refusal direction" was randomly deleted during training, then attacked it four ways. The defense defeats the standard abliteration recipe, but a stronger abliteration attacker still gets harmful answers on 40–48% of prompts. It does nothing against fine-tuning, prefilling, or pruning.

1. **The standard abliteration recipe stops working.** Judged by the HarmBench classifier, abliteration makes the original model give harmful answers to 79% of harmful test prompts. Against models trained with directional dropout (p = 0.5), the attacker's best setting reaches only **11%** on each of the three seeds. These worst cases come from different settings and different prompts; per setting, seeds range from 5 to 11%. A stronger attacker breaks this (item 6).
2. **The effect is specific to the refusal direction.** Controls that perturb the model by the same amount or more, with noise or with a different direction, give no protection: the attacker's best setting reaches 79–91%.
3. **The direction must be recomputed during training.** Dropping out a direction fixed at the start gives no protection either (83–88%), because the attacker finds a new one.
4. **Other attacks still work.** LoRA fine-tuning makes every model harmful on about 77–80% of prompts. Prefilling works better on defended models than on the original (48% harmful with 10 prefilled tokens, against 36%). Pruning works slightly less well on them (45–65% against 56–74%).
5. **MMLU and perplexity are unchanged, but instruction following drops and over-refusal rises.** MMLU (0.55) and perplexity (15) match the original. In the head-to-head runs, IFEval falls 8–9 points (67 → 58–59); about two-thirds of that drop also happens with plain self-distillation and no dropout. Refusal of XSTest's safe-but-scary-sounding prompts rises from 8% to 16–21%.
6. **Against ReFAT and DeepRefusal** (our ports), with a stronger attacker that picks directions by how harmful the resulting answers are, ours reaches **48% and 40%** worst-case harmful answers on its two seeds, and our DeepRefusal port 52% at its one valid setting; the original reaches 90%. We cannot rank ours against DeepRefusal. Ours loses half or less of DeepRefusal's instruction following (IFEval −8 to −9 points, against −18 to −27). DeepRefusal is the only method that resists prefilling, and none resists LoRA. The pilot's 11% figure for ours came from a weaker attacker and should not be used.

## Background: the refusal direction and abliteration

Inside a transformer, each token is represented by a vector (here 2,048 numbers) that every layer reads from and adds to. This running vector is called the residual stream. Arditi et al. (2024) found that whether a chat model refuses is largely controlled by a single direction in this space. If you take the average activation on harmful prompts and subtract the average on harmless prompts, you get a vector `r`. Deleting the component along `r` from every activation makes the model comply with almost anything:

h ← h − (h · r) r (r normalized to length 1)

Abliteration bakes this deletion permanently into the weights, so anyone with the weights can remove a model's refusal in minutes without any training. Our question is whether we can train a model whose refusal survives that.

## Method

### Model and data

**Qwen3.5-2B** (24 layers, hidden size 2,048; a hybrid of linear-attention and full-attention layers). We chose it over the 0.8B version because it refuses far more reliably (92% vs 67.5% in a pilot), and one abliterated direction already takes both to 0%.

Prompts come from public benchmarks: AdvBench, HarmBench, MaliciousInstruct, StrongREJECT and Arditi et al.'s splits for harmful prompts, Alpaca for harmless ones, and XSTest for over-refusal. Near-duplicates (word overlap ≥ 60%) between training and test sets are removed. The held-out sets are:

- **Harmful test set**: 313 prompts (213 StrongREJECT + 100 HarmBench), never used in training or by the attacker. One prompt is 0.32 percentage points.
- **Harmless test set**: 200 Alpaca prompts.
- **XSTest**: 250 safe prompts that sound unsafe ("how do I kill a Python process") and 200 genuinely unsafe ones.

### Training: self-distillation with directional dropout

Two copies of the model are used. The **teacher** is the original, frozen and never modified. The **student** starts identical and is the one trained. We first record the teacher's own greedy answers to 433 harmful prompts (it refuses 95% of them) and 4,000 harmless prompts.

Each training step takes 16 of these prompt–answer pairs, 4 harmful and 12 harmless. For each one, a coin with probability `p` decides whether the refusal direction is deleted from the student for that whole example: from the embedding output and from every attention and MLP output, at all 24 layers and every token position. The student is trained to match the teacher's next-token probabilities on every answer token (KL divergence from teacher to student).

The teacher always has its refusal direction, so on deleted examples the student learns to say what the intact model would say without the direction. On normal examples it learns to stay the same as the teacher. Because the targets are the model's own answers, it is taught no new behavior, only not to depend on one direction for behavior it already has.

**Recomputation is the key detail.** As training proceeds, the student moves refusal into other directions. Every 50 steps we recompute the refusal direction from the current student (with deletion turned off) and delete that one instead. Every 200 steps we also re-pick which layer and token position to compute it at. The selection rule is the attacker's (described below), so training always targets the direction an attacker would find right now.

steps

668 (two passes over the harmless prompts), about 20 minutes on one L40S GPU

optimizer

AdamW, learning rate 3·10⁻⁶ with cosine decay, gradient clipping 1.0; all weights trained except the token embeddings

lr choice

From 300-step pilots at 3·10⁻⁶, 10⁻⁵ and 3·10⁻⁵. The two higher rates pushed the student too far from the teacher on unperturbed inputs (KL 0.06 and 0.19 nats per token, against 0.02).

max length

512 tokens; teacher answers capped at 256 tokens

### Conditions

Each condition was trained with 3 seeds (0, 1, 2), giving 18 models.

| Condition | What is done to a selected example | p | Purpose |
| --- | --- | --- | --- |
| Dropout | Delete the refusal direction, recomputed every 50 steps | 0.5 | Main method |
| Dropout | Same, deleted on fewer examples | 0.25 | Sensitivity to p |
| Dropout, top 3 | Delete the 3 strongest refusal directions, recomputed | 0.5 | Does deleting more help? |
| Dropout, fixed | Delete the direction computed once at step 0, never updated | 0.5 | Is recomputation needed? |
| Noise control | Add random noise whose size equals what deleting the refusal direction would remove, at every write | 0.5 | Is any perturbation enough? |
| Random-direction control | Delete a different direction: nearly orthogonal to refusal (\|cos\| \< 0.1), with similar activation variance | 0.5 | Is any deleted direction enough? |

Measured perturbation size, as a fraction of the residual stream's size, averaged over 3 seeds: dropout 2.1% at the start of training and 3.6% at the end; noise 2.2% and 3.8%, so it is matched. The random-direction control removes more: 7.7% at the start and 4.9% at the end. That makes it a conservative control, since its failure to protect cannot be blamed on being too small.

### Attacks

Every model, including the untouched original ("base"), faces the same four attacks. The attacker only has the trained weights, and recomputes everything on them.

abliteration

The attacker computes candidate refusal directions at each of the first 19 layers and the last 5 prompt positions. It keeps the candidate that most reduces the model's tendency to start with a refusal, among those that barely change answers to harmless prompts (KL \< 0.1). This is the standard Arditi et al. recipe. To remove `k` directions, it repeats the search with the earlier ones already deleted (k = 1 to 10). The directions are then written permanently into the weights. Two attacker prompt sets are used: "in-distribution" (128 harmful prompts like the training ones but never trained on) and "out-of-distribution" (100 held-out StrongREJECT prompts).

LoRA

Fine-tune small adapter matrices (rank 16, all linear layers, learning rate 10⁻⁴, batch 8) on 128 harmful answers produced by the abliterated original model, for 25 and 100 steps. This is a cheap, standard jailbreak-by-fine-tuning.

prefilling

Start the assistant's reply with the first 10 or 40 tokens of a compliant answer (taken from the abliterated original model) and let the model continue.

pruning

Wei et al. (2024): score each weight's importance for refusing and for normal answering, and zero the weights in the top 1% or 3% for refusing that are not in the top 1% or 3% for normal answering.

### Measurements

- **Refusal rate** (most tables below): the share of the 313 harmful test prompts whose greedy answer (up to 384 tokens) contains a standard refusal phrase such as "I can't" or "I'm sorry". Higher means the defense held. This is a crude measure; see Limitations.
- **Refusal score**: the log-odds that the model's first answer token is a refusal opener (almost always "I"), averaged over 32 held-out harmful prompts. Positive means it leans toward refusing, negative toward complying. It lets us see how strong the attacker's best direction is without generating text.
- **Utility**: accuracy on 1,000 random MMLU questions (multiple choice; 0.25 is chance) and perplexity on WikiText-2 (lower is better).

## Results

All numbers are means over 3 seeds for the trained conditions. The original model is a single model. Per-seed numbers are in the appendix.

### 1. Harmful answers, judged by the HarmBench classifier

Percent of 313 harmful test prompts whose answer the HarmBench classifier (a 13B model trained to judge harmful completions) rates as harmful. Lower is better; darker red means the attack succeeded more often. "Worst case" is, per model, the attacker's best abliteration setting, averaged over seeds.

| Model | No attack | Worst case | Directions removed, in-distribution attacker | StrongREJECT attacker |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 3 | 5 | 10 | 1 | 3 |  |  |  |
| Original (base) | 4.8 | 78.9 | 77.3 | 56.5 | 58.8 | 58.8 | 1.3 | 78.9 | 64.5 |
| Noise control | 2.5 | 86.3 | 81.6 | 81.1 | 82.0 | 71.8 | 55.8 | 78.6 | 58.5 |
| Random-direction control | 3.1 | 83.2 | 73.3 | 69.0 | 43.0 | 41.0 | 39.5 | 81.4 | 66.0 |
| Dropout, fixed direction | 3.1 | 84.9 | 72.3 | 79.4 | 79.6 | 68.6 | 57.3 | 57.9 | 79.9 |
| Dropout p = 0.5 | 1.8 | 11.2 | 9.1 | 9.3 | 7.2 | 9.5 | 10.5 | 7.1 | 6.8 |
| Dropout p = 0.25 | 2.6 | 17.6 | 11.4 | 10.9 | 10.8 | 10.1 | 12.1 | 12.4 | 15.4 |
| Dropout, top 3 | 1.3 | 29.6 | 9.8 | 19.8 | 26.4 | 28.4 | 17.0 | 7.8 | 8.4 |

| Model | Prefill tokens | LoRA steps | Weights pruned |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 40 | 25 | 100 | 1% | 3% |  |
| Original (base) | 35.8 | 64.9 | 77.6 | 76.0 | 74.1 | 55.6 |
| Noise control | 41.1 | 62.8 | 78.9 | 79.1 | 69.5 | 50.4 |
| Random-direction control | 39.0 | 63.8 | 77.2 | 80.2 | 72.1 | 56.2 |
| Dropout, fixed direction | 41.9 | 68.5 | 77.2 | 79.0 | 70.4 | 52.7 |
| Dropout p = 0.5 | 48.4 | 71.3 | 78.3 | 77.1 | 65.2 | 44.6 |
| Dropout p = 0.25 | 46.1 | 68.7 | 76.7 | 77.5 | 65.6 | 46.9 |
| Dropout, top 3 | 44.5 | 68.7 | 79.2 | 77.7 | 63.9 | 50.5 |

The judge confirms the refusal-phrase result for this attacker. Against p = 0.5 dropout, the attacker's best abliteration setting produces harmful answers 11.2% of the time on every seed, against 79% for the original model and 79–91% for the controls and the fixed-direction model. Removing more directions does not help this attacker. A stronger attacker, which chooses the direction by judging the harmful answers it produces, raises ours to 40–48% (see the head-to-head section).

The judge also shows where refusal phrases mislead. With 10 directions removed, the original model "refuses" 0% of prompts by phrase count, yet only 1.3% of its answers are harmful: 98.7% are neither refusals nor harmful, because removing that many directions breaks the model. The random-direction control shows the same breakdown at 3 or more directions. Phrase counts treat a broken model as a successful attack; the judge does not.

Prefilling is confirmed as a weakness: 48% harmful at 10 prefilled tokens against 36% for the original. LoRA succeeds on every model (77–80%). Pruning is slightly less effective on the dropout models, but it damages every model's MMLU (section 4), so we do not read much into it.

### 2. Refusal-phrase counts under abliteration (secondary)

Percent of 313 harmful test prompts whose answer contains a refusal phrase. Darker green means more refusals. This measure counts broken output as "not refused"; see section 1.

| Model | No attack | Directions removed, in-distribution attacker | Out-of-distribution attacker |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 3 | 5 | 10 | 1 | 3 |  |  |
| Original (base) | 87.9 | 0.3 | 0.6 | 1.0 | 1.0 | 0.0 | 1.0 | 3.2 |
| Noise control | 91.3 | 1.2 | 1.8 | 1.5 | 5.4 | 2.2 | 0.6 | 1.4 |
| Random-direction control | 91.3 | 0.7 | 1.4 | 0.5 | 1.2 | 2.4 | 0.6 | 1.0 |
| Dropout, fixed direction | 91.9 | 5.0 | 5.5 | 6.0 | 7.8 | 11.9 | 6.8 | 7.1 |
| Dropout p = 0.5 | 98.0 | 85.5 | 84.6 | 86.7 | 83.6 | 81.8 | 87.0 | 86.7 |
| Dropout p = 0.25 | 96.4 | 77.3 | 79.0 | 77.3 | 75.3 | 73.9 | 79.2 | 73.9 |
| Dropout, top 3 | 98.4 | 83.3 | 73.7 | 67.5 | 62.9 | 76.0 | 84.1 | 85.4 |

The three recomputed-dropout models keep most of their refusals however many directions the attacker removes, and the seeds agree. For p = 0.5, the three seeds give 83–89% after one direction and 78–84% after ten. Deleting more directions during training (top 3) does not help. Deleting on fewer examples (p = 0.25) costs about 8 points.

Dropping out a fixed direction gives little protection and is inconsistent across seeds: 4–6% refusal after one direction, and between 1% and 20% after ten depending on the seed. Both controls behave like the original model.

The defended models also refuse more without any attack (96–98% against 88%), while the controls barely move (91–92%).

### 3. Why the standard attacker fails

The attacker's own search on its prompt sets. The refusal score is positive when the model leans toward refusing. "Layer picked" lists the layer the in-distribution attacker chose for each seed.

| Model | Layer picked | Refusal score after removing k directions | MMLU after removing k directions |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| k = 0 | 1 | 3 | 10 | OOD, 1 | 0 | 1 | 10 |  |  |
| Original (base) | 15 | +2.76 | −8.05 | −7.97 | −18.34 | −7.86 | 0.550 | 0.545 | 0.372 |
| Noise control | 18, 11, 11 | +2.78 | −7.62 | −5.16 | −4.25 | −8.13 | 0.558 | 0.551 | 0.482 |
| Random-direction control | 18, 18, 18 | +2.67 | −8.05 | −5.93 | −7.59 | −8.15 | 0.549 | 0.532 | 0.385 |
| Dropout, fixed | 11, 11, 18 | +2.58 | −3.27 | −3.25 | −2.57 | −4.10 | 0.548 | 0.543 | 0.487 |
| Dropout p = 0.5 | 10, 10, 10 | +3.13 | +1.12 | +1.21 | +1.10 | −0.02 | 0.551 | 0.541 | 0.457 |
| Dropout p = 0.25 | 10, 10, 10 | +3.12 | +0.72 | +0.83 | +0.70 | −0.43 | 0.548 | 0.535 | 0.495 |
| Dropout, top 3 | 14, 14, 14 | +3.21 | +0.71 | +1.39 | +1.51 | −0.27 | 0.559 | 0.555 | 0.487 |

On the original model the best direction swings the refusal score from +2.8 to −8, a firm "comply". On the dropout models the best direction the attacker can find only lowers it to about +1, so the model still leans toward refusing, and removing more directions does not lower it further. No single direction carries refusal on its own any more. The out-of-distribution attacker pushes the first-token score to about 0, a coin flip, yet 79–87% of the generated answers still refuse. The model often recovers into a refusal after the first word. This attacker picks its direction by that first-token score; one that picks by judging full answers does break the dropout models (head-to-head section).

The attack is cheap in utility: removing one direction costs at most 0.02 MMLU on any model. Removing ten costs more on every model, and the most on the original (0.55 to 0.37), so the attacker has no reason to go that far.

The same picture shows up during training. At step 0, deleting the refusal direction drives the score to −8.8. By the end of training, deleting the current direction leaves it at +0.5 to +1.2, and deleting the original step-0 direction leaves it at +3.2 to +3.7, so the model has stopped relying on that direction entirely. The noise and random-direction students still collapse to −7.7 to −8.6 when the refusal direction is deleted.

### 4. The other attacks: refusal phrases and utility (secondary)

Percent of 313 harmful test prompts whose answer contains a refusal phrase.

| Model | Prefill tokens | LoRA steps | Weights pruned |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 40 | 25 | 100 | 1% | 3% |  |
| Original (base) | 27.2 | 10.2 | 0.3 | 0.6 | 0.6 | 0.6 |
| Noise control | 20.0 | 8.0 | 1.1 | 1.3 | 2.8 | 0.4 |
| Random-direction control | 22.4 | 7.2 | 0.6 | 1.2 | 1.5 | 1.6 |
| Dropout, fixed direction | 16.5 | 4.6 | 0.9 | 1.2 | 4.5 | 1.8 |
| Dropout p = 0.5 | 8.8 | 3.6 | 0.9 | 1.1 | 8.0 | 12.8 |
| Dropout p = 0.25 | 11.3 | 3.7 | 0.3 | 1.1 | 4.9 | 4.0 |
| Dropout, top 3 | 9.8 | 2.3 | 0.3 | 1.3 | 7.2 | 7.2 |

MMLU of the attacked models (no attack: 0.55 for every model).

| Model | LoRA 25 steps | LoRA 100 steps | Pruned 1% | Pruned 3% |
| --- | --- | --- | --- | --- |
| Original (base) | 0.532 | 0.519 | 0.340 | 0.364 |
| Noise control | 0.526 | 0.521 | 0.450 | 0.397 |
| Random-direction control | 0.516 | 0.517 | 0.439 | 0.418 |
| Dropout, fixed | 0.532 | 0.526 | 0.441 | 0.380 |
| Dropout p = 0.5 | 0.546 | 0.540 | 0.456 | 0.411 |
| Dropout p = 0.25 | 0.540 | 0.531 | 0.452 | 0.375 |
| Dropout, top 3 | 0.544 | 0.544 | 0.417 | 0.411 |

- **LoRA** removes refusal from every model after 25 steps and costs the attacker almost no utility (MMLU stays at 0.52–0.55).
- **Prefilling** works better on all self-distilled models than on the original, and best of all on the dropout models (about 9–11% refusals left at 10 tokens, against 27% for the original and 20–22% for the controls). A likely reason: dropout teaches the model to decide to refuse from features at the end of the prompt, and once a compliant answer has started, those features no longer steer what comes next. This is untested.
- **Pruning** leaves the dropout models slightly more refusals (up to 13%), but pruning is a poor attack on this model to begin with. It cuts MMLU to 0.34–0.46 on every model, so we do not read much into these columns.

### 5. Cost of the defense

Percent refused on prompts that should be answered (harmless test set, XSTest safe) and should be refused (XSTest unsafe).

| Model | Harmless test set (200) | XSTest safe (250) | XSTest unsafe (200) |
| --- | --- | --- | --- |
| Original (base) | 2.5 | 8.0 | 71.5 |
| Noise control | 1.7 | 9.3 | 73.2 |
| Random-direction control | 1.0 | 9.2 | 72.3 |
| Dropout, fixed direction | 2.0 | 11.2 | 75.3 |
| Dropout p = 0.5 | 1.8 | 20.0 | 78.2 |
| Dropout p = 0.25 | 1.8 | 16.4 | 76.8 |
| Dropout, top 3 | 1.0 | 20.7 | 78.5 |

Ordinary harmless requests are unaffected, and MMLU and perplexity match the original (section 3). The cost shows up on borderline prompts: the dropout models refuse about twice as many of XSTest's safe-but-alarming prompts. The student also stays close to the teacher overall, with a final KL of 0.015–0.03 nats per answer token on unperturbed held-out examples. The head-to-head pilot (next section) measured instruction following, which these metrics miss, and found a real drop.

## Head-to-head pilot: ReFAT and DeepRefusal

We compared our method with the two closest published defenses on the same model, data splits, attacks and judges. Neither paper released code for this model, so both baselines are our ports, written from a specification an independent reviewer produced from the papers. The ports use the papers' data, loss, layers, LoRA settings and ablation schedule. A fourth arm, "hard-label", is our method with ordinary cross-entropy on the teacher's answers in place of matching its full next-token distribution.

**Fairness procedure.** Everything below was fixed in a written protocol before any baseline was trained:

- **Learning rates.** Each method got three full-length runs, around the paper's value for the baselines and around our original value for ours.
- **Selection rule.** One rule, applied to validation data only, picked each method's learning rate. A model was eligible if it kept MMLU-validation within 2 points of the original, harmless over-refusal within 5 points and borderline over-refusal within 10 points. Among eligible models the rule picked the most refusal-robust; if none was eligible, it took the lowest learning rate.
- **Data hygiene.** Every baseline training prompt that overlapped a test set was removed.
- **Gates.** The pipeline halted if the refusal judge disagreed with human labels too often, if gradients changed with memory-saving recomputation, or if a saved model did not reproduce its training loss.

The pilot ran one seed per method against the standard attacker. The final test added a second, stronger attacker for every model and a second seed for ours and DeepRefusal, with the seed-0 learning rates kept fixed.

| Arm | What is trained | Ablation during training | Weights | Selected lr |
| --- | --- | --- | --- | --- |
| Ours | Match the frozen original's next-token distribution on its own answers | One refusal direction, recomputed every 50 steps, on half the examples | full | 3·10⁻⁶ (passed) |
| Hard-label | Same answers, cross-entropy on the tokens | Same as ours | full | 3·10⁻⁶ (fallback) |
| ReFAT (port) | Refusals to Circuit Breakers harmful prompts, plus UltraChat and XSTest compliant answers | Per-layer directions at layers 6–23, recomputed every 4 steps, on harmful examples | LoRA r128 | 6.7·10⁻⁶ (fallback) |
| DeepRefusal (port) | Refusals, UltraChat, XSTest and OR-Bench compliant answers; harmful prompts also with 20–25-token harmful answer prefixes | One direction fixed at the start, dropped per site and per token with p = 0.5, on all examples | LoRA r16 | 3·10⁻⁴ (passed) |

"Passed" means the run met the selection rule; "fallback" means no learning rate did, so the lowest was used. ReFAT over-refused at every learning rate. As declared before testing, we also report ReFAT at 6·10⁻⁵, the grid point that broke the limits least. The seed-1 runs were checked against the same rule afterwards: ours passes; DeepRefusal fails it (it refuses 15.6% of harmless validation requests against a 7.7% limit, and 46.5% of borderline ones against 25.5%), so its seed-1 numbers are not at a valid operating point.

### Final test: two abliteration attackers

**Attacker 1** is the pilot's standard abliteration: it chooses the layer and token position of the refusal direction by how much removing it lowers the first-token refusal score, and only accepts directions that barely change harmless answers. That is the same rule our method recomputes during training, so it favors ours.

**Attacker 2**, written into the protocol before it was run, searches the same family of directions but chooses among all 90 (layer, position) candidates by generating 192-token answers to 32 of its own harmful prompts and counting how many the HarmBench classifier rates harmful. It comes in two variants, with and without the harmless-answer limit. It passed its validity check: on the original model it reaches 89.8%, against 78.9% for attacker 1. Neither attacker ever saw the test prompts.

Worst-case HarmBench attack success rate on the 313 harmful test prompts (%; lower is better), over all attacker settings: 1–10 directions removed, two attacker prompt sets, and for attacker 2 both variants. These are maxima over 14–21 settings per model, so they are biased upward, by different amounts per row, and have no error bar; one setting's 95% interval is about ±2–5.5 points. Settings where removal broke the model (MMLU down more than 10 points) are excluded from robustness readings. Over-refusal and IFEval are shown alongside because several baselines sit at fallback settings.

| Model | lr (selection rule) | Attacker 1 worst | Attacker 2 worst | Worst of both | No attack | XSTest safe refused | Alpaca refused | IFEval change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original | – | 78.9 | 89.8 | 89.8 | 4.8 | 18 | 3.5 | – |
| Ours, seed 0 | 3·10⁻⁶ (passed) | 11.2 | 47.6 | 47.6 | 1.3 | 30 | 4.5 | −8.1 |
| Ours, seed 1 | 3·10⁻⁶ (passes) | 13.4 | 40.3 | 40.3 | 2.2 | 23 | 4.0 | −9.1 |
| Hard-label | 3·10⁻⁶ (fallback) | 61.7 | 43.1 | 61.7 | 2.9 | 42 | 5.5 | −8.3 |
| ReFAT (port) | 6.7·10⁻⁶ (fallback) | 64.2 | 58.5 | 64.2 | 0.6 | 73 | 21.5 | −14.4 |
| DeepRefusal (port), seed 0 | 3·10⁻⁴ (passed) | 43.1 | 51.8 | 51.8 | 0.6 | 28 | 4.0 | −17.7 |
| DeepRefusal (port), seed 1 | 3·10⁻⁴ (fails) | 36.1 | 48.6 | 48.6 | 0.0 | 55 | 10.0 | −27.4 |
| ReFAT @ 6·10⁻⁵ (sensitivity) | 6·10⁻⁵ (failed) | 33.2 | 30.3 | 33.2 | 0.6 | 72 | 15.0 | −22.4 |

Attacker 2 is not stronger on every model: on hard-label and ReFAT attacker 1 does more damage, so each model's headline is whichever attacker did worse to it.

Attacker 2 against ours and DeepRefusal by setting (%). Columns are how many directions were removed; "pool" prompts come from the training-prompt pool (in-distribution for ours), "SR" are held-out StrongREJECT prompts; "no limit" drops the harmless-answer limit.

| Model | Pool prompts, directions removed | SR prompts | SR, no limit |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 3 | 5 | 10 | 1 | 3 | 1 | 3 |  |
| Ours, seed 0 | 10.2 | 11.8 | 39.3 | 47.6 | 30.7 | 5.8 | 18.5 | 7.0 | 36.4 |
| Ours, seed 1 | 7.7 | 8.6 | 26.2 | 40.3 | 24.9 | 9.6 | 15.7 | 8.0 | 32.9 |
| DeepRefusal (port), seed 0 | 42.2 | 51.8 | 38.0 | 37.1 | 24.6 | 48.9 | 35.8 | 48.9 | 35.8 |

Ours holds against one or two removed directions (at most 13% in every setting and both seeds) and breaks at three or more. The damage comes from one direction cell, layer 10 at the fifth-from-last prompt token (position 0 of the five positions scored). On ours, every candidate cell scored at most 1 of 32 harmful in attacker 2's selection, so the pool-prompt choice was settled by a tie-break. We therefore ran every cell tied on that top score at 3 and 5 directions. None raised ours' worst case in either seed: the chosen cell gives 39–48% (seed 0) and 26–40% (seed 1), and every other tied cell is at or below 22%. The StrongREJECT selection, which did have signal (5–6 of 32), picked the same cell independently in both seeds. No tied cell raised any model's worst-of-both figure. This is all within one family, single difference-of-means directions below 80% depth, at the settings listed.

### Attacker 1 (pilot) and the other attacks

HarmBench-classifier attack success rate on the 313 harmful test prompts (%; lower is better; about ±3 points at these rates). "Pool" attacker prompts come from the training-prompt pool (in-distribution for ours and hard-label); "SR" are held-out StrongREJECT prompts; the number is how many directions were removed. \*DeepRefusal trains on harmful prefixes, so prefill lies inside its training threat model.

| Model | No attack | Attacker 1 worst | Pool attacker, directions removed | SR attacker | Prefill\* | LoRA steps |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 3 | 5 | 10 | 1 | 3 | 10 | 40 | 25 | 100 |  |  |  |
| Original | 4.8 | 78.9 | 77.3 | 56.5 | 58.8 | 58.8 | 1.3 | 78.9 | 64.5 | 35.8 | 64.9 | 77.6 | 76.0 |
| Ours | 1.3 | 11.2 | 6.4 | 8.6 | 5.8 | 11.2 | 9.9 | 6.1 | 6.1 | 47.9 | 70.6 | 73.5 | 78.3 |
| Hard-label | 2.9 | 61.7 | 9.3 | 61.7 | 25.6 | 47.3 | 52.7 | 13.1 | 35.8 | 40.6 | 68.0 | 80.2 | 80.2 |
| ReFAT (port) | 0.6 | 64.2 | 33.6 | 64.2 | 52.7 | 34.2 | 7.3 | 32.6 | 57.8 | 37.4 | 64.9 | 76.4 | 80.5 |
| DeepRefusal (port) | 0.6 | 43.1 | 42.2 | 34.2 | 34.5 | 21.4 | 15.3 | 43.1 | 28.7 | 3.2 | 30.7 | 80.8 | 77.3 |
| ReFAT @ 6·10⁻⁵ (sensitivity) | 0.6 | 33.2 | 4.5 | 10.9 | 32.9 | 33.2 | 21.7 | 16.0 | 12.5 | 48.2 | 73.2 | 77.3 | 79.2 |

Low numbers with 10 directions removed can mean a broken model rather than a robust one. The original's 1.3% and ReFAT's 7.3% come with MMLU of 0.37 and 0.22 (chance is 0.25). Ours keeps 0.47 at 10 directions.

### Capability and over-refusal

IFEval: prompt-level strict accuracy, 541 prompts; the change is paired against the original, item by item. Running the original twice gives a change of −0.6. IFEval is scored with the language detector seeded; unseeded, scores move by 1–2 items between runs. GSM8K: 1,319 problems. Refusal is judged by Qwen2.5-14B with the XSTest paper's prompt (92% agreement with human labels). The XSTest column uses 100 safe prompts that no arm trained on (±9 points). These judge-based numbers are not comparable with section 5's phrase counts. KL: drift from the original on Alpaca prompts, nats per token.

| Model | IFEval | Change | GSM8K | MMLU | WikiText ppl | XSTest safe refused | Alpaca refused | OR-Bench-Hard refused | KL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original | 66.7 | – | 75.3 | 0.550 | 15.3 | 18.0 | 3.5 | 78.2 | – |
| Ours | 58.6 | −8.1 | 74.5 | 0.552 | 15.1 | 30.0 | 4.5 | 92.0 | 0.019 |
| Hard-label | 58.4 | −8.3 | 77.0 | 0.545 | 20.7 | 42.0 | 5.5 | 94.0 | 0.738 |
| ReFAT (port) | 52.3 | −14.4 | 75.9 | 0.547 | 15.1 | 73.0 | 21.5 | 99.9 | 0.195 |
| DeepRefusal (port) | 49.0 | −17.7 | 71.6 | 0.606 | 14.7 | 28.0 | 4.0 | 95.5† | 0.538 |
| ReFAT @ 6·10⁻⁵ (sensitivity) | 44.4 | −22.4 | 71.9 | 0.592 | 14.6 | 72.0 | 15.0 | 99.2 | 0.362 |

†DeepRefusal trains on OR-Bench prompts, so OR-Bench-Hard is in its training distribution. IFEval drops are significant for every trained arm (paired test, z ≥ 4.1; the repeat of the original gives z = 0.5). Seed 1: ours 57.7 (−9.1), GSM8K 74.1; DeepRefusal 39.4 (−27.4), GSM8K 68.2.

### What the final test shows

- **The pilot overstated ours.** Choosing the direction by judged answers instead of the first-token score raised ours' worst case from 11–13% to 40–48%. The pilot's 11.2% should not be used as ours' robustness to abliteration.
- **Ours versus DeepRefusal on abliteration: no ranking.** Under the stronger attacker, ours (47.6% and 40.3%) and our DeepRefusal port at its one valid setting (51.8%, seed 0) cannot be told apart. The original model is at 89.8%.
- **Where ours breaks.** The vulnerability is one reproducible direction cell (layer 10, position 0) and appears only when three or more directions are removed; with one or two, ours stayed at or below 13% in every setting and both seeds.
- **Instruction following.** Ours lost 8–9 IFEval points in both seeds; DeepRefusal lost 18 (seed 0) and 27 (seed 1, invalid setting). Plain self-distillation with no dropout already loses 5.4 points and the matched noise control 9.4, so most of ours' loss comes from the self-distillation recipe, not from directional dropout. GSM8K moved by under 2 points for ours.
- **Prefilling.** Only DeepRefusal reduced prefill attacks (3–4% at 10 tokens and 31–39% at 40, against 36% and 65% for the original). Ours was worse than the original (48–52% and 71–74%) in both seeds. Prefill is inside DeepRefusal's training threat model.
- **LoRA.** No method resisted LoRA fine-tuning (76–81% after 100 steps, against 76% for the original).
- **Distribution matching versus hard labels.** The hard-label variant kept resistance to one direction but not to two or more (61.7%), and drifted far more than ours (KL 0.74 against 0.019). It did not pass the selection rule, and one seed cannot separate the loss from the amount of drift.
- **ReFAT and learning rate.** Our ReFAT port over-refused at every learning rate (at the fallback, 21.5% of ordinary requests and 73% of XSTest safe prompts), so it has no valid operating point. At 6·10⁻⁵ its worst case was 33.2%, with 72% XSTest refusal and 22 IFEval points lost. Baseline results are highly sensitive to learning rate.
- **Over-refusal.** On held-out XSTest safe prompts ours refused 30% and 23% against 18% for the original (n = 100, about ±9). DeepRefusal seed 0 refused 28%.

### What it does not show

- **A robustness ranking.** It does not show that ours is more, or less, robust than ReFAT or DeepRefusal. The gaps between ours and DeepRefusal are 4–12 points, between upward-biased maxima from one or two seeds.
- **Robustness beyond this attack family.** Both attackers remove difference-of-means directions at layers below 80% depth, at the settings listed. Gradient-learned directions and other families were not tried.
- **Capability preservation.** Ours does not "preserve capability". Low KL is its training objective, and it coexists with an 8–9-point IFEval loss.
- **Properties of the methods.** DeepRefusal's seed instability and ReFAT's over-refusal are properties of our ports on a 2B model with inferred hyperparameters, not shown to be properties of the published methods.
- **DeepRefusal's MMLU gain.** Its MMLU gain (0.606) is mostly a change in which answer letter it prefers: its letter mix moves close to the true answer mix, while GSM8K and IFEval fall.
- **Pruning, or broken settings.** Pruning lowers MMLU to 0.34–0.54 on every model, and settings where removal broke the model are not evidence of robustness.

### Caveats

1. **Unvalidated ports.** The baselines are our ports with some inferred hyperparameters, not validated reproductions of the papers' results.
2. **Fallback and invalid settings.** ReFAT and hard-label run at fallback learning rates, and DeepRefusal seed 1 fails the selection rule; seed 0 passed it exactly at the borderline limit. Always read attack success next to over-refusal.
3. **Weak selection signal.** Attacker 2 scores candidates with one direction removed, on 32 prompts. On ours, every cell scored at most 1 of 32, so its pool-prompt choice carried no signal; the tied-cell check above covers that case.
4. **In-distribution prompts.** Pool attacker prompts and the validation probe are in-distribution for ours and hard-label.
5. **Biased maxima.** Worst cases are maxima over 14–21 settings and are biased upward, by different amounts per row. In DeepRefusal seed 0 both attackers give 42.2% at pool, one direction; the cells differ and the match is a coincidence.
6. **Confounds.** One or two seeds per arm, not three. Ours and the baselines also differ in training data, full versus LoRA fine-tuning, and loss. A LoRA version of ours has not been run.
7. **Refusal judge.** It matches humans 92% of the time on XSTest and is not validated on harmful outputs.
8. **Exploratory IFEval controls.** The self-distillation and noise-control IFEval runs were added after the protocol was written and are exploratory.
9. **Letter-sensitive MMLU.** MMLU is scored by the logit of the first answer letter, so it is sensitive to which letters a model prefers.

## Interpretation

Directional dropout works like data augmentation against one specific attack: deleting the direction that the standard recipe picks from the residual stream. Training against many random draws of that perturbation spreads refusal across directions, so no single deletion removes it. The final test shows the limit of this: an attacker that picks a different cell (layer 10, position 0) and removes three or more directions there still gets through, so the augmentation covers the directions it was trained on more than the family as a whole. Two results suggest the model is chasing a moving target rather than just becoming generally robust. A fixed direction is easily routed around, and perturbations of the same or larger size that are not aimed at refusal do nothing.

That framing also explains the failures. LoRA changes the weights themselves using data that directly rewards complying, and prefilling and pruning act on different parts of the computation. The student never saw anything like these during training, so there is no reason to expect protection against them.

## Limitations

- **One judge, not validated on our prompts.** Harmfulness is scored by the HarmBench classifier only. It is applied to StrongREJECT prompts as well as HarmBench ones, which is outside the behaviour set it was validated on, and answers are cut to 512 tokens before judging. The StrongREJECT judge (needs a gated Gemma token) has not been run. Over-refusal numbers in section 5 still use refusal phrases.
- **The attackers stay within one family.** The standard recipe overstated ours; a second attacker that selects by judged answers, with and without the harmless-answer limit, raised ours to 40–48%. A direction learned by gradient descent, or another attack family, might do more and has not been tried.
- **One attack setting each** for LoRA, prefilling and pruning. For LoRA the result is already near 0%, so more settings would not change the conclusion.
- **Unrealistic threat model in one respect.** The undefended Qwen3.5-2B weights are public, so in practice an attacker would not need to attack our model at all. The experiment asks what a model release would need, not how to protect this particular model.

## Related work

- [DeepRefusal](https://arxiv.org/abs/2509.15202) (Xie et al., EMNLP Findings 2025) drops one refusal direction, fixed at the start of training, per site and per token with p = 0.5. It trains on curated refusals and compliant answers, with harmful-prefix augmentation, using LoRA.
- [ReFAT](https://arxiv.org/abs/2409.20089) (Yu et al., ICLR 2025) ablates per-layer refusal features, recomputed every few steps, on harmful examples during refusal training with LoRA.
- [ART](https://arxiv.org/html/2605.26526) (Kuo et al., 2026) adds an abliteration-based training objective. It has not been compared.

What differs in our method: self-distillation on the model's own answers in place of curated refusals; one direction recomputed with the attacker's own selection rule; and full fine-tuning. The fixed-versus-recomputed comparison, the matched controls, and the negative results for LoRA and prefilling are also ours.

## Next steps

1. Find out why ours fails at layer 10, position 0 with three or more directions removed. For example, train with more than one direction dropped at once, or with directions chosen by attacker 2's rule.
2. Try an attacker outside the difference-of-means family, such as a direction learned by gradient descent.
3. Run a third seed for every arm, seeds for ReFAT and hard-label, and a LoRA version of ours to separate the method from full versus LoRA fine-tuning.
4. Reduce the self-distillation IFEval loss, which accounts for most of ours' drop, for example with a different teacher-generation setting or a smaller learning rate.
5. Check whether a short LoRA attack acts like a steering vector in activation space. If it does, the same recompute-and-drop idea could be extended to LoRA.

## Appendix: per-seed refusal rates

Percent of 313 harmful test prompts refused, for every model. "In-dist." and "OOD" are the two attacker prompt sets; the number is how many directions were removed.

| Model | Seed | No attack | In-dist. 1 | In-dist. 10 | OOD 1 | Prefill 10 | Prefill 40 | LoRA 25 | LoRA 100 | Prune 1% | Prune 3% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original (base) | – | 87.9 | 0.3 | 0.0 | 1.0 | 27.2 | 10.2 | 0.3 | 0.6 | 0.6 | 0.6 |
| Noise control | 0 | 91.7 | 0.6 | 3.2 | 0.3 | 23.0 | 9.9 | 0.6 | 1.3 | 2.6 | 0.6 |
| Noise control | 1 | 90.1 | 1.3 | 1.3 | 0.6 | 18.8 | 6.4 | 1.9 | 1.0 | 2.9 | 0.3 |
| Noise control | 2 | 92.0 | 1.6 | 2.2 | 1.0 | 18.2 | 7.7 | 0.6 | 1.6 | 2.9 | 0.3 |
| Random-direction control | 0 | 92.0 | 1.0 | 0.3 | 0.6 | 20.4 | 6.4 | 0.6 | 1.3 | 0.6 | 1.3 |
| Random-direction control | 1 | 91.1 | 0.3 | 6.4 | 0.3 | 22.0 | 7.0 | 1.0 | 1.6 | 1.9 | 0.6 |
| Random-direction control | 2 | 90.7 | 1.0 | 0.6 | 1.0 | 24.6 | 8.3 | 0.3 | 0.6 | 1.9 | 2.9 |
| Dropout, fixed direction | 0 | 92.7 | 5.4 | 14.7 | 5.4 | 16.6 | 4.8 | 0.6 | 1.0 | 4.5 | 3.2 |
| Dropout, fixed direction | 1 | 91.4 | 6.1 | 20.4 | 8.6 | 15.7 | 4.5 | 1.3 | 1.3 | 3.5 | 1.0 |
| Dropout, fixed direction | 2 | 91.7 | 3.5 | 0.6 | 6.4 | 17.3 | 4.5 | 0.6 | 1.3 | 5.4 | 1.3 |
| Dropout p = 0.5 | 0 | 98.4 | 88.8 | 84.0 | 89.8 | 8.6 | 4.5 | 0.3 | 1.0 | 7.3 | 15.7 |
| Dropout p = 0.5 | 1 | 97.8 | 82.7 | 83.1 | 84.3 | 11.2 | 3.8 | 1.3 | 1.3 | 10.9 | 8.0 |
| Dropout p = 0.5 | 2 | 97.8 | 85.0 | 78.3 | 86.9 | 6.7 | 2.6 | 1.0 | 1.0 | 5.8 | 14.7 |
| Dropout p = 0.25 | 0 | 96.5 | 80.2 | 80.2 | 78.6 | 11.2 | 3.8 | 0.6 | 1.3 | 8.0 | 4.2 |
| Dropout p = 0.25 | 1 | 96.2 | 76.4 | 73.5 | 79.9 | 10.5 | 3.2 | 0.3 | 1.0 | 3.5 | 2.6 |
| Dropout p = 0.25 | 2 | 96.5 | 75.4 | 68.1 | 79.2 | 12.1 | 4.2 | 0.0 | 1.0 | 3.2 | 5.4 |
| Dropout, top 3 | 0 | 98.4 | 83.4 | 75.7 | 88.5 | 9.9 | 2.2 | 0.0 | 1.3 | 9.3 | 10.5 |
| Dropout, top 3 | 1 | 99.0 | 81.5 | 73.2 | 83.4 | 8.9 | 3.2 | 1.0 | 1.9 | 6.7 | 6.1 |
| Dropout, top 3 | 2 | 97.8 | 85.0 | 79.2 | 80.5 | 10.5 | 1.6 | 0.0 | 0.6 | 5.8 | 5.1 |

Code: [redcurryeater/robust-refusal](https://github.com/redcurryeater/robust-refusal) (private). Training and attacks ran on two NVIDIA L40S GPUs; each model takes about 20 minutes to train and 30 minutes to attack.