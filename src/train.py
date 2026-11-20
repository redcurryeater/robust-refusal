import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import directions as D
from common import Perturber, fmt, load_model, load_split

CONDITIONS = ("noise", "randdir", "dropout", "dropout_fixed", "dropout_top3")
WARMUP, MAX_LEN, KL_CHUNK = 20, 512, 512


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--teacher_data", required=True)
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--harmful_frac", type=float, default=0.25)
    ap.add_argument("--recompute_every", type=int, default=50)
    ap.add_argument("--reselect_every", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def tokenize(tok, rows, eos_id):
    items = []
    for r in rows:
        p = tok(fmt(tok, r["prompt"]), add_special_tokens=False).input_ids
        a = tok(r["response"], add_special_tokens=False).input_ids
        if r.get("finish_reason") == "stop":
            a = a + [eos_id]
        ids = (p + a)[:MAX_LEN]
        if len(ids) > len(p):
            items.append((ids, len(p)))
    return items


def collate(items, pad_id):
    T = max(len(ids) for ids, _ in items)
    ids = torch.full((len(items), T), pad_id, dtype=torch.long)
    att = torch.zeros((len(items), T), dtype=torch.long)
    lm = torch.zeros((len(items), T), dtype=torch.bool)
    for i, (x, n_p) in enumerate(items):
        ids[i, :len(x)] = torch.tensor(x)
        att[i, :len(x)] = 1
        lm[i, n_p - 1:len(x) - 1] = True
    return ids.cuda(), att.cuda(), lm.cuda()


def cycle(items, rng):
    while True:
        order = list(range(len(items)))
        rng.shuffle(order)
        for i in order:
            yield items[i]


def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "config.json"), "w"), indent=1)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    logf = open(os.path.join(args.out, "log.jsonl"), "a")
    t0 = time.time()

    def log(rec):
        rec["time"] = round(time.time() - t0, 1)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(json.dumps(rec), flush=True)

    tok, teacher = load_model(args.model, dtype=torch.bfloat16)
    teacher.requires_grad_(False)
    _, student = load_model(args.model, dtype=torch.float32)
    student.get_input_embeddings().weight.requires_grad_(False)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.train()
    pert = Perturber(student)
    t_head, s_head = teacher.get_output_embeddings(), student.get_output_embeddings()

    eos_id = tok.convert_tokens_to_ids("<|im_end|>")
    if eos_id is None or eos_id == tok.unk_token_id:
        eos_id = tok.eos_token_id
    rows = [json.loads(line) for line in open(args.teacher_data, encoding="utf-8")]
    by = lambda s: [r for r in rows if r["split"] == s]
    train_h = tokenize(tok, by("distill_harmful"), eos_id)
    train_n = tokenize(tok, by("distill_harmless"), eos_id)
    val_items = tokenize(tok, by("val_harmful"), eos_id)[:64] + tokenize(tok, by("val_harmless"), eos_id)[:64]
    pad_id = tok.pad_token_id

    dir_h, dir_n = load_split(args.splits, "dir_harmful"), load_split(args.splits, "dir_harmless")
    sel_h, sel_n = load_split(args.splits, "val_harmful", 64), load_split(args.splits, "val_harmless", 64)

    n_h = max(1, round(args.bs * args.harmful_frac))
    total = args.max_steps or int(args.epochs * math.ceil(len(train_n) / (args.bs - n_h)))
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP) * (
        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total))))))
    log({"event": "start", "n_train_harmful": len(train_h), "n_train_harmless": len(train_n),
         "n_val": len(val_items), "steps": total, "trainable_params": sum(p.numel() for p in params)})

    ref_list, ref_tokens = D.refusal_token_ids(tok, teacher, dir_h)
    ref_idx = torch.tensor(ref_list, device="cuda")
    k = 3 if args.condition == "dropout_top3" else 1
    state = {"layer": None, "pos": None, "ref": None, "ctrl": None, "ref0": None}

    def recompute(reselect):
        info = {}
        if reselect or state["layer"] is None:
            with pert.override("none"):
                diff = D.mean_acts(tok, student, dir_h) - D.mean_acts(tok, student, dir_n)
            best, _ = D.select(tok, student, pert, diff, sel_h, sel_n, ref_idx)
            state["layer"], state["pos"] = best["layer"], best["pos"]
            info["selected"] = best
        dirs, norms = D.iterative_directions(tok, student, pert, dir_h, dir_n, state["layer"], state["pos"], k)
        if state["ref0"] is None:
            state["ref0"] = dirs[0].clone()
        info["cos_with_initial"] = abs((dirs[0] @ state["ref0"]).item())
        if state["ref"] is not None:
            info["cos_with_previous"] = abs((dirs[0] @ state["ref"][0]).item())
        info["diff_norms"] = norms
        state["ref"] = dirs
        if args.condition == "randdir":
            state["ctrl"], info["control"] = D.matched_direction(tok, student, pert, dir_h + dir_n,
                                                                  state["layer"], dirs[0])
        if args.condition == "noise":
            pert.set("noise", state["ref"])
        elif args.condition == "randdir":
            pert.set("project", state["ctrl"])
        else:
            pert.set("project", state["ref"])
        return info

    log({"event": "directions", "step": 0, "refusal_tokens": ref_tokens, **recompute(reselect=True)})
    torch.save({"ref": state["ref"].cpu(), "layer": state["layer"], "pos": state["pos"]},
               os.path.join(args.out, "directions_initial.pt"))

    def kl_chunk(hs, ht):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ls = s_head(hs).float()
            with torch.no_grad():
                lt = t_head(ht.to(torch.bfloat16)).float()
        lpt, lps = F.log_softmax(lt, -1), F.log_softmax(ls, -1)
        return (lpt.exp() * (lpt - lps)).sum(-1)

    def per_position_kl(ids, att, lm):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = student(input_ids=ids, attention_mask=att, output_hidden_states=True,
                         logits_to_keep=1).hidden_states[-1]
        with torch.no_grad():
            ht = teacher(input_ids=ids, attention_mask=att, output_hidden_states=True,
                         logits_to_keep=1).hidden_states[-1]
        hs, ht = hs[lm], ht[lm]
        return torch.cat([checkpoint(kl_chunk, hs[i:i + KL_CHUNK], ht[i:i + KL_CHUNK], use_reentrant=False)
                          for i in range(0, hs.shape[0], KL_CHUNK)])

    def evaluate(step):
        rec = {"event": "eval", "step": step}
        student.eval()
        with torch.no_grad():
            for name, mask_val in (("val_kl_clean", 0.0), ("val_kl_perturbed", 1.0)):
                tot, n = 0.0, 0
                for i in range(0, len(val_items), args.bs):
                    batch = val_items[i:i + args.bs]
                    pert.mask = torch.full((len(batch),), mask_val)
                    v = per_position_kl(*collate(batch, pad_id))
                    tot, n = tot + v.sum().item(), n + v.numel()
                rec[name] = tot / n
            pert.mask = None
        student.train()
        with pert.override("none"):
            rec["refusal_score_clean"] = D.refusal_score(D.last_logprobs(tok, student, sel_h), ref_idx)
        with pert.override("project", state["ref"]):
            rec["refusal_score_ablated"] = D.refusal_score(D.last_logprobs(tok, student, sel_h), ref_idx)
        with pert.override("project", state["ref0"]):
            rec["refusal_score_ablated_initial_dir"] = D.refusal_score(D.last_logprobs(tok, student, sel_h), ref_idx)
        log(rec)

    evaluate(0)

    it_h, it_n = cycle(train_h, rng), cycle(train_n, rng)
    g = torch.Generator().manual_seed(args.seed)
    pert.track = True
    acc = dict.fromkeys(("loss", "kl_pert", "kl_clean", "n_pert", "n_clean", "steps"), 0.0)
    for step in range(1, total + 1):
        batch = [next(it_h) for _ in range(n_h)] + [next(it_n) for _ in range(args.bs - n_h)]
        ids, att, lm = collate(batch, pad_id)
        mask = (torch.rand(args.bs, generator=g) < args.p).float()
        pert.mask = mask
        kl = per_position_kl(ids, att, lm)
        loss = kl.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

        pm = mask.to(lm.device)[lm.nonzero()[:, 0]].bool()
        acc["loss"] += loss.item()
        acc["kl_pert"] += kl[pm].sum().item()
        acc["kl_clean"] += kl[~pm].sum().item()
        acc["n_pert"] += int(pm.sum())
        acc["n_clean"] += int((~pm).sum())
        acc["steps"] += 1

        if step % 10 == 0 or step == total:
            s = pert.stats
            log({"event": "train", "step": step, "lr": sched.get_last_lr()[0], "loss": acc["loss"] / acc["steps"],
                 "kl_perturbed_seqs": acc["kl_pert"] / max(1, acc["n_pert"]),
                 "kl_clean_seqs": acc["kl_clean"] / max(1, acc["n_clean"]),
                 "perturb_rel_norm": math.sqrt(s["delta_sq"] / max(1e-9, s["h_sq"])),
                 "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)})
            acc = dict.fromkeys(acc, 0.0)
            pert.reset_stats()
        if args.condition != "dropout_fixed" and step % args.recompute_every == 0 and step < total:
            pert.track = False
            reselect = args.reselect_every and step % args.reselect_every == 0
            log({"event": "directions", "step": step, **recompute(reselect)})
            pert.track = True
        if step % args.eval_every == 0 or step == total:
            pert.track = False
            evaluate(step)
            pert.track = True

    pert.remove()
    torch.save({"ref": state["ref"].cpu(), "layer": state["layer"], "pos": state["pos"]},
               os.path.join(args.out, "directions_final.pt"))
    student.to(torch.bfloat16).save_pretrained(os.path.join(args.out, "model"), safe_serialization=True)
    tok.save_pretrained(os.path.join(args.out, "model"))
    log({"event": "done", "step": total})


if __name__ == "__main__":
    main()
