import argparse
import json
import os
import random

import torch
import torch.nn as nn

from common import fmt, get_layers, is_refusal, load_model


def batches(tok, rows, bs, max_len=512):
    for i in range(0, len(rows), bs):
        items = []
        for r in rows[i:i + bs]:
            p = tok(fmt(tok, r["prompt"]), add_special_tokens=False).input_ids
            a = tok(r["response"], add_special_tokens=False).input_ids
            items.append(((p + a)[:max_len], len(p)))
        T = max(len(x) for x, _ in items)
        ids = torch.full((len(items), T), tok.pad_token_id)
        lab = torch.full((len(items), T), -100)
        att = torch.zeros((len(items), T), dtype=torch.long)
        for j, (x, n_p) in enumerate(items):
            ids[j, :len(x)] = torch.tensor(x)
            lab[j, n_p:len(x)] = torch.tensor(x[n_p:])
            att[j, :len(x)] = 1
        yield ids.cuda(), att.cuda(), lab.cuda()


def importance(model, tok, rows, linears, bs):
    scores = {n: torch.zeros_like(m.weight, dtype=torch.float32) for n, m in linears.items()}
    for ids, att, lab in batches(tok, rows, bs):
        model(input_ids=ids, attention_mask=att, labels=lab).loss.backward()
        for n, m in linears.items():
            scores[n] += (m.weight.detach().float() * m.weight.grad.float()).abs()
            m.weight.grad = None
    return scores


def topk_mask(s, frac):
    k = max(1, int(frac * s.numel()))
    return s >= torch.topk(s.flatten(), k, sorted=False).values.min()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--teacher_data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--q", type=float, default=0.03)
    ap.add_argument("--p", type=float, default=0.03)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--bs", type=int, default=4)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.teacher_data, encoding="utf-8")]
    rng = random.Random(0)
    safety = [r for r in rows if r["split"] == "distill_harmful" and is_refusal(r["response"])]
    utility = [r for r in rows if r["split"] == "distill_harmless"]
    rng.shuffle(safety)
    rng.shuffle(utility)
    safety, utility = safety[:args.n], utility[:args.n]

    tok, model = load_model(args.model, dtype=torch.bfloat16)
    tok.padding_side = "right"
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    model.requires_grad_(False)
    linears = {}
    for i, layer in enumerate(get_layers(model)):
        for n, m in layer.named_modules():
            if isinstance(m, nn.Linear):
                m.weight.requires_grad_(True)
                linears[f"layers.{i}.{n}"] = m
    model.enable_input_require_grads()
    s_safe = importance(model, tok, safety, linears, args.bs)
    s_util = importance(model, tok, utility, linears, args.bs)
    pruned, total = 0, 0
    with torch.no_grad():
        for n, m in linears.items():
            mask = topk_mask(s_safe[n], args.q) & ~topk_mask(s_util[n], args.p)
            m.weight[mask] = 0
            pruned, total = pruned + int(mask.sum()), total + mask.numel()
    model.gradient_checkpointing_disable()
    model.requires_grad_(False)
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    json.dump({"q": args.q, "p": args.p, "pruned_fraction_of_decoder_linears": pruned / total, "n_safety": len(safety),
               "n_utility": len(utility)}, open(os.path.join(args.out, "prune_info.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
