import argparse
import json
import os
import random

import torch

from common import fmt, is_refusal, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--steps", default="25,100")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from peft import LoraConfig, PeftModel, get_peft_model

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    rows = [r for r in rows if r["config"] == "id_k1" and not is_refusal(r["response"])]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    torch.manual_seed(args.seed)

    tok, model = load_model(args.model, dtype=torch.bfloat16)
    items = []
    for r in rows[:args.n]:
        p = tok(fmt(tok, r["prompt"]), add_special_tokens=False).input_ids
        a = tok(r["response"], add_special_tokens=False).input_ids
        items.append(((p + a)[:768], len(p)))
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    budgets = sorted(map(int, args.steps.split(",")))
    os.makedirs(args.out, exist_ok=True)
    log = []
    for step in range(1, budgets[-1] + 1):
        batch = [items[rng.randrange(len(items))] for _ in range(args.bs)]
        T = max(len(x) for x, _ in batch)
        ids = torch.full((args.bs, T), tok.pad_token_id)
        labels = torch.full((args.bs, T), -100)
        att = torch.zeros((args.bs, T), dtype=torch.long)
        for i, (x, n_p) in enumerate(batch):
            ids[i, :len(x)] = torch.tensor(x)
            labels[i, n_p:len(x)] = torch.tensor(x[n_p:])
            att[i, :len(x)] = 1
        loss = model(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=labels.cuda()).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        log.append({"step": step, "loss": loss.item()})
        if step in budgets:
            model.save_pretrained(os.path.join(args.out, f"adapter_s{step}"))
    json.dump({"n_examples": len(items), "log": log, "args": vars(args)},
              open(os.path.join(args.out, "lora_log.json"), "w"))

    for s in budgets:
        _, base = load_model(args.model, dtype=torch.bfloat16)
        merged = PeftModel.from_pretrained(base, os.path.join(args.out, f"adapter_s{s}")).merge_and_unload()
        merged.save_pretrained(os.path.join(args.out, f"merged_s{s}"), safe_serialization=True)
        tok.save_pretrained(os.path.join(args.out, f"merged_s{s}"))
        del base, merged
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
