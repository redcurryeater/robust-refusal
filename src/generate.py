import argparse
import json
import os

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

import torch

from common import fmt, load_split

TARGETS = ("o_proj.weight", "out_proj.weight", "down_proj.weight", "embed_tokens.weight")


def make_editor(D):
    def edit(model):
        import torch as T
        if not hasattr(model, "_dd_orig"):
            model._dd_orig = {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters()
                              if n.endswith(TARGETS)}
        params = dict(model.named_parameters())
        hidden = params[next(n for n in model._dd_orig if n.endswith("embed_tokens.weight"))].shape[1]
        for name, w0 in model._dd_orig.items():
            p = params[name]
            if D is None:
                p.data.copy_(w0.to(p.device))
                continue
            Dg = D.to(p.device, T.float32)
            if name.endswith("embed_tokens.weight"):
                for i in range(0, w0.shape[0], 32768):
                    W = w0[i:i + 32768].to(p.device, T.float32)
                    p.data[i:i + 32768].copy_((W - (W @ Dg.T) @ Dg).to(p.dtype))
            else:
                assert p.shape[0] == hidden, name
                W = w0.to(p.device, T.float32)
                p.data.copy_((W - Dg.T @ (Dg @ W)).to(p.dtype))
        return len(model._dd_orig)
    return edit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--split", default="test_harmful")
    ap.add_argument("--name", default="base")
    ap.add_argument("--skip_base", action="store_true")
    ap.add_argument("--extras", action="store_true")
    ap.add_argument("--dirs")
    ap.add_argument("--sets", default="id,ood")
    ap.add_argument("--ks", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--prefill")
    ap.add_argument("--prefill_tokens", default="10,40")
    ap.add_argument("--max_tokens", type=int, default=384)
    ap.add_argument("--gpu_mem", type=float, default=0.7)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=args.gpu_mem,
              enable_prefix_caching=False, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    harm = load_split(args.splits, args.split)

    configs = [] if args.skip_base else [(args.name, None)]
    if args.dirs:
        saved = torch.load(args.dirs)
        configs += [(f"{s}_k{k}", saved[s]["dirs"][:k].float()) for s in args.sets.split(",")
                    for k in map(int, args.ks.split(","))]

    prefixes = {}
    if args.prefill:
        for line in open(args.prefill, encoding="utf-8"):
            r = json.loads(line)
            if r["config"] == "id_k1" and r["split"] == "test_harmful":
                prefixes[r["prompt"]] = r["response"]

    out = open(args.out, "w", encoding="utf-8")

    def run(config, split, prompts, prefill=0, meta=None):
        pre = [tok.decode(tok(prefixes.get(p, ""), add_special_tokens=False).input_ids[:prefill]) if prefill else ""
               for p in prompts]
        outs = llm.generate([fmt(tok, p) + x for p, x in zip(prompts, pre)], sp)
        for i, (p, o) in enumerate(zip(prompts, outs)):
            row = {"config": config, "split": split, "prompt": p, "prefill": pre[i],
                   "response": pre[i] + o.outputs[0].text, **(meta[i] if meta else {})}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()

    for name, D in configs:
        llm.apply_model(make_editor(D))
        if args.prefill:
            for t in map(int, args.prefill_tokens.split(",")):
                run(f"prefill{t}", args.split, harm, prefill=t)
            continue
        run(name, args.split, harm)
        if D is None and args.extras:
            run(name, "test_harmless", load_split(args.splits, "test_harmless"))
            xs = json.load(open(os.path.join(args.splits, "xstest.json")))
            run(name, "xstest", [x["prompt"] for x in xs],
                meta=[{"xs_type": x["type"], "xs_label": x["label"]} for x in xs])


if __name__ == "__main__":
    main()
