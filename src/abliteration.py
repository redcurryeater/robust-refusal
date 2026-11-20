import argparse
import json
import os
import random

import torch
import torch.nn.functional as F

import directions as D
from common import Perturber, encode, load_model, load_split


def load_mmlu(raw, n, seed=0):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    df = pd.read_parquet(hf_hub_download("cais/mmlu", "all/test-00000-of-00001.parquet", repo_type="dataset",
                                        cache_dir=raw))
    out = []
    for i in random.Random(seed).sample(range(len(df)), n):
        r = df.iloc[i]
        q = (f"{r['question']}\n\n" + "\n".join(f"{l}. {c}" for l, c in zip("ABCD", r["choices"]))
             + "\n\nAnswer with only the letter of the correct option.")
        out.append((q, int(r["answer"])))
    return out


def load_wikitext(raw, tok, n_seqs, seqlen=512):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                        repo_type="dataset", cache_dir=raw)
    ids = tok("".join(pd.read_parquet(p)["text"].tolist()), add_special_tokens=False).input_ids
    return torch.tensor(ids[:n_seqs * seqlen]).view(-1, seqlen)


@torch.no_grad()
def mmlu_acc(tok, model, items, bs=32):
    letters = [tok.encode(l, add_special_tokens=False)[0] for l in "ABCD"]
    correct = 0
    for i in range(0, len(items), bs):
        batch = items[i:i + bs]
        logits = model(**encode(tok, [q for q, _ in batch]), logits_to_keep=1).logits[:, -1, letters]
        correct += (logits.argmax(-1).cpu() == torch.tensor([a for _, a in batch])).sum().item()
    return correct / len(items)


@torch.no_grad()
def perplexity(model, chunks, bs=8):
    tot, n = 0.0, 0
    for i in range(0, len(chunks), bs):
        x = chunks[i:i + bs].cuda()
        logits = model(input_ids=x).logits[:, :-1].float()
        tot += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="sum").item()
        n += x[:, 1:].numel()
    return float(torch.exp(torch.tensor(tot / n)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--sets", default="id,ood")
    ap.add_argument("--mmlu_n", type=int, default=1000)
    ap.add_argument("--ppl_seqs", type=int, default=64)
    ap.add_argument("--utility_only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tok, model = load_model(args.model)
    pert = Perturber(model)
    mmlu = load_mmlu(args.raw, args.mmlu_n)
    chunks = load_wikitext(args.raw, tok, args.ppl_seqs)

    def utility():
        model.eval()
        return {"mmlu": mmlu_acc(tok, model, mmlu), "ppl": perplexity(model, chunks)}

    if args.utility_only:
        json.dump({"model": args.model, **utility()}, open(os.path.join(args.out, "utility.json"), "w"), indent=1)
        return

    ref_list, ref_tokens = D.refusal_token_ids(tok, model, load_split(args.splits, "attack_id_harmful"))
    ref_idx = torch.tensor(ref_list, device="cuda")
    res = {"model": args.model, "refusal_tokens": ref_tokens, "k0": utility()}
    saved = {}
    for s in args.sets.split(","):
        H, N = load_split(args.splits, f"attack_{s}_harmful"), load_split(args.splits, f"attack_{s}_harmless")
        dh, sh, dn, sn = H[:-32], H[-32:], N[:-32], N[-32:]
        with pert.override("none"):
            diff = D.mean_acts(tok, model, dh) - D.mean_acts(tok, model, dn)
            res[f"{s}_refusal_score_k0"] = D.refusal_score(D.last_logprobs(tok, model, sh), ref_idx)
        best, cands = D.select(tok, model, pert, diff, sh, sn, ref_idx)
        dirs, norms = D.iterative_directions(tok, model, pert, dh, dn, best["layer"], best["pos"], args.K)
        per_k = []
        for k in range(1, args.K + 1):
            with pert.override("project", dirs[:k]):
                per_k.append({"k": k, **utility(),
                              "refusal_score": D.refusal_score(D.last_logprobs(tok, model, sh), ref_idx)})
            print(s, per_k[-1], flush=True)
        res[s] = {"selected": best, "diff_norms": norms, "per_k": per_k,
                  "top_candidates": sorted(cands, key=lambda c: c["harm_score"])[:5]}
        saved[s] = {"dirs": dirs.cpu(), "layer": best["layer"], "pos": best["pos"]}
        json.dump(res, open(os.path.join(args.out, "attack_dirs.json"), "w"), indent=1)
    torch.save(saved, os.path.join(args.out, "attack_dirs.pt"))


if __name__ == "__main__":
    main()
