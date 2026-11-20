import contextlib
import math

import torch
import torch.nn.functional as F

from common import encode, generate, is_refusal

NPOS = 5


@contextlib.contextmanager
def evaluating(model):
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        model.train(was)


def _fwd(model, enc, **kw):
    try:
        return model(**enc, logits_to_keep=1, **kw)
    except TypeError:
        return model(**enc, **kw)


def last_logprobs(tok, model, prompts, bs=32):
    out = []
    with evaluating(model):
        for i in range(0, len(prompts), bs):
            out.append(F.log_softmax(_fwd(model, encode(tok, prompts[i:i + bs])).logits[:, -1].float(), -1))
    return torch.cat(out)


def refusal_score(lp, ref_idx):
    p = lp[:, ref_idx].exp().sum(-1).clamp(1e-8, 1 - 1e-8)
    return (p.log() - (1 - p).log()).mean().item()


def mean_acts(tok, model, prompts, bs=32):
    L = model.config.get_text_config().num_hidden_layers
    tot = None
    with evaluating(model):
        for i in range(0, len(prompts), bs):
            out = _fwd(model, encode(tok, prompts[i:i + bs]), output_hidden_states=True)
            s = torch.stack(out.hidden_states[:L], 0)[:, :, -NPOS:, :].float().sum(1)
            tot = s if tot is None else tot + s
    return tot / len(prompts)


def refusal_token_ids(tok, model, harmful_prompts):
    with evaluating(model):
        gens = generate(tok, model, harmful_prompts, max_new_tokens=32)
        first = []
        for i in range(0, len(harmful_prompts), 64):
            first += _fwd(model, encode(tok, harmful_prompts[i:i + 64])).logits[:, -1].argmax(-1).tolist()
    counts = {}
    for t, g in zip(first, gens):
        if is_refusal(g):
            counts[t] = counts.get(t, 0) + 1
    n = sum(counts.values())
    ids = [t for t, c in sorted(counts.items(), key=lambda x: -x[1]) if c >= max(2, 0.05 * n)]
    return ids or [tok.convert_tokens_to_ids("I")], {tok.decode([t]): counts.get(t, 0) for t in ids}


def select(tok, model, pert, diff, harm_val, harmless_val, ref_idx, max_layer_frac=0.8, kl_thr=0.1):
    with pert.override("none"):
        base_hl = last_logprobs(tok, model, harmless_val)
    cands = []
    for l in range(int(max_layer_frac * diff.shape[0])):
        for p in range(NPOS):
            norm = diff[l, p].norm()
            if not torch.isfinite(norm) or norm == 0:
                continue
            with pert.override("project", diff[l, p] / norm):
                lp_h = last_logprobs(tok, model, harm_val)
                lp_hl = last_logprobs(tok, model, harmless_val)
            kl = (base_hl.exp() * (base_hl - lp_hl)).sum(-1).mean().item()
            score = refusal_score(lp_h, ref_idx)
            if math.isfinite(score) and math.isfinite(kl):
                cands.append({"layer": l, "pos": p, "harm_score": score, "harmless_kl": kl, "norm": norm.item()})
    valid = [c for c in cands if c["harmless_kl"] < kl_thr] or cands
    return min(valid, key=lambda c: c["harm_score"]), cands


def iterative_directions(tok, model, pert, harm, harmless, layer, pos, k):
    dirs, norms = [], []
    for _ in range(k):
        with pert.override("project", torch.stack(dirs)) if dirs else pert.override("none"):
            d = (mean_acts(tok, model, harm) - mean_acts(tok, model, harmless))[layer, pos]
        for u in dirs:
            d = d - (d @ u) * u
        norms.append(d.norm().item())
        dirs.append(d / d.norm())
    return torch.stack(dirs), norms


def matched_direction(tok, model, pert, prompts, layer, d_ref, bs=32, max_cos=0.1, top=256):
    xs = []
    with pert.override("none"), evaluating(model):
        for i in range(0, len(prompts), bs):
            h = _fwd(model, encode(tok, prompts[i:i + bs]), output_hidden_states=True).hidden_states[layer]
            xs.append(h[:, -NPOS:, :].float().reshape(-1, h.shape[-1]))
    X = torch.cat(xs)
    M = X.T @ X / X.shape[0]
    vals, vecs = torch.linalg.eigh(M)
    d = d_ref.float().to(M.device)
    target = (d @ M @ d).item()
    best = None
    for j in range(1, min(top, len(vals)) + 1):
        u = vecs[:, -j]
        cos = abs((u @ d).item())
        if cos > max_cos:
            continue
        gap = abs(math.log(vals[-j].item() / target))
        if best is None or gap < best[0]:
            best = (gap, u, {"rank": j, "var": vals[-j].item(), "target_var": target, "cos_with_ref": cos})
    if best is None:
        raise RuntimeError("no eigenvector satisfied the orthogonality constraint")
    return best[1], best[2]
