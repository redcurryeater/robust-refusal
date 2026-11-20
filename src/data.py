import argparse
import csv
import json
import os
import random
import re
import urllib.request

ARDITI = "https://raw.githubusercontent.com/andyrdt/refusal_direction/main/dataset/splits/"
SOURCES = {
    "advbench.csv": "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
    "harmbench.csv": "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv",
    "maliciousinstruct.txt": "https://raw.githubusercontent.com/Princeton-SysML/Jailbreak_LLM/main/data/MaliciousInstruct.txt",
    "strongreject.csv": "https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_dataset.csv",
    "xstest.csv": "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv",
    **{f"arditi_{s}.json": f"{ARDITI}{s}.json" for s in
       ["harmful_train", "harmful_val", "harmless_train", "harmless_val", "harmless_test"]},
}
N_DISTILL_HARMLESS = 4000


def fetch(raw):
    os.makedirs(raw, exist_ok=True)
    for name, url in SOURCES.items():
        path = os.path.join(raw, name)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            urllib.request.urlretrieve(url, path)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def words(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def dedupe(items, against=(), thr=0.6):
    aw = [words(p) for p, _ in against]
    kept, kw = [], []
    for p, src in items:
        w = words(p)
        if w and not any(len(w & y) / len(w | y) >= thr for y in kw + aw):
            kept.append((p, src))
            kw.append(w)
    return kept


def rows(items):
    return [{"prompt": p, "source": s} for p, s in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    fetch(args.raw)
    rng = random.Random(args.seed)

    def raw(name):
        return os.path.join(args.raw, name)

    advbench = [(r["goal"].strip(), "advbench") for r in read_csv(raw("advbench.csv"))]
    hb = [(r["Behavior"].strip(), "harmbench") for r in read_csv(raw("harmbench.csv"))
          if r["FunctionalCategory"] == "standard"]
    mi = [(x.strip(), "maliciousinstruct") for x in open(raw("maliciousinstruct.txt"), encoding="utf-8") if x.strip()]
    sr = [(r["forbidden_prompt"].strip(), "strongreject:" + r["category"]) for r in read_csv(raw("strongreject.csv"))]
    arditi_h = [(r["instruction"].strip(), "arditi_train") for s in ["harmful_train", "harmful_val"]
                for r in read_json(raw(f"arditi_{s}.json"))]
    xs = [{"prompt": r["prompt"], "type": r["type"], "label": r["label"]} for r in read_csv(raw("xstest.csv"))]

    rng.shuffle(hb)
    rng.shuffle(sr)
    hb_test, hb_train = hb[:100], hb[100:]
    sr_attack, sr_test = sr[:100], sr[100:]
    test_harmful = sr_test + hb_test
    held_out = test_harmful + sr_attack + [(x["prompt"], "xstest") for x in xs]

    pool = advbench + mi + hb_train + arditi_h
    rng.shuffle(pool)
    pool = dedupe(pool, against=held_out)
    attack_id_harm, val_harm, distill_harm = pool[:128], pool[128:192], pool[192:]

    hl = {s: [(r["instruction"].strip(), "alpaca") for r in read_json(raw(f"arditi_harmless_{s}.json"))]
          for s in ("train", "val", "test")}
    for s in ("train", "val", "test"):
        rng.shuffle(hl[s])
    distill_harmless = hl["train"][:N_DISTILL_HARMLESS]

    splits = {
        "distill_harmful": rows(distill_harm),
        "distill_harmless": rows(distill_harmless),
        "dir_harmful": rows(distill_harm[:128]),
        "dir_harmless": rows(distill_harmless[:128]),
        "val_harmful": rows(val_harm),
        "val_harmless": rows(hl["val"][:256]),
        "attack_id_harmful": rows(attack_id_harm),
        "attack_id_harmless": rows(hl["val"][256:384]),
        "attack_ood_harmful": rows(sr_attack),
        "attack_ood_harmless": rows(hl["val"][384:484]),
        "test_harmful": rows(test_harmful),
        "test_harmless": rows(hl["test"][:200]),
        "xstest": xs,
    }
    os.makedirs(args.out, exist_ok=True)
    for name, data in splits.items():
        with open(os.path.join(args.out, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
    sources = {}
    for _, s in distill_harm:
        sources[s] = sources.get(s, 0) + 1
    manifest = {"counts": {k: len(v) for k, v in splits.items()}, "distill_harmful_sources": sources,
                "seed": args.seed}
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
