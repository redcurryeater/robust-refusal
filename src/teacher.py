import argparse
import json
import os

SPLITS = ["distill_harmful", "distill_harmless", "val_harmful", "val_harmless"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    items = [(s, r["prompt"]) for s in SPLITS for r in json.load(open(os.path.join(args.splits, f"{s}.json")))]
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False) for _, p in items]
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=args.gpu_mem,
              seed=0, limit_mm_per_prompt={"image": 0, "video": 0})
    outs = llm.generate(texts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for (s, p), o in zip(items, outs):
            c = o.outputs[0]
            f.write(json.dumps({"split": s, "prompt": p, "response": c.text, "finish_reason": c.finish_reason},
                               ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
