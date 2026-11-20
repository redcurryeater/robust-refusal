import argparse
import os

import torch
import torch.nn as nn

from common import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.path.exists(os.path.join(args.out, "config.json")):
        return
    tok, model = load_model(args.model, dtype=torch.bfloat16)
    emb, head = model.get_input_embeddings(), model.get_output_embeddings()
    if head.weight.data_ptr() == emb.weight.data_ptr():
        head.weight = nn.Parameter(emb.weight.detach().clone())
    for cfg in {id(c): c for c in (model.config, model.config.get_text_config())}.values():
        cfg.tie_word_embeddings = False
    model.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)


if __name__ == "__main__":
    main()
