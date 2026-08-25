"""Interactive prompt for a trained nanomath checkpoint.

The model was trained on `question<sep>answer<eos>` and nothing else, so the
question has to be phrased the way the dataset phrases it. Type `help` for
sample questions drawn from the held-out eval set.
"""
import argparse
import json
import random
from pathlib import Path

import torch

from mathlm.config import Config
from mathlm.model import GPT
from mathlm.tokenizer import CharTokenizer


@torch.no_grad()
def answer(model, tok, cfg, question, max_new_tokens=48):
    ids = tok.encode(question) + [tok.SEP]
    if len(ids) >= cfg.block_size:
        return "[question longer than block_size]"

    idx = torch.tensor([ids], dtype=torch.long, device=cfg.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(idx, max_new_tokens=max_new_tokens,
                             eos_id=tok.EOS, greedy=True)

    gen = out[0, len(ids):].tolist()
    hit_eos = tok.EOS in gen
    if hit_eos:
        gen = gen[:gen.index(tok.EOS)]

    # join directly rather than tok.decode, which silently drops ids <= EOS
    text = "".join(tok.itos[i] for i in gen)
    return text if hit_eos else text + "  [no <eos>, hit token limit]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/run_001/ckpt/final.pt")
    ap.add_argument("--eval_json", default="data/shards/eval.json")
    ap.add_argument("--max_new_tokens", type=int, default=48)
    args = ap.parse_args()

    cfg = Config()
    cfg.compile = False          # one question at a time, compile is a net loss
    tok = CharTokenizer()

    model = GPT(cfg).to(cfg.device)
    ck = torch.load(args.ckpt, map_location=cfg.device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {args.ckpt} (step {ck.get('step', -1)}), "
          f"{model.num_params()/1e6:.2f}M params on {cfg.device}")

    eval_path = Path(args.eval_json)
    eval_data = json.load(open(eval_path)) if eval_path.exists() else {}

    print("\ntype a question, or:")
    print("  help            sample question from each module")
    print("  test <module>   5 held-out questions with correct answers")
    print("  quit\n")

    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not q:
            continue
        if q in ("quit", "exit", "q"):
            break

        if q == "help":
            for m in sorted(eval_data):
                print(f"  [{m}]\n    {random.choice(eval_data[m])['question']}")
            continue

        if q.startswith("test "):
            module = q[5:].strip()
            if module not in eval_data:
                print("  no such module. try 'help'.")
                continue
            items = random.sample(eval_data[module], min(5, len(eval_data[module])))
            hits = 0
            for it in items:
                pred = answer(model, tok, cfg, it["question"], args.max_new_tokens)
                ok = pred.strip() == it["answer"].strip()
                hits += ok
                print(f"  {'OK ' if ok else 'X  '} {it['question'][:64]}")
                print(f"       want {it['answer']!r}  got {pred!r}")
            print(f"  {hits}/{len(items)} exact\n")
            continue

        print(f"  {answer(model, tok, cfg, q, args.max_new_tokens)}\n")


if __name__ == "__main__":
    main()