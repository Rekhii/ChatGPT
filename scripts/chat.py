# scripts/chat.py
"""Interactive prompt for a trained nanomath checkpoint.

The model was trained on `question<sep>answer<eos>` and nothing else, so the
question has to be phrased the way the dataset phrases it. Type `help` for
sample questions drawn from the held-out eval set.
"""
import argparse
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from mathlm.config import Config
from mathlm.model import GPT
from mathlm.tokenizer import CharTokenizer


def load(ckpt_path, cfg):
    model = GPT(cfg).to(cfg.device)
    ck = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck.get("step", -1)


@torch.no_grad()
def answer(model, tok, cfg, question, max_new_tokens=48, show_raw=False):
    ids = tok.encode(question) + [tok.SEP]
    if len(ids) >= cfg.block_size:
        return "[question longer than block_size, truncated context]"

    idx = torch.tensor([ids], dtype=torch.long, device=cfg.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(idx, max_new_tokens=max_new_tokens,
                             eos_id=tok.EOS, greedy=True)

    gen = out[0, len(ids):].tolist()
    hit_eos = tok.EOS in gen
    if hit_eos:
        gen = gen[: gen.index(tok.EOS)]

    # join directly rather than tok.decode, which silently drops ids <= EOS
    text = "".join(tok.itos[i] for i in gen)
    if show_raw:
        print(f"    [raw ids: {gen[:20]}{'...' if len(gen) > 20 else ''}]")
    if not hit_eos:
        text += "  [no <eos>, hit token limit]"
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/run_001/ckpt/final.pt")
    ap.add_argument("--eval_json", default="data/shards/eval.json")
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--raw", action="store_true", help="print raw token ids")
    args = ap.parse_args()

    cfg = Config()
    cfg.compile = False          # one question at a time, compile is a net loss
    tok = CharTokenizer()

    model, step = load(args.ckpt, cfg)
    print(f"loaded {args.ckpt} (step {step}), "
          f"{model.num_params()/1e6:.2f}M params on {cfg.device}")

    eval_data = {}
    p = pathlib.Path(args.eval_json)
    if p.exists():
        eval_data = json.load(open(p))

    print("\ntype a question, or:")
    print("  help          sample questions from each module")
    print("  test <module> run 5 held-out questions with correct answers")
    print("  modules       list module names")
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

        if q == "modules":
            for m in sorted(eval_data):
                print(f"  {m}")
            continue

        if q == "help":
            for m in sorted(eval_data):
                sample = random.choice(eval_data[m])
                print(f"  [{m}]\n    {sample['question']}")
            continue

        if q.startswith("test "):
            module = q[5:].strip()
            if module not in eval_data:
                print(f"  no such module. try 'modules'.")
                continue
            hits = 0
            for it in random.sample(eval_data[module], 5):
                pred = answer(model, tok, cfg, it["question"],
                              args.max_new_tokens, args.raw)
                ok = pred.strip() == it["answer"].strip()
                hits += ok
                print(f"  {'OK ' if ok else 'X  '} {it['question'][:64]}")
                print(f"       want {it['answer']!r}  got {pred!r}")
            print(f"  {hits}/5 exact\n")
            continue

        try:
            print(f"  {answer(model, tok, cfg, q, args.max_new_tokens, args.raw)}\n")
        except ValueError as e:
            print(f"  {e}  (vocab is digits, letters, punctuation, space)\n")


if __name__ == "__main__":
    main()