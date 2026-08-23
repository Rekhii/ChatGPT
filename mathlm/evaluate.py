# mathlm/evaluate.py
import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from mathlm.config import Config
from mathlm.model import GPT
from mathlm.tokenizer import CharTokenizer


@torch.no_grad()
def eval_checkpoint(model, tok, eval_data, cfg, n_per_module, batch_size=64,
                    max_new_tokens=40):
    """Returns {module: accuracy} plus overall.

    Prompts are grouped by identical length so no padding is ever fed to the
    model. The model never saw <pad> during training, so padded prompts put it
    in a state it has no representation for and generation degrades badly.
    """
    model.eval()
    results = {}
    sep_id = tok.SEP
    eos_id = tok.EOS

    for module, items in eval_data.items():
        items = items[:n_per_module]
        correct = 0

        by_len = defaultdict(list)
        for it in items:
            by_len[len(it["question"])].append(it)

        for group in by_len.values():
            for i in range(0, len(group), batch_size):
                chunk = group[i : i + batch_size]
                prompts = [tok.encode(it["question"]) + [sep_id] for it in chunk]
                idx = torch.tensor(prompts, dtype=torch.long, device=cfg.device)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model.generate(idx, max_new_tokens=max_new_tokens,
                                         eos_id=eos_id, greedy=True)

                gen = out[:, idx.shape[1]:]
                for row, it in zip(gen.tolist(), chunk):
                    if eos_id in row:
                        row = row[: row.index(eos_id)]
                    pred = "".join(tok.itos[i] for i in row).strip()
                    if pred == it["answer"].strip():
                        correct += 1

        results[module] = correct / len(items)

    results["overall"] = sum(results.values()) / len(results)
    model.train()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="runs/run_001")
    ap.add_argument("--n_per_module", type=int, default=200)
    ap.add_argument("--ckpt", default=None,
                    help="single checkpoint; default sweeps all")
    args = ap.parse_args()

    cfg = Config()
    cfg.compile = False
    tok = CharTokenizer()
    eval_data = json.load(open(Path(cfg.shard_dir) / "eval.json"))
    print(f"{len(eval_data)} modules, {args.n_per_module} questions each")

    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "ckpt"
    if args.ckpt:
        paths = [Path(args.ckpt)]
    else:
        paths = sorted(ckpt_dir.glob("step_*.pt")) + [ckpt_dir / "final.pt"]
        paths = [p for p in paths if p.exists()]

    modules = sorted(eval_data.keys())
    out_path = run_dir / "per_module_accuracy.csv"
    f = open(out_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["checkpoint", "step", "overall"] + modules)

    model = GPT(cfg).to(cfg.device)

    for p in paths:
        ck = torch.load(p, map_location=cfg.device)
        model.load_state_dict(ck["model"])
        step = ck.get("step", -1)

        t0 = time.perf_counter()
        res = eval_checkpoint(model, tok, eval_data, cfg, args.n_per_module)
        dt = time.perf_counter() - t0

        w.writerow([p.name, step, f"{res['overall']:.4f}"] +
                   [f"{res[m]:.4f}" for m in modules])
        f.flush()

        print(f"\n{p.name}  step {step}  overall {res['overall']:.1%}  ({dt:.0f}s)")
        for m in modules:
            bar = "#" * int(res[m] * 40)
            print(f"  {m:<28} {res[m]:>6.1%}  {bar}")

    f.close()
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()