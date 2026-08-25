import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from mathlm.config import Config
from mathlm.model import GPT
from mathlm.tokenizer import CharTokenizer


@torch.no_grad()
def eval_checkpoint(model, tok, eval_data, cfg, n_per_module,
                    batch_size=64, max_new_tokens=40):
    """Returns {module: accuracy} plus a macro-averaged 'overall'.

    Prompts are grouped by identical length so no padding is ever fed to the
    model. The model never saw <pad> during training, so a padded prompt puts
    it in a state it has no representation for and generation collapses. This
    single detail is the difference between 6.5% and 43.5% overall. Do not
    replace this with a padded collate.
    """
    model.eval()
    results = {}

    for module, items in eval_data.items():
        items = items[:n_per_module]
        correct = 0

        by_len = defaultdict(list)
        for it in items:
            by_len[len(it["question"])].append(it)

        for group in by_len.values():
            for start in range(0, len(group), batch_size):
                chunk = group[start:start + batch_size]
                prompts = [tok.encode(it["question"]) + [tok.SEP] for it in chunk]
                idx = torch.tensor(prompts, dtype=torch.long, device=cfg.device)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model.generate(idx, max_new_tokens=max_new_tokens,
                                         eos_id=tok.EOS, greedy=True)

                for row, it in zip(out[:, idx.shape[1]:].tolist(), chunk):
                    if tok.EOS in row:
                        row = row[:row.index(tok.EOS)]
                    pred = "".join(tok.itos[t] for t in row).strip()
                    correct += pred == it["answer"].strip()

        results[module] = correct / len(items)

    results["overall"] = sum(results.values()) / len(results)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="runs/run_001")
    ap.add_argument("--n_per_module", type=int, default=200)
    ap.add_argument("--ckpt", default=None, help="single ckpt; default sweeps all")
    args = ap.parse_args()

    cfg = Config()
    cfg.compile = False
    tok = CharTokenizer()
    eval_data = json.load(open(Path(cfg.shard_dir) / "eval.json"))

    ckpt_dir = Path(args.run_dir) / "ckpt"
    if args.ckpt:
        paths = [Path(args.ckpt)]
    else:
        paths = sorted(ckpt_dir.glob("step_*.pt")) + [ckpt_dir / "final.pt"]
        paths = [p for p in paths if p.exists()]

    modules = sorted(eval_data.keys())
    print(f"{len(modules)} modules x {args.n_per_module} questions, "
          f"{len(paths)} checkpoints")

    model = GPT(cfg).to(cfg.device)
    rows = []

    for p in paths:
        ck = torch.load(p, map_location=cfg.device)
        model.load_state_dict(ck["model"])
        res = eval_checkpoint(model, tok, eval_data, cfg, args.n_per_module)
        rows.append({"checkpoint": p.name, "step": ck.get("step", -1), **res})

        print(f"\n{p.name}  step {ck.get('step', -1)}  overall {res['overall']:.1%}")
        for m in modules:
            print(f"  {m:<28} {res[m]:>6.1%}")

    out_path = Path(args.run_dir) / "per_module_accuracy.json"
    json.dump(rows, open(out_path, "w"), indent=2)
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()