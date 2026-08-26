# scripts/preflight.py
import math
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mathllm.config import Config
from mathllm.model import GPT
from mathllm.dataset import ShardDataset
from mathllm.tokenizer import CharTokenizer


def check(name, passed, detail=""):
    tag = "PASS" if passed else "FAIL"
    print(f"[{tag}] {name}  {detail}")
    return passed


def main():
    cfg = Config()
    cfg.compile = False          # keep startup fast while debugging
    torch.manual_seed(cfg.seed)
    results = []

    # ---- 1. shapes ----
    model = GPT(cfg).to(cfg.device)
    n = model.num_params() / 1e6
    results.append(check("param count", 18.0 < n < 20.0, f"{n:.2f}M"))

    # ---- 2. data loads and decodes ----
    tok = CharTokenizer()
    train = ShardDataset(cfg.shard_dir, "train", cfg.block_size, cfg.device)
    x, y, m = train.get_batch(8)
    results.append(check("batch shapes",
                         x.shape == (8, cfg.block_size) and m.shape == x.shape,
                         f"{tuple(x.shape)}"))

    sample = tok.decode(x[0].tolist())
    print("\n--- decoded sample ---")
    print(repr(sample[:200]))
    print("--- answer positions only ---")
    ans = "".join(tok.itos[t] for t, k in zip(y[0].tolist(), m[0].tolist()) if k > 0)
    print(repr(ans[:200]))
    print()
    results.append(check("mask non-empty", m.sum().item() > 0,
                         f"{m.mean().item():.3f} of tokens are answers"))

    # ---- 3. initial loss ----
    model.eval()
    with torch.no_grad():
        losses = []
        for _ in range(10):
            xb, yb, mb = train.get_batch(cfg.micro_batch_size)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(xb, yb, mb)
            losses.append(loss.item())
    avg = sum(losses) / len(losses)
    expected = math.log(cfg.vocab_size)
    results.append(check("initial loss", abs(avg - expected) < 0.15,
                         f"{avg:.3f} vs ln({cfg.vocab_size})={expected:.3f}"))

    # ---- 4. overfit one batch ----
    model.train()
    opt = model.configure_optimizers(0.0, 1e-3, (cfg.beta1, cfg.beta2))
    xb, yb, mb = train.get_batch(8)
    history = []
    for step in range(400):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(xb, yb, mb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for g in opt.param_groups:  # decay lr so it settles
            g["lr"] = 1e-3 * (0.5 ** (step // 150))
        opt.step()
        history.append(loss.item())
        if step % 50 == 0:
            print(f"  overfit step {step:3d}  loss {loss.item():.4f}")
    best = min(history[-20:])
    results.append(check("overfits one batch", best < 0.05,
                         f"{history[0]:.3f} -> best-of-last-20 {best:.4f}"))

    # ---- 5. memory headroom at real batch size ----
    model = GPT(cfg).to(cfg.device)
    opt = model.configure_optimizers(cfg.weight_decay, cfg.learning_rate,
                                     (cfg.beta1, cfg.beta2))
    torch.cuda.reset_peak_memory_stats()
    xb, yb, mb = train.get_batch(cfg.micro_batch_size)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = model(xb, yb, mb)
    loss.backward()
    opt.step()
    peak = torch.cuda.max_memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    results.append(check("vram headroom", peak < total * 0.75,
                         f"{peak:.2f} GB peak of {total:.2f} GB"))

    print()
    print("ALL PASS" if all(results) else "SOMETHING FAILED, do not start the run")


if __name__ == "__main__":
    main()