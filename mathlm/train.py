import argparse
import json
import math
import time
from pathlib import Path

import torch

from mathlm.config import Config
from mathlm.model import GPT
from mathlm.dataset import ShardDataset


def lr_at(step, cfg):
    """Linear warmup, then cosine decay to min_lr."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_val(model, data, cfg):
    model.eval()
    total = 0.0
    for _ in range(cfg.eval_batches):
        x, y, m = data.get_batch(cfg.micro_batch_size)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y, m)
        total += loss.item()
    model.train()
    return total / cfg.eval_batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_minutes", type=float, default=50.0)
    args = ap.parse_args()

    cfg = Config()
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    run_dir = Path(cfg.run_dir)
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.json")

    train_data = ShardDataset(cfg.shard_dir, "train", cfg.block_size, cfg.device)
    val_data = ShardDataset(cfg.shard_dir, "val", cfg.block_size, cfg.device)

    raw_model = GPT(cfg).to(cfg.device)
    print(f"model: {raw_model.num_params()/1e6:.2f}M params")
    opt = raw_model.configure_optimizers(
        cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2)
    )
    model = torch.compile(raw_model) if cfg.compile else raw_model

    tokens_per_step = cfg.micro_batch_size * cfg.block_size * cfg.grad_accum_steps
    deadline = time.time() + args.max_minutes * 60
    t0 = time.perf_counter()
    history = []
    print(f"{cfg.max_steps} steps, {tokens_per_step:,} tok/step, "
          f"{args.max_minutes:.0f} min budget")

    model.train()
    for step in range(cfg.max_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)

        opt.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y, m = train_data.get_batch(cfg.micro_batch_size)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y, m)
            train_loss += loss.item() / cfg.grad_accum_steps
            (loss / cfg.grad_accum_steps).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % cfg.log_interval == 0:
            mins = (time.perf_counter() - t0) / 60
            print(f"step {step:5d}  loss {train_loss:.4f}  {mins:5.1f}m")

        # checkpoints are the per-module accuracy signal, so they stay
        if step > 0 and step % cfg.eval_interval == 0:
            vl = estimate_val(model, val_data, cfg)
            history.append({"step": step, "train_loss": train_loss, "val_loss": vl})
            torch.save({"model": raw_model.state_dict(), "step": step,
                        "config": vars(cfg)}, run_dir / "ckpt" / f"step_{step:06d}.pt")
            print(f"  >> val loss {vl:.4f}")

        if time.time() > deadline:
            print(f"stopping at step {step}: time budget")
            break

    vl = estimate_val(model, val_data, cfg)
    torch.save({"model": raw_model.state_dict(), "step": step,
                "config": vars(cfg), "val_loss": vl}, run_dir / "ckpt" / "final.pt")

    mins = (time.perf_counter() - t0) / 60
    tokens = (step + 1) * tokens_per_step
    print(f"done. {step+1} steps, {tokens/1e6:.0f}M tokens, "
          f"{mins:.1f} min, val loss {vl:.4f}")
    json.dump({"steps": step + 1, "tokens": tokens, "minutes": mins,
               "final_val_loss": vl, "history": history},
              open(run_dir / "summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()