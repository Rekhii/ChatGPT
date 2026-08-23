# mathlm/train.py
import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from mathlm.config import Config
from mathlm.model import GPT
from mathlm.dataset import ShardDataset


def telemetry():
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=temperature.gpu,clocks.sm,power.draw",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=5).strip()
        return [float(v) for v in out.split(",")]
    except Exception:
        return [None, None, None]


def lr_at(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_val(model, data, cfg):
    model.eval()
    losses = []
    for _ in range(cfg.eval_batches):
        x, y, m = data.get_batch(cfg.micro_batch_size)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y, m)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_minutes", type=float, default=50.0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    run_dir = Path(cfg.run_dir)
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.json")

    train_data = ShardDataset(cfg.shard_dir, "train", cfg.block_size, cfg.device)
    val_data = ShardDataset(cfg.shard_dir, "val", cfg.block_size, cfg.device)

    model = GPT(cfg).to(cfg.device)
    print(f"model: {model.num_params()/1e6:.2f}M params")
    raw_model = model
    if cfg.compile:
        model = torch.compile(model)

    opt = raw_model.configure_optimizers(cfg.weight_decay, cfg.learning_rate,
                                         (cfg.beta1, cfg.beta2))

    start_step = 0
    if args.resume and (run_dir / "ckpt" / "last.pt").exists():
        ck = torch.load(run_dir / "ckpt" / "last.pt", map_location=cfg.device)
        raw_model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_step = ck["step"] + 1
        print(f"resumed at step {start_step}")

    metrics_path = run_dir / "metrics.csv"
    new_file = not metrics_path.exists()
    mf = open(metrics_path, "a", newline="")
    mw = csv.writer(mf)
    if new_file:
        mw.writerow(["step", "tokens", "train_loss", "val_loss", "lr",
                     "tok_per_s", "elapsed_s", "temp_c", "sm_mhz", "power_w"])

    tokens_per_step = cfg.micro_batch_size * cfg.block_size * cfg.grad_accum_steps
    deadline = time.time() + args.max_minutes * 60
    t_start = time.perf_counter()
    t_log = t_start
    tokens_seen = start_step * tokens_per_step
    stop_reason = "max_steps"

    print(f"training to step {cfg.max_steps}, budget {args.max_minutes:.0f} min, "
          f"{tokens_per_step:,} tok/step")

    model.train()
    for step in range(start_step, cfg.max_steps):
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y, m = train_data.get_batch(cfg.micro_batch_size)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y, m)
            accum_loss += loss.item() / cfg.grad_accum_steps
            (loss / cfg.grad_accum_steps).backward()
            tokens_seen += x.numel()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % cfg.log_interval == 0:
            torch.cuda.synchronize()
            now = time.perf_counter()
            rate = (cfg.log_interval * tokens_per_step) / (now - t_log) if step else 0
            t_log = now
            elapsed = now - t_start
            temp, sm, pw = telemetry()
            mw.writerow([step, tokens_seen, f"{accum_loss:.4f}", "", f"{lr:.2e}",
                         f"{rate:.0f}", f"{elapsed:.1f}", temp, sm, pw])
            mf.flush()
            remain = (cfg.max_steps - step) * tokens_per_step / max(rate, 1) / 60
            print(f"step {step:5d}/{cfg.max_steps}  loss {accum_loss:.4f}  "
                  f"lr {lr:.2e}  {rate:>8,.0f} tok/s  "
                  f"{elapsed/60:5.1f}m  eta {remain:4.1f}m  {temp:.0f}C {sm:.0f}MHz")

        if step > 0 and step % cfg.eval_interval == 0:
            vl = estimate_val(model, val_data, cfg)
            now = time.perf_counter()
            temp, sm, pw = telemetry()
            mw.writerow([step, tokens_seen, f"{accum_loss:.4f}", f"{vl:.4f}",
                         f"{lr:.2e}", "", f"{now-t_start:.1f}", temp, sm, pw])
            mf.flush()
            print(f"  >> val loss {vl:.4f} at step {step}")
            torch.save({"model": raw_model.state_dict(), "step": step,
                        "config": vars(cfg)},
                       run_dir / "ckpt" / f"step_{step:06d}.pt")
            torch.save({"model": raw_model.state_dict(),
                        "optimizer": opt.state_dict(), "step": step,
                        "config": vars(cfg)},
                       run_dir / "ckpt" / "last.pt")
            t_log = time.perf_counter()

        if time.time() > deadline:
            stop_reason = "time budget"
            print(f"\nstopping at step {step}: {stop_reason} reached")
            break

    vl = estimate_val(model, val_data, cfg)
    torch.save({"model": raw_model.state_dict(), "optimizer": opt.state_dict(),
                "step": step, "config": vars(cfg), "val_loss": vl},
               run_dir / "ckpt" / "final.pt")
    mw.writerow([step, tokens_seen, f"{accum_loss:.4f}", f"{vl:.4f}", f"{lr:.2e}",
                 "", f"{time.perf_counter()-t_start:.1f}", *telemetry()])
    mf.close()

    total = (time.perf_counter() - t_start) / 60
    print(f"\ndone. {step+1} steps, {tokens_seen/1e6:.0f}M tokens, "
          f"{total:.1f} min, final val loss {vl:.4f}, stopped on {stop_reason}")
    json.dump({"steps": step + 1, "tokens": tokens_seen, "minutes": total,
               "final_val_loss": vl, "stop_reason": stop_reason},
              open(run_dir / "summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()