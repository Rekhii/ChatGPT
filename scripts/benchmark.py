"""Sustained throughput measurement.

Short benchmarks lie on a laptop GPU: the first minute runs at boost clocks
that the cooling cannot hold. Only the tail of a long run reflects what a
real training run will actually get, so max_steps is derived from that.
"""
import argparse
import csv
import subprocess
import time
from pathlib import Path

import torch

from mathlm.config import Config
from mathlm.model import GPT
from mathlm.dataset import ShardDataset


def telemetry():
    """(temp_c, sm_mhz, power_w) or Nones. sm_mhz dropping means throttling."""
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=temperature.gpu,clocks.sm,power.draw",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=5).strip()
        return [float(v) for v in out.split(",")]
    except Exception:
        return [None, None, None]


def train_steps(model, opt, data, cfg, micro_bs, accum, n_steps):
    """Runs n_steps optimiser steps, returns tokens processed."""
    tokens = 0
    for _ in range(n_steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            x, y, m = data.get_batch(micro_bs)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y, m)
            (loss / accum).backward()
            tokens += x.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
    torch.cuda.synchronize()
    return tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--micro_bs", type=int, default=None)
    ap.add_argument("--accum", type=int, default=None)
    ap.add_argument("--budgets", type=int, nargs="+", default=[45, 60],
                    help="minutes to project max_steps for")
    ap.add_argument("--out", default="runs/benchmark/throughput.csv")
    args = ap.parse_args()

    cfg = Config()
    micro_bs = args.micro_bs or cfg.micro_batch_size
    accum = args.accum or cfg.grad_accum_steps
    torch.manual_seed(cfg.seed)

    model = GPT(cfg).to(cfg.device)
    opt = model.configure_optimizers(cfg.weight_decay, cfg.learning_rate,
                                     (cfg.beta1, cfg.beta2))
    if cfg.compile:
        model = torch.compile(model)
    data = ShardDataset(cfg.shard_dir, "train", cfg.block_size, cfg.device)
    train_steps(model, opt, data, cfg, micro_bs, accum, 3)  # compile warmup

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(out_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["elapsed_s", "step", "tok_per_s", "temp_c", "sm_mhz", "power_w"])

    print(f"running {args.minutes} min at micro_bs={micro_bs}, accum={accum}")
    deadline = time.time() + args.minutes * 60
    t_start = last_t = time.perf_counter()
    step = 0
    samples = []

    while time.time() < deadline:
        tok = train_steps(model, opt, data, cfg, micro_bs, accum, 10)
        now = time.perf_counter()
        rate = tok / (now - last_t)
        last_t = now
        step += 10
        elapsed = now - t_start
        temp, sm, pw = telemetry()
        w.writerow([f"{elapsed:.1f}", step, f"{rate:.0f}", temp, sm, pw])
        f.flush()
        samples.append((elapsed, rate, temp, sm))
        if step % 50 == 0:
            print(f"  {elapsed/60:5.1f} min  step {step:5d}  {rate:>9,.0f} tok/s  "
                  f"{temp:.0f}C  {sm:.0f} MHz")
    f.close()

    # steady state is the last 5 minutes, after thermals have settled
    total = samples[-1][0]
    tail = [s for s in samples if s[0] > total - 300] or samples[len(samples)//2:]
    steady = sum(s[1] for s in tail) / len(tail)
    cold = [s[1] for s in samples if s[0] < 60]
    cold = sum(cold) / len(cold) if cold else steady
    temps = [s[2] for s in tail if s[2]]
    clocks = [s[3] for s in tail if s[3]]

    print(f"\nsteady {steady:,.0f} tok/s   cold {cold:,.0f} "
          f"({100*(cold-steady)/cold:+.1f}% drift)")
    if temps:
        print(f"temp {min(temps):.0f} to {max(temps):.0f} C")
    if clocks:
        print(f"sm clock {min(clocks):.0f} to {max(clocks):.0f} MHz "
              f"(spread {max(clocks)-min(clocks):.0f})")

    tokens_per_step = micro_bs * cfg.block_size * accum
    for budget in args.budgets:
        total_tok = steady * budget * 60
        print(f"{budget} min: {total_tok/1e6:,.0f}M tokens, "
              f"max_steps = {int(total_tok / tokens_per_step):,}")
    print(f"\nlog written to {out_path}")


if __name__ == "__main__":
    main()