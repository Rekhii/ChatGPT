# scripts/benchmark.py
import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mathlm.config import Config
from mathlm.model import GPT
from mathlm.dataset import ShardDataset


def gpu_telemetry():
    """Returns (temp_c, sm_clock_mhz, power_w, util_pct) or Nones."""
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=temperature.gpu,clocks.sm,power.draw,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=5).strip()
        t, c, p, u = [float(v) for v in out.split(",")]
        return t, c, p, u
    except Exception:
        return None, None, None, None


def build(cfg, micro_bs):
    torch.manual_seed(cfg.seed)
    model = GPT(cfg).to(cfg.device)
    if cfg.compile:
        model = torch.compile(model)
    opt = model.configure_optimizers(cfg.weight_decay, cfg.learning_rate,
                                     (cfg.beta1, cfg.beta2))
    data = ShardDataset(cfg.shard_dir, "train", cfg.block_size, cfg.device)
    return model, opt, data


def train_steps(model, opt, data, cfg, micro_bs, accum, n_steps):
    """Runs n_steps optimiser steps. Returns tokens processed."""
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


def sweep(cfg):
    """Short runs at increasing batch size to find what fits and what's fastest."""
    print(f"{'micro_bs':>9} {'tok/s':>10} {'peak GB':>9}  status")
    best = None
    for micro_bs in [64, 96, 128, 192, 256]:
        try:
            model, opt, data = build(cfg, micro_bs)
            train_steps(model, opt, data, cfg, micro_bs, 1, 5)   # warmup + compile
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            tok = train_steps(model, opt, data, cfg, micro_bs, 1, 20)
            dt = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated() / 1e9
            rate = tok / dt
            print(f"{micro_bs:>9} {rate:>10,.0f} {peak:>9.2f}  ok")
            if best is None or rate > best[1]:
                best = (micro_bs, rate)
        except torch.cuda.OutOfMemoryError:
            print(f"{micro_bs:>9} {'-':>10} {'-':>9}  OOM")
            break
        finally:
            del model, opt, data
            torch.cuda.empty_cache()
    print(f"\nfastest: micro_batch_size={best[0]} at {best[1]:,.0f} tok/s")
    print("note: sweep numbers are cold. run --sustained for the real figure.")


def sustained(cfg, micro_bs, accum, minutes, out_path):
    model, opt, data = build(cfg, micro_bs)
    train_steps(model, opt, data, cfg, micro_bs, accum, 3)  # compile warmup

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    f = open(out_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["elapsed_s", "step", "tok_per_s", "temp_c", "sm_mhz", "power_w", "util"])

    deadline = time.time() + minutes * 60
    step = 0
    samples = []
    t_start = time.perf_counter()
    last_t = t_start

    print(f"running {minutes} min at micro_bs={micro_bs}, accum={accum}")
    while time.time() < deadline:
        tok = train_steps(model, opt, data, cfg, micro_bs, accum, 10)
        now = time.perf_counter()
        rate = tok / (now - last_t)
        last_t = now
        step += 10
        elapsed = now - t_start
        temp, sm, pw, util = gpu_telemetry()
        w.writerow([f"{elapsed:.1f}", step, f"{rate:.0f}", temp, sm, pw, util])
        f.flush()
        samples.append((elapsed, rate, temp, sm))
        if step % 50 == 0:
            print(f"  {elapsed/60:5.1f} min  step {step:5d}  "
                  f"{rate:>9,.0f} tok/s  {temp:.0f}C  {sm:.0f} MHz")
    f.close()

    total = samples[-1][0]
    tail = [s for s in samples if s[0] > total - 300]      # last 5 minutes
    if not tail:
        tail = samples[len(samples)//2:]
    rates = [s[1] for s in tail]
    temps = [s[2] for s in tail if s[2]]
    clocks = [s[3] for s in tail if s[3]]

    steady = sum(rates) / len(rates)
    first_min = [s[1] for s in samples if s[0] < 60]
    cold = sum(first_min) / len(first_min) if first_min else steady

    print("\n--- steady state, last 5 minutes ---")
    print(f"  tok/s        {steady:,.0f}")
    print(f"  cold tok/s   {cold:,.0f}  ({100*(cold-steady)/cold:+.1f}% drift)")
    if temps:
        print(f"  temp         {min(temps):.0f} to {max(temps):.0f} C")
    if clocks:
        print(f"  sm clock     {min(clocks):.0f} to {max(clocks):.0f} MHz  "
              f"(spread {max(clocks)-min(clocks):.0f})")

    tokens_per_step = micro_bs * cfg.block_size * accum
    for budget in [45, 60]:
        total_tok = steady * budget * 60
        print(f"  {budget} min budget: {total_tok/1e6:,.0f}M tokens, "
              f"max_steps = {int(total_tok / tokens_per_step):,}")
    print(f"\nlog written to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--sustained", action="store_true")
    ap.add_argument("--micro_bs", type=int, default=None)
    ap.add_argument("--accum", type=int, default=None)
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--out", default="runs/benchmark/throughput.csv")
    args = ap.parse_args()

    cfg = Config()
    mb = args.micro_bs or cfg.micro_batch_size
    ac = args.accum or cfg.grad_accum_steps

    if args.sweep:
        sweep(cfg)
    elif args.sustained:
        sustained(cfg, mb, ac, args.minutes, args.out)
    else:
        print("pass --sweep or --sustained")