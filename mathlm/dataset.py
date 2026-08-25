# mathlm/dataset.py
import numpy as np
import torch
from pathlib import Path

class ShardDataset:
    """Random-offset sampler over a memory-mapped token stream.
ShardDataset
    ↓
opens token files
    ↓
samples random chunks
    ↓
creates x, y, mask
    ↓
sends them to CPU/GPU

    """

    def __init__(self, shard_dir, split, block_size, device,
                 n_chunks=1000, val_every=100, seed=1234):
        d = Path(shard_dir)
        self.tokens = np.memmap(d / "train_tokens.bin", dtype=np.uint16, mode="r")
        self.mask   = np.memmap(d / "train_mask.bin",   dtype=np.uint8,  mode="r")
        assert len(self.tokens) == len(self.mask), "token/mask length mismatch"

        self.block_size = block_size
        self.device = device
        self.rng = np.random.default_rng(seed if split == "train" else seed + 1)

        N = len(self.tokens)
        size = N // n_chunks
        want_val = (split != "train")

        lo, hi = [], []
        for i in range(n_chunks):
            if (i % val_every == 0) != want_val:
                continue
            start, end = i * size, (i + 1) * size
            if end - start > block_size + 1:
                lo.append(start)
                hi.append(end - block_size - 1)

        self.lo = np.array(lo, dtype=np.int64)
        self.hi = np.array(hi, dtype=np.int64)
        self.n = int((self.hi - self.lo).sum())
        print(f"  {split}: {len(self.lo)} chunks, {self.n/1e6:.1f}M sampleable positions")

    def __len__(self):
        return self.n

    def get_batch(self, batch_size):
        c = self.rng.integers(0, len(self.lo), size=batch_size)
        starts = self.rng.integers(self.lo[c], self.hi[c])

        x = torch.from_numpy(np.stack([
            self.tokens[i : i + self.block_size] for i in starts
        ]).astype(np.int64))

        y = torch.from_numpy(np.stack([
            self.tokens[i + 1 : i + 1 + self.block_size] for i in starts
        ]).astype(np.int64))

        m = torch.from_numpy(np.stack([
            self.mask[i + 1 : i + 1 + self.block_size] for i in starts
        ]).astype(np.float32))

        if self.device == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
            m = m.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y, m = x.to(self.device), y.to(self.device), m.to(self.device)

        return x, y, m