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

    def __init__(self, shard_dir, split, block_size, device):
        d = Path(shard_dir)
        self.tokens = np.memmap(d / f"{split}_tokens.bin", dtype=np.uint16, mode="r")
        self.mask   = np.memmap(d / f"{split}_mask.bin",   dtype=np.uint8,  mode="r")
        assert len(self.tokens) == len(self.mask), "token/mask length mismatch"
        self.block_size = block_size
        self.device = device
        self.n = len(self.tokens) - block_size - 1

    def __len__(self):
        # len of dataset
        return self.n

    def get_batch(self, batch_size, generator=None):
        ix = torch.randint(self.n, (batch_size,), generator=generator)

        x = torch.stack([
            torch.from_numpy(self.tokens[i : i + self.block_size].astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(self.tokens[i + 1 : i + 1 + self.block_size].astype(np.int64))
            for i in ix
        ])
        m = torch.stack([
            torch.from_numpy(self.mask[i + 1 : i + 1 + self.block_size].astype(np.float32))
            for i in ix
        ])

        if self.device == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
            m = m.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y, m = x.to(self.device), y.to(self.device), m.to(self.device)

        return x, y, m