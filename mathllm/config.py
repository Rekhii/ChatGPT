# mathllm/config.py
from dataclasses import dataclass, asdict
import json

@dataclass
class Config:
    # data
    shard_dir: str = "data/shards"
    vocab_size: int = 98
    block_size: int = 256

    # model (~19.1M params)
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = False                  # slightly reduce parameter count and computation  and arch

    # optimisation
    micro_batch_size: int = 96
    grad_accum_steps: int = 2
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 300
    max_steps: int = 7386  # from benchmark: 134,460 tok/s x 45 min / 49,152
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # runtime
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = True
    seed: int = 1337

    #logging
    run_dir: str = "runs/run_001"
    log_interval: int = 50
    eval_interval: int = 1000
    eval_batches: int = 50

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)