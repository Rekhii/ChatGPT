---
license: mit
datasets:
  - deepmind/math_dataset
language:
  - en
tags:
  - mathematics
  - character-level
  - gpt
  - from-scratch
  - nanogpt
pipeline_tag: text-generation
---

# nanomath

A 18.93M parameter decoder-only transformer trained from scratch on the DeepMind
Mathematics Dataset, character level, in 45 minutes on a single RTX 4060 laptop GPU.

This is a learning-by-doing project: no `transformers`, no `lightning`, no config
framework. Tokenizer, data pipeline, model, training loop and evaluation are all
hand-written. Source: https://github.com/Rekhii/ChatGPT

## What it does

Answers maths questions phrased exactly the way the DeepMind dataset phrases them:

```
What is the first derivative of 13*a**2 - 627434*a + 1191410?  ->  26*a - 627434
Sort 53, -2.2, -3, 14, 36 in increasing order.                 ->  -3, -2.2, 14, 36, 53
What is the hundred thousands digit of 82923295?               ->  9
```

It is not a chat model. It saw one format during training, `question<sep>answer<eos>`,
and nothing else. Free-form phrasing produces garbage.

## Results

Exact string match on 200 held-out questions per module, final checkpoint (step 7385).

| Module | Accuracy | Regime |
|---|---|---|
| `numbers__place_value` | 96.5% | solved |
| `calculus__differentiate` | 86.0% | solved |
| `comparison__sort` | 85.5% | solved |
| `arithmetic__add_or_sub` | 62.0% | still improving at deadline |
| `arithmetic__div` | 53.0% | still improving at deadline |
| `numbers__gcd` | 33.5% | plateaued |
| `arithmetic__mul` | 29.0% | still improving at deadline |
| `polynomials__evaluate` | 9.0% | barely learned |
| `arithmetic__add_sub_multiple` | 8.5% | barely learned |
| `algebra__linear_1d` | 8.0% | barely learned |
| `arithmetic__mixed` | 1.0% | not learned |
| `numbers__is_prime` | 50.0% | chance (binary output) |
| **overall** | **43.5%** | |
| overall excluding `is_prime` | 42.7% | |

Final validation loss 0.339, down from 4.705 at initialisation.

## The interesting finding

The model learned every task expressible as a fixed-depth transformation and failed
every task requiring iterative search.

`comparison__sort` (85.5%) and `numbers__gcd` (33.5%) are both integer tasks, but
sorting is comparison and gcd requires factorization. `numbers__is_prime` sat at
50.0%, 50.0%, 50.5%, 54.0%, 49.0%, 56.0%, 49.0%, 50.0% across all eight checkpoints,
perfectly flat for 7,386 steps while every other module climbed. Primality cannot be
pattern-matched from digit strings, and a fixed-depth forward pass with no scratchpad
has nowhere to run a factorization loop.

Failure modes are also informative. On `gcd(1569427, 657)` the model answers `657`,
having learned the shortcut "when one number is much smaller, the gcd is often that
number", which is true whenever the smaller divides the larger. On
`gcd(464690, 48980)` it answers `2210` against a true `310`: wrong, but the right
order of magnitude. It estimates the size of a plausible answer without being able
to compute it.

Arithmetic errors show the classic character-level signature. `-125266 - 1686191`
returns `-1707797` against a true `-1811457`: correct sign, correct digit count,
wrong middle digits. Meanwhile `-2769526541 - -0.5` is exactly right. Alignment and
formatting are reliable; carry propagation is not.

## Model

| | |
|---|---|
| Parameters | 18,930,176 |
| Layers | 6 |
| Heads | 8 |
| Embedding dim | 512 |
| Context | 256 |
| Vocab | 98 (char level: 95 printable ASCII + `<pad>` `<sep>` `<eos>`) |
| Attention | FlashAttention via `F.scaled_dot_product_attention` |
| Norm | pre-norm LayerNorm, no bias |
| Weight tying | input embedding shared with output projection |

Character level was chosen over BPE deliberately: digit-level tokenization gives the
model positional resolution over numbers that a subword vocabulary destroys.

## Training

| | |
|---|---|
| Steps | 7,386 (schedule completed, not truncated) |
| Tokens | 363M (~17 passes over 21M tokens) |
| Wall clock | 45.2 minutes |
| Hardware | 1x RTX 4060 Laptop (8.59 GB, sm_89) |
| Throughput | 134,460 tok/s sustained |
| Effective batch | 49,152 tokens (96 x 256 x 2 accum) |
| Optimiser | AdamW fused, lr 6e-4 cosine to 6e-5, 300 warmup |
| Precision | bfloat16 autocast, `torch.compile` |
| Loss | cross entropy masked to answer tokens only |

Only 14.9% of tokens are answer tokens. Loss is masked so the model is not rewarded
for autocompleting question phrasing.

`max_steps` was derived from measurement rather than guessed: a 15 minute sustained
throughput benchmark run to thermal equilibrium (86-87C, SM clock stable at
1920-2100 MHz) gave 134,460 tok/s, and the step count was set from that. Predicted
45 minutes, actual 45.2.

## Usage

The checkpoint is a raw `state_dict`, not a `transformers` model. You need `model.py`,
`config.py` and `tokenizer.py` from the GitHub repo.

```python
import torch
from mathlm.config import Config
from mathlm.model import GPT
from mathlm.tokenizer import CharTokenizer

cfg = Config(); cfg.compile = False
tok = CharTokenizer()
model = GPT(cfg).to(cfg.device)
model.load_state_dict(torch.load("final.pt", map_location=cfg.device)["model"])
model.eval()

def ask(question, max_new_tokens=48):
    ids = tok.encode(question) + [tok.SEP]
    idx = torch.tensor([ids], device=cfg.device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(idx, max_new_tokens=max_new_tokens,
                             eos_id=tok.EOS, greedy=True)
    gen = out[0, len(ids):].tolist()
    if tok.EOS in gen:
        gen = gen[:gen.index(tok.EOS)]
    return "".join(tok.itos[i] for i in gen)

print(ask("What is the first derivative of 13*a**2 - 627434*a + 1191410?"))
```

Generate with batch size 1, or group prompts by identical length. The model never saw
`<pad>` during training, so left-padded batches put it in a state it has no
representation for and generation degrades badly. In this project that bug alone cost
6.7x on the headline number: the same checkpoint scored 6.5% padded and 43.5%
unpadded.

Decode greedily. There is one correct answer to an arithmetic question, so sampling
is strictly worse.

## Files

- `final.pt` — model weights, optimiser state, config, final val loss
- `config.json` — full hyperparameter dump
- `metrics.csv` — per-step loss, lr, throughput, GPU temperature and SM clock
- `per_module_accuracy.csv` — accuracy per module at all eight checkpoints
- `throughput_4060.csv` — the 15 minute thermal benchmark

## Limitations

Roughly 17 passes over 21M tokens rather than one pass over 363M. Chinchilla-optimal
for this parameter count is around 380M tokens, so the run was compute-matched but
data-repeated. Some of the reported accuracy may be memorisation, which the held-out
eval set bounds but does not eliminate.

Trained on easy and medium difficulty splits only. Hard was excluded.

`arithmetic__add_or_sub`, `arithmetic__div` and `arithmetic__mul` were still gaining
roughly 5 points per 1,000 steps when the clock ran out. Those numbers are a floor.

Not a calculator, not a chat model, and no use for anything beyond understanding how
small transformers learn structured tasks.