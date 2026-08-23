# scripts/debug_gen.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, json
from mathlm.config import Config
from mathlm.model import GPT
from mathlm.tokenizer import CharTokenizer

cfg = Config(); cfg.compile = False
tok = CharTokenizer()
model = GPT(cfg).to(cfg.device)
model.load_state_dict(torch.load("runs/run_001/ckpt/final.pt",
                                 map_location=cfg.device)["model"])
model.eval()

data = json.load(open("data/shards/eval.json"))
for module in ["arithmetic__add_or_sub", "calculus__differentiate", "numbers__is_prime"]:
    print(f"\n=== {module} ===")
    for it in data[module][:5]:
        ids = tok.encode(it["question"]) + [tok.SEP]
        idx = torch.tensor([ids], dtype=torch.long, device=cfg.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(idx, max_new_tokens=40, eos_id=tok.EOS, greedy=True)
        gen = out[0, len(ids):].tolist()
        if tok.EOS in gen:
            gen = gen[:gen.index(tok.EOS)]
        pred = "".join(tok.itos[i] for i in gen)
        print(f"  Q: {it['question'][:60]}")
        print(f"  want: {it['answer']!r}")
        print(f"  got:  {pred!r}")