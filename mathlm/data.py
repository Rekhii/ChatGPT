"""Download DeepMind Mathematics modules and pack them into uint16 shards."""
import ast, json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from mathlm.tokenizer import CharTokenizer

REPO = "deepmind/math_dataset"
REVISION = "refs/convert/parquet"

# train split = easy | medium | hard, concatenated in that order
PER_DIFFICULTY = 666_666
DIFFICULTY = {
    "easy":   (0, PER_DIFFICULTY),
    "medium": (PER_DIFFICULTY, 2 * PER_DIFFICULTY),
    "hard":   (2 * PER_DIFFICULTY, 3 * PER_DIFFICULTY),
}

MODULES = [
    "arithmetic__add_or_sub",
    "arithmetic__mul",
    "arithmetic__div",
    "arithmetic__add_sub_multiple",
    "arithmetic__mixed",
    "algebra__linear_1d",
    "polynomials__evaluate",
    "calculus__differentiate",
    "comparison__sort",
    "numbers__gcd",
    "numbers__is_prime",
    "numbers__place_value",
]


def clean(s):
    """Records are stringified bytes literals: "b'What is 5 times 3?\\n'"."""
    return ast.literal_eval(s).decode("utf-8").strip()


def load_module(module, split="train"):
    path = hf_hub_download(REPO, f"{module}/{split}/0000.parquet",
                           repo_type="dataset", revision=REVISION)
    return pq.read_table(path)


def take(table, start, end, limit):
    """Read rows [start, end) up to `limit` as cleaned (question, answer) pairs."""
    end = min(end, table.num_rows)
    n = min(limit, end - start)
    sl = table.slice(start, n)
    qs = sl.column("question").to_pylist()
    ans = sl.column("answer").to_pylist()
    return [(clean(q), clean(a)) for q, a in zip(qs, ans)]


def build(out_dir, modules=MODULES, difficulties=("easy", "medium"),
          per_module=40_000, eval_per_module=200):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = CharTokenizer()

    tokens, mask, counts = [], [], {}
    eval_sets = {}

    for module in modules:
        train_tbl = load_module(module, "train")
        n_each = per_module // len(difficulties)
        pairs = []
        for d in difficulties:
            lo, hi = DIFFICULTY[d]
            pairs += take(train_tbl, lo, hi, n_each)

        for q, a in pairs:
            ids = tok.encode_pair(q, a)
            n_ans = len(a) + 1                      # answer characters + EOS
            m = [0] * (len(ids) - n_ans) + [1] * n_ans
            tokens.extend(ids)
            mask.extend(m)
        counts[module] = len(pairs)

        test_tbl = load_module(module, "test")
        eval_sets[module] = [
            {"question": q, "answer": a}
            for q, a in take(test_tbl, 0, test_tbl.num_rows, eval_per_module)
        ]
        print(f"{module:30s} train={len(pairs):6d} eval={len(eval_sets[module])}")

    tokens = np.array(tokens, dtype=np.uint16)
    mask = np.array(mask, dtype=np.uint8)
    tokens.tofile(out_dir / "train_tokens.bin")
    mask.tofile(out_dir / "train_mask.bin")

    with open(out_dir / "eval.json", "w") as f:
        json.dump(eval_sets, f)

    meta = {
        "vocab_size": len(tok),
        "n_tokens": int(tokens.size),
        "answer_fraction": float(mask.mean()),
        "modules": counts,
        "difficulties": list(difficulties),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\ntokens={tokens.size:,}  answer_fraction={mask.mean():.3f}")
    return meta