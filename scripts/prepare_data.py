import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mathlm.data import build

if __name__ == "__main__":
    build("data/shards", per_module=40_000, eval_per_module=200)