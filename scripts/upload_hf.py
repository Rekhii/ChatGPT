# scripts/upload_hf.py
"""Push the trained checkpoint and run artifacts to a Hugging Face model repo.

  pip install huggingface_hub
  huggingface-cli login
  python scripts/upload_hf.py --repo <your-username>/nanomath
"""
import argparse
import pathlib
import sys

from huggingface_hub import HfApi, create_repo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. Rekhii/nanomath")
    ap.add_argument("--run_dir", default="runs/run_001")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry_run", action="store_true",
                    help="list what would be uploaded and stop")
    args = ap.parse_args()

    root = pathlib.Path(".")
    run = pathlib.Path(args.run_dir)

    # (local path, path inside the repo)
    wanted = [
        (run / "ckpt" / "final.pt",          "final.pt"),
        (run / "config.json",                "config.json"),
        (run / "metrics.csv",                "metrics.csv"),
        (run / "per_module_accuracy.csv",    "per_module_accuracy.csv"),
        (run / "summary.json",               "summary.json"),
        (root / "docs" / "throughput_4060.csv", "throughput_4060.csv"),
        (root / "README.md",                 "README.md"),
    ]

    found, missing = [], []
    for local, remote in wanted:
        (found if local.exists() else missing).append((local, remote))

    print("will upload:")
    total = 0
    for local, remote in found:
        mb = local.stat().st_size / 1e6
        total += mb
        print(f"  {str(local):<44} -> {remote:<28} {mb:>8.2f} MB")
    if missing:
        print("\nnot found, skipping:")
        for local, remote in missing:
            print(f"  {local}")
    print(f"\ntotal {total:.1f} MB")

    if not found:
        sys.exit("nothing to upload")
    if args.dry_run:
        return

    if not any(r == "README.md" for _, r in found):
        print("\nwarning: no README.md, the model page will be blank")

    create_repo(args.repo, repo_type="model", private=args.private,
                exist_ok=True)
    print(f"\nrepo ready: https://huggingface.co/{args.repo}")

    api = HfApi()
    for local, remote in found:
        print(f"  uploading {remote} ...", flush=True)
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=args.repo,
            repo_type="model",
        )

    print(f"\ndone: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()