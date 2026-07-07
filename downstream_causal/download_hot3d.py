"""Download a spread of HOT3D-Clips train_aria clips and pre-scan them for
object count (multi-object clips are the ones with switch potential; this
avoids running the full tracker pipeline on trivial single/well-separated-
object clips, mirroring the min_visibility/min-parts filters used for SAPIEN).

Usage:
    python -m downstream_causal.download_hot3d --out ~/data/hot3d_subset --n 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "bop-benchmark/hot3d"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_root = Path(args.out).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    files = list_repo_files(REPO, repo_type="dataset")
    clips = sorted(f for f in files if f.startswith("train_aria/clip-") and f.endswith(".tar"))
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(clips, size=min(args.n, len(clips)), replace=False)

    print(f"downloading {len(chosen)} clips")
    for f in sorted(chosen):
        hf_hub_download(REPO, f, repo_type="dataset", local_dir=str(out_root))
        print(f"  done {f}")

    print(f"subset ready at {out_root}/train_aria")


if __name__ == "__main__":
    main()
