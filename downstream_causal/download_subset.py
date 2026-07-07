"""Download a stratified subset of the QueST-PartNetMobility-SAPIEN benchmark
dataset (frames + affordance NPZs + metadata only; skips video.mp4) for E1/E2.

Set QUEST_DATASET_REPO to the Hugging Face dataset repo id (see the top-level
README's Dataset section).

Usage:
    python -m downstream_causal.download_subset --out ~/data/quest_partnet_subset \
        --per-level 6
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

REPO = os.environ.get("QUEST_DATASET_REPO", "<hf-dataset-repo-id>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-level", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_root = Path(args.out).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    files = list_repo_files(REPO, repo_type="dataset")
    file_count = defaultdict(int)
    for f in files:
        m = re.match(r"(manipulation_(\d+)/\d+/take_\d+)/", f)
        if m:
            file_count[m.group(1)] += 1

    seqs_by_level = defaultdict(list)
    for seq, n in file_count.items():
        level = seq.split("/")[0].split("_")[1]
        # a real (frames+affordance) sequence has >= ~2 files/frame; sequences
        # with only metadata.json+video.mp4 (n<=3) are unusable for us.
        if n >= 20:
            seqs_by_level[level].append(seq)

    import random

    rng = random.Random(args.seed)
    chosen = []
    for level, seqs in sorted(seqs_by_level.items()):
        seqs = sorted(seqs)
        rng.shuffle(seqs)
        picked = seqs[: args.per_level]
        if len(picked) < args.per_level:
            print(f"WARNING: only {len(picked)} usable sequences for level {level} "
                  f"(requested {args.per_level})")
        chosen.extend(picked)

    print(f"downloading {len(chosen)} sequences: {chosen}")

    for seq in chosen:
        seq_files = [
            f for f in files
            if f.startswith(seq + "/") and not f.endswith("video.mp4")
        ]
        for f in seq_files:
            hf_hub_download(
                REPO, f, repo_type="dataset", local_dir=str(out_root),
            )
        print(f"  done {seq} ({len(seq_files)} files)")

    print(f"subset ready at {out_root}")


if __name__ == "__main__":
    main()
