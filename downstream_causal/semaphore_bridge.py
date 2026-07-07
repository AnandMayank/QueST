"""Subprocess bridge: run downstream_causal's tracker adapters (CoTracker2/3,
which need the vidbot conda env's torch) from SEMAPHORE's napari process,
which runs in a separate venv (~/movement/.venv) that has napari+magicgui+Qt
but not torch.

Executed inside the vidbot conda env:
    <vidbot_python> semaphore_bridge.py --tracker cotracker3 --device cpu \
        --in in.npz --out out.npz

in.npz:  video (T, H, W, 3) uint8, queries (N, 2) float32 xy at frame 0
out.npz: coords (N, T, 2) float32
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from downstream_causal.trackers import get_tracker

    data = np.load(args.inp)
    video = data["video"]
    queries = data["queries"].astype(np.float64)

    tracker = get_tracker(args.tracker, device=args.device)
    coords = tracker.track(video, queries, query_frame=0)  # (N, T, 2)
    np.savez_compressed(args.out, coords=coords.astype(np.float32))
    print(f"bridge OK: {coords.shape[0]} tracks x {coords.shape[1]} frames")


if __name__ == "__main__":
    main()
