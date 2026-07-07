"""Subprocess bridge: run STIR-challenge tracker wrappers (which live in their
own uv venv at ~/stir-challenge-2026-inference/.venv) from the vidbot conda
env, exchanging arrays via a temporary NPZ.

Executed as a script INSIDE the stir venv:
    .venv/bin/python <path-to-this-repo>/downstream_causal/stir_bridge.py \
        --wrapper models.mono.alltracker_wrapper.AllTrackerWrapper \
        --in in.npz --out out.npz

in.npz:  video (T, H, W, 3) uint8, queries (N, 2) float32
out.npz: coords (N, T, 2) float32, visibs (N, T) bool
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

import numpy as np

STIR_ROOT = os.environ.get("STIR_CHALLENGE_ROOT", "<path-to-stir-challenge-repo>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrapper", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, STIR_ROOT)

    if "tapir" in args.wrapper:
        # tapnet/__init__.py eagerly imports the full JAX+TF training stack,
        # but the wrapper only needs the torch submodule. Pre-register a stub
        # package so `from tapnet.torch import tapir_model` skips __init__.
        import types

        tapnet_root = os.environ.get("TAPNET_ROOT", "<path-to-tapnet-repo>")
        sys.path.insert(0, tapnet_root)
        stub = types.ModuleType("tapnet")
        stub.__path__ = [f"{tapnet_root}/tapnet"]
        sys.modules["tapnet"] = stub

    module_name, cls_name = args.wrapper.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), cls_name)
    tracker = cls()

    data = np.load(args.inp)
    video = data["video"]
    queries = data["queries"].astype(np.float32)

    coords, visibs = tracker.track_offline(video, queries)  # (T, N, 2), (T, N)
    np.savez_compressed(
        args.out,
        coords=coords.transpose(1, 0, 2).astype(np.float32),  # (N, T, 2)
        visibs=visibs.transpose(1, 0),
    )
    print(f"bridge OK: {coords.shape[1]} tracks x {coords.shape[0]} frames")


if __name__ == "__main__":
    main()
