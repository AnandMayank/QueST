"""E3 probe: does a frozen video backbone's trajectory-conditioned
representation carry the same low-ISR/high-ISR downstream gap that E1/E2
found for tracking-derived segmentation?

For each (sequence, ISR condition) pair from downstream_causal's E2
injection results (switch arm; APE-matched against the drift arm), pool
backbone patch tokens along the trajectory and compute:
  (a) latent similarity: cosine distance between the low-ISR (clean) and
      high-ISR (switched) trajectory-pooled latents -- does the backbone even
      "notice" the switch in feature space?
  (b) downstream predictor: a linear probe trained to predict a verifiable
      future physical state (here: joint position at the last GT frame,
      binarized into "opened past the midpoint" vs not -- free from SAPIEN
      metadata) from the trajectory-pooled latent sequence. Brier score
      compares low-ISR vs high-ISR conditioning.

This mirrors E2's design (matched geometric error, varying only identity) so
any gap is attributable to identity preservation, not drift magnitude.

Usage (run inside the worldmodel-eval venv, with vidbot on PYTHONPATH):
    HF_HOME=<your-hf-cache-dir> \
    PYTHONPATH=<path-to-this-repo> \
    ~/worldmodel-eval/.venv/bin/python -m worldmodel_eval.probe \
        --data-root ~/data/quest_partnet_subset \
        --backbones dinov2 vjepa2 \
        --out ~/worldmodel-eval/results/e3_probe.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from downstream_causal.data import SyntheticSequence, discover_sequences  # noqa: E402
from downstream_causal.interventions.inject_switches import inject_switch  # noqa: E402

from .backbones import ClipTokens, get_backbone  # noqa: E402


def pool_trajectory(clip: ClipTokens, traj_px: np.ndarray, gt_frame_indices: list[int]) -> torch.Tensor:
    """Bilinearly pool backbone tokens along a trajectory.

    traj_px: (G, 2) xy pixel positions at each GT frame.
    gt_frame_indices: source-video frame index for each row of traj_px.
    Returns (G, C) pooled features (nearest backbone time-step per GT frame).
    """
    tokens = clip.tokens  # (T', H', W', C)
    t_prime, h, w, c = tokens.shape
    backbone_frames = np.array(clip.frame_indices)

    out = torch.zeros(len(traj_px), c)
    for g, (x, y) in enumerate(traj_px):
        # nearest backbone time-step to this GT frame
        ti = int(np.argmin(np.abs(backbone_frames - gt_frame_indices[g])))
        gx = np.clip(x / clip.patch_size, 0, w - 1.001)
        gy = np.clip(y / clip.patch_size, 0, h - 1.001)
        x0, y0 = int(gx), int(gy)
        fx, fy = gx - x0, gy - y0
        t00 = tokens[ti, y0, x0]
        t10 = tokens[ti, y0, min(x0 + 1, w - 1)]
        t01 = tokens[ti, min(y0 + 1, h - 1), x0]
        t11 = tokens[ti, min(y0 + 1, h - 1), min(x0 + 1, w - 1)]
        out[g] = (
            t00 * (1 - fx) * (1 - fy) + t10 * fx * (1 - fy)
            + t01 * (1 - fx) * fy + t11 * fx * fy
        )
    return out


JOINT_OPEN_THRESHOLD = 1.0  # radians (or normalized units); ~fully-open (1.57) vs partially-open (0.4-0.8)


def joint_open_label(seq: SyntheticSequence) -> int:
    """Binary verifiable-state label: did the most-active joint's absolute
    final position cross a fixed threshold (roughly: opened most of the way
    vs only partially)? Free ground truth from the simulator's own
    joint_positions metadata. Uses an absolute threshold (not a fraction of
    each sequence's own final value, which would be tautological)."""
    jp0 = seq.gt_frames[0].joint_positions
    jpT = seq.gt_frames[-1].joint_positions
    if jp0 is None or jpT is None:
        return -1
    displacement = np.abs(jpT - jp0)
    if displacement.max() < 1e-6:
        return -1
    primary = int(np.argmax(displacement))
    return int(abs(jpT[primary]) > JOINT_OPEN_THRESHOLD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--backbones", nargs="+", default=["dinov2", "vjepa2"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--k-per-part", type=int, default=12)
    ap.add_argument("--p-high", type=float, default=0.3)
    ap.add_argument("--max-seqs", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    seq_dirs = []
    for sdir in discover_sequences(args.data_root):
        try:
            seq = SyntheticSequence.load(sdir)
        except Exception:
            continue
        if len(seq.part_ids) >= 2 and joint_open_label(seq) >= 0:
            seq_dirs.append(seq)
    seq_dirs = seq_dirs[: args.max_seqs]
    print(f"{len(seq_dirs)} usable sequences")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["sequence"], r["backbone"]))
        print(f"resuming: {len(done)} rows already present")

    for backbone_name in args.backbones:
        print(f"loading backbone {backbone_name}")
        backbone = get_backbone(backbone_name, device=args.device)

        with out_path.open("a") as fh:
            for seq in seq_dirs:
                if (seq.name, backbone_name) in done:
                    continue
                try:
                    video = seq.load_frames()
                except Exception as e:
                    print(f"SKIP {seq.name}: {e}")
                    continue
                clip = backbone.encode_clip(video)
                rng = np.random.default_rng(0)
                pts, labels = seq.sample_query_points(args.k_per_part, rng)
                base = seq.transported_gt_trajectories(pts, labels)
                high, _ = inject_switch(base, labels, seq, args.p_high, np.random.default_rng(1))
                gt_idx = seq.gt_frame_indices
                label = joint_open_label(seq)

                # pool each point's own trajectory, then average the resulting
                # *feature vectors* across points (averaging pixel coordinates
                # first would land on meaningless positions between parts).
                low_feat = torch.stack(
                    [pool_trajectory(clip, base[i], gt_idx) for i in range(len(base))]
                ).mean(0)   # (G, C) low-ISR
                high_feat = torch.stack(
                    [pool_trajectory(clip, high[i], gt_idx) for i in range(len(high))]
                ).mean(0)   # (G, C) high-ISR (APE-matched switch)

                cos = torch.nn.functional.cosine_similarity(
                    low_feat.flatten(), high_feat.flatten(), dim=0
                ).item()
                fd = (low_feat - high_feat).norm(dim=-1).mean().item()

                row = dict(
                    sequence=seq.name,
                    backbone=backbone_name,
                    label=label,
                    cosine_low_high=cos,
                    feature_distance=fd,
                    low_feat=low_feat.mean(0).tolist(),
                    high_feat=high_feat.mean(0).tolist(),
                )
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(f"{seq.name} {backbone_name}: cos={cos:.4f} fd={fd:.3f} label={label}")

        del backbone
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
