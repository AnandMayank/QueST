# QueST

Official implementation of *Identity Matters: Identity-Aware Query-Based
Point Tracking under Ambiguity*, submitted to NeurIPS 2026.

This repo has two parts, deliberately separated:

## 1. Evidence that identity switch rate (ISR) matters as a metric — `benchmark/`, `downstream_causal/`
Independent of any specific tracker, this validates that ISR carries real,
statistically significant downstream signal (segmentation IoU, ARI,
boundary F1, articulated consistency), evaluated entirely on standard
off-the-shelf trackers (CoTracker2/3, AllTracker, DenseTrack2D, BootsTAPIR):
- `benchmark/isr_evaluation.py`, `benchmark/ambiguity_eval.py` — the ISR /
  ISR-AUC metric (paper Eq. 10) and its symmetry/occlusion/articulation
  breakdown (Tables 4/9).
- `benchmark/motion_segmentation.py`, `benchmark/metrics/` — the downstream
  motion-segmentation and articulated-consistency pipeline (Tables 10/11).
- `downstream_causal/` — a follow-on causal validation package: does ISR
  predict downstream quality beyond geometric metrics (E1), do identity
  switches *cause* downstream degradation at matched geometric error (E2),
  does that hold for real trackers' natural mistakes when repaired with an
  automatic corrector (E4)? See `downstream_causal/results/CONSOLIDATED_REPORT.md`
  for full results and honestly-reported limitations.

## 2. The QueST model — `quest/`
The identity-aware tracker architecture itself (Section 3 of the paper).
See `quest/README.md` for the file-by-file breakdown and how it plugs into
a `facebookresearch/co-tracker` checkout (not vendored here — see below).

## Also in this repo
- `dataset_prep/` — how the PartNet-Mobility → SAPIEN synthetic benchmark
  data was generated (code only; PartNet-Mobility/SAPIEN assets themselves
  are not redistributed and must be obtained under their own license).
- `interactive_repair/` — SEMAPHORE, a human-in-the-loop point-tracking
  repair platform (napari GUI), extended with an IRE (Identity Recovery
  Efficiency) metric and an automatic corrector built on
  [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) (CC BY-NC-SA
  4.0, not vendored — see `interactive_repair/nninteractive_corrector.py`).
- `worldmodel_eval/` — a probe testing whether identity preservation
  determines the downstream predictive utility of frozen video-backbone
  representations (V-JEPA2, DINOv2).

## What's not in this repo
Trained checkpoints, PartNet-Mobility/SAPIEN mesh assets, and nnInteractive
model weights are not committed here (size and third-party licensing). Code
in `quest/` and `dataset_prep/` depends on the upstream
`facebookresearch/co-tracker` package, which is also not vendored — each
subfolder's README states exactly what external checkout/weights it expects.

## License
This repo's own code is released under CC BY-NC 4.0, matching the license of
`facebookresearch/co-tracker` that `quest/` and `benchmark/` build on. See
`LICENSE` and `NOTICE` for third-party attributions.
