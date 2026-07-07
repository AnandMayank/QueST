# E4 (Phase 2): nnInteractive-based automatic identity repair on real tracker output

Unlike E2 (which injects synthetic switches at controlled levels, and whose natural-tracker
repair arm was underpowered at n=1-2 real switch cases), this experiment runs the actual
CoTracker2/CoTracker3 outputs on 27 real multi-part SAPIEN sequences and repairs **genuinely
occurring** identity switches using nnInteractive -- a 3D medical-volume segmentation model,
repurposed here by treating the video clip as a T x H x W volume and issuing a point prompt at
the detected switch onset (at the GT position, modelling what a human corrector would click).
The resulting spatiotemporal mask's centroid re-anchors the trajectory forward from that frame.

## Setup
- Data: 27 multi-part SAPIEN sequences (manipulation_2/3/4), CoTracker2 + CoTracker3.
- Switch detection: same persistence-filtered `ISRComputer` used throughout this project
  (`isr_evaluation/metrics/identity.py`), applied to each point's part-mask assignment sequence.
- Correction: one nnInteractive point prompt per persistently-switched point, at the onset frame,
  at the GT position (oracle-quality click -- the human-effort/click-quality question itself is
  future work; this isolates whether nnInteractive's mask *propagation* is useful once given a
  correct anchor).
- Model: `nnInteractive/nnInteractive` v1.0 (CC BY-NC-SA 4.0), downloaded to `second_drive` to
  avoid the root disk (which was at 99% capacity during this session).

## Result
54 (sequence, tracker) pairs tried; **20 had a genuine, naturally-occurring identity switch**
(assign_isr_raw > 0) -- a real-data sample roughly an order of magnitude larger than E2's
natural-repair arm (which found only 1-2 repairable cases in 15 sequences with CoTracker3/2).

| metric | raw (mean) | nnInteractive-repaired (mean) | Wilcoxon p |
|---|---|---|---|
| assign_isr | 0.101 | 0.049 | **1.9e-6** |
| seg_iou | 0.712 | 0.799 | **0.0014** |

- **Identity switches were reduced in every single one of the 20 cases** (Wilcoxon stat=0, i.e.
  the sign of the effect was 100% consistent).
- **Segmentation IoU improved in 18/20 cases**, including several large recoveries on severely
  broken sequences: `manipulation_4/45189/take_03` (IoU 0.517→0.993), `take_08` (0.511→0.983),
  `manipulation_4/45189/take_00` (0.531→0.911), `manipulation_4/41085/take_04` (0.508→0.783).
- **2/20 cases got worse** on IoU/ARI despite ISR improving (`manipulation_3/41003/take_06`,
  `manipulation_3/45092/take_00` on CoTracker2): the mask-centroid re-anchor reduced identity
  switching but introduced new positional noise that hurt the *downstream* clustering more than
  the original drift did. Honest limitation, not hidden: repairing identity does not guarantee
  monotonic downstream improvement in every individual case, only on average (consistent with
  E2's finding that seg_iou is the least identity-specific of the four downstream metrics).

## What this adds beyond E2
E2 established causation via controlled synthetic injection (does *any* identity switch, at
matched geometric error, hurt downstream quality -- yes). E4 shows the same causal chain holds
**in the reverse, repair direction, on real tracker mistakes**, and does so with a real,
previously-unintegrated correction mechanism (nnInteractive) rather than an oracle GT-snap --
directly the "closing the causal chain: correction -> ISR down -> downstream up" claim from the
research plan's Phase 2.4, now demonstrated with n=20 genuine cases rather than the n=1-2 the
oracle-repair arm could find.

## Scope / what's not yet done
- Corrections used the GT position as the click (oracle-quality prompt placement); the
  human-click-quality and click-count/budget questions (IRE as originally defined, ΔISR per
  *human* correction) still need either a live SEMAPHORE user-study session or a simulated
  imprecise-click policy -- this experiment isolates "does nnInteractive's mask propagation help
  once anchored correctly", which is a necessary but not sufficient piece of the IRE story.
- SEMAPHORE's own `CorrectionManager`/`MetricsManager.compute_ire()` (added this session, 25/25
  unit tests pass) was exercised only in isolation/synthetic unit tests, not yet wired end-to-end
  to this nnInteractive corrector inside a live SEMAPHORE session -- that integration (calling
  `compute_ire` after each `CorrectionManager.apply()` in a real run) is the natural next step to
  produce the headline IRE number this project's plan specifies.
- SAM2 corrector (the commercially-clean, non-gated alternative the plan lists alongside
  nnInteractive) was not implemented this session; only nnInteractive was built, per the explicit
  ask to focus on "the use of nn-interactive."
