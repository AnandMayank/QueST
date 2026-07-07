# QueST: Identity-Aware Query-Based Point Tracking

Core model code for QueST (Section 3 of the paper), extracted from a fork of
[`facebookresearch/co-tracker`](https://github.com/facebookresearch/co-tracker).
These files implement the identity-aware additions described in the paper;
they are **not** a standalone package — they depend on co-tracker's base
CoTracker3 modules (`cotracker3_offline.py`, `model_utils.py`, `blocks.py`,
`embeddings.py`, the base `cotracker.py`) which are upstream code, not part
of this contribution, and are not vendored here. To run this code, drop
these files into the corresponding paths inside a
`facebookresearch/co-tracker` checkout:

```
cotracker/models/core/quest.py
cotracker/models/core/identity_head.py
cotracker/models/identity_aware_predictor.py
cotracker/models/core/cotracker/cotracker3_offline_identity_aware.py
cotracker/models/core/cotracker/hierarchical_blocks.py
init_hierarchical_queries.py                          # repo root
```

## Files

| File | Corresponds to (paper) | Description |
|---|---|---|
| `quest.py` | Sec. 3.3, Eq. 1–2 | `DualPathQuery`: persistent identity query `q_id` + adaptive appearance query `q_app`, fused via the confidence-adaptive gate (Eq. 1) and smoothed recurrent state (Eq. 2). Also implements the prototype-codebook soft retrieval described in Sec. 3.3. |
| `identity_head.py` | Sec. 3.5, Eq. 8 | `IdentityMatchingHead`: projects correlation-volume features into identity embedding space; used to compute the positive/hard-negative terms of the identity loss `L_id`. |
| `cotracker3_offline_identity_aware.py` | Sec. 3, full pipeline | Wires `DualPathQuery` + `IdentityMatchingHead` into the CoTracker3 offline architecture; this is the concrete `Φ` + trajectory-decoding pipeline of Sec. 3.4, with the ICQE fusion inserted into the per-iteration update. |
| `identity_aware_predictor.py` | — | Thin inference-time wrapper (checkpoint loading, pre/post-processing) around the model above, mirroring `cotracker.predictor.CoTrackerPredictor`'s interface. |
| `hierarchical_blocks.py` | — (QueST-H variant, not in the main paper) | Parent/child cross-attention blocks for an exploratory hierarchical extension (e.g. coupled articulated parts). Included for completeness; not part of the main-paper results. |
| `init_hierarchical_queries.py` | — | Initializes the QueST-H hierarchy's query set from PartNet-Mobility affordance annotations (part parent/child relations). Also exploratory, not part of the main-paper results. |

## Status

The architecture above is implemented and runs; it is the model referred to
throughout the paper as QueST. The paper's Table 2 identity-switch-rate
numbers are the authoritative reported results — we do not repeat or restate
performance claims here to avoid any risk of drift between this README and
the paper. Training/evaluation is still an active area of the project; see
the paper's Limitations section (Appendix J) for an honest account of scope.

For the evaluation pipeline that produces ISR/ISR-AUC/downstream metrics
(independent of which tracker is being scored), see `../benchmark/`. For
evidence that the ISR metric itself carries real, statistically significant
signal — evaluated entirely on standard off-the-shelf trackers, not QueST —
see `../downstream_causal/`.
