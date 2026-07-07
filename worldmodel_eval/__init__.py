"""E3: does identity preservation determine the downstream predictive utility
of trajectory-conditioned world-model representations?

Backbones expose per-clip spatial patch tokens; probe.py bilinearly pools
tokens along a 2D pixel trajectory and compares low-ISR vs high-ISR
conditioning on (a) latent similarity and (b) a downstream verifiable-state
predictor. Run only after downstream_causal's E1/E2 trajectories exist.
"""
