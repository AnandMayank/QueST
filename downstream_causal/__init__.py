"""Downstream-causal evaluation of identity preservation (E1/E2 of the
minimal experiment program).

E1: does ISR carry downstream signal beyond geometric metrics? (run_matrix.py
    + analysis/correlate.py)
E2: do identity switches cause downstream degradation at matched geometric
    error? (interventions/)

Reuses ISR/persistence logic from isr_evaluation and evaluates only standard
pretrained trackers (CoTracker2/3, TAPIR).
"""
