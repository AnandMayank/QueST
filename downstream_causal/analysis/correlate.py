"""E1 analysis: does ISR carry downstream signal beyond geometric metrics?

Reads downstream_causal/results/matrix.jsonl (one row per sequence x tracker
x variant) and reports, for each downstream outcome:
  - Pearson/Spearman correlation of each predictor
  - hierarchical regression: outcome ~ geometric metrics, then + ISR -> ΔR²
    with an F-test on the nested model comparison
  - cluster-bootstrap CIs on ΔR² (resampling sequences, since trackers/variants
    share sequences and are not independent observations)
  - same breakdown restricted to high-ambiguity sequences (manipulation_3/4,
    which have more active joints -> more symmetry/occlusion opportunities)

Usage:
    python -m downstream_causal.analysis.correlate \
        --matrix downstream_causal/results/matrix.jsonl \
        --out downstream_causal/results/e1_report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

GEOMETRIC_PREDICTORS = ["ape_px", "oa", "drift_at_100"]
ISR_PREDICTORS = ["isr_tau_mid", "isr_auc_pct", "assign_isr"]
OUTCOMES = ["seg_iou", "ari", "boundary_f1", "articulated_consistency"]

N_BOOT = 2000


def load_matrix(path: str) -> pd.DataFrame:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    return df.dropna(subset=OUTCOMES + GEOMETRIC_PREDICTORS + ISR_PREDICTORS)


def r2(y: np.ndarray, X: np.ndarray) -> float:
    if X.shape[1] == 0:
        return 0.0
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def nested_f_test(y: np.ndarray, X_base: np.ndarray, X_full: np.ndarray) -> Dict[str, float]:
    n = len(y)
    p_base, p_full = X_base.shape[1], X_full.shape[1]
    r2_base, r2_full = r2(y, X_base), r2(y, X_full)
    delta = r2_full - r2_base
    df1, df2 = p_full - p_base, n - p_full - 1
    if df1 <= 0 or df2 <= 0 or (1 - r2_full) <= 0:
        f_stat, p_val = float("nan"), float("nan")
    else:
        f_stat = ((r2_full - r2_base) / df1) / ((1 - r2_full) / df2)
        p_val = float(1 - stats.f.cdf(f_stat, df1, df2))
    return {"r2_base": r2_base, "r2_full": r2_full, "delta_r2": delta, "f": f_stat, "p": p_val}


def cluster_bootstrap_delta_r2(
    df: pd.DataFrame, outcome: str, seed: int = 0, n_boot: int = N_BOOT
) -> np.ndarray:
    """Resample whole sequences (with replacement) to respect the fact that
    multiple tracker/variant rows share a sequence."""
    rng = np.random.default_rng(seed)
    seqs = df["sequence"].unique()
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.choice(seqs, size=len(seqs), replace=True)
        sub = pd.concat([df[df["sequence"] == s] for s in chosen], ignore_index=True)
        y = sub[outcome].to_numpy()
        X_base = sub[GEOMETRIC_PREDICTORS].to_numpy()
        X_full = sub[GEOMETRIC_PREDICTORS + ["isr_tau_mid"]].to_numpy()
        deltas[b] = r2(y, X_full) - r2(y, X_base)
    return deltas


def holm_correction(pvals: List[float]) -> List[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min((m - rank) * pvals[idx], 1.0)
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj.tolist()


def analyze(df: pd.DataFrame, label: str) -> str:
    lines = [f"### {label} (n={len(df)} rows, {df['sequence'].nunique()} sequences)\n"]
    lines.append("| outcome | Pearson(ISR) | Spearman(ISR) | R2(geom) | R2(geom+ISR) | ΔR2 | ΔR2 95% CI | p(F-test) |")
    lines.append("|---|---|---|---|---|---|---|---|")

    pvals = []
    rows_for_holm = []
    for outcome in OUTCOMES:
        if df[outcome].nunique() < 2:
            continue
        y = df[outcome].to_numpy()
        isr = df["isr_tau_mid"].to_numpy()
        pear = stats.pearsonr(isr, y)
        spear = stats.spearmanr(isr, y)

        X_base = df[GEOMETRIC_PREDICTORS].to_numpy()
        X_full = df[GEOMETRIC_PREDICTORS + ["isr_tau_mid"]].to_numpy()
        ftest = nested_f_test(y, X_base, X_full)
        boot = cluster_bootstrap_delta_r2(df, outcome)
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

        pvals.append(ftest["p"])
        rows_for_holm.append(
            (outcome, pear, spear, ftest, ci_lo, ci_hi)
        )

    adj_p = holm_correction([r[3]["p"] for r in rows_for_holm]) if rows_for_holm else []
    for (outcome, pear, spear, ftest, ci_lo, ci_hi), p_adj in zip(rows_for_holm, adj_p):
        lines.append(
            f"| {outcome} | {pear.statistic:.3f} (p={pear.pvalue:.3g}) | "
            f"{spear.statistic:.3f} (p={spear.pvalue:.3g}) | {ftest['r2_base']:.3f} | "
            f"{ftest['r2_full']:.3f} | {ftest['delta_r2']:.3f} | "
            f"[{ci_lo:.3f}, {ci_hi:.3f}] | {p_adj:.3g} |"
        )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ambiguity-levels", nargs="+", default=["manipulation_3", "manipulation_4"])
    args = ap.parse_args()

    df = load_matrix(args.matrix)
    report = ["# E1: ISR variance-decomposition report\n"]
    report.append(analyze(df, "All sequences"))

    amb = df[df["manipulation_level"].isin(args.ambiguity_levels)]
    if len(amb) > 5:
        report.append(analyze(amb, f"Ambiguity subset ({', '.join(args.ambiguity_levels)})"))

    Path(args.out).write_text("\n".join(report))
    print(f"wrote {args.out}")
    print("\n".join(report))


if __name__ == "__main__":
    main()
