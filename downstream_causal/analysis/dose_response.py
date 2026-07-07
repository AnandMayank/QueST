"""E2 analysis: dose-response of injected identity switches on downstream
segmentation, at matched geometric error (switch vs drift arms), plus the
natural-tracker repair arm.

Usage:
    python -m downstream_causal.analysis.dose_response \
        --injection downstream_causal/results/e2_injection.jsonl \
        --repair downstream_causal/results/e2_repair.jsonl \
        --out downstream_causal/results/e2_report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.DataFrame(json.loads(l) for l in p.read_text().splitlines() if l.strip())


def manipulation_check(df: pd.DataFrame) -> str:
    """APE-vs-base must be flat across injected levels within each arm."""
    lines = ["#### Manipulation check: ape_vs_base by arm and level (must be ~equal within a level)\n"]
    piv = df.pivot_table(index="p_injected", columns="arm", values="ape_vs_base", aggfunc="mean")
    lines.append(piv.to_markdown())
    return "\n".join(lines) + "\n"


def dose_response(df: pd.DataFrame, metric: str) -> str:
    lines = [f"#### Dose-response on {metric} (switch arm; mean over sequences x seeds)\n"]
    sw = df[df["arm"] == "switch"]
    levels = sorted(sw["p_injected"].unique())
    means = sw.groupby("p_injected")[metric].mean()
    lines.append("| p_injected | mean " + metric + " |")
    lines.append("|---|---|")
    for lv in levels:
        lines.append(f"| {lv} | {means[lv]:.3f} |")

    # paired Wilcoxon between adjacent levels (paired by sequence x seed)
    lines.append("\nPaired Wilcoxon (adjacent levels, paired by sequence+seed):\n")
    lines.append("| level_a vs level_b | W | p |")
    lines.append("|---|---|---|")
    for a, b in zip(levels[:-1], levels[1:]):
        da = sw[sw["p_injected"] == a].set_index(["sequence", "seed"])[metric]
        db = sw[sw["p_injected"] == b].set_index(["sequence", "seed"])[metric]
        common = da.index.intersection(db.index)
        if len(common) < 3:
            continue
        w, p = stats.wilcoxon(da.loc[common], db.loc[common])
        lines.append(f"| {a} vs {b} | {w:.1f} | {p:.3g} |")
    return "\n".join(lines) + "\n"


def switch_vs_drift(df: pd.DataFrame, metric: str) -> str:
    lines = [f"#### Switch vs drift (APE-matched) on {metric}, at each injected level\n"]
    lines.append("| p_injected | switch mean | drift mean | Wilcoxon p |")
    lines.append("|---|---|---|---|")
    for p_lv in sorted(df[df["p_injected"] > 0]["p_injected"].unique()):
        sub = df[df["p_injected"] == p_lv]
        sw = sub[sub["arm"] == "switch"].set_index(["sequence", "seed"])[metric]
        dr = sub[sub["arm"] == "drift"].set_index(["sequence", "seed"])[metric]
        common = sw.index.intersection(dr.index)
        if len(common) < 3:
            continue
        w, pval = stats.wilcoxon(sw.loc[common], dr.loc[common])
        lines.append(f"| {p_lv} | {sw.loc[common].mean():.3f} | {dr.loc[common].mean():.3f} | {pval:.3g} |")
    return "\n".join(lines) + "\n"


def repair_effect(df: pd.DataFrame, metric: str) -> str:
    if df.empty:
        return "(no repair-arm data)\n"
    lines = [f"#### Natural-tracker repair effect on {metric} (paired raw vs repaired)\n"]
    lines.append("| tracker | raw mean | repaired mean | Wilcoxon p |")
    lines.append("|---|---|---|---|")
    for tracker in df["tracker"].unique():
        sub = df[df["tracker"] == tracker]
        raw = sub[sub["arm"] == "raw"].set_index("sequence")[metric]
        rep = sub[sub["arm"] == "repaired"].set_index("sequence")[metric]
        common = raw.index.intersection(rep.index)
        if len(common) < 3:
            continue
        w, pval = stats.wilcoxon(raw.loc[common], rep.loc[common])
        lines.append(f"| {tracker} | {raw.loc[common].mean():.3f} | {rep.loc[common].mean():.3f} | {pval:.3g} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--injection", required=True)
    ap.add_argument("--repair", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    inj = load(args.injection)
    rep = load(args.repair)

    report = ["# E2: switch-injection dose-response and repair report\n"]
    if not inj.empty:
        report.append(manipulation_check(inj))
        for metric in ["seg_iou", "ari", "boundary_f1", "articulated_consistency"]:
            report.append(dose_response(inj, metric))
            report.append(switch_vs_drift(inj, metric))
    if not rep.empty:
        for metric in ["seg_iou", "ari", "boundary_f1"]:
            report.append(repair_effect(rep, metric))

    Path(args.out).write_text("\n".join(report))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
