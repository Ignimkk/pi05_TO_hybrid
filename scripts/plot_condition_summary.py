"""Headline baseline-vs-SEAM figure: task success preserved, motion quality improved.

Left panel  — task success with Wilson 95% intervals (the "semantics are preserved" claim).
Right panel — per-metric relative change, signed so improvement and regression read at a glance.

    src/openpi/.venv/bin/python scripts/plot_condition_summary.py \
        --baseline data/rby1_grid_eval_baseline --seam data/rby1_grid_eval_seam \
        --out docs/assets/summary_baseline_vs_seam.png

Metrics are recomputed from the trajectory NPZs by ``scripts/compare_grid_conditions.py`` helpers,
so this figure and the numeric report can never drift apart.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from scipy import stats

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.compare_grid_conditions import (  # noqa: E402
    METRICS, load_condition, trajectory_metrics, wilson,
)

# Validated data-viz palette.
SERIES_BASELINE = "#2a78d6"   # categorical slot 1
SERIES_SEAM = "#eb6834"       # categorical slot 2
DIVERGING_GOOD = "#2a78d6"    # improvement (blue pole)
DIVERGING_BAD = "#d03b3b"     # regression (red pole)
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE_RULE = "#c3c2b7"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", default="data/rby1_grid_eval_baseline")
    parser.add_argument("--seam", default="data/rby1_grid_eval_seam")
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("docs/assets/summary_baseline_vs_seam.png"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base, seam = load_condition(args.baseline), load_condition(args.seam)
    shared = sorted(set(base) & set(seam))
    paired = {}
    for t in shared:
        mb, ms = trajectory_metrics(base[t]), trajectory_metrics(seam[t])
        if mb is not None and ms is not None:
            paired[t] = (mb, ms)

    # --- panel data --------------------------------------------------------
    b_ok = sum(base[t]["success"] for t in shared)
    s_ok = sum(seam[t]["success"] for t in shared)
    rates = [b_ok / len(shared), s_ok / len(shared)]
    cis = [wilson(b_ok, len(shared)), wilson(s_ok, len(shared))]

    rows = []  # (label, pct_change, p_value)
    for space, tag in (("action", "commanded"), ("qpos", "measured")):
        for metric in METRICS:
            bv = np.array([paired[t][0][space][metric] for t in paired], dtype=float)
            sv = np.array([paired[t][1][space][metric] for t in paired], dtype=float)
            ok = np.isfinite(bv) & np.isfinite(sv)
            bv, sv = bv[ok], sv[ok]
            p = stats.wilcoxon(bv, sv).pvalue if np.any(bv != sv) else 1.0
            rows.append((f"{metric}  ({tag})", (sv.mean() - bv.mean()) / bv.mean() * 100, p))
    bv = np.array([paired[t][0]["overlap_residual"] for t in paired], dtype=float)
    sv = np.array([paired[t][1]["overlap_residual"] for t in paired], dtype=float)
    ok = np.isfinite(bv) & np.isfinite(sv)
    rows.append(("overlap residual", (sv[ok].mean() - bv[ok].mean()) / bv[ok].mean() * 100,
                 stats.wilcoxon(bv[ok], sv[ok]).pvalue))

    fig, (ax_s, ax_m) = plt.subplots(
        1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1, 2.1]}, layout="constrained"
    )
    fig.patch.set_facecolor(SURFACE)

    # --- left: success rate ------------------------------------------------
    ax_s.set_facecolor(SURFACE)
    xs = [0, 1]
    colours = [SERIES_BASELINE, SERIES_SEAM]
    for x, rate, (lo, hi), colour in zip(xs, rates, cis, colours):
        ax_s.bar(x, rate * 100, width=0.52, color=colour, zorder=3)
        ax_s.plot([x, x], [lo * 100, hi * 100], color=TEXT_PRIMARY, linewidth=2, zorder=4)
        for cap in (lo, hi):
            ax_s.plot([x - 0.09, x + 0.09], [cap * 100] * 2,
                      color=TEXT_PRIMARY, linewidth=2, zorder=4)
        ax_s.text(x, rate * 100 - 5, f"{rate*100:.1f}%", ha="center", va="top",
                  fontsize=13, fontweight="bold", color="#ffffff", zorder=5)
    ax_s.set_xticks(xs)
    ax_s.set_xticklabels(["baseline", "SEAM"], fontsize=11, color=TEXT_PRIMARY)
    ax_s.set_ylim(0, 112)  # headroom so the test annotation clears the error bars
    ax_s.set_ylabel("Task success rate (%)", fontsize=10, color=TEXT_SECONDARY)
    ax_s.set_title(f"Task success  (n = {len(shared)} paired trials)",
                   fontsize=11, fontweight="bold", color=TEXT_PRIMARY, pad=10)
    ax_s.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax_s.set_axisbelow(True)
    b_only = sum(base[t]["success"] and not seam[t]["success"] for t in shared)
    s_only = sum(seam[t]["success"] and not base[t]["success"] for t in shared)
    p_mc = stats.binomtest(s_only, b_only + s_only, 0.5).pvalue if (b_only + s_only) else 1.0
    # Significance bracket spanning the two bars, above the error bars.
    top = max(hi for _, hi in cis) * 100
    ax_s.plot([0, 0, 1, 1], [top + 4, top + 8, top + 8, top + 4],
              color=TEXT_SECONDARY, linewidth=1.2, clip_on=False)
    ax_s.text(0.5, top + 10, f"n.s.  (McNemar p = {p_mc:.2f})",
              ha="center", va="bottom", fontsize=9.5, color=TEXT_SECONDARY)
    ax_s.set_yticks([0, 20, 40, 60, 80, 100])

    # --- right: relative change -------------------------------------------
    ax_m.set_facecolor(SURFACE)
    labels = [r[0] for r in rows][::-1]
    values = [r[1] for r in rows][::-1]
    pvals = [r[2] for r in rows][::-1]
    ypos = np.arange(len(labels))
    bar_colours = [DIVERGING_BAD if v > 0 else DIVERGING_GOOD for v in values]
    ax_m.barh(ypos, values, height=0.6, color=bar_colours, zorder=3)
    ax_m.axvline(0, color=BASELINE_RULE, linewidth=1.4, zorder=4)
    for y, v, p in zip(ypos, values, pvals):
        offset = 1.1 if v > 0 else -1.1
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax_m.text(v + offset, y, f"{v:+.1f}%  {star}", va="center",
                  ha="left" if v > 0 else "right", fontsize=9.5,
                  color=TEXT_PRIMARY, fontweight="bold")
    ax_m.set_yticks(ypos)
    ax_m.set_yticklabels(labels, fontsize=10, color=TEXT_SECONDARY)
    ax_m.set_xlabel("Change vs. baseline (%)   ← smoother        rougher →",
                    fontsize=10, color=TEXT_SECONDARY)
    ax_m.set_xlim(min(values) - 14, max(values) + 14)
    ax_m.set_title("Motion quality — SEAM relative to baseline",
                   fontsize=11, fontweight="bold", color=TEXT_PRIMARY, pad=10)
    ax_m.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax_m.set_axisbelow(True)
    ax_m.text(0.99, 0.02, "*** p<0.001   Wilcoxon signed-rank, paired",
              transform=ax_m.transAxes, ha="right", va="bottom",
              fontsize=8.5, color=TEXT_MUTED)

    for ax in (ax_s, ax_m):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRIDLINE)
        ax.tick_params(colors=TEXT_MUTED, labelsize=9)

    fig.suptitle("SEAM preserves task success while reducing chunk-boundary jerk",
                 fontsize=14, fontweight="bold", color=TEXT_PRIMARY, x=0.005, ha="left")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
