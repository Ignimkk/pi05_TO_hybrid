"""Jerk-focused baseline-vs-SEAM figures for the RB-Y1 grid evaluation.

Two figures, both aggregated over every paired trial:

``--out-profile`` — **chunk-boundary-aligned profiles.** Every chunk boundary in every trial is
    overlaid on a common axis (step 0 = the boundary) and averaged, for per-step jerk (2nd
    difference) and per-step displacement (1st difference), in commanded and measured space. This
    is what makes the boundary artifact visible: the baseline spike at step 0 and its removal.
    Single-trial plots cannot show this because baseline and SEAM episodes differ in length.

``--out-metrics`` — **aggregate metric bars** with 95% CI, one small multiple per metric.

    src/openpi/.venv/bin/python scripts/plot_jerk_comparison.py \
        --baseline data/rby1_grid_eval_baseline --seam data/rby1_grid_eval_seam

Arm joints only (grippers excluded); jerk is the SEAM paper's discrete 2nd difference of the
absolute joint target, not SI rad/s^3.
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
from benchmark.seam_vla.metrics.motion import compute_motion_metrics  # noqa: E402
from scripts.compare_grid_conditions import load_condition  # noqa: E402

ARM_DIMS = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)
METRICS = ("BJ", "IJ", "CD", "AVb")
METRIC_LABEL = {
    "BJ": "BJ — boundary jerk",
    "IJ": "IJ — interior jerk",
    "CD": "CD — boundary discontinuity",
    "AVb": "AVb — boundary jerk variance",
}
SPACES = (("action", "executed_actions", "commanded"),
          ("qpos", "measured_qpos", "measured"))

BASELINE_C = "#2a78d6"
SEAM_C = "#eb6834"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
RULE = "#c3c2b7"


def per_step_series(path, key):
    """Return (jerk[t], displacement[t], K) with index i corresponding to absolute step i+1."""
    data = np.load(path, allow_pickle=True)
    series = np.asarray(data[key], dtype=np.float64)[:, ARM_DIMS]
    if series.shape[0] < 3:
        return None
    jerk = np.linalg.norm(series[2:] - 2.0 * series[1:-1] + series[:-2], axis=-1)
    disp = np.linalg.norm(np.diff(series, axis=0), axis=-1)
    return jerk, disp, int(data["execution_length"])


def boundary_profile(records, key, half_window):
    """Average every chunk boundary onto a common axis.

    Returns (offsets, jerk_mean, jerk_sem, disp_mean, disp_sem) where offset 0 is the boundary.
    """
    offsets = np.arange(-half_window, half_window + 1)
    jerk_acc = [[] for _ in offsets]
    disp_acc = [[] for _ in offsets]
    for record in records:
        got = per_step_series(record["trajectory"], key)
        if got is None:
            continue
        jerk, disp, K = got
        # jerk index i <-> absolute step i+1 ; disp index i <-> absolute step i+1
        for boundary in range(K, len(disp), K):
            for slot, offset in enumerate(offsets):
                t = boundary + offset
                if 1 <= t <= len(jerk):
                    jerk_acc[slot].append(jerk[t - 1])
                if 1 <= t <= len(disp):
                    disp_acc[slot].append(disp[t - 1])

    def summarise(acc):
        mean = np.array([np.mean(v) if v else np.nan for v in acc])
        sem = np.array([stats.sem(v) if len(v) > 1 else np.nan for v in acc])
        return mean, sem

    jm, js = summarise(jerk_acc)
    dm, ds = summarise(disp_acc)
    return offsets, jm, js, dm, ds


def metric_table(base, seam, shared):
    """{(space, metric): (baseline_values, seam_values)} recomputed from the NPZs."""
    out = {(sp, m): ([], []) for sp, _, _ in SPACES for m in METRICS}
    for trial in shared:
        for space, key, _ in SPACES:
            vals = []
            for src in (base, seam):
                data = np.load(src[trial]["trajectory"], allow_pickle=True)
                series = np.asarray(data[key], dtype=np.float64)[:, ARM_DIMS]
                vals.append(compute_motion_metrics(series, int(data["execution_length"])))
            for metric in METRICS:
                mb = vals[0]["paper_avb" if metric == "AVb" else metric]
                ms = vals[1]["paper_avb" if metric == "AVb" else metric]
                if np.isfinite(mb) and np.isfinite(ms):
                    out[(space, metric)][0].append(mb)
                    out[(space, metric)][1].append(ms)
    return {k: (np.array(a), np.array(b)) for k, (a, b) in out.items()}


def style(ax):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRIDLINE)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", default="data/rby1_grid_eval_baseline")
    parser.add_argument("--seam", default="data/rby1_grid_eval_seam")
    parser.add_argument("--out-profile", type=pathlib.Path,
                        default=pathlib.Path("docs/assets/jerk_boundary_profile.png"))
    parser.add_argument("--out-metrics", type=pathlib.Path,
                        default=pathlib.Path("docs/assets/jerk_metrics_bars.png"))
    parser.add_argument("--out-table", type=pathlib.Path,
                        default=pathlib.Path("docs/assets/jerk_table.md"))
    parser.add_argument("--half-window", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base, seam = load_condition(args.baseline), load_condition(args.seam)
    shared = sorted(set(base) & set(seam))
    b_recs = [base[t] for t in shared]
    s_recs = [seam[t] for t in shared]

    # ---------------- figure 1: boundary-aligned profiles -------------------
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.2), layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    for col, (space, key, tag) in enumerate(SPACES):
        off, bj, bjs, bd, bds = boundary_profile(b_recs, key, args.half_window)
        _, sj, sjs, sd, sds = boundary_profile(s_recs, key, args.half_window)
        for row, (bm, bs, sm, ss, name) in enumerate((
            (bj, bjs, sj, sjs, "per-step jerk  (2nd difference)"),
            (bd, bds, sd, sds, "per-step displacement  (1st difference)"),
        )):
            ax = axes[row][col]
            style(ax)
            ax.axvline(0, color=RULE, linewidth=1.4, linestyle=(0, (4, 3)), zorder=1)
            for mean, sem, colour, label in ((bm, bs, BASELINE_C, "baseline"),
                                             (sm, ss, SEAM_C, "SEAM")):
                ax.fill_between(off, mean - sem, mean + sem, color=colour, alpha=0.18, zorder=2)
                ax.plot(off, mean, color=colour, linewidth=2, zorder=3, label=label)
            ax.set_title(f"{name} — {tag}", fontsize=10,
                         color=TEXT_PRIMARY, fontweight="bold", pad=6)
            ax.grid(color=GRIDLINE, linewidth=0.7)
            ax.set_xticks(range(-args.half_window, args.half_window + 1, 2))
            if row == 1:
                ax.set_xlabel("step relative to chunk boundary", fontsize=9, color=TEXT_SECONDARY)
            if col == 0:
                ax.set_ylabel("mean ‖·‖  (± SE)", fontsize=9, color=TEXT_SECONDARY)
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
    fig.suptitle("Chunk-boundary-aligned motion profile  "
                 f"(all {len(shared)} paired trials, boundary at step 0)",
                 fontsize=13, fontweight="bold", color=TEXT_PRIMARY, x=0.005, ha="left")
    args.out_profile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_profile, dpi=args.dpi, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {args.out_profile}")

    # ---------------- figure 2 + table: aggregate metrics -------------------
    table = metric_table(base, seam, shared)
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 6.4), layout="constrained")
    fig.patch.set_facecolor(SURFACE)
    lines = ["# Jerk metrics — baseline vs SEAM",
             "",
             f"Recomputed from {len(shared)} paired trajectory NPZs, 12 arm joints. "
             "Wilcoxon signed-rank, paired.",
             ""]
    for row, (space, _, tag) in enumerate(SPACES):
        lines += [f"## {space} space ({tag})", "",
                  "| metric | baseline | SEAM | change | p |", "|---|---:|---:|---:|---:|"]
        for col, metric in enumerate(METRICS):
            bv, sv = table[(space, metric)]
            ax = axes[row][col]
            style(ax)
            means = [bv.mean(), sv.mean()]
            errs = [1.96 * stats.sem(bv), 1.96 * stats.sem(sv)]
            ax.bar([0, 1], means, yerr=errs, width=0.55,
                   color=[BASELINE_C, SEAM_C], zorder=3,
                   error_kw={"ecolor": TEXT_PRIMARY, "elinewidth": 1.6, "capsize": 5})
            change = (sv.mean() - bv.mean()) / bv.mean() * 100 if bv.mean() else np.nan
            p = stats.wilcoxon(bv, sv).pvalue if np.any(bv != sv) else 1.0
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["baseline", "SEAM"], fontsize=8.5, color=TEXT_PRIMARY)
            ax.set_title(METRIC_LABEL[metric], fontsize=9.5,
                         color=TEXT_PRIMARY, fontweight="bold", pad=16)
            ax.grid(axis="y", color=GRIDLINE, linewidth=0.7)
            ax.set_ylim(0, max(means) * 1.42)
            ax.annotate(f"{change:+.1f}%  {star}",
                        xy=(0.5, 0.965), xycoords="axes fraction", ha="center", va="top",
                        fontsize=10, fontweight="bold",
                        color=("#d03b3b" if change > 0 else "#006300"))
            if col == 0:
                ax.set_ylabel(f"{tag}\n(arm-joint norm)", fontsize=9, color=TEXT_SECONDARY)
            lines.append(f"| {metric} | {bv.mean():.5f} ± {bv.std(ddof=1):.5f} | "
                         f"{sv.mean():.5f} ± {sv.std(ddof=1):.5f} | {change:+.1f}% | "
                         f"{'<0.001' if p < 0.001 else f'{p:.3f}'} |")
        lines.append("")
    fig.suptitle("Motion-quality metrics  (mean ± 95% CI over "
                 f"{len(shared)} paired trials)",
                 fontsize=13, fontweight="bold", color=TEXT_PRIMARY, x=0.005, ha="left")
    fig.savefig(args.out_metrics, dpi=args.dpi, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {args.out_metrics}")

    args.out_table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out_table}")


if __name__ == "__main__":
    main()
