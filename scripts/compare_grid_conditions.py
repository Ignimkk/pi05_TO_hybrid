"""Paired baseline-vs-SEAM analysis of an RBY1 grid evaluation.

Recomputes every motion metric from the saved trajectory NPZs (not from the summary written at
run time), pairs the two conditions by ``trial_id``, and reports the statistics a paper needs:
success with a Wilson interval, McNemar's exact test on the paired success outcomes, and
Wilcoxon signed-rank tests on the paired per-trial jerk/discontinuity metrics.

    src/openpi/.venv/bin/python scripts/compare_grid_conditions.py \
        --baseline data/rby1_grid_eval_baseline \
        --seam     data/rby1_grid_eval_seam \
        --out      docs/assets/comparison_summary.md

Metrics follow the SEAM paper (Eqs. 9-13) via ``benchmark/seam_vla/metrics/motion.py`` and are
computed on the 12 arm joints only; grippers are near-binary and would dominate the jerk norm.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from scipy import stats

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from benchmark.seam_vla.metrics.motion import compute_motion_metrics  # noqa: E402

ARM_DIMS = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)
METRICS = ("BJ", "IJ", "CD", "AVb")


def load_condition(directory):
    """Return {trial_id: record} for one condition directory."""
    path = pathlib.Path(directory) / "results.jsonl"
    out = {}
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") in ("setup_error", "interrupted"):
            continue
        out[record["trial_id"]] = record
    return out


def trajectory_metrics(record):
    """Recompute BJ/IJ/CD/AVb from the trial's NPZ, in both action and measured-qpos space.

    Also returns the overlap residual: the mean distance between the previous chunk's unexecuted
    tail and the new chunk's head at the same absolute timestep, which is what SEAM's guidance
    directly targets.
    """
    path = record.get("trajectory")
    if not path or not pathlib.Path(path).exists():
        return None
    data = np.load(path, allow_pickle=True)
    execution_length = int(data["execution_length"])
    out = {}
    for space, key in (("action", "executed_actions"), ("qpos", "measured_qpos")):
        series = np.asarray(data[key], dtype=np.float64)
        if series.ndim != 2 or series.shape[0] < 3:
            return None
        values = compute_motion_metrics(series[:, ARM_DIMS], execution_length)
        out[space] = {
            "BJ": values["BJ"], "IJ": values["IJ"],
            "CD": values["CD"], "AVb": values["paper_avb"],
        }

    chunks = np.asarray(data["predicted_chunks"], dtype=np.float64)  # [N, H, D]
    residuals = []
    for i in range(1, chunks.shape[0]):
        tail = chunks[i - 1][execution_length:]          # previous plan for the upcoming steps
        head = chunks[i][: tail.shape[0]]                # new plan for those same steps
        if tail.size:
            residuals.append(
                float(np.mean(np.linalg.norm(tail[:, ARM_DIMS] - head[:, ARM_DIMS], axis=-1)))
            )
    out["overlap_residual"] = float(np.mean(residuals)) if residuals else float("nan")
    out["num_chunks"] = int(chunks.shape[0])
    out["steps"] = int(np.asarray(data["executed_actions"]).shape[0])
    return out


def wilson(successes, total, z=1.96):
    """Wilson score interval — well behaved at proportions near 0 and 1, unlike the normal approx."""
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", default="data/rby1_grid_eval_baseline")
    parser.add_argument("--seam", default="data/rby1_grid_eval_seam")
    parser.add_argument("--out", type=pathlib.Path, default=None,
                        help="also write the report as markdown")
    parser.add_argument("--far-x", type=float, default=0.625,
                        help="x value treated as the far/limit column in the zone split")
    args = parser.parse_args()

    base, seam = load_condition(args.baseline), load_condition(args.seam)
    shared = sorted(set(base) & set(seam))
    lines = []

    def emit(text=""):
        print(text)
        lines.append(text)

    emit(f"# Baseline vs SEAM — RBY1 grid evaluation")
    emit()
    emit(f"- paired trials: **{len(shared)}** "
         f"(baseline {len(base)}, seam {len(seam)}, unmatched {len(set(base) ^ set(seam))})")
    fps = {r["grid_fingerprint"] for r in list(base.values()) + list(seam.values())}
    emit(f"- grid revision: `{'`, `'.join(sorted(fps))}`")
    emit()

    # ---- success -----------------------------------------------------------
    b_ok = sum(base[t]["success"] for t in shared)
    s_ok = sum(seam[t]["success"] for t in shared)
    emit("## 1. Task success")
    emit()
    emit("| condition | success | rate | 95% CI (Wilson) |")
    emit("|---|---:|---:|---|")
    for name, ok in (("baseline", b_ok), ("SEAM", s_ok)):
        lo, hi = wilson(ok, len(shared))
        emit(f"| {name} | {ok}/{len(shared)} | {ok/len(shared)*100:.1f}% | "
             f"[{lo*100:.1f}%, {hi*100:.1f}%] |")
    emit()

    # McNemar on the paired outcomes: only the discordant pairs carry information.
    b_only = sum(base[t]["success"] and not seam[t]["success"] for t in shared)
    s_only = sum(seam[t]["success"] and not base[t]["success"] for t in shared)
    both = sum(base[t]["success"] and seam[t]["success"] for t in shared)
    neither = sum(not base[t]["success"] and not seam[t]["success"] for t in shared)
    p_mcnemar = stats.binomtest(s_only, b_only + s_only, 0.5).pvalue if (b_only + s_only) else 1.0
    emit("Paired outcomes (same grid cell, same repeat index):")
    emit()
    emit("| | SEAM success | SEAM fail |")
    emit("|---|---:|---:|")
    emit(f"| **baseline success** | {both} | {b_only} |")
    emit(f"| **baseline fail** | {s_only} | {neither} |")
    emit()
    emit(f"McNemar exact test on the {b_only + s_only} discordant pairs: "
         f"**p = {fmt_p(p_mcnemar)}**")
    emit()

    # ---- zone split --------------------------------------------------------
    def is_far(t):
        return abs(base[t]["requested_xyz"][0] - args.far_x) < 1e-6
    emit(f"## 2. Success by workspace zone")
    emit()
    emit(f"| zone | n | baseline | SEAM | delta |")
    emit("|---|---:|---:|---:|---:|")
    for label, sel in (("near (x < %.3f)" % args.far_x, [t for t in shared if not is_far(t)]),
                       ("far  (x = %.3f)" % args.far_x, [t for t in shared if is_far(t)])):
        bo = sum(base[t]["success"] for t in sel)
        so = sum(seam[t]["success"] for t in sel)
        emit(f"| {label} | {len(sel)} | {bo}/{len(sel)} ({bo/len(sel)*100:.1f}%) | "
             f"{so}/{len(sel)} ({so/len(sel)*100:.1f}%) | {(so-bo)/len(sel)*100:+.1f} pp |")
    emit()

    # ---- motion metrics from the NPZs -------------------------------------
    emit("## 3. Motion quality (recomputed from trajectory NPZs, 12 arm joints)")
    emit()
    paired = {}
    skipped = 0
    for t in shared:
        mb, ms = trajectory_metrics(base[t]), trajectory_metrics(seam[t])
        if mb is None or ms is None:
            skipped += 1
            continue
        paired[t] = (mb, ms)
    emit(f"Trials with a usable NPZ in both conditions: **{len(paired)}** "
         f"(skipped {skipped})")
    emit()

    for space in ("action", "qpos"):
        emit(f"### {space} space "
             f"({'commanded joint targets' if space == 'action' else 'measured physical response'})")
        emit()
        emit("| metric | baseline (mean ± sd) | SEAM (mean ± sd) | change | Wilcoxon p |")
        emit("|---|---:|---:|---:|---:|")
        for metric in METRICS:
            bv = np.array([paired[t][0][space][metric] for t in paired], dtype=float)
            sv = np.array([paired[t][1][space][metric] for t in paired], dtype=float)
            ok = np.isfinite(bv) & np.isfinite(sv)
            bv, sv = bv[ok], sv[ok]
            change = (sv.mean() - bv.mean()) / bv.mean() * 100 if bv.mean() else float("nan")
            p = stats.wilcoxon(bv, sv).pvalue if np.any(bv != sv) else 1.0
            emit(f"| {metric} | {bv.mean():.5f} ± {bv.std(ddof=1):.5f} | "
                 f"{sv.mean():.5f} ± {sv.std(ddof=1):.5f} | {change:+.1f}% | {fmt_p(p)} |")
        emit()

    # overlap residual — the quantity SEAM's guidance directly minimises
    bv = np.array([paired[t][0]["overlap_residual"] for t in paired], dtype=float)
    sv = np.array([paired[t][1]["overlap_residual"] for t in paired], dtype=float)
    ok = np.isfinite(bv) & np.isfinite(sv)
    bv, sv = bv[ok], sv[ok]
    p = stats.wilcoxon(bv, sv).pvalue if np.any(bv != sv) else 1.0
    emit("### Overlap residual (previous chunk's tail vs. new chunk's head)")
    emit()
    emit("| baseline | SEAM | change | Wilcoxon p |")
    emit("|---:|---:|---:|---:|")
    emit(f"| {bv.mean():.5f} | {sv.mean():.5f} | "
         f"{(sv.mean()-bv.mean())/bv.mean()*100:+.1f}% | {fmt_p(p)} |")
    emit()

    # ---- cost --------------------------------------------------------------
    emit("## 4. Inference cost")
    emit()
    emit("| condition | mean latency (ms) | max (ms) | chunks/trial |")
    emit("|---|---:|---:|---:|")
    for name, src in (("baseline", base), ("SEAM", seam)):
        m = np.array([src[t]["inference_ms_mean"] for t in shared
                      if src[t].get("inference_ms_mean") is not None])
        x = np.array([src[t]["inference_ms_max"] for t in shared
                      if src[t].get("inference_ms_max") is not None])
        c = np.array([src[t].get("num_chunks", 0) for t in shared])
        emit(f"| {name} | {m.mean():.1f} | {x.max():.1f} | {c.mean():.1f} |")
    emit()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n[wrote {args.out}]")


if __name__ == "__main__":
    main()
