"""Do the risky moments of a π0.5 rollout sit at action-chunk boundaries?

This is the diagnostic behind the claim that **smoothness and safety are the same phenomenon**.
π0.5 replans every ``K`` steps; a boundary is the only place where the policy is free to contradict
its own previous plan. If clearance minima concentrate there, then chunk-boundary discontinuity is
not merely a comfort problem — it is the mechanism by which the robot gets close to things.

Nothing here needs the policy, the simulator's dynamics, or a GPU: it recomputes clearance
kinematically from the recorded ``measured_qpos`` using the **same collision model AG3S and trajopt
use** (`UrdfSphereChain`), so a number reported here is a number the optimizer would have seen.

What is measured, per executed step ``t``:

    phase(t)      t minus the start step of the chunk that produced t, in [0, K)
    d_min(t)      min clearance over {table, container, self}, metres; negative is interposition
    channels      the same split by source, so "which thing" is answerable, not just "how close"

and per chunk boundary ``b`` (the step where a new chunk takes over):

    CD_b          ‖a_b − a_{b−1}‖ over the 12 arm joints — the realized boundary discontinuity
    drop_b        min d_min just before b  −  min d_min just after b; positive means clearance lost

Five questions, each a test rather than a plot to squint at:

    Q1  Are per-trial clearance minima uniformly distributed over the phase? (chi-square GOF)
    Q2  Are the globally worst steps (lowest quantile) uniformly distributed over the phase?
    Q3  Is clearance at boundary steps lower than at interior steps? (paired Wilcoxon per trial)
    Q4  Do boundaries with larger discontinuity lose more clearance? (Spearman over all boundaries)
    Q5  SEAM cuts CD by ~21% without changing success. Does clearance improve with it?

Q5 is the point of running this on the *existing* paired grid: SEAM is a treatment that reduces
boundary discontinuity and nothing else, already applied to 216 matched trials. If clearance follows
CD down, the causal direction in Q4 is not merely correlational.

    src/openpi/.venv/bin/python src/scripts/analyze_boundary_risk.py \
        --baseline data/rby1_grid_eval_baseline \
        --seam     data/rby1_grid_eval_seam \
        --out      docs/assets/boundary_risk.md

Blocks are excluded from the clearance model on purpose. Touching the block is the task; the
recorded NPZs do not carry per-step block poses anyway, so including them would measure grasping
and call it a collision. Table, container and the robot's own body are static or kinematic, so they
are exactly recoverable from ``measured_qpos``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

import numpy as np
from scipy import stats

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.ag3s.robot_models.urdf_sphere_chain import (  # noqa: E402
    RBY1_URDF, UrdfSphereChain, parse_urdf,
)
from benchmark.trajopt.types import ChunkLayout  # noqa: E402

MODEL_XML = REPO_ROOT / "src/rby1_description/models/rby1a/mujoco/model.xml"
ARM_DIMS = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)

# The URDF ships collision capsules for the torso and arm_0..5 only. Everything past arm_5 — the
# wrist and the four fingers — is where the robot actually reaches the table, so it cannot be left
# uncovered; `gap_filling_capsules` derives it from the simulator's own meshes. Base and wheels are
# omitted: they never approach the table in this task and would triple the sphere count.
EXTRA_CAPSULE_LINKS = (
    "link_right_arm_6", "link_left_arm_6",
    "ee_finger_r1", "ee_finger_r2", "ee_finger_l1", "ee_finger_l2",
)

# Joints the 14-D action format cannot address. Both are held at the 'teleop' keyframe for the whole
# grid evaluation, and forward kinematics needs them even though the policy never moves them.
FIXED_Q = {**{f"torso_{i}": 0.0 for i in range(6)},
           "right_arm_6": -1.57, "left_arm_6": -1.57}

CHANNELS = ("table", "container", "self")
"""Clearance sources, and the order the report lists them in.

They are kept separate rather than collapsed at source because they do not mean the same thing:
entering the container is the *place* action succeeding, while touching the table or folding an arm
onto the torso never is. `--channels` picks which ones enter the aggregate `d_min`.
"""

MIN_CHAIN_SEPARATION = 4
"""How many joints apart two links must be before their clearance means anything.

Nearby links overlap by construction, and their "clearance" is then a constant of the URDF rather
than anything the policy did. Measured over six trials, the minimum self-clearance and its
variability across the whole rollout:

    separation >= 3   6.3 mm, std 4.2 mm   <- `link_torso_5` vs `link_left_arm_2`, the shoulder
                                              mount: structurally close at every configuration
    separation >= 4  55.6 mm, std 10.8 mm  <- `link_torso_4` vs `link_left_arm_2`, motion-driven
    separation >= 5  99.6 mm, std 10.7 mm  <- saturating; real fold-backs are now excluded too

Three is therefore too permissive: it pins the aggregate minimum to a constant that owns ~65% of
all steps and buries the table and container, which are what this task actually risks. Four is the
smallest threshold whose minimum still moves with the arm.
"""


# ------------------------------------------------------------------------- collision environment

@dataclasses.dataclass(frozen=True)
class Box:
    """A static axis-aligned obstacle box, in the URDF base frame (== MuJoCo world here)."""

    name: str
    source: str          # 'table' | 'container'
    center: np.ndarray   # (3,)
    half: np.ndarray     # (3,)


def load_static_boxes() -> list[Box]:
    """Read the table and container boxes out of the MuJoCo scene at the 'teleop' keyframe.

    Both bodies are welded to the world, so one `mj_forward` fixes them for every trial. Reading
    them from the simulator rather than hard-coding numbers is what keeps this analysis honest if
    the scene is ever re-tuned.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML.resolve()))
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "teleop")
    if key < 0:
        raise RuntimeError("model has no 'teleop' keyframe")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)

    boxes: list[Box] = []
    for source in ("table", "container"):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, source)
        if body < 0:
            raise RuntimeError(f"scene has no {source!r} body")
        for geom in range(model.ngeom):
            if model.geom_bodyid[geom] != body or model.geom_contype[geom] == 0:
                continue
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
                raise RuntimeError(
                    f"{source} geom {geom} is type {model.geom_type[geom]}, not a box; this "
                    "analysis's closed-form distance assumes boxes"
                )
            rot = np.asarray(data.geom_xmat[geom], np.float64).reshape(3, 3)
            if not np.allclose(rot, np.eye(3), atol=1e-9):
                raise RuntimeError(f"{source} geom {geom} is rotated; add R to the box distance")
            boxes.append(Box(
                name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or f"geom{geom}",
                source=source,
                center=np.asarray(data.geom_xpos[geom], np.float64).copy(),
                half=np.asarray(model.geom_size[geom], np.float64).copy(),
            ))
    return boxes


def box_clearance(points: np.ndarray, radii: np.ndarray, box: Box) -> np.ndarray:
    """Signed clearance from spheres to an axis-aligned box: ``sdf(centre) − radius``.

    `points` is ``(T, S, 3)``; the result is ``(T, S)``. The exterior term is the usual
    ``‖max(|p − c| − h, 0)‖`` and the interior term ``min(max(|p − c| − h), 0)`` makes the sign
    continuous through the surface, so a penetration is a negative number rather than a zero that
    looks like a graze.
    """
    delta = np.abs(points - box.center) - box.half
    outside = np.linalg.norm(np.maximum(delta, 0.0), axis=-1)
    inside = np.minimum(delta.max(axis=-1), 0.0)
    return outside + inside - radii[None, :]


def chain_depth(link: str) -> int:
    """Depth of a link along RB-Y1's kinematic chain, counted from the base."""
    if link.startswith("link_torso_"):
        return int(link.rsplit("_", 1)[1])
    if "_arm_" in link:
        return 6 + int(link.rsplit("_", 1)[1])
    if link.startswith("ee_finger_"):
        return 13
    raise KeyError(f"no chain depth known for link {link!r}")


def limb_of(link: str) -> str:
    if link.startswith("link_torso_"):
        return "torso"
    if "right" in link or link.startswith("ee_finger_r"):
        return "right"
    if "left" in link or link.startswith("ee_finger_l"):
        return "left"
    raise KeyError(f"no limb known for link {link!r}")


def self_pairs(link_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Sphere index pairs whose distance is a meaningful self-collision measurement.

    Two links on the same limb are separated by ``|depth_i − depth_j|`` joints; two links on
    opposite arms are separated through the shoulder, ``(depth_i − 5) + (depth_j − 5)``. Pairs
    closer than `MIN_CHAIN_SEPARATION` are dropped as structurally overlapping.
    """
    depths = np.asarray([chain_depth(name) for name in link_names])
    limbs = [limb_of(name) for name in link_names]
    rows, cols = [], []
    for i in range(len(link_names)):
        for j in range(i + 1, len(link_names)):
            if limbs[i] == limbs[j] or "torso" in (limbs[i], limbs[j]):
                separation = abs(depths[i] - depths[j])
            else:
                separation = (depths[i] - 5) + (depths[j] - 5)
            if separation >= MIN_CHAIN_SEPARATION:
                rows.append(i)
                cols.append(j)
    return np.asarray(rows, np.int64), np.asarray(cols, np.int64)


def build_robot() -> UrdfSphereChain:
    from benchmark.ag3s.experiments.sources.mujoco_source import gap_filling_capsules
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML.resolve()))
    extra = gap_filling_capsules(model, links=EXTRA_CAPSULE_LINKS)
    return UrdfSphereChain(parse_urdf(REPO_ROOT / RBY1_URDF), extra_capsules=extra)


# ------------------------------------------------------------------------------ per-trial passes

def actions_to_q(actions: np.ndarray, layout: ChunkLayout) -> np.ndarray:
    """``(T, 14)`` in the policy's action layout -> ``(T, 20)`` in the collision model's joint order."""
    out = np.zeros((actions.shape[0], layout.nq_model), np.float64)
    for name, value in FIXED_Q.items():
        out[:, layout.joint_names.index(name)] = value
    out[:, layout.q_indices] = actions[:, layout.action_columns]
    return out


def phase_of_step(n_steps: int, chunk_start_steps: np.ndarray, execution_length: int) -> np.ndarray:
    """Position of each executed step within the chunk that produced it, in ``[0, K)``.

    Derived from the recorded chunk starts rather than ``t % K``: the last chunk of a trial is
    truncated when the episode ends, and a stalled inference can stretch one chunk, so modular
    arithmetic would silently mislabel exactly the steps this analysis is about.
    """
    phase = np.full(n_steps, -1, np.int64)
    starts = np.asarray(chunk_start_steps, np.int64)
    for index, start in enumerate(starts):
        end = int(starts[index + 1]) if index + 1 < starts.shape[0] else n_steps
        span = np.arange(start, min(end, n_steps))
        phase[span] = span - start
    return np.clip(phase, -1, execution_length - 1)


def trial_signals(record: dict, robot: UrdfSphereChain, boxes: list[Box],
                  pairs: tuple[np.ndarray, np.ndarray], layout: ChunkLayout,
                  window: int, channel_order: tuple[str, ...] = CHANNELS) -> dict | None:
    """Per-step clearance and per-boundary (discontinuity, clearance drop) for one trial."""
    path = record.get("trajectory")
    if not path or not (REPO_ROOT / path).exists():
        return None
    data = np.load(REPO_ROOT / path, allow_pickle=True)
    qpos = np.asarray(data["measured_qpos"], np.float64)
    actions = np.asarray(data["executed_actions"], np.float64)
    starts = np.asarray(data["chunk_start_steps"], np.int64)
    execution_length = int(data["execution_length"])
    n_steps = qpos.shape[0]
    if n_steps < 3 * window or starts.shape[0] < 2:
        return None

    q_full = actions_to_q(qpos, layout)
    centres = np.empty((n_steps, robot.n_spheres, 3), np.float64)
    for t in range(n_steps):
        centres[t], _ = robot.sphere_centers_numeric(q_full[t])
    radii = robot.radii

    channels: dict[str, np.ndarray] = {}
    for source in ("table", "container"):
        stack = [box_clearance(centres, radii, box) for box in boxes if box.source == source]
        channels[source] = np.min(np.stack(stack, axis=0), axis=(0, 2))

    rows, cols = pairs
    gaps = np.linalg.norm(centres[:, rows, :] - centres[:, cols, :], axis=-1)
    channels["self"] = np.min(gaps - (radii[rows] + radii[cols])[None, :], axis=1)

    active = [k for k in channel_order if k in channels]
    stacked = np.stack([channels[k] for k in active], axis=0)
    d_min = stacked.min(axis=0)
    owner = np.asarray(active)[stacked.argmin(axis=0)]
    phase = phase_of_step(n_steps, starts, execution_length)

    # Which robot link is closest to something, at the trial's worst step. The URDF's arm_5 capsule
    # is a fat bounding cylinder, so it will dominate; saying so in the report is the difference
    # between a modelling artefact the reader can discount and one that quietly sets the scale.
    worst = int(np.argmin(d_min))
    per_channel = {k: {"min_m": float(v.min()), "median_m": float(np.median(v))}
                   for k, v in channels.items()}

    # Per boundary: what the policy did to itself, and what happened to clearance right after.
    boundaries = [int(s) for s in starts[1:] if window <= int(s) <= n_steps - window]
    discontinuity, drop, at_step = [], [], []
    for b in boundaries:
        discontinuity.append(float(np.linalg.norm(actions[b, ARM_DIMS] - actions[b - 1, ARM_DIMS])))
        drop.append(float(d_min[b - window:b].min() - d_min[b:b + window].min()))
        at_step.append(b)

    return {
        "trial_id": record["trial_id"],
        "success": bool(record.get("success", False)),
        "phase": phase,
        "d_min": d_min,
        "channels": channels,
        "channel_min_owner": owner,
        "per_channel": per_channel,
        "worst_channel": str(owner[worst]),
        "boundary_step": np.asarray(at_step, np.int64),
        "boundary_cd": np.asarray(discontinuity, np.float64),
        "boundary_drop": np.asarray(drop, np.float64),
        "execution_length": execution_length,
    }


def load_condition(directory: pathlib.Path) -> dict[str, dict]:
    out = {}
    for line in (directory / "results.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") in ("setup_error", "interrupted"):
            continue
        out[record["trial_id"]] = record
    return out


def analyse(directory: pathlib.Path, robot, boxes, pairs, layout, window: int,
            limit: int | None, channel_order: tuple[str, ...]) -> dict[str, dict]:
    records = load_condition(directory)
    ids = sorted(records)[: limit or None]
    out = {}
    for index, trial_id in enumerate(ids, 1):
        signals = trial_signals(records[trial_id], robot, boxes, pairs, layout, window,
                                channel_order)
        if signals is not None:
            out[trial_id] = signals
        print(f"\r  {directory.name}: {index}/{len(ids)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return out


# ------------------------------------------------------------------------------------ statistics

def uniform_gof(phases: np.ndarray, execution_length: int) -> dict:
    """Chi-square goodness of fit of a phase histogram against the uniform null."""
    counts = np.bincount(phases[phases >= 0], minlength=execution_length)[:execution_length]
    expected = np.full(execution_length, counts.sum() / execution_length)
    chi2, p = stats.chisquare(counts, expected)
    return {"counts": counts.tolist(), "expected": float(expected[0]),
            "chi2": float(chi2), "p": float(p), "dof": int(execution_length - 1)}


def condition_stats(signals: dict[str, dict], quantile: float) -> dict:
    execution_length = next(iter(signals.values()))["execution_length"]

    argmin_phase = np.asarray([s["phase"][int(np.argmin(s["d_min"]))] for s in signals.values()])
    all_d = np.concatenate([s["d_min"] for s in signals.values()])
    all_phase = np.concatenate([s["phase"] for s in signals.values()])
    threshold = float(np.quantile(all_d, quantile))
    worst_phase = all_phase[all_d <= threshold]

    # Q3 is paired *within* a trial: absolute clearance varies hugely with where the block sits, so
    # an unpaired boundary-vs-interior comparison would mostly measure the grid.
    boundary_mean, interior_mean = [], []
    for s in signals.values():
        at_boundary = s["d_min"][s["phase"] == 0]
        interior = s["d_min"][s["phase"] >= 2]
        if at_boundary.size and interior.size:
            boundary_mean.append(float(at_boundary.mean()))
            interior_mean.append(float(interior.mean()))
    boundary_mean = np.asarray(boundary_mean)
    interior_mean = np.asarray(interior_mean)
    w_stat, w_p = stats.wilcoxon(boundary_mean, interior_mean)

    cd = np.concatenate([s["boundary_cd"] for s in signals.values()])
    drop = np.concatenate([s["boundary_drop"] for s in signals.values()])
    rho, rho_p = stats.spearmanr(cd, drop)

    profile = np.full(execution_length, np.nan)
    profile_sem = np.full(execution_length, np.nan)
    for p in range(execution_length):
        values = all_d[all_phase == p]
        if values.size:
            profile[p] = values.mean()
            profile_sem[p] = values.std(ddof=1) / np.sqrt(values.size)

    owners = np.concatenate([s["channel_min_owner"] for s in signals.values()])
    owner_share = {k: float((owners == k).mean()) for k in np.unique(owners)}
    channel_summary = {}
    for channel in CHANNELS:
        mins = [s["per_channel"][channel]["min_m"] for s in signals.values()
                if channel in s["per_channel"]]
        medians = [s["per_channel"][channel]["median_m"] for s in signals.values()
                   if channel in s["per_channel"]]
        if mins:
            channel_summary[channel] = {"worst_min_m": float(np.min(mins)),
                                        "mean_min_m": float(np.mean(mins)),
                                        "mean_median_m": float(np.mean(medians))}

    return {
        "n_trials": len(signals),
        "execution_length": int(execution_length),
        "channel_summary": channel_summary,
        "owner_share": owner_share,
        "q1_argmin_gof": uniform_gof(argmin_phase, execution_length),
        "q2_worst_gof": uniform_gof(worst_phase, execution_length),
        "q2_threshold_m": threshold,
        "q3": {"boundary_mean_m": float(boundary_mean.mean()),
               "interior_mean_m": float(interior_mean.mean()),
               "delta_mm": float((boundary_mean.mean() - interior_mean.mean()) * 1000),
               "wilcoxon_stat": float(w_stat), "p": float(w_p), "n_pairs": int(boundary_mean.size)},
        "q4": {"spearman_rho": float(rho), "p": float(rho_p), "n_boundaries": int(cd.size),
               "cd_mean": float(cd.mean()), "drop_mean_mm": float(drop.mean() * 1000)},
        "profile_m": profile.tolist(),
        "profile_sem_m": profile_sem.tolist(),
        "global_min_m": float(all_d.min()),
        "per_trial_min_m": {tid: float(s["d_min"].min()) for tid, s in signals.items()},
        "per_trial_boundary_mean_m": {
            tid: float(s["d_min"][s["phase"] == 0].mean()) for tid, s in signals.items()
        },
        "per_trial_drop_mean_mm": {
            tid: float(s["boundary_drop"].mean() * 1000) for tid, s in signals.items()
        },
    }


def paired_compare(a: dict, b: dict, key: str) -> dict:
    """Wilcoxon signed-rank over the trial ids the two conditions share."""
    shared = sorted(set(a[key]) & set(b[key]))
    xa = np.asarray([a[key][t] for t in shared])
    xb = np.asarray([b[key][t] for t in shared])
    stat, p = stats.wilcoxon(xa, xb)
    return {"n": len(shared), "baseline": float(xa.mean()), "seam": float(xb.mean()),
            "delta": float(xb.mean() - xa.mean()), "stat": float(stat), "p": float(p)}


# --------------------------------------------------------------------------------- presentation

def figures(stats_by_condition: dict, signals_by_condition: dict, out_dir: pathlib.Path,
            stem: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # Fig 1 — clearance as a function of position in the chunk. If the story is right, phase 0 dips.
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for name, st in stats_by_condition.items():
        phases = np.arange(st["execution_length"])
        mean = np.asarray(st["profile_m"]) * 1000
        sem = np.asarray(st["profile_sem_m"]) * 1000
        ax.errorbar(phases, mean, yerr=sem, marker="o", capsize=3, label=name)
    ax.set_xlabel("position within chunk (0 = boundary, the first step of a new chunk)")
    ax.set_ylabel("mean min clearance [mm]")
    ax.set_title("Clearance by position in the action chunk")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"{stem}_profile.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    # Fig 2 — where in the chunk the worst moment of a trial falls, against the uniform null.
    fig, axes = plt.subplots(1, len(stats_by_condition), figsize=(4.4 * len(stats_by_condition), 3.6),
                             squeeze=False)
    for ax, (name, st) in zip(axes[0], stats_by_condition.items()):
        counts = np.asarray(st["q1_argmin_gof"]["counts"])
        ax.bar(np.arange(counts.size), counts, color="#4C72B0")
        ax.axhline(st["q1_argmin_gof"]["expected"], color="#C44E52", ls="--", label="uniform null")
        ax.set_title(f"{name}\nchi2 p = {st['q1_argmin_gof']['p']:.3g}")
        ax.set_xlabel("phase of per-trial clearance minimum")
        ax.set_ylabel("trials")
        ax.legend()
    fig.tight_layout()
    path = out_dir / f"{stem}_argmin.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    # Fig 3 — the causal link: bigger boundary discontinuity, bigger clearance loss?
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for name, signals in signals_by_condition.items():
        cd = np.concatenate([s["boundary_cd"] for s in signals.values()])
        drop = np.concatenate([s["boundary_drop"] for s in signals.values()]) * 1000
        edges = np.quantile(cd, np.linspace(0, 1, 11))
        idx = np.clip(np.searchsorted(edges, cd, side="right") - 1, 0, 9)
        centres = [cd[idx == k].mean() for k in range(10)]
        means = [drop[idx == k].mean() for k in range(10)]
        sems = [drop[idx == k].std(ddof=1) / np.sqrt(max((idx == k).sum(), 1)) for k in range(10)]
        rho = stats_by_condition[name]["q4"]["spearman_rho"]
        ax.errorbar(centres, means, yerr=sems, marker="o", capsize=3,
                    label=f"{name} (rho = {rho:+.3f})")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("boundary discontinuity ‖a_b − a_{b−1}‖ [rad], decile bins")
    ax.set_ylabel("clearance lost across the boundary [mm]")
    ax.set_title("Does a rougher chunk boundary cost clearance?")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"{stem}_coupling.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))
    return written


def render(stats_by_condition: dict, comparisons: dict, figure_paths: list[str],
           window: int, quantile: float, channel_order: tuple[str, ...]) -> str:
    lines = [
        "# Are the risky moments at the chunk boundaries?",
        "",
        "Recomputed kinematically from `measured_qpos` with the AG3S/trajopt collision model "
        "(`UrdfSphereChain`, torso + arms + wrists + fingers). The aggregate `d_min` here is the "
        f"minimum over **{', '.join(channel_order)}**; the blocks are excluded because contacting "
        "them is the task.",
        "",
        f"Boundary window: ±{window} steps. Worst-step quantile: {quantile:.0%}.",
        "",
        "## Channel breakdown — what is `d_min` actually made of?",
        "",
        "| condition | channel | worst over all trials [mm] | mean per-trial min [mm] | "
        "mean median [mm] | share of steps it owns |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, st in stats_by_condition.items():
        for channel, values in st["channel_summary"].items():
            share = st["owner_share"].get(channel, 0.0)
            lines.append(
                f"| {name} | {channel} | {values['worst_min_m']*1000:.2f} | "
                f"{values['mean_min_m']*1000:.2f} | {values['mean_median_m']*1000:.2f} | "
                f"{share:.1%} |"
            )
    lines += [
        "",
        "Negative values are expected and are not all collisions: the URDF's `arm_5` capsule is a "
        "conservative bounding cylinder, and reaching into the container is the place action "
        "working. Every test below is paired *within* a trial or compares conditions on the same "
        "trial, so a constant modelling offset cancels; the absolute millimetres are the part to "
        "read with care.",
        "",
        "## Q1/Q2 — is the phase of a risky moment uniform?",
        "",
        "| condition | n trials | argmin phase chi2 | p | worst-{q} phase chi2 | p |".replace(
            "{q}", f"{quantile:.0%}"),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, st in stats_by_condition.items():
        lines.append(
            f"| {name} | {st['n_trials']} | {st['q1_argmin_gof']['chi2']:.1f} | "
            f"{st['q1_argmin_gof']['p']:.3g} | {st['q2_worst_gof']['chi2']:.1f} | "
            f"{st['q2_worst_gof']['p']:.3g} |"
        )
    lines += ["", "Phase histogram of the per-trial clearance minimum:", ""]
    for name, st in stats_by_condition.items():
        lines.append(f"- **{name}**: {st['q1_argmin_gof']['counts']} "
                     f"(uniform would be {st['q1_argmin_gof']['expected']:.1f} each)")

    lines += ["", "## Q3 — boundary steps vs interior steps (paired within trial)", "",
              "| condition | boundary [mm] | interior [mm] | delta [mm] | Wilcoxon p | n |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, st in stats_by_condition.items():
        q3 = st["q3"]
        lines.append(
            f"| {name} | {q3['boundary_mean_m']*1000:.2f} | {q3['interior_mean_m']*1000:.2f} | "
            f"{q3['delta_mm']:+.2f} | {q3['p']:.3g} | {q3['n_pairs']} |"
        )

    lines += ["", "## Q4 — does a rougher boundary cost clearance?", "",
              "| condition | Spearman rho | p | n boundaries | mean CD [rad] | mean drop [mm] |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, st in stats_by_condition.items():
        q4 = st["q4"]
        lines.append(
            f"| {name} | {q4['spearman_rho']:+.4f} | {q4['p']:.3g} | {q4['n_boundaries']} | "
            f"{q4['cd_mean']:.4f} | {q4['drop_mean_mm']:+.3f} |"
        )
    lines += ["", "A negative rho means larger discontinuity is followed by larger clearance "
                  "loss. Read it with the decile figure: at n > 7000 a non-monotone pattern still "
                  "reaches significance, so the sign alone is not the finding.", ""]

    if comparisons:
        lines += ["## Q5 — SEAM reduces CD by ~21%. Does clearance follow?", "",
                  "| quantity | baseline | SEAM | delta | Wilcoxon p | n pairs |",
                  "|---|---:|---:|---:|---:|---:|"]
        labels = {"per_trial_min_m": "worst clearance in a trial [m]",
                  "per_trial_boundary_mean_m": "mean clearance at boundary steps [m]",
                  "per_trial_drop_mean_mm": "mean clearance lost per boundary [mm]"}
        for key, label in labels.items():
            c = comparisons[key]
            lines.append(f"| {label} | {c['baseline']:.5g} | {c['seam']:.5g} | "
                         f"{c['delta']:+.5g} | {c['p']:.3g} | {c['n']} |")
        lines.append("")

    lines += ["## Figures", ""]
    for path in figure_paths:
        rel = pathlib.Path(path).name
        lines.append(f"![{rel}]({rel})")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline", type=pathlib.Path, required=True)
    ap.add_argument("--seam", type=pathlib.Path, default=None)
    ap.add_argument("--out", type=pathlib.Path, required=True, help="markdown report path")
    ap.add_argument("--window", type=int, default=2,
                    help="steps either side of a boundary used for the clearance drop")
    ap.add_argument("--quantile", type=float, default=0.05,
                    help="which tail of the clearance distribution counts as a 'worst' step")
    ap.add_argument("--channels", default=",".join(CHANNELS),
                    help="comma-separated clearance sources entering d_min: " + ",".join(CHANNELS))
    ap.add_argument("--limit", type=int, default=None, help="only the first N trials (smoke test)")
    ap.add_argument("--json-out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    channel_order = tuple(c.strip() for c in args.channels.split(",") if c.strip())
    unknown = set(channel_order) - set(CHANNELS)
    if unknown:
        ap.error(f"unknown channels {sorted(unknown)}; known: {list(CHANNELS)}")

    robot = build_robot()
    boxes = load_static_boxes()
    pairs = self_pairs(robot.sphere_link_names)
    layout = ChunkLayout.rby1(robot.joint_names)
    print(f"collision model: {robot.n_spheres} spheres, {len(boxes)} static boxes, "
          f"{pairs[0].size} self pairs", file=sys.stderr)

    signals_by_condition: dict[str, dict] = {}
    for name, directory in (("baseline", args.baseline), ("seam", args.seam)):
        if directory is None:
            continue
        signals_by_condition[name] = analyse(directory, robot, boxes, pairs, layout,
                                             args.window, args.limit, channel_order)

    stats_by_condition = {name: condition_stats(signals, args.quantile)
                          for name, signals in signals_by_condition.items()}

    comparisons = {}
    if {"baseline", "seam"} <= set(stats_by_condition):
        for key in ("per_trial_min_m", "per_trial_boundary_mean_m", "per_trial_drop_mean_mm"):
            comparisons[key] = paired_compare(stats_by_condition["baseline"],
                                              stats_by_condition["seam"], key)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure_paths = figures(stats_by_condition, signals_by_condition, args.out.parent,
                          args.out.stem)
    args.out.write_text(render(stats_by_condition, comparisons, figure_paths,
                               args.window, args.quantile, channel_order), encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)

    if args.json_out:
        payload = {"stats": stats_by_condition, "comparisons": comparisons,
                   "window": args.window, "quantile": args.quantile}
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
