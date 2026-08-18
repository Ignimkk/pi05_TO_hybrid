"""Dual-arm waypoint execution on top of the existing single-arm IK.

Why two independent `solve_kinematic_ik` calls rather than one dual-task solver:
the left and right arm DoF sets are disjoint, and the torso and base are excluded
from both masks, so the two masked solutions compose exactly - measured 4.2-4.6 mm
simultaneous hover accuracy on both arms. A second FrameTask would in fact be
worse here, because mink's velocity is masked to the arm DoFs *after* the QP is
solved, and adding tasks makes that post-hoc projection less faithful, not more.

Two things this module does that the single-arm path does not:
  * it plans with `max_iters=DEFAULT_IK_ITERS` (1200), because 300-400 leaves tens
    of mm of residual near the workspace edge;
  * it lets a waypoint carry a *callable* target, so the handle sites are re-read
    right before each solve. The crate shifts a few cm during contact, and
    planning every waypoint against the reset pose lets the error run away.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

import mink
import mujoco
import numpy as np

from rby1_manipulation.control.ik import (
    ArmHandles,
    DEFAULT_IK_ITERS,
    se3_at,
    set_gripper,
    solve_kinematic_ik,
)

# Distance from an EE site to the finger-pad centre, along the gripper's +z axis.
# The fingers point along -z_ee, so a grasp target is the pad position pushed
# back by this much.
#
# Measured on the compiled model: the two large flat pad geoms
# (0.0081 x 0.016 x 0.0369 and 0.005 x 0.0163 x 0.0319) sit at +0.0032 and
# -0.0019 along z_ee from the EE site, i.e. the pad centre is essentially AT the
# site. Getting this wrong is expensive: an assumed 0.0244 put the bar 22 mm below
# the pad centre, leaving only ~8 mm of pad beneath it, and the crate crept out of
# the grippers within a second of being lifted.
GRASP_PAD_OFFSET = 0.003

# A target is either a fixed world position or something re-evaluated per
# waypoint (e.g. "wherever the left handle is right now").
TargetLike = Union[np.ndarray, Sequence[float],
                   Callable[[mujoco.MjModel, mujoco.MjData], np.ndarray]]


def resolve_target(target: Optional[TargetLike],
                   model: mujoco.MjModel, data: mujoco.MjData) -> Optional[np.ndarray]:
    if target is None:
        return None
    if callable(target):
        return np.asarray(target(model, data), dtype=float)
    return np.asarray(target, dtype=float)


def grasp_site_target(grasp_pos: np.ndarray, rot: mink.SO3,
                      pad_offset: float = GRASP_PAD_OFFSET) -> np.ndarray:
    """EE-site target that puts the finger pads on `grasp_pos`."""
    return np.asarray(grasp_pos, dtype=float) - pad_offset * rot.as_matrix()[:, 2]


def approach_target(grasp_pos: np.ndarray, rot: mink.SO3, standoff: float,
                    pad_offset: float = GRASP_PAD_OFFSET) -> np.ndarray:
    """Same as grasp_site_target but backed off `standoff` along the approach axis."""
    return grasp_site_target(grasp_pos, rot, pad_offset) + standoff * rot.as_matrix()[:, 2]


@dataclass
class BiWaypoint:
    """One synchronized pose for both arms, the grippers and the base.

    `right_pos` / `left_pos` are EE *site* targets (already pad-compensated). A
    None target holds that arm where it is. `base` is an absolute (x, y, yaw)
    ctrl triple; None holds the base.
    """
    label: str
    right_pos: Optional[TargetLike] = None
    left_pos: Optional[TargetLike] = None
    right_quat: Optional[mink.SO3] = None
    left_quat: Optional[mink.SO3] = None
    # 'open' | 'close' | 'hold', or a float in [0, 1] for a partial opening.
    gripper: Union[str, float] = "hold"
    base: Optional[Sequence[float]] = None
    duration: float = 1.5
    wait_after: float = 0.4


def solve_bimanual_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    right_arm: ArmHandles,
    left_arm: ArmHandles,
    right_mask: np.ndarray,
    left_mask: np.ndarray,
    right_target: Optional[mink.SE3] = None,
    left_target: Optional[mink.SE3] = None,
    max_iters: int = DEFAULT_IK_ITERS,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Solve each arm against its own target. Returns (right_q7, left_q7).

    A None target yields None for that arm, meaning "leave this arm's ctrl alone".
    """
    seed = data.qpos.copy()
    right_q = left_q = None
    if right_target is not None:
        q = solve_kinematic_ik(model, seed, right_arm.ee_site, right_target,
                               right_mask, max_iters=max_iters)
        right_q = np.array([q[i] for i in right_arm.qidx])
    if left_target is not None:
        q = solve_kinematic_ik(model, seed, left_arm.ee_site, left_target,
                               left_mask, max_iters=max_iters)
        left_q = np.array([q[i] for i in left_arm.qidx])
    return right_q, left_q


def execute_bimanual_waypoints(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    right_arm: ArmHandles,
    left_arm: ArmHandles,
    right_mask: np.ndarray,
    left_mask: np.ndarray,
    waypoints: Sequence[BiWaypoint],
    base=None,
    max_iters: int = DEFAULT_IK_ITERS,
    viewer=None,
    on_step: Optional[Callable[[], None]] = None,
    stop_condition: Optional[Callable[[], bool]] = None,
    verbose: bool = True,
) -> bool:
    """Plan and run a waypoint list, ramping both arms (and the base) together."""
    dt = model.opt.timestep

    def should_stop() -> bool:
        if stop_condition is None or not stop_condition():
            return False
        if base is not None and base.aid:
            for aid, qidx in zip(base.aid, base.qidx):
                data.ctrl[aid] = data.qpos[qidx]
        return True

    for wp in waypoints:
        if should_stop():
            return False
        r_pos = resolve_target(wp.right_pos, model, data)
        l_pos = resolve_target(wp.left_pos, model, data)

        r_target = se3_at(r_pos, wp.right_quat) if r_pos is not None else None
        l_target = se3_at(l_pos, wp.left_quat) if l_pos is not None else None
        r_q, l_q = solve_bimanual_ik(
            model, data,
            right_arm=right_arm, left_arm=left_arm,
            right_mask=right_mask, left_mask=left_mask,
            right_target=r_target, left_target=l_target,
            max_iters=max_iters,
        )

        if not (isinstance(wp.gripper, str) and wp.gripper == "hold"):
            set_gripper(data, right_arm, wp.gripper)
            set_gripper(data, left_arm, wp.gripper)

        # Everything below ramps in ctrl space from the current command, so a
        # held arm simply keeps its last target.
        plans = []
        if r_q is not None:
            plans.append((right_arm.aid, np.array([data.ctrl[a] for a in right_arm.aid]), r_q))
        if l_q is not None:
            plans.append((left_arm.aid, np.array([data.ctrl[a] for a in left_arm.aid]), l_q))
        base_plan = None
        if wp.base is not None:
            if base is None:
                raise ValueError(f"waypoint {wp.label!r} commands the base but none was passed")
            base_plan = (base.aid,
                         np.array([data.ctrl[a] for a in base.aid]),
                         np.asarray(wp.base, dtype=float))

        steps = max(1, int(round(wp.duration / dt)))
        for k in range(steps):
            alpha = (k + 1) / steps
            for aids, start, target in plans:
                for i, aid in enumerate(aids):
                    data.ctrl[aid] = (1.0 - alpha) * start[i] + alpha * target[i]
            if base_plan is not None:
                aids, start, target = base_plan
                for i, aid in enumerate(aids):
                    data.ctrl[aid] = (1.0 - alpha) * start[i] + alpha * target[i]
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
            if on_step is not None:
                on_step()
            if should_stop():
                return False

        for aids, _, target in plans:
            for i, aid in enumerate(aids):
                data.ctrl[aid] = target[i]
        for _ in range(max(0, int(round(wp.wait_after / dt)))):
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
            if on_step is not None:
                on_step()
            if should_stop():
                return False

        if verbose:
            parts = []
            for tag, arm, want in (("R", right_arm, r_pos), ("L", left_arm, l_pos)):
                if want is None:
                    continue
                got = data.site_xpos[arm.ee_site_id]
                parts.append(f"{tag} err={np.linalg.norm(got - want) * 1000:5.1f}mm")
            base_txt = ""
            if base is not None:
                base_txt = f" base={np.array([data.qpos[q] for q in base.qidx]).round(3).tolist()}"
            print(f"  wp {wp.label:14s} {'  '.join(parts)} gripper={wp.gripper}{base_txt}")
    return True


def plan_error(model: mujoco.MjModel, data: mujoco.MjData, arm: ArmHandles,
               target_pos: np.ndarray, mask: np.ndarray, rot: mink.SO3,
               max_iters: int = DEFAULT_IK_ITERS) -> float:
    """Residual (m) of a dry-run IK solve - used by the reach report.

    Solves without touching `data`: the returned qpos is written into a scratch
    MjData so the caller's simulation state is untouched.
    """
    q = solve_kinematic_ik(model, data.qpos.copy(), arm.ee_site,
                           se3_at(target_pos, rot), mask, max_iters=max_iters)
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = q
    mujoco.mj_forward(model, scratch)
    return float(np.linalg.norm(scratch.site_xpos[arm.ee_site_id] - target_pos))
