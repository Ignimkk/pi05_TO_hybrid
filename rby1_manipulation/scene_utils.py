"""Scene randomization + auto arm selection helpers.

Reach envelope for each arm (measured empirically on the teleop keyframe):
    right arm reachable xy (roughly, at z ~ 0.87):
        x in [0.35, 0.65]
        y in [-0.35, +0.10]      (crosses centerline by ~10 cm)
    left  arm reachable xy:
        x in [0.35, 0.65]
        y in [-0.10, +0.35]

These bounds are used both for auto-arm selection (per block y) and for jitter
sampling. Blocks whose spawn falls outside a given arm's envelope will fail to
be picked up (see the blue-block case in scenario 1 tests).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import mujoco
import numpy as np


# Per-arm reach envelope (min_x, max_x, min_y, max_y) for block SPAWNS.
# Constraints layered together:
#   * table top spans x=[0.40, 0.90] -> stay well inside to avoid falling off
#   * arm reach at grasp height maxes around 0.60 -> cap max_x
#   * container at (0.60, 0) with interior half-width 0.10 -> keep |y| >= 0.15
#     so a randomly-spawned block never lands inside the container
RIGHT_ARM_REACH = dict(x=(0.45, 0.60), y=(-0.32, -0.15))
LEFT_ARM_REACH  = dict(x=(0.45, 0.60), y=( 0.15,  0.32))

# Freejoint qpos start indices in the RBY1 model (they are the last 3 freejoints).
# Populated at first call to _freejoint_qadr_map().
_FREE_QADR_CACHE: Dict[str, int] = {}


def _freejoint_qadr_map(model: mujoco.MjModel) -> Dict[str, int]:
    if _FREE_QADR_CACHE:
        return _FREE_QADR_CACHE
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            _FREE_QADR_CACHE[name] = model.jnt_qposadr[j]
    return _FREE_QADR_CACHE


def set_block_pose(model: mujoco.MjModel, data: mujoco.MjData,
                   joint_name: str, xyz: np.ndarray,
                   quat_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)) -> None:
    """Move a block via its freejoint qpos (both position and orientation)."""
    qadr = _freejoint_qadr_map(model)[joint_name]
    data.qpos[qadr:qadr + 3] = xyz
    data.qpos[qadr + 3:qadr + 7] = quat_wxyz
    # zero velocity so it doesn't inherit motion from a previous episode
    dofadr = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)]
    data.qvel[dofadr:dofadr + 6] = 0.0


def sample_in_reach(rng: np.random.Generator, envelope: dict,
                    other_positions: list, min_sep: float = 0.10,
                    z: float = 0.87, max_tries: int = 50) -> np.ndarray:
    """Sample an xy inside a reach envelope, keeping min_sep from other blocks."""
    x_lo, x_hi = envelope["x"]
    y_lo, y_hi = envelope["y"]
    for _ in range(max_tries):
        x = rng.uniform(x_lo, x_hi)
        y = rng.uniform(y_lo, y_hi)
        p = np.array([x, y, z])
        if all(np.linalg.norm(p[:2] - q[:2]) >= min_sep for q in other_positions):
            return p
    # give up: return last candidate
    return p


def pick_arm_for_block(block_pos: np.ndarray, default: str = "right") -> str:
    """Return 'right' if block is on the robot's right side, 'left' otherwise."""
    y = block_pos[1]
    if y < -0.05:
        return "right"
    if y > 0.05:
        return "left"
    return default


def randomize_blocks(model: mujoco.MjModel, data: mujoco.MjData,
                     rng: Optional[np.random.Generator] = None,
                     z: float = 0.87) -> Dict[str, np.ndarray]:
    """Sample fresh positions for all 3 blocks inside per-arm reach envelopes.

    Rule: red -> right-arm envelope, green -> left, blue -> left (matches the
    default teleop keyframe layout). Callers can still override arm selection.
    """
    if rng is None:
        rng = np.random.default_rng()

    placements: Dict[str, np.ndarray] = {}
    others: list = []
    for jname, envelope in [
        ("red_block_free",   RIGHT_ARM_REACH),
        ("green_block_free", LEFT_ARM_REACH),
        ("blue_block_free",  LEFT_ARM_REACH),
    ]:
        p = sample_in_reach(rng, envelope, others, z=z)
        set_block_pose(model, data, jname, p)
        others.append(p)
        placements[jname] = p

    mujoco.mj_forward(model, data)
    return placements
