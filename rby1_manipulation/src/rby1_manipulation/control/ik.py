"""IK helpers shared by scripted demo-collection scenarios.

Design decisions (learned during smoke testing, see follow_mocap.py header):

* mink solves over ALL model DoFs by default. RBY1 has a planar base
  (joint_x/y/th) and a 6-DoF torso. If we let mink use those, kinematic IK
  reports convergence but the sim EE never reaches the target — the position
  actuators only drive the arms directly. So every IK call must be masked
  down to the arm DoFs we actually control.

* Joint damping is set to 50 in the default class. Position tracking is
  therefore slow (~1-2 s per waypoint). For scripted trajectories we run
  IK once per waypoint to get a target joint config, then interpolate ctrl
  linearly between waypoints so the actuators receive a smooth ramp.

* For each waypoint we also want a stable "hold" period after the ramp so
  the arm actually settles at the commanded pose before we move on. This is
  what makes reliable grasp/release timing possible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import mujoco
import mink
import numpy as np

# ---------- constants ----------

RIGHT_ARM_JOINTS = [f"right_arm_{i}" for i in range(7)]
LEFT_ARM_JOINTS  = [f"left_arm_{i}"  for i in range(7)]
RIGHT_ARM_ACTS   = [f"right_arm_{i+1}_act" for i in range(7)]
LEFT_ARM_ACTS    = [f"left_arm_{i+1}_act"  for i in range(7)]

RIGHT_EE_SITE = "right_ee"
LEFT_EE_SITE  = "left_ee"

GRIPPER_R_ACT   = "gripper_r_act"
GRIPPER_L_ACT   = "gripper_l_act"
GRIPPER_R_JOINT = "gripper_finger_r1"
GRIPPER_L_JOINT = "gripper_finger_l1"

# Position-actuator ctrl values driving gripper_finger_r1/l1.
# The equality constraint mirrors this onto _r2/_l2.
GRIPPER_OPEN   = -0.045
GRIPPER_CLOSED = 0.0

# Clear gap between the finger pads as a function of the actuated joint value:
#   finger body separation = 0.006 + 2*|q|,  pad half-thickness = 0.0048 each
#   => gap(q) = 2*|q| - 0.0036
# At the joint limit q = -0.05 that is 96.4 mm; at GRIPPER_OPEN it is 86.4 mm.
# Measured on the compiled model.
_GRIPPER_GAP_OFFSET = 0.0036
GRIPPER_MAX_WIDTH = 2 * 0.05 - _GRIPPER_GAP_OFFSET

# solve_kinematic_ik's own default (400) is under-converged: at the edge of the
# workspace it reports ~110 mm of residual where 1200 iterations reach ~15 mm.
# The default is left alone so existing scenario trajectories stay bit-identical;
# new callers pass max_iters=DEFAULT_IK_ITERS explicitly.
DEFAULT_IK_ITERS = 1200

# ---------- model index helpers ----------

@dataclass
class ArmHandles:
    """Cached model / actuator / joint indices for one arm + its gripper."""
    joint_names: List[str]
    act_names: List[str]
    ee_site: str
    gripper_act: str
    gripper_joint: str
    # populated by resolve_arm_handles():
    qidx: List[int] = field(default_factory=list)
    dofidx: List[int] = field(default_factory=list)
    aid:   List[int] = field(default_factory=list)
    gripper_aid: int = -1
    gripper_qidx: int = -1
    ee_site_id: int = -1


def resolve_arm_handles(model: mujoco.MjModel, handles: ArmHandles) -> ArmHandles:
    handles.qidx   = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
                      for j in handles.joint_names]
    handles.dofidx = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
                      for j in handles.joint_names]
    handles.aid    = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
                      for a in handles.act_names]
    handles.gripper_aid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, handles.gripper_act)
    handles.gripper_qidx = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, handles.gripper_joint)
    ]
    handles.ee_site_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, handles.ee_site)
    return handles


def right_arm_handles(model: mujoco.MjModel) -> ArmHandles:
    return resolve_arm_handles(model, ArmHandles(
        joint_names=RIGHT_ARM_JOINTS, act_names=RIGHT_ARM_ACTS,
        ee_site=RIGHT_EE_SITE,
        gripper_act=GRIPPER_R_ACT, gripper_joint=GRIPPER_R_JOINT,
    ))


def left_arm_handles(model: mujoco.MjModel) -> ArmHandles:
    return resolve_arm_handles(model, ArmHandles(
        joint_names=LEFT_ARM_JOINTS, act_names=LEFT_ARM_ACTS,
        ee_site=LEFT_EE_SITE,
        gripper_act=GRIPPER_L_ACT, gripper_joint=GRIPPER_L_JOINT,
    ))


def build_dof_mask(model: mujoco.MjModel, joint_names: Sequence[str]) -> np.ndarray:
    """Boolean mask over model.nv, True at DoFs of the listed joints."""
    mask = np.zeros(model.nv, dtype=bool)
    for jn in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        mask[model.jnt_dofadr[jid]] = True
    return mask


# ---------- kinematic IK ----------

def solve_kinematic_ik(
    model: mujoco.MjModel,
    seed_qpos: np.ndarray,
    ee_site: str,
    target_pose: mink.SE3,
    dof_mask: np.ndarray,
    *,
    position_cost: float = 100.0,
    orientation_cost: float = 10.0,
    lm_damping: float = 1e-3,
    posture_cost: float = 1e-4,
    max_iters: int = 400,
    dt: float = 1e-2,
    pos_tol: float = 2e-3,       # 2 mm
    ori_tol: float = 5e-2,       # ~3 deg
) -> np.ndarray:
    """Iterate mink until the EE pose matches `target_pose` (arm-only masked).

    Returns a full qpos vector (same shape as seed_qpos). Only DoFs in
    `dof_mask` are changed from `seed_qpos`.
    """
    config = mink.Configuration(model)
    config.update(seed_qpos)

    task = mink.FrameTask(ee_site, "site",
                          position_cost=position_cost,
                          orientation_cost=orientation_cost,
                          lm_damping=lm_damping)
    task.set_target(target_pose)
    posture = mink.PostureTask(model, cost=posture_cost)
    posture.set_target_from_configuration(config)

    for _ in range(max_iters):
        vel = mink.solve_ik(config, [task, posture], dt,
                            solver="daqp", damping=1e-4)
        vel = vel * dof_mask
        config.integrate_inplace(vel, dt)

        cur_T = config.get_transform_frame_to_world(ee_site, "site")
        pos_err = np.linalg.norm(cur_T.translation() - target_pose.translation())
        # ori_err via rotation delta angle
        R_err = target_pose.rotation().inverse() @ cur_T.rotation()
        ori_err = np.linalg.norm(R_err.log())
        if pos_err < pos_tol and ori_err < ori_tol:
            break

    return config.q.copy()


# ---------- pose helpers ----------

def site_pose(data: mujoco.MjData, model: mujoco.MjModel, site_name: str) -> mink.SE3:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    pos = np.asarray(data.site_xpos[sid]).copy()
    mat = np.asarray(data.site_xmat[sid]).reshape(3, 3).copy()
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat.reshape(-1))
    return mink.SE3.from_rotation_and_translation(mink.SO3(q), pos)


def se3_at(position: np.ndarray, orientation: mink.SO3) -> mink.SE3:
    return mink.SE3.from_rotation_and_translation(orientation, np.asarray(position))


# ---------- ctrl helpers ----------

def set_arm_ctrl(data: mujoco.MjData, arm: ArmHandles, target_qpos_all: np.ndarray) -> None:
    """Copy target arm joint positions from a full-qpos vector into ctrl."""
    for i, aid in enumerate(arm.aid):
        data.ctrl[aid] = target_qpos_all[arm.qidx[i]]


def gripper_ctrl_for_fraction(fraction: float) -> float:
    """0.0 = fully closed, 1.0 = GRIPPER_OPEN. Values are clamped."""
    return float(np.clip(fraction, 0.0, 1.0)) * GRIPPER_OPEN


def gripper_ctrl_for_width(width_m: float) -> float:
    """Gripper ctrl that leaves `width_m` of clear gap between the pads.

    Inverts gap(q) = 2*|q| - 0.0036. Clamped to the joint limit, so the widest
    achievable gap is GRIPPER_MAX_WIDTH (96.4 mm); GRIPPER_OPEN corresponds to
    86.4 mm.
    """
    q = (float(width_m) + _GRIPPER_GAP_OFFSET) / 2.0
    return -float(np.clip(q, 0.0, 0.05))


def gripper_width_from_qpos(qpos: float) -> float:
    """Current clear pad gap, in metres, from the actuated finger joint value."""
    return max(0.0, 2.0 * abs(float(qpos)) - _GRIPPER_GAP_OFFSET)


def set_gripper(data: mujoco.MjData, arm: ArmHandles, action) -> None:
    """Command the gripper.

    `action` accepts:
      'open'  / 'close' / 'hold'   - the original three-way interface
      a float in [0, 1]            - fraction of the open stroke
      a string like '0.4'          - same, so CLI values pass straight through

    'hold' (and any unrecognised string) leaves ctrl untouched.
    """
    if isinstance(action, (int, float)):
        data.ctrl[arm.gripper_aid] = gripper_ctrl_for_fraction(float(action))
        return
    if action == "open":
        data.ctrl[arm.gripper_aid] = GRIPPER_OPEN
    elif action == "close":
        data.ctrl[arm.gripper_aid] = GRIPPER_CLOSED
    elif action != "hold":
        try:
            data.ctrl[arm.gripper_aid] = gripper_ctrl_for_fraction(float(action))
        except (TypeError, ValueError):
            pass  # unknown label -> hold


def set_gripper_width(data: mujoco.MjData, arm: ArmHandles, width_m: float) -> None:
    """Open the gripper to a specific clear pad gap in metres."""
    data.ctrl[arm.gripper_aid] = gripper_ctrl_for_width(width_m)
