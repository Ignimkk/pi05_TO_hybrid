"""Reusable single-arm waypoint execution for MuJoCo position control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import mink
import mujoco
import numpy as np

from rby1_manipulation.control.ik import (
    ArmHandles,
    se3_at,
    set_gripper,
    solve_kinematic_ik,
)


@dataclass
class Waypoint:
    label: str
    pos: np.ndarray
    quat: mink.SO3
    gripper: str
    duration: float
    wait_after: float = 0.4


def read_ctrl_snapshot(data: mujoco.MjData, arm: ArmHandles) -> np.ndarray:
    return np.array([data.ctrl[actuator_id] for actuator_id in arm.aid])


def execute_waypoints(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: ArmHandles,
    arm_mask: np.ndarray,
    waypoints: Sequence[Waypoint],
    *,
    viewer=None,
    on_step: Optional[Callable[[], None]] = None,
) -> None:
    """Solve each Cartesian waypoint and ramp position controls to it."""
    dt = model.opt.timestep
    prev_target = read_ctrl_snapshot(data, arm)

    for waypoint in waypoints:
        target_pose = se3_at(waypoint.pos, waypoint.quat)
        q_target = solve_kinematic_ik(
            model,
            data.qpos,
            arm.ee_site,
            target_pose,
            arm_mask,
            max_iters=300,
        )
        target_arm_ctrl = np.array([q_target[qidx] for qidx in arm.qidx])

        if waypoint.gripper != "hold":
            set_gripper(data, arm, waypoint.gripper)

        ramp_steps = max(1, int(round(waypoint.duration / dt)))
        for step in range(ramp_steps):
            alpha = (step + 1) / ramp_steps
            for index, actuator_id in enumerate(arm.aid):
                data.ctrl[actuator_id] = (
                    (1.0 - alpha) * prev_target[index]
                    + alpha * target_arm_ctrl[index]
                )
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
            if on_step is not None:
                on_step()

        for index, actuator_id in enumerate(arm.aid):
            data.ctrl[actuator_id] = target_arm_ctrl[index]
        hold_steps = max(0, int(round(waypoint.wait_after / dt)))
        for _ in range(hold_steps):
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
            if on_step is not None:
                on_step()

        prev_target = target_arm_ctrl
        ee_now = data.site_xpos[arm.ee_site_id]
        error_mm = np.linalg.norm(ee_now - waypoint.pos) * 1000.0
        print(
            f"  wp {waypoint.label:9s} target={waypoint.pos.round(3).tolist()} "
            f"EE={ee_now.round(3).tolist()} err={error_mm:5.1f}mm "
            f"gripper={waypoint.gripper}"
        )
