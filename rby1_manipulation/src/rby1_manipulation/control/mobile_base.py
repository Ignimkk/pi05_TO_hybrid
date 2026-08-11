"""Experimental differential-drive base mode.

Selected with `--base-mode wheel`, which loads model_transport_wheels.xml (no
base position actuators) and steers with the two existing wheel servos.

Read the banner below before using this for anything: the RB-Y1 MuJoCo model
cannot actually do traction drive without changes to rby1.xml, so this mode is
provided for experimentation and is deliberately excluded from data collection.
The kinematic mode (the default) is what every dataset and evaluation uses.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import mujoco
import numpy as np

from rby1_manipulation.control.motion import WHEEL_HALF_TRACK, WHEEL_RADIUS

BANNER = """
  ----------------------------------------------------------------------------
  --base-mode wheel: the base is driven by the two wheels, not by position
  servos. The pose is closed-loop but not exactly repeatable, so use the default
  kinematic mode for dataset collection.
  ----------------------------------------------------------------------------
"""


def wheel_mode_banner() -> None:
    """Print the experimental wheel-mode warning."""
    print(BANNER)


def wheel_actuator_ids(model: mujoco.MjModel) -> Dict[str, int]:
    return {
        "left": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel_act"),
        "right": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel_act"),
    }


# The wheel joint axis is "0 -1 0": a positive wheel velocity rolls the robot
# BACKWARDS, so the body-forward velocity maps to a negated wheel command.
WHEEL_DIRECTION = -1.0


def unicycle_to_wheels(v: float, w: float) -> tuple[float, float]:
    """(forward m/s, yaw rad/s) -> (left, right) wheel angular velocity."""
    left = WHEEL_DIRECTION * (v - w * WHEEL_HALF_TRACK) / WHEEL_RADIUS
    right = WHEEL_DIRECTION * (v + w * WHEEL_HALF_TRACK) / WHEEL_RADIUS
    return left, right


def drive_base_with_wheels(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    base,
    target_xyyaw,
    duration: float,
    *,
    k_lin: float = 1.2,
    k_ang: float = 2.5,
    max_lin: float = 0.35,
    max_ang: float = 1.0,
    pos_tol: float = 0.02,
    align_first: float = 0.35,
    viewer=None,
    on_step: Optional[Callable[[], None]] = None,
) -> np.ndarray:
    """Closed-loop unicycle controller on the two wheel velocity actuators.

    Three phases, chosen automatically from the current error:
      1. turn in place until the robot points at the goal,
      2. drive toward it while correcting heading,
      3. once within `pos_tol`, turn in place to the goal yaw.

    Runs for at most `duration` seconds and stops early once the pose is reached.
    Returns the final (x, y, yaw) error.
    """
    wheels = wheel_actuator_ids(model)
    target = np.asarray(target_xyyaw, dtype=float)
    dt = model.opt.timestep

    def wrap(a: float) -> float:
        return float(np.arctan2(np.sin(a), np.cos(a)))

    for _ in range(int(round(duration / dt))):
        x, y, yaw = (float(data.qpos[q]) for q in base.qidx)
        dx, dy = target[0] - x, target[1] - y
        dist = float(np.hypot(dx, dy))

        if dist > pos_tol:
            heading_err = wrap(float(np.arctan2(dy, dx)) - yaw)
            w = float(np.clip(k_ang * heading_err, -max_ang, max_ang))
            # Point roughly at the goal before translating, otherwise the robot
            # drives a long arc instead of a straight line.
            v = 0.0 if abs(heading_err) > align_first else \
                float(np.clip(k_lin * dist, -max_lin, max_lin))
        else:
            yaw_err = wrap(target[2] - yaw)
            w = float(np.clip(k_ang * yaw_err, -max_ang, max_ang))
            v = 0.0
            if abs(yaw_err) < 0.01:
                data.ctrl[wheels["left"]] = 0.0
                data.ctrl[wheels["right"]] = 0.0
                break

        left, right = unicycle_to_wheels(v, w)
        data.ctrl[wheels["left"]] = left
        data.ctrl[wheels["right"]] = right

        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        if on_step is not None:
            on_step()

    data.ctrl[wheels["left"]] = 0.0
    data.ctrl[wheels["right"]] = 0.0
    for _ in range(int(round(0.5 / dt))):
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        if on_step is not None:
            on_step()

    pose = np.array([float(data.qpos[q]) for q in base.qidx])
    return np.array([pose[0] - target[0], pose[1] - target[1], wrap(pose[2] - target[2])])
