"""Experimental differential-drive base mode.

Selected with `--base-mode wheel`, which loads model_transport_wheels.xml (no
base position actuators) and steers with the two existing wheel servos.

Read the banner below before using this for anything: the RB-Y1 MuJoCo model
cannot actually do traction drive without changes to rby1.xml, so this mode is
provided for experimentation and is deliberately excluded from data collection.
The kinematic mode (the default) is what every dataset and evaluation uses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class WheelPreflightReport:
    ready: bool
    actuator_ids: Dict[str, int]
    ctrl_ranges: Dict[str, tuple[float, float]]
    issues: tuple[str, ...] = field(default_factory=tuple)


def wheel_preflight(model: mujoco.MjModel) -> WheelPreflightReport:
    """Validate the compiled differential-drive interface before motion."""
    issues: list[str] = []
    actuators = wheel_actuator_ids(model)
    ctrl_ranges: Dict[str, tuple[float, float]] = {}
    expected_joints = {"left": "left_wheel", "right": "right_wheel"}
    for side, actuator_id in actuators.items():
        if actuator_id < 0:
            issues.append(f"missing {side}_wheel_act")
            continue
        ctrl_ranges[side] = tuple(float(v) for v in model.actuator_ctrlrange[actuator_id])
        if not bool(model.actuator_ctrllimited[actuator_id]):
            issues.append(f"{side} wheel actuator has no ctrl limit")
        elif ctrl_ranges[side][0] > -10.0 or ctrl_ranges[side][1] < 10.0:
            issues.append(f"{side} wheel ctrl range {ctrl_ranges[side]} is too narrow")
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name != expected_joints[side]:
            issues.append(f"{side} actuator drives {joint_name!r}, not {expected_joints[side]!r}")
        # MuJoCo <velocity> actuators compile to fixed gain kv and bias -kv*qvel.
        gain = float(model.actuator_gainprm[actuator_id, 0])
        velocity_bias = float(model.actuator_biasprm[actuator_id, 2])
        if gain <= 0.0 or not np.isclose(velocity_bias, -gain):
            issues.append(f"{side} wheel actuator is not velocity-servo configured")

    for name in ("base_x_act", "base_y_act", "base_yaw_act"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) >= 0:
            issues.append(f"wheel model must not contain kinematic actuator {name}")
    if WHEEL_RADIUS <= 0.0 or WHEEL_HALF_TRACK <= 0.0:
        issues.append("wheel geometry constants must be positive")
    return WheelPreflightReport(
        ready=not issues,
        actuator_ids=actuators,
        ctrl_ranges=ctrl_ranges,
        issues=tuple(issues),
    )


@dataclass(frozen=True)
class WheelDriveResult:
    error: np.ndarray
    reached: bool
    reason: str
    elapsed_seconds: float
    path_length: float


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
    max_lin_accel: float = 0.60,
    max_ang_accel: float = 1.50,
    pos_tol: float = 0.02,
    yaw_tol: float = 0.01,
    align_first: float = 0.35,
    progress_timeout: float = 2.0,
    safety_stop: Optional[Callable[[], bool]] = None,
    return_result: bool = False,
    viewer=None,
    on_step: Optional[Callable[[], None]] = None,
) -> np.ndarray | WheelDriveResult:
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

    current_v = 0.0
    current_w = 0.0
    elapsed = 0.0
    path_length = 0.0
    previous_xy = np.array([float(data.qpos[base.qidx[0]]), float(data.qpos[base.qidx[1]])])
    best_dist = float("inf")
    last_progress_time = 0.0
    reason = "timeout"
    reached = False

    for step_index in range(int(round(duration / dt))):
        if safety_stop is not None and safety_stop():
            reason = "safety_stop"
            break
        x, y, yaw = (float(data.qpos[q]) for q in base.qidx)
        dx, dy = target[0] - x, target[1] - y
        dist = float(np.hypot(dx, dy))

        if dist > pos_tol:
            heading_err = wrap(float(np.arctan2(dy, dx)) - yaw)
            target_w = float(np.clip(k_ang * heading_err, -max_ang, max_ang))
            # Point roughly at the goal before translating, otherwise the robot
            # drives a long arc instead of a straight line.
            target_v = 0.0 if abs(heading_err) > align_first else \
                float(np.clip(k_lin * dist, -max_lin, max_lin))
        else:
            yaw_err = wrap(target[2] - yaw)
            target_w = float(np.clip(k_ang * yaw_err, -max_ang, max_ang))
            target_v = 0.0
            if abs(yaw_err) < yaw_tol:
                data.ctrl[wheels["left"]] = 0.0
                data.ctrl[wheels["right"]] = 0.0
                reached = True
                reason = "reached"
                break

        current_v += float(np.clip(target_v - current_v,
                                   -max_lin_accel * dt, max_lin_accel * dt))
        current_w += float(np.clip(target_w - current_w,
                                   -max_ang_accel * dt, max_ang_accel * dt))
        left, right = unicycle_to_wheels(current_v, current_w)
        data.ctrl[wheels["left"]] = left
        data.ctrl[wheels["right"]] = right

        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        if on_step is not None:
            on_step()

        elapsed = (step_index + 1) * dt
        current_xy = np.array([float(data.qpos[base.qidx[0]]), float(data.qpos[base.qidx[1]])])
        path_length += float(np.linalg.norm(current_xy - previous_xy))
        previous_xy = current_xy
        if target_v == 0.0:
            # Turning in place is valid progress even though distance is fixed.
            last_progress_time = elapsed
            best_dist = min(best_dist, dist)
        elif dist < best_dist - 0.002:
            best_dist = dist
            last_progress_time = elapsed
        elif progress_timeout > 0.0 and elapsed - last_progress_time > progress_timeout:
            reason = "stalled"
            break

    data.ctrl[wheels["left"]] = 0.0
    data.ctrl[wheels["right"]] = 0.0
    for _ in range(int(round(0.5 / dt))):
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        if on_step is not None:
            on_step()

    pose = np.array([float(data.qpos[q]) for q in base.qidx])
    error = np.array([pose[0] - target[0], pose[1] - target[1], wrap(pose[2] - target[2])])
    result = WheelDriveResult(
        error=error,
        reached=reached,
        reason=reason,
        elapsed_seconds=elapsed,
        path_length=path_length,
    )
    return result if return_result else result.error
