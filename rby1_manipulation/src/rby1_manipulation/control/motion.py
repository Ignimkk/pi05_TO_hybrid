"""Ramping / holding / gripper primitives generalized to two arms and a base.

These mirror the helpers in scenario1_single_arm.py, widened to take a sequence
of arms, a tunable squeeze offset, and the mobile base. The scenario-1 originals
are deliberately left untouched: the block dataset and the pi05_rby1_lora grid
evaluation both depend on their exact trajectories, and a shared refactor would
put that at risk for no gain.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import mujoco
import numpy as np

from rby1_manipulation.control.ik import ArmHandles, set_gripper, set_gripper_width

# Latched squeeze offsets: after the fingers stall on the object we set
# ctrl = qpos + offset, so the holding force settles at kp * offset instead of
# growing without bound.
#
# 0.012 for the crate was chosen by test: it holds both a 0.82 kg and a 1.48 kg
# crate through the lift, while 0.002 and 0.006 slip at 1.5 kg. The small-object
# value matches scenario 1's GRIPPER_ADAPT_OFFSET.
CRATE_SQUEEZE = 0.012
# 0.006 measured for the small objects: 0.001 lets a 32 mm apple roll out of the
# pads mid-carry, while 0.006 and 0.012 both carry it into the crate.
SMALL_OBJ_SQUEEZE = 0.006

# Wheel geometry. WHEEL_RADIUS was measured by rolling the robot open loop and
# dividing base travel by wheel rotation: 10 rad of commanded wheel angle moved
# the base 0.2836 m over 2.824 rad of actual rotation, i.e. 0.1004 m/rad. An
# earlier value of 0.0602 (taken from the wheel body offset rather than measured)
# was 40% low and made every wheel-drive command undershoot.
# The wheel joint axis is "0 -1 0", so POSITIVE wheel velocity drives the base
# backwards; forward motion needs a negative wheel command.
WHEEL_RADIUS = 0.1004
WHEEL_HALF_TRACK = 0.265


def settle_scene(model: mujoco.MjModel, data: mujoco.MjData, seconds: float = 1.5,
                 viewer=None, on_step: Optional[Callable[[], None]] = None) -> None:
    for _ in range(int(round(seconds / model.opt.timestep))):
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        if on_step is not None:
            on_step()


def hold_ctrl_for_secs(model: mujoco.MjModel, data: mujoco.MjData, secs: float,
                       viewer=None, on_step: Optional[Callable[[], None]] = None) -> None:
    """Freeze ctrl as-is and step the sim."""
    settle_scene(model, data, secs, viewer=viewer, on_step=on_step)


def hold_all_ctrl_from_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Point every actuator at its own joint's current qpos (a no-motion hold)."""
    for aid in range(model.nu):
        data.ctrl[aid] = data.qpos[model.jnt_qposadr[model.actuator_trnid[aid, 0]]]


def read_ctrl_snapshot(data: mujoco.MjData, arm: ArmHandles) -> np.ndarray:
    return np.array([data.ctrl[a] for a in arm.aid])


def return_arms_to_rest(model: mujoco.MjModel, data: mujoco.MjData,
                        arms: Sequence[ArmHandles],
                        rest_ctrl: Dict[int, float], duration: float,
                        viewer=None, on_step: Optional[Callable[[], None]] = None) -> None:
    """Ramp every listed arm's joints back to a stored ctrl snapshot.

    `rest_ctrl` maps actuator id -> ctrl value (see snapshot_rest_ctrl). Gripper
    ctrl is left alone so a release stays released.
    """
    steps = max(1, int(round(duration / model.opt.timestep)))
    aids = [aid for arm in arms for aid in arm.aid]
    current = np.array([float(data.ctrl[a]) for a in aids])
    target = np.array([float(rest_ctrl[a]) for a in aids])
    for k in range(steps):
        alpha = (k + 1) / steps
        for i, aid in enumerate(aids):
            data.ctrl[aid] = (1.0 - alpha) * current[i] + alpha * target[i]
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        if on_step is not None:
            on_step()


def wait_arms_at_rest(model: mujoco.MjModel, data: mujoco.MjData,
                      arms: Sequence[ArmHandles], rest_ctrl: Dict[int, float], *,
                      tolerance: float, timeout: float,
                      viewer=None, on_step: Optional[Callable[[], None]] = None) -> float:
    """Hold rest targets until measured arm qpos converges within tolerance."""
    if tolerance <= 0.0:
        raise ValueError("tolerance must be greater than zero")
    if timeout <= 0.0:
        raise ValueError("timeout must be greater than zero")
    pairs = [
        (aid, qidx)
        for arm in arms
        for aid, qidx in zip(arm.aid, arm.qidx)
    ]
    max_steps = max(1, int(round(timeout / model.opt.timestep)))
    error = float("inf")
    for _ in range(max_steps + 1):
        error = max(
            abs(float(data.qpos[qidx]) - float(rest_ctrl[aid]))
            for aid, qidx in pairs
        )
        if error <= tolerance:
            return error
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        if on_step is not None:
            on_step()
    raise RuntimeError(
        f"arm failed to reach rest within {timeout:.2f}s: "
        f"max joint error={error:.5f} rad > {tolerance:.5f} rad"
    )


def snapshot_rest_ctrl(data: mujoco.MjData, arms: Sequence[ArmHandles]) -> Dict[int, float]:
    return {aid: float(data.ctrl[aid]) for arm in arms for aid in arm.aid}


def adaptive_close(model: mujoco.MjModel, data: mujoco.MjData,
                   arms: Sequence[ArmHandles], *, squeeze: float,
                   press_secs: float = 0.7, settle_secs: float = 0.4,
                   viewer=None, on_step: Optional[Callable[[], None]] = None) -> Dict[int, float]:
    """Close the listed grippers onto whatever is between the pads.

    Two stages, matching the scenarios' proven pattern:
      (a) command a full close so the fingers drive into contact,
      (b) read the stalled qpos and latch ctrl = qpos + squeeze, which drops the
          holding force to kp * squeeze - enough to carry, not enough to crush
          or creep the object out of the finger cage.

    Returns the contact qpos per gripper actuator id.
    """
    dt = model.opt.timestep
    for arm in arms:
        set_gripper(data, arm, "close")
    for _ in range(int(round(press_secs / dt))):
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        if on_step is not None:
            on_step()

    contact_qpos: Dict[int, float] = {}
    for arm in arms:
        q = float(data.qpos[arm.gripper_qidx])
        contact_qpos[arm.gripper_aid] = q
        data.ctrl[arm.gripper_aid] = q + squeeze
    for _ in range(int(round(settle_secs / dt))):
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        if on_step is not None:
            on_step()
    return contact_qpos


def open_grippers(model: mujoco.MjModel, data: mujoco.MjData,
                  arms: Sequence[ArmHandles], *, secs: float = 0.5,
                  opening=1.0,
                  viewer=None, on_step: Optional[Callable[[], None]] = None) -> None:
    """Open the listed grippers and let them settle.

    `opening` is a fraction of the full stroke (1.0 = GRIPPER_OPEN, 0.0 = shut)
    or the string 'open'/'close'. Pass a float to release only partially.
    """
    for arm in arms:
        set_gripper(data, arm, opening)
    settle_scene(model, data, secs, viewer=viewer, on_step=on_step)


def set_grippers_width(data: mujoco.MjData, arms: Sequence[ArmHandles],
                       width_m: float) -> None:
    """Command a specific clear pad gap, in metres, on every listed gripper."""
    for arm in arms:
        set_gripper_width(data, arm, width_m)


def base_ctrl(data: mujoco.MjData, base) -> np.ndarray:
    return np.array([float(data.ctrl[a]) for a in base.aid])


def base_qpos(data: mujoco.MjData, base) -> np.ndarray:
    return np.array([float(data.qpos[q]) for q in base.qidx])


def sync_wheel_ctrl(data: mujoco.MjData, wheel_aid: Dict[str, int],
                    d_forward: float, d_yaw: float) -> None:
    """Cosmetic only: advance the wheel servos so they look like they rolled.

    In kinematic mode the base is moved by its own actuators and the wheels are
    excluded from floor contact, so this changes nothing physically.
    """
    d_left = (d_forward + d_yaw * WHEEL_HALF_TRACK) / WHEEL_RADIUS
    d_right = (d_forward - d_yaw * WHEEL_HALF_TRACK) / WHEEL_RADIUS
    data.ctrl[wheel_aid["left"]] += d_left
    data.ctrl[wheel_aid["right"]] += d_right


def ramp_base(model: mujoco.MjModel, data: mujoco.MjData, base,
              target_xyyaw: Sequence[float], duration: float, *,
              wheel_aid: Optional[Dict[str, int]] = None,
              viewer=None, on_step: Optional[Callable[[], None]] = None) -> None:
    """Linearly ramp the base ctrl to (x, y, yaw) over `duration` seconds.

    Arm ctrl is untouched, so a carried object rides along with the base.
    """
    steps = max(1, int(round(duration / model.opt.timestep)))
    start = base_ctrl(data, base)
    target = np.asarray(target_xyyaw, dtype=float)
    prev = start.copy()
    for k in range(steps):
        alpha = (k + 1) / steps
        cmd = (1.0 - alpha) * start + alpha * target
        for i, aid in enumerate(base.aid):
            data.ctrl[aid] = cmd[i]
        if wheel_aid is not None:
            yaw = 0.5 * (prev[2] + cmd[2])
            d_xy = cmd[:2] - prev[:2]
            sync_wheel_ctrl(data, wheel_aid,
                            float(d_xy[0] * np.cos(yaw) + d_xy[1] * np.sin(yaw)),
                            float(cmd[2] - prev[2]))
        prev = cmd
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        if on_step is not None:
            on_step()
