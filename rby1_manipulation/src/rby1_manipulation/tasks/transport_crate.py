"""Scenario: bimanual crate transport.

    grasp the crate on the table with both arms
      -> lift it
      -> drive the mobile base to the shelf
      -> place it on the target shelf level

Mirrors scenario1_single_arm.py's structure (argparse -> reset -> phases ->
success print -> optional dataset logging) so the two are easy to read side by
side, but runs on model_transport.xml and the 17-D state/action layout.

    python -m rby1_manipulation.tasks.transport_crate --headless
    python -m rby1_manipulation.tasks.transport_crate --headless --record /tmp/crate.mp4
    python -m rby1_manipulation.tasks.transport_crate --headless --random --seed 3
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Optional

import mujoco
import numpy as np

from rby1_manipulation.control.bimanual import execute_bimanual_waypoints
from rby1_manipulation.control.ik import (
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    build_dof_mask,
    left_arm_handles,
    right_arm_handles,
)
from rby1_manipulation.control.motion import (
    CRATE_SQUEEZE,
    adaptive_close,
    hold_ctrl_for_secs,
    open_grippers,
    ramp_base,
)
from rby1_manipulation.evaluation.transport import check_grasp, crate_on_shelf
from rby1_manipulation.planning.transport import (
    capture_grasp_frames,
    crate_approach_waypoints,
    crate_lift_waypoints,
    crate_place_waypoints,
    crate_retract_waypoints,
    drive_waypoints,
)
from rby1_manipulation.simulation.transport_scene import (
    CRATE_BODY,
    MODEL_XML,
    MODEL_XML_WHEELS,
    RandomizationSpec,
    base_handles,
    body_position,
    build_action_17,
    build_state_17,
    load_layout_config,
    reset_transport_scene,
)

DEFAULT_PROMPT = "pick up the crate with both arms and put it on the shelf"

# Cameras, unchanged from the block pipeline: the policy interface stays the same.
CAM_NAME_MAP = {
    "cam_high": "zed_left",
    "cam_left_wrist": "wrist_cam_l",
    "cam_right_wrist": "wrist_cam_r",
}


def default_randomization(args) -> Optional[RandomizationSpec]:
    if not args.random:
        return None
    return RandomizationSpec(
        crate_xy_jitter=args.crate_jitter,
        crate_yaw_jitter=args.crate_yaw_jitter,
        crate_mass_range=(args.mass_min, args.mass_max),
        friction_range=(args.friction_min, args.friction_max),
        object_pose=True,
        shelf_y_jitter=args.shelf_jitter,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None, help="transport_layout.json override")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--record", default=None, help="save a third-person mp4 here")
    ap.add_argument("--random", action="store_true",
                    help="randomize crate pose/mass/friction and the shelf pose")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--crate-jitter", type=float, default=0.02)
    ap.add_argument("--crate-yaw-jitter", type=float, default=0.10)
    ap.add_argument("--mass-min", type=float, default=0.6)
    ap.add_argument("--mass-max", type=float, default=1.5)
    ap.add_argument("--friction-min", type=float, default=0.8)
    ap.add_argument("--friction-max", type=float, default=1.3)
    ap.add_argument("--shelf-jitter", type=float, default=0.05)
    ap.add_argument("--level", type=int, default=None, help="target shelf level override")
    ap.add_argument("--log-dataset", default=None,
                    help="append this episode to a LeRobot dataset root (17-D schema)")
    ap.add_argument("--log-fps", type=int, default=15)
    ap.add_argument("--task-prompt", default=None)
    ap.add_argument("--save-failed", action="store_true",
                    help="also write the episode when SUCCESS=False, prompt prefixed [FAIL]")
    ap.add_argument("--grip-open", type=float, default=1.0,
                    help="gripper opening used for approach and release, as a "
                         "fraction of full stroke (1.0 = 86.4 mm, 0.5 = 41 mm)")
    ap.add_argument("--base-mode", choices=["kinematic", "wheel"], default="kinematic",
                    help="'wheel' is experimental; see model_transport_wheels.xml")
    return ap


def drive_to_shelf(model, data, config, base, wheel_mode, *,
                   right, left, rmask, lmask, kw) -> None:
    """Move the base to the shelf dock, holding whatever the arms are carrying.

    Kinematic mode ramps the base position servos through the waypoint list.
    Wheel mode hands the same two targets to the experimental unicycle
    controller instead and reports the resulting pose error, which is expected
    to be large - see wheel_drive.py.
    """
    waypoints = drive_waypoints(config)
    if not wheel_mode:
        execute_bimanual_waypoints(
            model, data, right_arm=right, left_arm=left,
            right_mask=rmask, left_mask=lmask, waypoints=waypoints, base=base, **kw)
        return
    from rby1_manipulation.control.mobile_base import drive_base_with_wheels
    for wp in waypoints:
        err = drive_base_with_wheels(model, data, base, wp.base, wp.duration, **kw)
        print(f"  wp {wp.label:14s} [wheel] pose error "
              f"x={err[0]:+.3f} y={err[1]:+.3f} yaw={err[2]:+.3f}")


def main() -> int:
    args = build_arg_parser().parse_args()

    wheel_mode = args.base_mode == "wheel"
    if wheel_mode:
        from rby1_manipulation.control.mobile_base import wheel_mode_banner
        wheel_mode_banner()

    config = load_layout_config(args.config) if args.config else load_layout_config()
    model = mujoco.MjModel.from_xml_path(MODEL_XML_WHEELS if wheel_mode else MODEL_XML)
    data = mujoco.MjData(model)

    rng = np.random.default_rng(args.seed)
    spec = default_randomization(args)
    if args.level is not None:
        spec = spec or RandomizationSpec()
        spec.target_level = args.level

    state = reset_transport_scene(model, data, config, rng=rng, randomize=spec)
    print(f"=== scenario: bimanual crate transport (level {state.target_level}) ===")
    print(f"  crate  {state.crate_pose[:3].round(3).tolist()}  mass {state.crate_mass:.3f} kg")
    print(f"  shelf  {state.shelf_pos.round(3).tolist()}  dock {state.dock_pose.round(3).tolist()}")

    right, left = right_arm_handles(model), left_arm_handles(model)
    base = base_handles(model, require_actuators=not wheel_mode)
    rmask = build_dof_mask(model, RIGHT_ARM_JOINTS)
    lmask = build_dof_mask(model, LEFT_ARM_JOINTS)
    both = {"right": right, "left": left}

    # ---- dataset / video plumbing (optional) ----
    recorder = None
    logger = None
    if args.record or args.log_dataset:
        from rby1_manipulation.data.recording import EpisodeRecorder
        recorder = EpisodeRecorder(
            model, data, record_path=args.record,
            dataset_root=args.log_dataset, fps=args.log_fps,
            cam_name_map=CAM_NAME_MAP,
            task=args.task_prompt or DEFAULT_PROMPT,
            state_fn=lambda: build_state_17(model, data, left, right, base),
            action_fn=lambda: build_action_17(model, data, left, right, base),
        )
        logger = recorder.on_step

    def run(viewer=None) -> bool:
        kw = dict(viewer=viewer, on_step=logger)
        frames = capture_grasp_frames(model, data)

        print("\n--- phase A: approach the handles ---")
        execute_bimanual_waypoints(
            model, data, right_arm=right, left_arm=left,
            right_mask=rmask, left_mask=lmask,
            waypoints=crate_approach_waypoints(frames), base=base, **kw)

        print("--- phase B: close both grippers on the handles ---")
        contact = adaptive_close(model, data, [right, left], squeeze=CRATE_SQUEEZE, **kw)
        print("    contact qpos: " + ", ".join(f"{v:+.4f}" for v in contact.values()))
        grasp = check_grasp(model, data, both, CRATE_BODY)
        print(f"    grasp ok={grasp.ok} contacts={grasp.contacts} "
              f"gripper={ {k: round(v, 4) for k, v in grasp.gripper_qpos.items()} }")

        print("--- phase C: lift ---")
        execute_bimanual_waypoints(
            model, data, right_arm=right, left_arm=left,
            right_mask=rmask, left_mask=lmask,
            waypoints=crate_lift_waypoints(model, data, frames,
                                           clear_z=config["carry"]["clear_z"]),
            base=base, **kw)
        lifted = check_grasp(model, data, both, CRATE_BODY, require_lift=True, lift_margin=0.08)
        print(f"    crate z={body_position(model, data, CRATE_BODY)[2]:.3f} "
              f"held={lifted.ok}")

        print("--- phase D: drive to the shelf ---")
        drive_to_shelf(model, data, config, base, wheel_mode,
                       right=right, left=left, rmask=rmask, lmask=lmask, kw=kw)

        print("--- phase E: place on the shelf ---")
        # Re-capture the wrist frames here: the arms rotated with the base, so
        # their current EE orientation is the one the IK will converge near.
        dock_frames = capture_grasp_frames(model, data)
        execute_bimanual_waypoints(
            model, data, right_arm=right, left_arm=left,
            right_mask=rmask, left_mask=lmask,
            waypoints=crate_place_waypoints(model, data, dock_frames, state.target_level),
            base=base, **kw)

        print("--- phase F: release and retract ---")
        open_grippers(model, data, [right, left], secs=0.6, opening=args.grip_open, **kw)
        execute_bimanual_waypoints(
            model, data, right_arm=right, left_arm=left,
            right_mask=rmask, left_mask=lmask,
            waypoints=crate_retract_waypoints(model, data, dock_frames), base=base, **kw)
        hold_ctrl_for_secs(model, data, 1.5, **kw)

        shelf = crate_on_shelf(model, data, level=state.target_level)
        print(f"\n    crate final {body_position(model, data, CRATE_BODY).round(3).tolist()}")
        print(f"    xy_err={shelf.xy_err:.3f} z_err={shelf.z_err:.3f} "
              f"tilt={shelf.tilt_deg:.1f}deg speed={shelf.speed:.4f}")
        return shelf.ok

    if args.headless:
        success = run(None)
    else:
        # `from mujoco import viewer` rather than `import mujoco.viewer`: the
        # latter rebinds `mujoco` as a local and shadows the module-level import.
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as viewer:
            success = run(viewer)

    print(f"\n>>> SUCCESS = {success}")

    if recorder is not None:
        recorder.finish(success=success, save_failed=args.save_failed)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
