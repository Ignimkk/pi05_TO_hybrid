"""Scenario: object packing and transport.

    grasp a small object on the table with one hand
      -> drop it into the crate
      -> grasp the crate with both arms
      -> drive the mobile base to the shelf
      -> place the crate on the target shelf level

The crate half is identical to scenario_transport_crate.py and shares the same
waypoint builders in transport_plan.py; only the packing phase in front is new.

    python -m rby1_manipulation.tasks.transport_load_and_carry --headless --object apple
    python -m rby1_manipulation.tasks.transport_load_and_carry --headless --object banana
    python -m rby1_manipulation.tasks.transport_load_and_carry --headless --object orange
    python -m rby1_manipulation.tasks.transport_load_and_carry --headless --object pear
"""
from __future__ import annotations

import argparse
import pathlib
import sys

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
    SMALL_OBJ_SQUEEZE,
    adaptive_close,
    hold_ctrl_for_secs,
    open_grippers,
)
from rby1_manipulation.simulation.common import pick_arm_for_block
from rby1_manipulation.evaluation.transport import check_grasp, crate_on_shelf, object_in_crate
from rby1_manipulation.planning.transport import (
    arm_retract_waypoint,
    capture_grasp_frames,
    crate_approach_waypoints,
    crate_lift_waypoints,
    crate_place_waypoints,
    crate_retract_waypoints,
    drive_waypoints,
    object_into_crate_waypoints,
    object_pick_waypoints,
)
from rby1_manipulation.simulation.transport_scene import (
    CRATE_BODY,
    MODEL_XML,
    MODEL_XML_WHEELS,
    OBJECT_BODIES,
    RandomizationSpec,
    base_handles,
    body_position,
    build_action_17,
    build_state_17,
    load_layout_config,
    reset_transport_scene,
)
from rby1_manipulation.simulation.obstacles import (
    ObstacleCollisionMonitor,
    TransportObstacleManager,
    load_obstacle_config,
)

CAM_NAME_MAP = {
    "cam_high": "zed_left",
    "cam_left_wrist": "wrist_cam_l",
    "cam_right_wrist": "wrist_cam_r",
}


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--object", choices=sorted(OBJECT_BODIES), default="apple")
    ap.add_argument("--arm", choices=["auto", "right", "left"], default="auto",
                    help="'auto' picks the arm from the object's y coordinate")
    ap.add_argument("--config", default=None)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--record", default=None)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--crate-jitter", type=float, default=0.02)
    ap.add_argument("--crate-yaw-jitter", type=float, default=0.10)
    ap.add_argument("--mass-min", type=float, default=0.6)
    ap.add_argument("--mass-max", type=float, default=1.5)
    ap.add_argument("--friction-min", type=float, default=0.8)
    ap.add_argument("--friction-max", type=float, default=1.3)
    ap.add_argument("--shelf-jitter", type=float, default=0.05)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--log-dataset", default=None)
    ap.add_argument("--log-fps", type=int, default=15)
    ap.add_argument("--task-prompt", default=None)
    ap.add_argument("--save-failed", action="store_true")
    ap.add_argument("--grip-open", type=float, default=1.0,
                    help="gripper opening used for approach and release, as a "
                         "fraction of full stroke (1.0 = 86.4 mm, 0.5 = 41 mm)")
    ap.add_argument("--base-mode", choices=["kinematic", "wheel"], default="kinematic")
    ap.add_argument("--obstacle-config", default=None)
    ap.add_argument("--obstacle-profile", default="clear",
                    help="evaluation-only profile: clear, static_offset, "
                         "static_blocked, dynamic_crossing, or mixed")
    return ap


def drive_to_shelf(model, data, config, base, wheel_mode, *,
                   right, left, rmask, lmask, kw, safety_stop=None) -> bool:
    """Move the base to the shelf dock, holding whatever the arms are carrying.

    Kinematic mode ramps the base position servos through the waypoint list.
    Wheel mode hands the same two targets to the experimental unicycle
    controller instead and reports the resulting pose error, which is expected
    to be large - see wheel_drive.py.
    """
    waypoints = drive_waypoints(config)
    if not wheel_mode:
        return execute_bimanual_waypoints(
            model, data, right_arm=right, left_arm=left,
            right_mask=rmask, left_mask=lmask, waypoints=waypoints, base=base,
            stop_condition=safety_stop, **kw,
        )
    from rby1_manipulation.control.mobile_base import drive_base_with_wheels
    reached = True
    for wp in waypoints:
        result = drive_base_with_wheels(
            model, data, base, wp.base, wp.duration,
            safety_stop=safety_stop, return_result=True, **kw
        )
        print(f"  wp {wp.label:14s} [wheel] pose error "
              f"x={result.error[0]:+.3f} y={result.error[1]:+.3f} "
              f"yaw={result.error[2]:+.3f} reached={result.reached} "
              f"reason={result.reason}")
        reached &= result.reached
        if result.reason == "safety_stop":
            break
    return reached


def main() -> int:
    args = build_arg_parser().parse_args()

    wheel_mode = args.base_mode == "wheel"
    if wheel_mode:
        from rby1_manipulation.control.mobile_base import wheel_mode_banner
        wheel_mode_banner()

    config = load_layout_config(args.config) if args.config else load_layout_config()
    obstacle_config = load_obstacle_config(args.obstacle_config) \
        if args.obstacle_config else load_obstacle_config()
    if args.obstacle_profile not in obstacle_config["profiles"]:
        raise SystemExit(
            f"unknown --obstacle-profile {args.obstacle_profile!r}; known: "
            f"{tuple(obstacle_config['profiles'])}"
        )
    if args.log_dataset and args.obstacle_profile != "clear":
        raise SystemExit(
            "obstacle profiles are evaluation-only and cannot be combined with --log-dataset"
        )
    model = mujoco.MjModel.from_xml_path(MODEL_XML_WHEELS if wheel_mode else MODEL_XML)
    data = mujoco.MjData(model)

    rng = np.random.default_rng(args.seed)
    spec = None
    if args.random:
        spec = RandomizationSpec(
            crate_xy_jitter=args.crate_jitter,
            crate_yaw_jitter=args.crate_yaw_jitter,
            crate_mass_range=(args.mass_min, args.mass_max),
            friction_range=(args.friction_min, args.friction_max),
            object_pose=True,
            shelf_y_jitter=args.shelf_jitter,
        )
    if args.level is not None:
        spec = spec or RandomizationSpec()
        spec.target_level = args.level

    state = reset_transport_scene(model, data, config, rng=rng, randomize=spec)
    obstacles = TransportObstacleManager(model, data, obstacle_config)
    obstacles.activate(args.obstacle_profile)
    obstacle_monitor = ObstacleCollisionMonitor(model, data, obstacles)
    obstacle_start_time = float(data.time)

    object_body = OBJECT_BODIES[args.object]
    obj_pos = body_position(model, data, object_body)
    arm_side = args.arm if args.arm != "auto" else pick_arm_for_block(obj_pos)

    prompt = args.task_prompt or (
        f"put the {args.object} in the crate and carry the crate to the shelf")
    print(f"=== scenario: pack the {args.object} and transport (level {state.target_level}) ===")
    print(f"  {args.object:7s} {obj_pos.round(3).tolist()} -> {arm_side} arm")
    print(f"  crate  {state.crate_pose[:3].round(3).tolist()}  mass {state.crate_mass:.3f} kg")
    print(f"  shelf  {state.shelf_pos.round(3).tolist()}  dock {state.dock_pose.round(3).tolist()}")
    print(f"  obstacles    {args.obstacle_profile}: "
          f"{tuple(obstacles.active)}")

    right, left = right_arm_handles(model), left_arm_handles(model)
    base = base_handles(model, require_actuators=not wheel_mode)
    rmask = build_dof_mask(model, RIGHT_ARM_JOINTS)
    lmask = build_dof_mask(model, LEFT_ARM_JOINTS)
    both = {"right": right, "left": left}
    picking = {arm_side: both[arm_side]}

    recorder = None
    logger = None
    if args.record or args.log_dataset:
        from rby1_manipulation.data.recording import EpisodeRecorder
        recorder = EpisodeRecorder(
            model, data, record_path=args.record,
            dataset_root=args.log_dataset, fps=args.log_fps,
            cam_name_map=CAM_NAME_MAP, task=prompt,
            state_fn=lambda: build_state_17(model, data, left, right, base),
            action_fn=lambda: build_action_17(model, data, left, right, base),
        )
        logger = recorder.on_step

    def on_step() -> None:
        obstacles.update(float(data.time) - obstacle_start_time)
        obstacle_monitor.observe()
        obstacle_monitor.observe_base_clearance(
            [data.qpos[base.qidx[0]], data.qpos[base.qidx[1]]]
        )
        if logger is not None:
            logger()

    def print_obstacle_metrics() -> None:
        print(f"    obstacle collision={obstacle_monitor.collided} "
              f"contact_steps={obstacle_monitor.contact_steps} "
              f"min_clearance={obstacle_monitor.min_planar_clearance:.3f} m "
              f"pairs={tuple(sorted(obstacle_monitor.geom_pairs))}")

    def run(viewer=None) -> bool:
        kw = dict(viewer=viewer, on_step=on_step)
        ex = lambda wps: execute_bimanual_waypoints(
            model, data, right_arm=right, left_arm=left,
            right_mask=rmask, left_mask=lmask, waypoints=wps, base=base, **kw)
        frames = capture_grasp_frames(model, data)

        print(f"\n--- phase 1: pick the {args.object} with the {arm_side} hand ---")
        ex(object_pick_waypoints(model, data, arm_side, frames, object_body))
        adaptive_close(model, data, [both[arm_side]], squeeze=SMALL_OBJ_SQUEEZE, **kw)
        picked = check_grasp(model, data, picking, object_body)
        print(f"    grasp ok={picked.ok} contacts={picked.contacts}")

        print("--- phase 2: drop it into the crate ---")
        ex(object_into_crate_waypoints(model, data, arm_side, frames))
        open_grippers(model, data, [both[arm_side]], secs=0.8, opening=args.grip_open, **kw)
        ex([arm_retract_waypoint(arm_side, frames, model, data)])
        hold_ctrl_for_secs(model, data, 0.8, **kw)
        packed = object_in_crate(model, data, object_body)
        print(f"    {args.object} in crate = {packed}   "
              f"pos {body_position(model, data, object_body).round(3).tolist()}")

        # The crate is a free body: dropping an object into it nudges it, so the
        # bimanual phase re-reads the handle sites rather than trusting the reset.
        print("--- phase 3: grasp the crate with both arms ---")
        ex(crate_approach_waypoints(frames))
        contact = adaptive_close(model, data, [right, left], squeeze=CRATE_SQUEEZE, **kw)
        print("    contact qpos: " + ", ".join(f"{v:+.4f}" for v in contact.values()))

        print("--- phase 4: lift and drive to the shelf ---")
        ex(crate_lift_waypoints(model, data, frames, clear_z=config["carry"]["clear_z"]))
        print(f"    crate z={body_position(model, data, CRATE_BODY)[2]:.3f}")
        def navigation_stop() -> bool:
            base_xy = [data.qpos[base.qidx[0]], data.qpos[base.qidx[1]]]
            clearance = obstacle_monitor.observe_base_clearance(base_xy)
            return obstacle_monitor.collided or clearance < 0.05

        drive_ok = drive_to_shelf(
            model, data, config, base, wheel_mode,
            right=right, left=left, rmask=rmask, lmask=lmask, kw=kw,
            safety_stop=navigation_stop,
        )
        if not drive_ok or obstacle_monitor.collided:
            print("--- navigation aborted: place phase skipped ---")
            print_obstacle_metrics()
            return False

        print("--- phase 5: place on the shelf ---")
        dock_frames = capture_grasp_frames(model, data)
        ex(crate_place_waypoints(model, data, dock_frames, state.target_level))
        open_grippers(model, data, [right, left], secs=0.6, opening=args.grip_open, **kw)
        ex(crate_retract_waypoints(model, data, dock_frames))
        hold_ctrl_for_secs(model, data, 1.5, **kw)

        shelf = crate_on_shelf(model, data, level=state.target_level)
        still_packed = object_in_crate(model, data, object_body)
        print(f"\n    crate final {body_position(model, data, CRATE_BODY).round(3).tolist()}")
        print(f"    xy_err={shelf.xy_err:.3f} z_err={shelf.z_err:.3f} "
              f"tilt={shelf.tilt_deg:.1f}deg speed={shelf.speed:.4f}")
        print(f"    {args.object} still in crate = {still_packed}")
        print_obstacle_metrics()
        return shelf.ok and still_packed and drive_ok and not obstacle_monitor.collided

    if args.headless:
        success = run(None)
    else:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as viewer:
            success = run(viewer)

    print(f"\n>>> SUCCESS = {success}")

    if recorder is not None:
        recorder.finish(success=success, save_failed=args.save_failed)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
