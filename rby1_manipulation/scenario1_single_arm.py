"""Scenario 1: single-arm pick a colored block, place it in the brown container.

Default: right arm picks the red block.

Approach:
1. Reset to the teleop keyframe, settle physics so blocks land on the table.
2. Read the current block and container positions from `data.xpos`.
3. Generate a waypoint sequence for the chosen arm/block.
4. For each waypoint:
     a. Solve masked kinematic IK for the arm joints (`ik_utils.solve_kinematic_ik`).
     b. Linearly interpolate ctrl from the previous target to the new target over
        `duration` seconds (joint-space ramp).
     c. Hold at the target for `wait_after` seconds so the actuators settle.
     d. Set gripper ctrl at waypoint entry.
5. Success = block ends up inside the container xy-bounds and above its base.

Usage:
    python scenario1_single_arm.py                   # viewer on, right + red
    python scenario1_single_arm.py --arm left --block green
    python scenario1_single_arm.py --headless --record /tmp/scenario1.mp4
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import mujoco
import mujoco.viewer
import mink

# Force offscreen GL if we're going to be headless.
if "--headless" in sys.argv and "MUJOCO_GL" not in __import__("os").environ:
    import os
    os.environ["MUJOCO_GL"] = "osmesa"

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ik_utils import (
    ArmHandles,
    right_arm_handles, left_arm_handles,
    RIGHT_ARM_JOINTS, LEFT_ARM_JOINTS,
    build_dof_mask, solve_kinematic_ik,
    site_pose, se3_at,
    set_arm_ctrl, set_gripper,
    GRIPPER_OPEN, GRIPPER_CLOSED,
)
from scene_utils import (
    pick_arm_for_block,
    randomize_blocks,
)
from episode_logger import (
    LeRobotWriter,
    EpisodeBuffer,
    Frame,
    CAMERAS as ALOHA_CAMERAS,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_XML = str(REPO_ROOT / "rby1_description" / "models" / "rby1a" / "mujoco" / "model.xml")

BLOCK_BODIES = {"red": "red_block", "green": "green_block", "blue": "blue_block"}
CONTAINER_BODY = "container"
BLOCK_HALF_SIZE = 0.025             # 5 cm cubes

# Vertical offsets used by the waypoint plan.
APPROACH_HEIGHT = 0.10               # hover above block on approach
GRASP_HEIGHT_ABOVE_CENTER = 0.02     # descend to slightly above block center (~5 mm above top)
LIFT_HEIGHT = 0.15                   # hover above table after grasp
RELEASE_HEIGHT = 0.08                # release just above container top
RETRACT_HEIGHT = 0.20


@dataclass
class Waypoint:
    label: str
    pos: np.ndarray          # target EE world position
    quat: mink.SO3           # target EE orientation
    gripper: str             # 'open' | 'close' | 'hold'
    duration: float          # seconds to ramp from previous target
    wait_after: float = 0.4  # seconds to hold after ramp


def settle_scene(model, data, seconds: float = 1.5) -> None:
    """Let blocks fall onto the table before we plan anything."""
    steps = int(round(seconds / model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(model, data)


def body_pos(model, data, name: str) -> np.ndarray:
    return data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)].copy()


def make_waypoints(block_pos: np.ndarray,
                   container_pos: np.ndarray,
                   ee_down_quat: mink.SO3) -> List[Waypoint]:
    """Right-way-up pick-and-place waypoint sequence."""
    # NOTE: for the RBY1 gripper the fingers hang ~10-12 cm below the EE site,
    # so "grasp" means the SITE is above the block, not at its center.
    approach = block_pos + np.array([0.0, 0.0, APPROACH_HEIGHT])
    grasp    = block_pos + np.array([0.0, 0.0, GRASP_HEIGHT_ABOVE_CENTER])
    lift     = block_pos + np.array([0.0, 0.0, LIFT_HEIGHT])
    carry    = container_pos + np.array([0.0, 0.0, LIFT_HEIGHT])
    release  = container_pos + np.array([0.0, 0.0, RELEASE_HEIGHT])
    retract  = container_pos + np.array([0.0, 0.0, RETRACT_HEIGHT])

    return [
        Waypoint("approach", approach, ee_down_quat, gripper="open",  duration=1.5),
        Waypoint("descend",  grasp,    ee_down_quat, gripper="open",  duration=1.2),
        Waypoint("grasp",    grasp,    ee_down_quat, gripper="close", duration=0.1, wait_after=0.8),
        Waypoint("lift",     lift,     ee_down_quat, gripper="hold",  duration=1.2),
        Waypoint("carry",    carry,    ee_down_quat, gripper="hold",  duration=1.8),
        Waypoint("descend2", release,  ee_down_quat, gripper="hold",  duration=1.0),
        Waypoint("release",  release,  ee_down_quat, gripper="open",  duration=0.1, wait_after=0.6),
        Waypoint("retract",  retract,  ee_down_quat, gripper="open",  duration=1.2),
    ]


def read_ctrl_snapshot(data: mujoco.MjData, arm: ArmHandles) -> np.ndarray:
    return np.array([data.ctrl[a] for a in arm.aid])


def execute_waypoints(
    model: mujoco.MjModel,
    data:  mujoco.MjData,
    arm:   ArmHandles,
    arm_mask: np.ndarray,
    waypoints: List[Waypoint],
    *,
    viewer=None,
    on_step=None,
) -> None:
    dt = model.opt.timestep
    # Anchor for the ramp: the current arm ctrl values.
    prev_target = read_ctrl_snapshot(data, arm)

    for wp in waypoints:
        # 1) Plan target arm joints for this waypoint via kinematic IK.
        target_pose = se3_at(wp.pos, wp.quat)
        q_target = solve_kinematic_ik(model, data.qpos, arm.ee_site, target_pose, arm_mask,
                                      max_iters=300)
        target_arm_ctrl = np.array([q_target[q] for q in arm.qidx])

        # 2) Gripper command is applied at the start of the waypoint.
        if wp.gripper != "hold":
            set_gripper(data, arm, wp.gripper)

        # 3) Linear joint-space ramp from prev_target to target_arm_ctrl.
        ramp_steps = max(1, int(round(wp.duration / dt)))
        for k in range(ramp_steps):
            alpha = (k + 1) / ramp_steps
            for i, aid in enumerate(arm.aid):
                data.ctrl[aid] = (1 - alpha) * prev_target[i] + alpha * target_arm_ctrl[i]
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
            if on_step is not None:
                on_step()

        # 4) Hold at target for wait_after seconds so actuators can settle.
        hold_steps = max(0, int(round(wp.wait_after / dt)))
        for i, aid in enumerate(arm.aid):
            data.ctrl[aid] = target_arm_ctrl[i]
        for _ in range(hold_steps):
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
            if on_step is not None:
                on_step()

        prev_target = target_arm_ctrl

        # Small status print.
        ee_now = data.site_xpos[arm.ee_site_id]
        err = np.linalg.norm(ee_now - wp.pos) * 1000
        print(f"  wp {wp.label:9s} target={wp.pos.round(3).tolist()} "
              f"EE={ee_now.round(3).tolist()} err={err:5.1f}mm gripper={wp.gripper}")


def check_success(model, data, block_name: str, container_pos: np.ndarray,
                  container_half_xy: float = 0.11, table_top_z: float = 0.82) -> bool:
    p = body_pos(model, data, block_name)
    inside_xy = (abs(p[0] - container_pos[0]) < container_half_xy and
                 abs(p[1] - container_pos[1]) < container_half_xy)
    above_table = p[2] > table_top_z - 0.01
    return inside_xy and above_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm",   choices=["right", "left", "auto"], default="auto",
                    help="'auto' selects the arm from the block's y coordinate")
    ap.add_argument("--block", choices=["red", "green", "blue"], default="red")
    ap.add_argument("--random", action="store_true",
                    help="jitter block spawn positions inside per-arm reach envelopes")
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed (only used with --random)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--record",   default=None, help="save third-person mp4")
    ap.add_argument("--log-dataset", default=None,
                    help="path to a LeRobot dataset root; append this episode there")
    ap.add_argument("--log-fps", type=int, default=15,
                    help="frames-per-second when writing dataset videos + parquet")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data  = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "teleop")
    mujoco.mj_resetDataKeyframe(model, data, key)

    # Optionally sample fresh block positions inside each arm's reach.
    if args.random:
        rng = np.random.default_rng(args.seed)
        placements = randomize_blocks(model, data, rng)
        print("  randomized block placements:")
        for name, p in placements.items():
            print(f"    {name}: {p.round(3).tolist()}")

    # Hold the entire pose with position ctrl (keyframe qpos -> ctrl).
    for i in range(model.nu):
        data.ctrl[i] = data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]

    # Let blocks settle onto the table.
    settle_scene(model, data, seconds=1.5)

    # Auto arm selection uses the block's y coord after settling.
    block_name_for_arm = BLOCK_BODIES[args.block]
    resolved_arm = args.arm
    if args.arm == "auto":
        resolved_arm = pick_arm_for_block(body_pos(model, data, block_name_for_arm))
        print(f"  auto-arm: block y={body_pos(model, data, block_name_for_arm)[1]:+.2f} -> {resolved_arm}")

    # Pick arm handles + DoF mask.
    if resolved_arm == "right":
        arm = right_arm_handles(model)
        joints = RIGHT_ARM_JOINTS
    else:
        arm = left_arm_handles(model)
        joints = LEFT_ARM_JOINTS
    arm_mask = build_dof_mask(model, joints)

    # EE "down-facing" orientation = whatever the initial pose gives us. Good
    # enough as long as the keyframe puts the fingers pointing at the workspace.
    ee_down = site_pose(data, model, arm.ee_site).rotation()

    block_name = BLOCK_BODIES[args.block]
    block_pos_now     = body_pos(model, data, block_name)
    container_pos_now = body_pos(model, data, CONTAINER_BODY)

    print(f"=== scenario 1  arm={args.arm}  block={args.block} ===")
    print(f"  block    world pos: {block_pos_now.round(3)}")
    print(f"  container world pos: {container_pos_now.round(3)}")

    waypoints = make_waypoints(block_pos_now, container_pos_now, ee_down)

    # Optional third-person mp4 (independent of dataset logging).
    third_person_frames: list = []
    recorder = None
    if args.record:
        recorder = mujoco.Renderer(model, height=480, width=640)

    # Optional dataset logging: renders 3 policy cameras at log_fps and records
    # state + action per frame.
    writer: LeRobotWriter | None = None
    episode: EpisodeBuffer | None = None
    log_renderer: mujoco.Renderer | None = None
    if args.log_dataset:
        writer = LeRobotWriter(args.log_dataset, fps=args.log_fps, image_wh=(224, 224))
        prompt = f"pick up the {args.block} block and put it in the brown box"
        episode = writer.new_episode(task=prompt)
        log_renderer = mujoco.Renderer(model, height=224, width=224)

    # Right/left arm handles (needed for state assembly regardless of active arm).
    right_h = right_arm_handles(model)
    left_h  = left_arm_handles(model)

    log_interval_sec = 1.0 / args.log_fps
    log_state = {"last_t": 0.0, "frame_idx": 0}

    def on_step():
        t_sim = data.time
        # 1) third-person recording (~30 Hz)
        if recorder is not None and int(t_sim / model.opt.timestep) % max(1, int(round(1.0 / 30.0 / model.opt.timestep))) == 0:
            recorder.update_scene(data, camera=-1)
            third_person_frames.append(recorder.render())

        # 2) dataset logger
        if episode is None:
            return
        if t_sim - log_state["last_t"] + 1e-9 < log_interval_sec and log_state["frame_idx"] > 0:
            return

        # ALOHA state layout: [L 6 joint, L grip, R 6 joint, R grip]
        state = np.array([
            *(data.qpos[q] for q in left_h.qidx[:6]),
            abs(data.qpos[left_h.gripper_qidx]) / abs(GRIPPER_OPEN),
            *(data.qpos[q] for q in right_h.qidx[:6]),
            abs(data.qpos[right_h.gripper_qidx]) / abs(GRIPPER_OPEN),
        ], dtype=np.float32)
        action = np.array([
            *(data.ctrl[a] for a in left_h.aid[:6]),
            (abs(data.ctrl[left_h.gripper_aid]) / abs(GRIPPER_OPEN)),
            *(data.ctrl[a] for a in right_h.aid[:6]),
            (abs(data.ctrl[right_h.gripper_aid]) / abs(GRIPPER_OPEN)),
        ], dtype=np.float32)

        images = {}
        for cam in ALOHA_CAMERAS:
            cam_name = {"cam_high": "zed_left",
                        "cam_left_wrist": "wrist_cam_l",
                        "cam_right_wrist": "wrist_cam_r"}[cam]
            log_renderer.update_scene(data, camera=cam_name)
            images[cam] = log_renderer.render()

        episode.append(Frame(state=state, action=action, images=images,
                             timestamp=float(t_sim - 0.0),
                             frame_index=log_state["frame_idx"]))
        log_state["frame_idx"] += 1
        log_state["last_t"] = t_sim

    if args.headless:
        execute_waypoints(model, data, arm, arm_mask, waypoints, on_step=on_step)
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            execute_waypoints(model, data, arm, arm_mask, waypoints,
                              viewer=viewer, on_step=on_step)

    # Assign local aliases so the mp4-saving block below keeps working.
    frames = third_person_frames

    # Success check.
    success = check_success(model, data, block_name, container_pos_now)
    print(f"\n>>> SUCCESS = {success}")
    print(f"    final block pos: {body_pos(model, data, block_name).round(3)}")

    if args.record and frames:
        try:
            import imageio
            imageio.mimsave(args.record, frames, fps=30)
            print(f"    video -> {args.record}")
        except ImportError:
            print("    (imageio missing; install it to save mp4)")

    if writer is not None and episode is not None:
        if success:
            writer.save_episode(episode)
            writer.finalize()
            print(f"    episode ({len(episode)} frames) -> {args.log_dataset}")
        else:
            print("    (episode NOT saved: success=False)")


if __name__ == "__main__":
    main()
