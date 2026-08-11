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
    python -m rby1_manipulation.tasks.block_pick                   # viewer on, right + red
    python -m rby1_manipulation.tasks.block_pick --arm left --block green
    python -m rby1_manipulation.tasks.block_pick --headless --record /tmp/scenario1.mp4
"""
from __future__ import annotations

import argparse
import sys
from typing import List

import numpy as np
import mujoco
import mujoco.viewer
import mink

# Force offscreen GL if we're going to be headless.
if "--headless" in sys.argv and "MUJOCO_GL" not in __import__("os").environ:
    import os
    os.environ["MUJOCO_GL"] = "osmesa"

from rby1_manipulation.control.ik import (
    ArmHandles,
    right_arm_handles, left_arm_handles,
    RIGHT_ARM_JOINTS, LEFT_ARM_JOINTS,
    build_dof_mask,
    site_pose,
    GRIPPER_OPEN,
)
from rby1_manipulation.control.motion import settle_scene
from rby1_manipulation.control.single_arm import Waypoint, execute_waypoints
from rby1_manipulation.evaluation.block import check_success
from rby1_manipulation.simulation.block_scene import (
    BLOCK_BODIES,
    BLOCK_HALF_SIZE,
    CONTAINER_BODY,
    MODEL_XML,
    body_pos,
)
from rby1_manipulation.simulation.common import (
    pick_arm_for_block,
    randomize_blocks,
    sample_in_reach,
    set_block_pose,
    LEFT_ARM_REACH,
    RIGHT_ARM_REACH,
)
from rby1_manipulation.data.episode import (
    LeRobotWriter,
    EpisodeBuffer,
    Frame,
    CAMERAS as ALOHA_CAMERAS,
)
# Vertical offsets used by the waypoint plan.
APPROACH_HEIGHT = 0.10               # hover above block on approach
GRASP_HEIGHT_ABOVE_CENTER = 0.02     # descend to slightly above block center (~5 mm above top)
LIFT_HEIGHT = 0.15                   # hover above table after grasp
RELEASE_HEIGHT = 0.08                # release just above container top
RETRACT_HEIGHT = 0.20


# Adaptive gripper close constants (mirror of scenario 2/3 approach).
# GRIPPER_TIGHT_INIT = 0.0: initial ctrl to drive fingers into contact.
# GRIPPER_ADAPT_OFFSET = 0.001: after contact, ctrl = qpos + offset gives
# a tiny continuous squeeze (~kp*0.001 force) that holds the block without
# crushing / creeping it inside the finger cage.
GRIPPER_TIGHT_INIT   = 0.0
GRIPPER_ADAPT_OFFSET = 0.001


def make_waypoints_pre_grasp(block_pos: np.ndarray,
                             ee_down_quat: mink.SO3) -> List[Waypoint]:
    """Pre-grasp: approach + descend with gripper OPEN. The grasp waypoint
    itself is replaced by an explicit adaptive close performed manually in
    main() (see scenarios 2/3 Phase A2 for the same pattern)."""
    approach = block_pos + np.array([0.0, 0.0, APPROACH_HEIGHT])
    grasp    = block_pos + np.array([0.0, 0.0, GRASP_HEIGHT_ABOVE_CENTER])
    return [
        Waypoint("approach", approach, ee_down_quat, gripper="open", duration=1.5),
        Waypoint("descend",  grasp,    ee_down_quat, gripper="open", duration=1.2, wait_after=0.1),
    ]


def make_waypoints_post_grasp(block_pos: np.ndarray,
                              container_pos: np.ndarray,
                              ee_down_quat: mink.SO3) -> List[Waypoint]:
    """Post-grasp: lift block from grasp pose, carry over container,
    descend, release, retract. Gripper ctrl is HELD (whatever the
    adaptive close set it to) until r_release."""
    lift     = block_pos + np.array([0.0, 0.0, LIFT_HEIGHT])
    carry    = container_pos + np.array([0.0, 0.0, LIFT_HEIGHT])
    release  = container_pos + np.array([0.0, 0.0, RELEASE_HEIGHT])
    retract  = container_pos + np.array([0.0, 0.0, RETRACT_HEIGHT])
    return [
        Waypoint("lift",     lift,    ee_down_quat, gripper="hold", duration=1.2),
        Waypoint("carry",    carry,   ee_down_quat, gripper="hold", duration=1.8),
        Waypoint("descend2", release, ee_down_quat, gripper="hold", duration=1.0),
        Waypoint("release",  release, ee_down_quat, gripper="open", duration=0.1, wait_after=0.6),
        Waypoint("retract",  retract, ee_down_quat, gripper="open", duration=1.2),
    ]


def hold_ctrl_for_secs(model, data, secs, viewer=None, on_step=None):
    """Freeze ctrl (positions + grippers as-is) and step sim for `secs`."""
    steps = int(round(secs / model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        if on_step is not None:
            on_step()


def return_arm_to_rest(model, data, arm, rest_ctrl_snapshot, duration,
                       viewer=None, on_step=None):
    """Ramp all 7 arm joints back to their keyframe ctrl values over
    `duration` seconds. Gripper ctrl is left as-is (open after release)."""
    dt = model.opt.timestep
    steps = max(1, int(round(duration / dt)))
    current = np.array([float(data.ctrl[a]) for a in arm.aid])
    target  = np.array([float(rest_ctrl_snapshot[a]) for a in arm.aid])
    for k in range(steps):
        alpha = (k + 1) / steps
        for i, aid in enumerate(arm.aid):
            data.ctrl[aid] = (1.0 - alpha) * current[i] + alpha * target[i]
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        if on_step is not None:
            on_step()


def adaptive_close_gripper(model: mujoco.MjModel, data: mujoco.MjData,
                           arm: ArmHandles, viewer=None, on_step=None) -> None:
    """Two-step close matching scenarios 2/3:
    (a) drive ctrl to GRIPPER_TIGHT_INIT so fingers make contact with the block,
    (b) read actual gripper qpos, then set ctrl = qpos + tiny offset so the
        holding force reduces to kp*offset (no continuous squeeze / creep).
    """
    dt = model.opt.timestep
    # (a) initial hard close so fingers reach the block
    data.ctrl[arm.gripper_aid] = GRIPPER_TIGHT_INIT
    for _ in range(int(round(0.5 / dt))):
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        if on_step is not None:
            on_step()
    # (b) match ctrl to contact qpos plus a small squeeze offset
    contact_qpos = float(data.qpos[arm.gripper_qidx])
    data.ctrl[arm.gripper_aid] = contact_qpos + GRIPPER_ADAPT_OFFSET
    for _ in range(int(round(0.3 / dt))):
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        if on_step is not None:
            on_step()
    print(f"    [GRIPPER] contact qpos={contact_qpos:+.4f}  "
          f"ctrl={data.ctrl[arm.gripper_aid]:+.4f}  (tiny squeeze)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm",   choices=["right", "left", "auto"], default="auto",
                    help="'auto' selects the arm from the block's y coordinate")
    ap.add_argument("--block", choices=["red", "green", "blue"], default="red")
    ap.add_argument("--random", action="store_true",
                    help="jitter block spawn positions inside per-arm reach envelopes")
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed (only used with --random)")
    ap.add_argument("--spawn-side", choices=["auto", "left", "right"], default="auto",
                    help="If 'left' or 'right', respawn the chosen --block into "
                         "that arm's reach envelope (overrides the natural color "
                         "-> side mapping used by --random). Use with --arm to "
                         "collect data where either arm handles any color.")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--record",   default=None, help="save third-person mp4")
    ap.add_argument("--log-dataset", default=None,
                    help="path to a LeRobot dataset root; append this episode there")
    ap.add_argument("--log-fps", type=int, default=15,
                    help="frames-per-second when writing dataset videos + parquet")
    ap.add_argument("--task-prompt", default=None,
                    help="override the LeRobot task prompt string (else auto-built)")
    ap.add_argument("--save-failed", action="store_true",
                    help="Also save the LeRobot episode when SUCCESS=False. "
                         "The task prompt gets a '[FAIL] ' prefix so failed "
                         "episodes are easy to filter out later.")
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

    # Override: force the target block into a specific arm's envelope. Used
    # for data collection where each color must be reachable by either arm.
    # Collision avoidance uses the current positions of the OTHER two blocks
    # so the target doesn't spawn on top of them.
    if args.spawn_side != "auto":
        envelope = LEFT_ARM_REACH if args.spawn_side == "left" else RIGHT_ARM_REACH
        override_rng = np.random.default_rng(args.seed if args.seed is not None else 0)
        target_joint = f"{BLOCK_BODIES[args.block]}_free"
        other_positions = [
            body_pos(model, data, BLOCK_BODIES[c])
            for c in ("red", "green", "blue") if c != args.block
        ]
        override_pos = sample_in_reach(override_rng, envelope,
                                       other_positions=other_positions, z=0.87)
        set_block_pose(model, data, target_joint, override_pos)
        print(f"  spawn-side override: {args.block} -> {args.spawn_side} envelope "
              f"at {override_pos.round(3).tolist()}")

    # Hold the entire pose with position ctrl (keyframe qpos -> ctrl).
    for i in range(model.nu):
        data.ctrl[i] = data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]

    # Let blocks settle onto the table.
    settle_scene(model, data, seconds=1.5)

    # Snapshot the ctrl values from the keyframe/rest pose so we can send
    # the arm back to its original posture after placement.
    rest_ctrl_snapshot = data.ctrl.copy()

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

    waypoints_pre  = make_waypoints_pre_grasp(block_pos_now, ee_down)
    waypoints_post = make_waypoints_post_grasp(block_pos_now, container_pos_now, ee_down)

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
        prompt = args.task_prompt or f"pick up the {args.block} block and put it in the brown box"
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

    def run(viewer=None):
        # Phase 1: approach + descend (gripper open)
        execute_waypoints(model, data, arm, arm_mask, waypoints_pre,
                          viewer=viewer, on_step=on_step)
        # Phase 2: adaptive close (matches scenarios 2/3 pattern)
        print("  --- grasp: adaptive gripper close (0.0 -> contact + 0.001) ---")
        adaptive_close_gripper(model, data, arm, viewer=viewer, on_step=on_step)
        # Phase 3: lift + carry + release + retract (gripper hold, then open at release)
        execute_waypoints(model, data, arm, arm_mask, waypoints_post,
                          viewer=viewer, on_step=on_step)
        # Phase 4: return arm to rest posture
        print("  --- rest return ---")
        return_arm_to_rest(model, data, arm, rest_ctrl_snapshot,
                           duration=2.5, viewer=viewer, on_step=on_step)
        # Phase 5: tail hold 1.5s so the sim settles before check_success
        print("  --- tail hold 1.5s ---")
        hold_ctrl_for_secs(model, data, 1.5, viewer=viewer, on_step=on_step)

    if args.headless:
        run()
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            run(viewer=viewer)

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
        elif args.save_failed:
            # Prefix the task string so failures are trivially filterable later.
            episode.task = f"[FAIL] {episode.task}"
            writer.save_episode(episode)
            writer.finalize()
            print(f"    [FAIL] episode ({len(episode)} frames) -> {args.log_dataset}")
        else:
            print("    (episode NOT saved: success=False)")


if __name__ == "__main__":
    main()
