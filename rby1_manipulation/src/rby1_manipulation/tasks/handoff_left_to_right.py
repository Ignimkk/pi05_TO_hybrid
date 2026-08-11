"""Scenario 2: LEFT arm picks a block, hands off MID-AIR to the RIGHT arm
along the Y axis (both arms facing each other), right places it in the box.

*** Y-axis aligned aerial handoff. Non-overlapping. ***

Design (per user's feedback):
    - LEFT arm sits at (x, +Y, z) facing the block along the -Y direction.
      Its wrist is rotated -90 deg around world X so the tool axis points
      in world -Y and its finger-close axis is world Z. Left grips the
      +/-Z (top/bottom) faces of the block.
    - RIGHT arm sits at (x, -Y, z) facing the block along the +Y direction.
      Its wrist is rotated +90 deg around world X (tool -> +Y) and then
      +90 deg around world Y (grip axis Z -> X). Right grips the +/-X
      (front/back) faces of the block.
    - The two arms only differ in their y coordinate and their orientation.
      Their housings are ~22 cm apart along Y. Their fingers curl inward
      to the same target block position but with PERPENDICULAR grip axes,
      so the finger volumes never intersect.

Handoff sequence:
    A) LEFT picks the block off the table with the default fingers-down
       orientation, lifts.
    B) LEFT carries the block up and rotates its wrist to the "face right"
       pose LEFT_HANDOFF_EE_POS. The block reorients with the wrist.
    C) RIGHT swings to a staging pose off to the -Y side already in the
       "face left" orientation, gripper OPEN.
    D) RIGHT approaches along -Y -> handoff pose RIGHT_HANDOFF_EE_POS,
       fingers open, straddling the block on its +/-X faces.
    E) RIGHT CLOSES the gripper. Both arms are now holding the block.
    F) LEFT OPENS the gripper (block held only by right).
    G) LEFT retracts up and away.
    H) RIGHT unwinds its wrist back to the default fingers-down orientation
       while carrying, delivers to the container, releases.

Usage:
    python -m rby1_manipulation.tasks.handoff_left_to_right                   # viewer, red
    python -m rby1_manipulation.tasks.handoff_left_to_right --block green --random --seed 3
    python -m rby1_manipulation.tasks.handoff_left_to_right --headless \
        --log-dataset /tmp/rby1_scenario2

NOTE: this design has not yet been tuned to 100% success. The 90 deg wrist
rotations on both arms cost some IK accuracy; the block reorientation
during left's Phase B may cause slippage on its gripper. Tune
LEFT_HANDOFF_EE_POS.y / RIGHT_HANDOFF_EE_POS.y and the pre-approach
offsets if the fingers miss the block.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np
import mujoco
import mujoco.viewer
import mink

if "--headless" in sys.argv and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "osmesa"

from rby1_manipulation.control.ik import (
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
    CONTAINER_BODY,
    MODEL_XML,
    body_pos,
)
from rby1_manipulation.simulation.common import set_block_pose
from rby1_manipulation.data.episode import LeRobotWriter, Frame, CAMERAS as ALOHA_CAMERAS

# --- Configuration ---
LEFT_PICK_ENVELOPE = dict(x=(0.45, 0.55), y=(0.17, 0.22))

# Y-axis facing handoff. Both EE sites are at the same (x, z), separated
# only in y. Their tool axes point at each other along Y. Finger length is
# ~10 cm from the EE site along the tool axis, so the grip zone lands
# roughly at y = 0 for these values.
# Handoff column: mirror-symmetric between the two arms. LEFT and RIGHT
# reach the same (x, z) from opposite Y sides; only the sign of y differs.
# Because arm poses are mirror-symmetric, both IK problems are equally
# reachable.
HANDOFF_X = 0.50
HANDOFF_Z = 1.15

# LEFT is nudged toward the center (from +0.10 to +0.04) so the block
# ends up sitting on the -Y side of the workspace midline. This gives
# RIGHT arm a more natural, extended pose at engage instead of forcing
# its shoulder to swing inward (which was making RIGHT's armpit collide
# with either its own torso or LEFT's forearm on the way in).
# RIGHT's target Y is now computed dynamically from the block pose in
# build_phase_c_right_approach_and_close(), so the constant below is
# no longer read by phase C — it only labels the "expected" pose for
# print output.
LEFT_HANDOFF_Y   = +0.04   # LEFT tool -Y, grip zone y = +0.04 - 0.10 = -0.06
RIGHT_HANDOFF_Y  = -0.16   # (informational only) RIGHT grip zone y = -0.06

LEFT_HANDOFF_EE_POS  = np.array([HANDOFF_X, LEFT_HANDOFF_Y,  HANDOFF_Z])
RIGHT_HANDOFF_EE_POS = np.array([HANDOFF_X, RIGHT_HANDOFF_Y, HANDOFF_Z])

# Timings (sim seconds)
LEFT_RELEASE_HOLD_SECS = 0.6   # after left opens, before left retracts

# Gripper close target for the RIGHT arm's pick. Pushed past 0.0 (which is
# the "just touching" position for the finger joints) so the pads squeeze
# the block more firmly than the default GRIPPER_CLOSED value.
GRIPPER_TIGHT = 0.02

CAM_NAME_MAP = {
    "cam_high":        "zed_left",
    "cam_left_wrist":  "wrist_cam_l",
    "cam_right_wrist": "wrist_cam_r",
}


def spawn_pick_block_in_left_envelope(model, data, joint_name, rng, z=0.87):
    env = LEFT_PICK_ENVELOPE
    if rng is None:
        x = 0.5 * (env["x"][0] + env["x"][1])
        y = 0.5 * (env["y"][0] + env["y"][1])
    else:
        x = rng.uniform(*env["x"])
        y = rng.uniform(*env["y"])
    set_block_pose(model, data, joint_name, [x, y, z])


def build_phase_a1_left_descend(block_pos, left_ee_down) -> list:
    """LEFT phase A1: approach + descend (gripper open). Fingers are
    positioned around the block, ready for the explicit adaptive close
    performed manually in Phase A2 (see run())."""
    approach = block_pos + np.array([0.0, 0.0, 0.10])
    grasp    = block_pos + np.array([0.0, 0.0, 0.02])
    return [
        Waypoint("l_approach", approach, left_ee_down, "open", 1.5),
        Waypoint("l_descend",  grasp,    left_ee_down, "open", 1.2, wait_after=0.1),
    ]


def build_phase_a3_left_lift(block_pos, left_ee_down) -> list:
    """LEFT phase A3: lift block straight up while HOLDING the gripper
    ctrl at whatever tight value Phase A2 set."""
    lift = block_pos + np.array([0.0, 0.0, 0.18])
    return [
        Waypoint("l_lift", lift, left_ee_down, "hold", 1.2, wait_after=0.4),
    ]


def build_phase_b_left_carry_to_handoff(left_ee_face) -> list:
    """LEFT: single-shot carry from the lift pose directly to the handoff
    pose. Translation + wrist rotation happen simultaneously over a long
    ramp so the block stays gripped."""
    return [
        Waypoint("l_carry_to_handoff", LEFT_HANDOFF_EE_POS, left_ee_face,
                 duration=3.5, wait_after=1.0, gripper="hold"),
    ]


def build_phase_c1_right_pre_approach(right_ee_face_mirror, block_p) -> list:
    """RIGHT pre_approach with MIRROR orientation (tool +Y, grip Z, no
    wrist twist). EE at block.y - 0.05 (5 cm -Y from block). Easy IK
    because it's just a mirror of LEFT's face-center pose."""
    pre_approach = block_p + np.array([0.0, -0.15, 0.0])
    return [
        Waypoint("r_pre_approach", pre_approach, right_ee_face_mirror,
                 "open", 3.0, wait_after=0.4),
    ]


def build_phase_c3_right_approach(target_rot, block_p) -> list:
    """RIGHT approach to the block position with target orientation aligned
    to the block's actual face normals (built by
    compute_right_target_rot_aligned_to_block below)."""
    approach = block_p - np.array([0.03, 0.017, 0.0])
    return [
        Waypoint("r_approach", approach, target_rot,
                 "open", 1.5, wait_after=0.4),
    ]


def compute_right_target_rot_aligned_to_block(model, data, block_name):
    """Build a target rotation for RIGHT's EE such that:
      * tool axis (world) = +Y (approach block from -Y side)
      * grip axis (world) = block's X-face normal, projected perpendicular
        to +Y. This way when the gripper closes, its fingers pinch two
        opposite faces of the block flush — no diagonal slip.

    Also prints the block's world-frame axes for diagnostics.
    """
    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, block_name)
    block_xmat = data.xmat[block_id].reshape(3, 3).copy()
    print(f"    [BLOCK ORIENT] block axes in world:")
    print(f"      block +X in world: {block_xmat[:, 0].round(3)}")
    print(f"      block +Y in world: {block_xmat[:, 1].round(3)}")
    print(f"      block +Z in world: {block_xmat[:, 2].round(3)}")

    tool_axis = np.array([0.0, 1.0, 0.0])   # world +Y (RIGHT approach direction)

    # Pick the block face normal LEAST parallel to tool_axis (so grip axis
    # actually has room to be perpendicular to tool and aligned with a
    # face). LEFT is gripping the block-frame axis that is now most
    # aligned with world Z, so we should pick something else here.
    best_axis = None
    best_score = -1.0
    for i in range(3):
        for sign in (+1.0, -1.0):
            cand = sign * block_xmat[:, i]
            # Rejection score: prefer axes that are perpendicular to tool
            # AND perpendicular to world Z (LEFT's grip direction).
            perp_to_tool = 1.0 - abs(np.dot(cand, tool_axis))
            perp_to_left_grip = 1.0 - abs(np.dot(cand, np.array([0.0, 0.0, 1.0])))
            score = perp_to_tool * perp_to_left_grip
            if score > best_score:
                best_score = score
                best_axis = cand

    # Project the chosen block face normal to be perpendicular to tool
    grip_axis = best_axis - np.dot(best_axis, tool_axis) * tool_axis
    grip_axis = grip_axis / np.linalg.norm(grip_axis)
    third_axis = np.cross(grip_axis, tool_axis)
    third_axis = third_axis / np.linalg.norm(third_axis)

    # Column convention: [body_x, body_y, body_z] = [third, grip, tool].
    # This matches the RBY1 gripper site (body +z along finger direction,
    # body +y along grip-close direction).
    R_target = np.column_stack([third_axis, grip_axis, tool_axis])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R_target.reshape(-1))
    print(f"    [C3 TARGET] tool={tool_axis.round(3)} grip={grip_axis.round(3)} third={third_axis.round(3)}")
    return mink.SO3(q)


def do_right_wrist_twist_j6(model, data, arm, delta_rad, duration,
                            viewer=None, on_step=None):
    """Ramp ONLY the last wrist joint (arm.aid[6]) by delta_rad over
    `duration` seconds. All other arm ctrl values (0-5) and both grippers
    stay put — pure wrist roll around the tool axis with no shoulder or
    elbow reconfiguration."""
    dt = model.opt.timestep
    steps = max(1, int(round(duration / dt)))
    j6_aid = arm.aid[6]
    q6_start = float(data.ctrl[j6_aid])
    q6_end   = q6_start + delta_rad
    for k in range(steps):
        alpha = (k + 1) / steps
        data.ctrl[j6_aid] = (1.0 - alpha) * q6_start + alpha * q6_end
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        if on_step is not None:
            on_step()


def return_arm_to_rest(model, data, arm, rest_ctrl_snapshot, duration,
                       viewer=None, on_step=None):
    """Ramp all 7 arm joints back to their initial (rest / keyframe) ctrl
    values over `duration` seconds. Gripper ctrl is left as-is so the
    block stays gripped during the return motion."""
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


def make_bg_ramp_on_step(model, data, arm, target_ctrl_snapshot, ramp_secs,
                         existing_on_step=None):
    """Return an on_step callback that ramps `arm` ctrl from its CURRENT
    values toward `target_ctrl_snapshot[arm.aid]` over `ramp_secs` sim
    seconds. Once the ramp completes, the callback stops touching arm
    ctrl. Chains to `existing_on_step` (which is called every step)."""
    dt = model.opt.timestep
    steps_total = max(1, int(round(ramp_secs / dt)))
    start_ctrl  = np.array([float(data.ctrl[a]) for a in arm.aid])
    target_ctrl = np.array([float(target_ctrl_snapshot[a]) for a in arm.aid])
    step_counter = {"n": 0}

    def bg_step():
        n = step_counter["n"]
        if n < steps_total:
            alpha = (n + 1) / steps_total
            for i, aid in enumerate(arm.aid):
                data.ctrl[aid] = (1.0 - alpha) * start_ctrl[i] + alpha * target_ctrl[i]
            step_counter["n"] = n + 1
        if existing_on_step is not None:
            existing_on_step()
    return bg_step


def build_phase_g_left_retract(left_ee_face) -> list:
    """LEFT: single short retract (~15 cm +Y from handoff) just to clear
    the space above the block. Actual return to rest posture happens in
    ctrl-space right after this, in parallel with RIGHT's delivery."""
    retract_short = LEFT_HANDOFF_EE_POS + np.array([0.0, 0.20, 0.0])
    return [
        Waypoint("l_retract_short", retract_short, left_ee_face, "open", 0.8, wait_after=0.1),
    ]


def build_phase_h_right_deliver(container_pos, right_ee_down) -> list:
    """RIGHT: carry the block to the container while IK unwinds the wrist
    from whatever twisted orientation it ended phase C in back to the
    default fingers-down orientation. The long ramp on the first waypoint
    gives IK time to smoothly untwist joint 6 + reorient the shoulder."""
    lift_down   = np.array([HANDOFF_X, -0.10, HANDOFF_Z + 0.05])
    over_cont   = container_pos + np.array([0.0, 0.0, 0.15])
    descend     = container_pos + np.array([0.0, 0.0, 0.08])
    retract     = container_pos + np.array([0.0, 0.0, 0.20])
    return [
        Waypoint("r_lift_and_unwind", lift_down,  right_ee_down, "hold", 3.5, wait_after=0.5),
        Waypoint("r_over_container",  over_cont,  right_ee_down, "hold", 1.8),
        Waypoint("r_descend_cont",    descend,    right_ee_down, "hold", 1.0),
        Waypoint("r_release",         descend,    right_ee_down, "open", 0.1, wait_after=0.5),
        Waypoint("r_retract",         retract,    right_ee_down, "open", 1.2),
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


def interactive_right_control(model, data, viewer, right_arm, right_mask,
                              right_target_body="link_right_arm_6_target",
                              print_interval_secs=1.0):
    """LEFT stays frozen at its current ctrl. RIGHT follows a mocap sphere
    (drag it in the viewer). Prints RIGHT joint qpos + EE pose every
    `print_interval_secs` sim seconds so you can read off good joint values.

    Controls (in the passive viewer):
        double-click the small red target sphere for the RIGHT arm to
        select it, then Ctrl + right-click + drag  -> translate
                       Ctrl + left-click  + drag  -> rotate.
    Ctrl+C in terminal or close the viewer window to exit.
    """
    import time

    body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, right_target_body)
    mocap_id = model.body_mocapid[body_id]
    ee_site_id = right_arm.ee_site_id

    # Snap mocap to current EE so nothing jumps.
    data.mocap_pos[mocap_id] = data.site_xpos[ee_site_id].copy()
    mat = data.site_xmat[ee_site_id].reshape(3, 3)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat.reshape(-1))
    data.mocap_quat[mocap_id] = q

    # Make the mocap markers visible.
    viewer.opt.geomgroup[3] = 1
    viewer.opt.sitegroup[4] = 1

    print()
    print("=" * 70)
    print("INTERACTIVE MODE — RIGHT arm follows mocap; LEFT is frozen holding block.")
    print("  Double-click the red target sphere for RIGHT arm, then:")
    print("    Ctrl + right-click + drag  -> translate")
    print("    Ctrl + left-click  + drag  -> rotate")
    print(f"  RIGHT joint qpos + EE pose printed every {print_interval_secs:.1f}s.")
    print("  Close the viewer window to end interactive mode.")
    print("=" * 70)
    print()

    # mink IK setup (arm-masked)
    config = mink.Configuration(model)
    config.update(data.qpos)

    task = mink.FrameTask(right_arm.ee_site, "site",
                          position_cost=100.0, orientation_cost=10.0,
                          lm_damping=1e-3)
    posture = mink.PostureTask(model, cost=1e-4)
    posture.set_target_from_configuration(config)

    CTRL_HZ = 60
    IK_INNER_ITERS = 10
    IK_INNER_DT = 1e-2
    dt = 1.0 / CTRL_HZ
    sim_steps_per_ctrl = max(1, int(round(dt / model.opt.timestep)))

    last_print_t = data.time
    while viewer.is_running():
        step_start = time.time()

        # Read mocap target
        target = mink.SE3.from_rotation_and_translation(
            mink.SO3(np.asarray(data.mocap_quat[mocap_id]).copy()),
            np.asarray(data.mocap_pos[mocap_id]).copy(),
        )
        task.set_target(target)

        # Solve IK (right arm DoFs only)
        config.update(data.qpos)
        for _ in range(IK_INNER_ITERS):
            vel = mink.solve_ik(config, [task, posture], IK_INNER_DT,
                                solver="daqp", damping=1e-4)
            vel = vel * right_mask
            config.integrate_inplace(vel, IK_INNER_DT)

        # Push RIGHT joint targets into ctrl. LEFT ctrl is untouched
        # so LEFT arm keeps holding the block at its current pose.
        for i, aid in enumerate(right_arm.aid):
            data.ctrl[aid] = config.q[right_arm.qidx[i]]

        # Step sim
        for _ in range(sim_steps_per_ctrl):
            mujoco.mj_step(model, data)
        viewer.sync()

        # Periodic print
        if data.time - last_print_t >= print_interval_secs:
            last_print_t = data.time
            qpos_right = np.array([data.qpos[q] for q in right_arm.qidx])
            ctrl_right = np.array([data.ctrl[a] for a in right_arm.aid])
            ee_pos = data.site_xpos[ee_site_id].copy()
            ee_mat = data.site_xmat[ee_site_id].reshape(3, 3)
            print(f"[t={data.time:6.2f}s] RIGHT qpos = "
                  f"[{', '.join(f'{q:+.3f}' for q in qpos_right)}]")
            print(f"           RIGHT ctrl = "
                  f"[{', '.join(f'{c:+.3f}' for c in ctrl_right)}]")
            print(f"           EE pos = {ee_pos.round(3)}  "
                  f"tool_world(col2)={ee_mat[:, 2].round(3)}  "
                  f"grip_world(col1)={ee_mat[:, 1].round(3)}")

        # Real-time pacing
        wait = dt - (time.time() - step_start)
        if wait > 0:
            time.sleep(wait)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", choices=["red", "green", "blue"], default="red")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--record", default=None)
    ap.add_argument("--log-dataset", default=None)
    ap.add_argument("--log-fps", type=int, default=15)
    ap.add_argument("--interactive-right", action="store_true",
                    help="Run phases A+B only, then hand control of the "
                         "RIGHT arm to the user via a mocap sphere in the "
                         "viewer. Prints RIGHT joint values periodically.")
    ap.add_argument("--task-prompt", default=None,
                    help="override the LeRobot task prompt string (else auto-built)")
    ap.add_argument("--save-failed", action="store_true",
                    help="Also save the LeRobot episode when SUCCESS=False "
                         "(task prompt prefixed with '[FAIL] ').")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data  = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "teleop")
    mujoco.mj_resetDataKeyframe(model, data, key)

    block_name = BLOCK_BODIES[args.block]
    joint_name = f"{block_name}_free"
    rng = np.random.default_rng(args.seed) if args.random else None
    spawn_pick_block_in_left_envelope(model, data, joint_name, rng)

    for i in range(model.nu):
        data.ctrl[i] = data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]
    settle_scene(model, data, seconds=1.5)

    # Snapshot the ctrl values from the keyframe/rest pose. Used at the
    # end to send the RIGHT arm back to its original posture after handoff.
    rest_ctrl_snapshot = data.ctrl.copy()

    left_arm  = left_arm_handles(model)
    right_arm = right_arm_handles(model)
    left_mask  = build_dof_mask(model, LEFT_ARM_JOINTS)
    right_mask = build_dof_mask(model, RIGHT_ARM_JOINTS)

    # Orientations
    left_ee_down  = site_pose(data, model, left_arm.ee_site).rotation()
    right_ee_down = site_pose(data, model, right_arm.ee_site).rotation()

    # LEFT face-center: R_x(-pi/2) rotates fingers-down to fingers-toward
    # -Y. Grip axis becomes world Z (fingers close on block's world +/-Z
    # faces after the wrist rotation carries the block through).
    left_ee_face  = mink.SO3.from_x_radians(-np.pi / 2) @ left_ee_down

    # RIGHT face_mirror: mirror of LEFT. Tool +Y, grip Z (no wrist twist).
    # Used only for phase C1 pre_approach. Phase C3's target orientation
    # is read live from the sim after C2's manual joint-6 twist, so no
    # separate "perp" orientation matrix is needed here.
    right_ee_face_mirror = mink.SO3.from_x_radians(+np.pi / 2) @ right_ee_down

    block_pos_now     = body_pos(model, data, block_name)
    container_pos_now = body_pos(model, data, CONTAINER_BODY)

    print(f"=== scenario 2 Y-axis handoff  block={args.block} ===")
    print(f"  block start        : {block_pos_now.round(3)}")
    print(f"  LEFT  handoff EE  : {LEFT_HANDOFF_EE_POS.round(3)}  (tool -Y, grip Z, top/bottom faces)")
    print(f"  RIGHT (info only) : {RIGHT_HANDOFF_EE_POS.round(3)}  (tool +Y, grip X @ approach, +/-X side faces)")

    phase_a1 = build_phase_a1_left_descend(block_pos_now, left_ee_down)
    phase_a3 = build_phase_a3_left_lift(block_pos_now, left_ee_down)
    phase_b = build_phase_b_left_carry_to_handoff(left_ee_face)
    phase_g = build_phase_g_left_retract(left_ee_face)
    # phase_c is built inside run() using the actual block pose after
    # phase B (see build_phase_c_right_arm_approach docstring).
    # Container delivery (phase H) is currently disabled — after handoff
    # both arms just return to their rest posture for validation.
    _ = container_pos_now  # kept for future container-delivery phase

    third_person_frames: list = []
    recorder = None
    if args.record:
        recorder = mujoco.Renderer(model, height=480, width=640)

    writer = None
    episode = None
    log_renderer = None
    if args.log_dataset:
        writer = LeRobotWriter(args.log_dataset, fps=args.log_fps, image_wh=(224, 224))
        prompt = args.task_prompt or (
            f"pick up the {args.block} block with the left hand, hand it off "
            f"to the right hand in mid-air, and place it in the brown box")
        episode = writer.new_episode(task=prompt)
        log_renderer = mujoco.Renderer(model, height=224, width=224)

    log_state = {"last_t": 0.0, "frame_idx": 0}
    log_interval = 1.0 / args.log_fps

    def on_step():
        t_sim = data.time
        if recorder is not None and int(t_sim / model.opt.timestep) % max(1, int(round(1.0 / 30.0 / model.opt.timestep))) == 0:
            recorder.update_scene(data, camera=-1)
            third_person_frames.append(recorder.render())
        if episode is None:
            return
        if t_sim - log_state["last_t"] + 1e-9 < log_interval and log_state["frame_idx"] > 0:
            return
        state = np.array([
            *(data.qpos[q] for q in left_arm.qidx[:6]),
            abs(data.qpos[left_arm.gripper_qidx]) / abs(GRIPPER_OPEN),
            *(data.qpos[q] for q in right_arm.qidx[:6]),
            abs(data.qpos[right_arm.gripper_qidx]) / abs(GRIPPER_OPEN),
        ], dtype=np.float32)
        action = np.array([
            *(data.ctrl[a] for a in left_arm.aid[:6]),
            abs(data.ctrl[left_arm.gripper_aid]) / abs(GRIPPER_OPEN),
            *(data.ctrl[a] for a in right_arm.aid[:6]),
            abs(data.ctrl[right_arm.gripper_aid]) / abs(GRIPPER_OPEN),
        ], dtype=np.float32)
        images = {}
        for cam in ALOHA_CAMERAS:
            log_renderer.update_scene(data, camera=CAM_NAME_MAP[cam])
            images[cam] = log_renderer.render()
        episode.append(Frame(state=state, action=action, images=images,
                             timestamp=float(t_sim),
                             frame_index=log_state["frame_idx"]))
        log_state["frame_idx"] += 1
        log_state["last_t"] = t_sim

    def run(viewer=None):
        # A1: approach + descend (gripper open)
        print("  --- phase A1: LEFT approach + descend (gripper open) ---")
        execute_waypoints(model, data, left_arm, left_mask, phase_a1,
                          viewer=viewer, on_step=on_step)

        # A2: adaptive close — initial close (0.0, 0.5s) then match ctrl to
        # contact qpos + 0.001 so grip force reduces to ~kp * 0.001. This
        # is the same pattern used in scenarios 1 and 3.
        print("  --- phase A2: LEFT adaptive close (0.0 -> contact + 0.001) ---")
        dt = model.opt.timestep
        data.ctrl[left_arm.gripper_aid] = 0.0
        for _ in range(int(round(0.5 / dt))):
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
            on_step()
        contact_qpos = float(data.qpos[left_arm.gripper_qidx])
        data.ctrl[left_arm.gripper_aid] = contact_qpos + 0.001
        for _ in range(int(round(0.3 / dt))):
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
            on_step()
        print(f"    [LEFT GRIPPER] contact qpos={contact_qpos:+.4f}  "
              f"ctrl={data.ctrl[left_arm.gripper_aid]:+.4f}  (tiny squeeze)")

        # A3: lift block straight up
        print("  --- phase A3: LEFT lift ---")
        execute_waypoints(model, data, left_arm, left_mask, phase_a3,
                          viewer=viewer, on_step=on_step)

        print("  --- phase B: LEFT carries + rotates wrist to handoff pose ---")
        execute_waypoints(model, data, left_arm, left_mask, phase_b,
                          viewer=viewer, on_step=on_step)
        p = body_pos(model, data, block_name)
        left_ee_after_b = data.site_xpos[left_arm.ee_site_id].copy()
        print(f"    [LEFT STOPPED] block pos    : {p.round(3)}")
        print(f"    [LEFT STOPPED] LEFT EE pos  : {left_ee_after_b.round(3)}")
        print(f"    [LEFT STOPPED] block - EE   : {(p - left_ee_after_b).round(3)}")

        # Interactive mode: skip phase C onwards. Freeze LEFT, hand RIGHT
        # over to the user via mocap. Only usable with the viewer.
        if args.interactive_right:
            if viewer is None:
                print("!! --interactive-right requires viewer (do not pass --headless)")
                return
            interactive_right_control(model, data, viewer, right_arm, right_mask)
            return

        # C1: RIGHT pre_approach with MIRROR orientation. IK computes a
        # natural arm shape 5 cm -Y from the block.
        phase_c1 = build_phase_c1_right_pre_approach(right_ee_face_mirror, p)
        print("  --- phase C1: RIGHT pre_approach (mirror, 5 cm -Y from block) ---")
        execute_waypoints(model, data, right_arm, right_mask, phase_c1,
                          viewer=viewer, on_step=on_step)

        # C2: Rotate joint 6 by -pi/2 (wrist roll around tool axis, no
        # shoulder/elbow motion). This puts grip axis on X ready to grab
        # the block's SIDE faces.
        print("  --- phase C2: RIGHT wrist twist -pi/2 (joint 6 only) ---")
        do_right_wrist_twist_j6(model, data, right_arm, np.pi / 2, 1.5,
                                viewer=viewer, on_step=on_step)

        # C3: Read the actual EE orientation AFTER C2's joint-6 twist and
        # use it directly as the IK target. IK won't try to redo the twist
        # — just translates to block position.
        right_current_rot = site_pose(data, model, right_arm.ee_site).rotation()
        phase_c3 = build_phase_c3_right_approach(right_current_rot, p)
        print("  --- phase C3: RIGHT approach (target orient = current post-twist EE) ---")
        execute_waypoints(model, data, right_arm, right_mask, phase_c3,
                          viewer=viewer, on_step=on_step)

        # C4: Close gripper tight around the block's side faces.
        print(f"  --- phase C4: RIGHT close gripper tight (ctrl={GRIPPER_TIGHT}) ---")
        data.ctrl[right_arm.gripper_aid] = GRIPPER_TIGHT
        hold_ctrl_for_secs(model, data, 0.8, viewer=viewer, on_step=on_step)
        p = body_pos(model, data, block_name)
        right_ee_after_pick = data.site_xpos[right_arm.ee_site_id].copy()
        print(f"    [RIGHT PICKED] block pos    : {p.round(3)}")
        print(f"    [RIGHT PICKED] RIGHT EE pos : {right_ee_after_pick.round(3)}")
        print(f"    [RIGHT PICKED] block - EE   : {(p - right_ee_after_pick).round(3)}")
        # Compare LEFT vs RIGHT EE positions to verify Y-axis alignment.
        delta_ee = right_ee_after_pick - left_ee_after_b
        print(f"    [ALIGNMENT]    RIGHT EE - LEFT EE = {delta_ee.round(3)}  "
              f"(x={delta_ee[0]*1000:+.1f}mm, y={delta_ee[1]*1000:+.1f}mm, z={delta_ee[2]*1000:+.1f}mm)")

        print("  --- phase D: LEFT OPENS gripper (block held by right only) ---")
        data.ctrl[left_arm.gripper_aid] = GRIPPER_OPEN
        hold_ctrl_for_secs(model, data, LEFT_RELEASE_HOLD_SECS,
                           viewer=viewer, on_step=on_step)
        p = body_pos(model, data, block_name)
        z_dropped = p[2] < 0.90
        print(f"    block after left-open: {p.round(3)}   "
              f"{'FELL' if z_dropped else 'held by right'}")

        print("  --- phase E: LEFT retracts ---")
        execute_waypoints(model, data, left_arm, left_mask, phase_g,
                          viewer=viewer, on_step=on_step)

        # F: RIGHT delivers block to container. LEFT ramps back to its
        # rest posture IN PARALLEL during the first ~1.5 s of the delivery
        # (chained via the on_step callback that fires every sim step).
        print("  --- phase F: RIGHT delivers block; LEFT ramps to rest in parallel ---")
        parallel_on_step = make_bg_ramp_on_step(
            model, data, left_arm, rest_ctrl_snapshot,
            ramp_secs=1.5, existing_on_step=on_step)
        phase_h = build_phase_h_right_deliver(container_pos_now, right_ee_down)
        execute_waypoints(model, data, right_arm, right_mask, phase_h,
                          viewer=viewer, on_step=parallel_on_step)
        p = body_pos(model, data, block_name)
        print(f"    block after delivery: {p.round(3)}")

        # Finally send RIGHT back to rest so the scene ends cleanly.
        print("  --- phase G: RIGHT returns to rest posture ---")
        return_arm_to_rest(model, data, right_arm, rest_ctrl_snapshot,
                           duration=2.5, viewer=viewer, on_step=on_step)

        # Small tail hold so the sim settles at the rest pose before we
        # exit the run() function (viewer close / check_success / logs).
        print("  --- tail hold 1.5s ---")
        hold_ctrl_for_secs(model, data, 1.5, viewer=viewer, on_step=on_step)

    if args.headless:
        run()
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            run(viewer=viewer)

    success = check_success(model, data, block_name, container_pos_now)
    print(f"\n>>> SUCCESS = {success}")
    print(f"    final block pos: {body_pos(model, data, block_name).round(3)}")

    if args.record and third_person_frames:
        try:
            import imageio
            imageio.mimsave(args.record, third_person_frames, fps=30)
            print(f"    video -> {args.record}")
        except ImportError:
            print("    (imageio missing; install it to save mp4)")

    if writer is not None and episode is not None:
        if success:
            writer.save_episode(episode)
            writer.finalize()
            print(f"    episode ({len(episode)} frames) -> {args.log_dataset}")
        elif args.save_failed:
            episode.task = f"[FAIL] {episode.task}"
            writer.save_episode(episode)
            writer.finalize()
            print(f"    [FAIL] episode ({len(episode)} frames) -> {args.log_dataset}")
        else:
            print("    (episode NOT saved: success=False)")


if __name__ == "__main__":
    main()
