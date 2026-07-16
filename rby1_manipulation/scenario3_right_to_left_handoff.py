"""Scenario 3: RIGHT arm picks a block, hands off MID-AIR to the LEFT arm
along the Y axis (both arms facing each other), left places it in the box.

Mirror of scenario 2. See scenario2_left_to_right_handoff.py for the full
design rationale. Only the arm roles are swapped:

    Handoff sequence (mirror of scenario 2):
      A1) RIGHT approach + descend (fingers-down) with gripper OPEN.
      A2) RIGHT close gripper tight (explicit ctrl setpoint).
      A3) RIGHT lift straight up.
      B ) RIGHT carries block to handoff column, rotating the wrist so the
          tool axis points to world +Y (facing LEFT).
      C1) LEFT pre_approach (mirror orientation, tool -Y, grip Z).
      C2) LEFT joint-6 twist -pi/2 (wrist roll to align grip axis with the
          block's SIDE face — opposite direction from RIGHT in scenario 2).
      C3) LEFT approach — IK translates to the block position while
          preserving the twisted wrist.
      C4) LEFT close gripper tight around the +/-X side faces.
      D ) RIGHT OPENS gripper (block now held by LEFT only).
      E ) RIGHT quick retract (~15 cm -Y from handoff).
      F ) LEFT delivers block to container; RIGHT ramps back to rest in
          parallel during the first ~1.5 s of delivery.
      G ) LEFT returns to rest.

Usage:
    python scenario3_right_to_left_handoff.py                   # viewer, red
    python scenario3_right_to_left_handoff.py --block green --random --seed 3
    python scenario3_right_to_left_handoff.py --headless \
        --log-dataset /tmp/rby1_scenario3
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

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ik_utils import (
    right_arm_handles, left_arm_handles,
    RIGHT_ARM_JOINTS, LEFT_ARM_JOINTS,
    build_dof_mask,
    site_pose,
    GRIPPER_OPEN,
)
from scene_utils import set_block_pose
from episode_logger import LeRobotWriter, Frame, CAMERAS as ALOHA_CAMERAS
from scenario1_single_arm import (
    MODEL_XML, BLOCK_BODIES, CONTAINER_BODY,
    Waypoint, settle_scene, body_pos, execute_waypoints, check_success,
)

# --- Configuration ---
# Mirror of scenario 2's LEFT_PICK_ENVELOPE: RIGHT arm picks from the
# -Y side of the workspace.
RIGHT_PICK_ENVELOPE = dict(x=(0.45, 0.55), y=(-0.22, -0.17))

# Y-axis facing handoff. Mirror of scenario 2.
HANDOFF_X = 0.50
HANDOFF_Z = 1.15

# RIGHT holds the block after pick + wrist rotation. LEFT approaches from
# the +Y side. Mirror of scenario 2's LEFT_HANDOFF_Y = +0.04.
RIGHT_HANDOFF_Y  = -0.04   # RIGHT tool +Y, grip zone y = -0.04 + 0.10 = +0.06
LEFT_HANDOFF_Y   = +0.16   # (informational only) LEFT grip zone y ~ +0.06

RIGHT_HANDOFF_EE_POS = np.array([HANDOFF_X, RIGHT_HANDOFF_Y, HANDOFF_Z])
LEFT_HANDOFF_EE_POS  = np.array([HANDOFF_X, LEFT_HANDOFF_Y,  HANDOFF_Z])

# Timings (sim seconds)
RIGHT_RELEASE_HOLD_SECS = 0.6   # after RIGHT opens, before RIGHT retracts

# Gripper close targets. RIGHT is set to 0.0 (just-touching) so the pads
# stop applying force once contact is made — prevents the block from
# creeping inside the finger cage under continuous squeeze. LEFT (side
# pick during handoff) still uses tight close because the block's face
# orientation isn't world-aligned there and a firm grip helps.
RIGHT_GRIPPER_TIGHT = 0.0    # RIGHT close during Phase A2 (top-down pick, no over-close)
LEFT_GRIPPER_TIGHT  = 0.02   # LEFT close during Phase C4 (side pick, firm)

CAM_NAME_MAP = {
    "cam_high":        "zed_left",
    "cam_left_wrist":  "wrist_cam_l",
    "cam_right_wrist": "wrist_cam_r",
}


def spawn_pick_block_in_right_envelope(model, data, joint_name, rng, z=0.87):
    env = RIGHT_PICK_ENVELOPE
    if rng is None:
        x = 0.5 * (env["x"][0] + env["x"][1])
        y = 0.5 * (env["y"][0] + env["y"][1])
    else:
        x = rng.uniform(*env["x"])
        y = rng.uniform(*env["y"])
    set_block_pose(model, data, joint_name, [x, y, z])


def build_phase_a1_right_descend(block_pos, right_ee_down) -> list:
    """RIGHT phase A1: approach + descend with gripper OPEN. Fingers are
    positioned around the block, ready for the explicit tight close in
    phase A2."""
    approach = block_pos + np.array([0.0, 0.0, 0.10])
    grasp    = block_pos + np.array([0.0, 0.0, 0.02])
    return [
        Waypoint("r_approach", approach, right_ee_down, "open", 1.5),
        Waypoint("r_descend",  grasp,    right_ee_down, "open", 1.2, wait_after=0.1),
    ]


def build_phase_a3_right_lift(block_pos, right_ee_down) -> list:
    """RIGHT phase A3: lift the block straight up while HOLDING the tight
    gripper ctrl from phase A2."""
    lift = block_pos + np.array([0.0, 0.0, 0.18])
    return [
        Waypoint("r_lift", lift, right_ee_down, "hold", 1.2, wait_after=0.4),
    ]


def build_phase_b_right_carry_to_handoff(right_ee_face) -> list:
    """RIGHT single-shot carry from lift pose to handoff with wrist
    rotation (fingers-down -> tool +Y face-center)."""
    return [
        Waypoint("r_carry_to_handoff", RIGHT_HANDOFF_EE_POS, right_ee_face,
                 duration=3.5, wait_after=1.0, gripper="hold"),
    ]


def build_phase_c1_left_pre_approach(left_ee_face_mirror, block_p) -> list:
    """LEFT pre_approach with MIRROR orientation (tool -Y, grip Z, no
    wrist twist). EE at block.y + 0.15 (15 cm +Y from block). Easy IK
    because it's a mirror of RIGHT's face-center pose during Phase B."""
    pre_approach = block_p + np.array([-0.05, +0.15, 0.0])
    return [
        Waypoint("l_pre_approach", pre_approach, left_ee_face_mirror,
                 "open", 3.0, wait_after=0.4),
    ]


def build_phase_c3_left_approach(target_rot, block_p) -> list:
    """LEFT approach to the block position with target orientation aligned
    to the block's actual face normals. Empirical XY offset (tunable)."""
    # Mirror of scenario 2's RIGHT approach offset (-0.03, -0.017); Y sign
    # flipped since LEFT approaches from +Y side.
    approach = block_p - np.array([0.015, -0.030, 0.0])
    return [
        Waypoint("l_approach", approach, target_rot,
                 "open", 1.5, wait_after=0.4),
    ]


def compute_left_target_rot_aligned_to_block(model, data, block_name):
    """Build a target rotation for LEFT's EE (mirror of scenario 2's RIGHT
    version):
      * tool axis (world) = -Y (approach block from +Y side)
      * grip axis (world) = block's face normal perpendicular to tool AND
        perpendicular to world Z (which is RIGHT's grip direction here).

    Prints the block's world-frame axes for diagnostics.
    """
    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, block_name)
    block_xmat = data.xmat[block_id].reshape(3, 3).copy()
    print(f"    [BLOCK ORIENT] block axes in world:")
    print(f"      block +X in world: {block_xmat[:, 0].round(3)}")
    print(f"      block +Y in world: {block_xmat[:, 1].round(3)}")
    print(f"      block +Z in world: {block_xmat[:, 2].round(3)}")

    tool_axis = np.array([0.0, 1.0, 0.0])   # world -Y (LEFT approach direction)

    best_axis = None
    best_score = -1.0
    for i in range(3):
        for sign in (+1.0, -1.0):
            cand = sign * block_xmat[:, i]
            perp_to_tool = 1.0 - abs(np.dot(cand, tool_axis))
            perp_to_right_grip = 1.0 - abs(np.dot(cand, np.array([0.0, 0.0, 1.0])))
            score = perp_to_tool * perp_to_right_grip
            if score > best_score:
                best_score = score
                best_axis = cand

    grip_axis = best_axis - np.dot(best_axis, tool_axis) * tool_axis
    grip_axis = grip_axis / np.linalg.norm(grip_axis)
    third_axis = np.cross(grip_axis, tool_axis)
    third_axis = third_axis / np.linalg.norm(third_axis)

    R_target = np.column_stack([third_axis, grip_axis, tool_axis])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R_target.reshape(-1))
    print(f"    [C3 TARGET] tool={tool_axis.round(3)} grip={grip_axis.round(3)} third={third_axis.round(3)}")
    return mink.SO3(q)


def do_left_wrist_twist_j6(model, data, arm, delta_rad, duration,
                           viewer=None, on_step=None):
    """Ramp ONLY the last wrist joint (arm.aid[6]) by delta_rad over
    `duration` seconds. Pure wrist roll — no shoulder/elbow motion."""
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
    """Ramp all 7 arm joints back to their initial (keyframe) ctrl values
    over `duration` seconds. Gripper ctrl is left as-is."""
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
    """on_step chained callback that ramps `arm` ctrl toward target over
    `ramp_secs`. Runs in parallel with the "foreground" waypoint arm."""
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


def build_phase_e_right_retract(right_ee_face) -> list:
    """RIGHT: single short retract (~15 cm -Y from handoff) to clear the
    space so LEFT can lift the block. Rest posture return happens in
    parallel with LEFT's delivery."""
    retract_short = RIGHT_HANDOFF_EE_POS + np.array([0.0, -0.20, 0.0])
    return [
        Waypoint("r_retract_short", retract_short, right_ee_face, "open", 0.8, wait_after=0.1),
    ]


def build_phase_f_left_deliver(container_pos, left_ee_down) -> list:
    """LEFT: carry block to container, unwinding wrist to fingers-down."""
    lift_down   = np.array([HANDOFF_X, +0.10, HANDOFF_Z + 0.05])
    over_cont   = container_pos + np.array([0.0, 0.0, 0.15])
    descend     = container_pos + np.array([0.0, 0.0, 0.08])
    retract     = container_pos + np.array([0.0, 0.0, 0.20])
    return [
        Waypoint("l_lift_and_unwind", lift_down, left_ee_down, "hold", 3.5, wait_after=0.5),
        Waypoint("l_over_container",  over_cont, left_ee_down, "hold", 1.8),
        Waypoint("l_descend_cont",    descend,   left_ee_down, "hold", 1.0),
        Waypoint("l_release",         descend,   left_ee_down, "open", 0.1, wait_after=0.5),
        Waypoint("l_retract",         retract,   left_ee_down, "open", 1.2),
    ]


def hold_ctrl_for_secs(model, data, secs, viewer=None, on_step=None):
    steps = int(round(secs / model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        if on_step is not None:
            on_step()


def interactive_left_control(model, data, viewer, left_arm, left_mask,
                             left_target_body="link_left_arm_6_target",
                             print_interval_secs=1.0):
    """LEFT follows a mocap sphere in the viewer while RIGHT stays frozen
    holding the block. Prints LEFT joint qpos/ctrl + EE pose periodically."""
    import time

    body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, left_target_body)
    mocap_id = model.body_mocapid[body_id]
    ee_site_id = left_arm.ee_site_id

    data.mocap_pos[mocap_id] = data.site_xpos[ee_site_id].copy()
    mat = data.site_xmat[ee_site_id].reshape(3, 3)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat.reshape(-1))
    data.mocap_quat[mocap_id] = q

    viewer.opt.geomgroup[3] = 1
    viewer.opt.sitegroup[4] = 1

    print()
    print("=" * 70)
    print("INTERACTIVE MODE — LEFT arm follows mocap; RIGHT is frozen holding block.")
    print("  Double-click the red LEFT target sphere, then:")
    print("    Ctrl + right-click + drag  -> translate")
    print("    Ctrl + left-click  + drag  -> rotate")
    print(f"  LEFT joint qpos + EE pose printed every {print_interval_secs:.1f}s.")
    print("=" * 70)
    print()

    config = mink.Configuration(model)
    config.update(data.qpos)

    task = mink.FrameTask(left_arm.ee_site, "site",
                          position_cost=100.0, orientation_cost=10.0, lm_damping=1e-3)
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

        target = mink.SE3.from_rotation_and_translation(
            mink.SO3(np.asarray(data.mocap_quat[mocap_id]).copy()),
            np.asarray(data.mocap_pos[mocap_id]).copy(),
        )
        task.set_target(target)

        config.update(data.qpos)
        for _ in range(IK_INNER_ITERS):
            vel = mink.solve_ik(config, [task, posture], IK_INNER_DT,
                                solver="daqp", damping=1e-4)
            vel = vel * left_mask
            config.integrate_inplace(vel, IK_INNER_DT)

        for i, aid in enumerate(left_arm.aid):
            data.ctrl[aid] = config.q[left_arm.qidx[i]]

        for _ in range(sim_steps_per_ctrl):
            mujoco.mj_step(model, data)
        viewer.sync()

        if data.time - last_print_t >= print_interval_secs:
            last_print_t = data.time
            qpos_left = np.array([data.qpos[q] for q in left_arm.qidx])
            ctrl_left = np.array([data.ctrl[a] for a in left_arm.aid])
            ee_pos = data.site_xpos[ee_site_id].copy()
            ee_mat = data.site_xmat[ee_site_id].reshape(3, 3)
            print(f"[t={data.time:6.2f}s] LEFT qpos = "
                  f"[{', '.join(f'{q:+.3f}' for q in qpos_left)}]")
            print(f"           LEFT ctrl = "
                  f"[{', '.join(f'{c:+.3f}' for c in ctrl_left)}]")
            print(f"           EE pos = {ee_pos.round(3)}  "
                  f"tool_world(col2)={ee_mat[:, 2].round(3)}  "
                  f"grip_world(col1)={ee_mat[:, 1].round(3)}")

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
    ap.add_argument("--interactive-left", action="store_true",
                    help="Run phases A+B only, then hand control of the "
                         "LEFT arm to the user via a mocap sphere.")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data  = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "teleop")
    mujoco.mj_resetDataKeyframe(model, data, key)

    block_name = BLOCK_BODIES[args.block]
    joint_name = f"{block_name}_free"
    rng = np.random.default_rng(args.seed) if args.random else None
    spawn_pick_block_in_right_envelope(model, data, joint_name, rng)

    for i in range(model.nu):
        data.ctrl[i] = data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]
    settle_scene(model, data, seconds=1.5)

    rest_ctrl_snapshot = data.ctrl.copy()

    left_arm  = left_arm_handles(model)
    right_arm = right_arm_handles(model)
    left_mask  = build_dof_mask(model, LEFT_ARM_JOINTS)
    right_mask = build_dof_mask(model, RIGHT_ARM_JOINTS)

    # Orientations
    left_ee_down  = site_pose(data, model, left_arm.ee_site).rotation()
    right_ee_down = site_pose(data, model, right_arm.ee_site).rotation()

    # RIGHT face-center: R_x(+pi/2) rotates fingers-down so tool axis
    # points to world +Y. Grip axis becomes world Z. (Mirror of scenario
    # 2's LEFT face-center R_x(-pi/2).)
    right_ee_face = mink.SO3.from_x_radians(+np.pi / 2) @ right_ee_down

    # LEFT face_mirror: mirror of RIGHT face-center. Tool -Y, grip Z.
    # Used for LEFT pre_approach — natural mirror pose, easy IK.
    left_ee_face_mirror = mink.SO3.from_x_radians(-np.pi / 2) @ left_ee_down

    block_pos_now     = body_pos(model, data, block_name)
    container_pos_now = body_pos(model, data, CONTAINER_BODY)

    print(f"=== scenario 3 Y-axis handoff  block={args.block} ===")
    print(f"  block start        : {block_pos_now.round(3)}")
    print(f"  RIGHT handoff EE  : {RIGHT_HANDOFF_EE_POS.round(3)}  (tool +Y, grip Z, top/bottom faces)")
    print(f"  LEFT  (info only) : {LEFT_HANDOFF_EE_POS.round(3)}  (tool -Y, grip X @ approach, +/-X side faces)")

    phase_a1 = build_phase_a1_right_descend(block_pos_now, right_ee_down)
    phase_a3 = build_phase_a3_right_lift(block_pos_now, right_ee_down)
    phase_b  = build_phase_b_right_carry_to_handoff(right_ee_face)
    phase_e  = build_phase_e_right_retract(right_ee_face)
    _ = container_pos_now  # used inside run()

    third_person_frames: list = []
    recorder = None
    if args.record:
        recorder = mujoco.Renderer(model, height=480, width=640)

    writer = None
    episode = None
    log_renderer = None
    if args.log_dataset:
        writer = LeRobotWriter(args.log_dataset, fps=args.log_fps, image_wh=(224, 224))
        prompt = (f"pick up the {args.block} block with the right hand, hand it off "
                  f"to the left hand in mid-air, and place it in the brown box")
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
        print("  --- phase A1: RIGHT approach + descend (gripper open) ---")
        execute_waypoints(model, data, right_arm, right_mask, phase_a1,
                          viewer=viewer, on_step=on_step)

        # A2a: command close briefly so fingers make contact with block
        print(f"  --- phase A2: RIGHT close gripper (ctrl={RIGHT_GRIPPER_TIGHT}) ---")
        data.ctrl[right_arm.gripper_aid] = RIGHT_GRIPPER_TIGHT
        hold_ctrl_for_secs(model, data, 0.5, viewer=viewer, on_step=on_step)

        # A2b: match ctrl to the actual qpos at contact, PLUS a small
        # positive offset (+0.001) so force = kp * 0.001 -> tiny continuous
        # squeeze that holds the block firmly without over-crushing it.
        contact_qpos = float(data.qpos[right_arm.gripper_qidx])
        data.ctrl[right_arm.gripper_aid] = contact_qpos + 0.0007
        hold_ctrl_for_secs(model, data, 0.3, viewer=viewer, on_step=on_step)
        print(f"    [RIGHT GRIPPER] contact qpos={contact_qpos:+.4f}  "
              f"ctrl set to qpos+0.001={data.ctrl[right_arm.gripper_aid]:+.4f}  "
              f"(tiny squeeze)")

        print("  --- phase A3: RIGHT lift ---")
        execute_waypoints(model, data, right_arm, right_mask, phase_a3,
                          viewer=viewer, on_step=on_step)

        print("  --- phase B: RIGHT carries + rotates wrist to handoff pose ---")
        execute_waypoints(model, data, right_arm, right_mask, phase_b,
                          viewer=viewer, on_step=on_step)
        p = body_pos(model, data, block_name)
        right_ee_after_b = data.site_xpos[right_arm.ee_site_id].copy()
        print(f"    [RIGHT STOPPED] block pos     : {p.round(3)}")
        print(f"    [RIGHT STOPPED] RIGHT EE pos  : {right_ee_after_b.round(3)}")
        print(f"    [RIGHT STOPPED] block - EE    : {(p - right_ee_after_b).round(3)}")

        # Interactive mode: skip phase C onwards. Freeze RIGHT, hand LEFT
        # over to the user via mocap.
        if args.interactive_left:
            if viewer is None:
                print("!! --interactive-left requires viewer (do not pass --headless)")
                return
            interactive_left_control(model, data, viewer, left_arm, left_mask)
            return

        # C1: LEFT pre_approach with MIRROR orientation.
        phase_c1 = build_phase_c1_left_pre_approach(left_ee_face_mirror, p)
        print("  --- phase C1: LEFT pre_approach (mirror, 15 cm +Y from block) ---")
        execute_waypoints(model, data, left_arm, left_mask, phase_c1,
                          viewer=viewer, on_step=on_step)

        # C2: LEFT joint 6 twist by -pi/2 (mirror of scenario 2 RIGHT's
        # +pi/2). Physically rotates grip axis from world Z to world X.
        print("  --- phase C2: LEFT wrist twist -pi/2 (joint 6 only) ---")
        do_left_wrist_twist_j6(model, data, left_arm, -np.pi / 2, 1.5,
                               viewer=viewer, on_step=on_step)

        # C3: Read the actual LEFT EE orientation AFTER C2's joint-6 twist
        # and use it directly as the IK target. IK won't try to redo the
        # twist — just translates to block position.
        left_current_rot = site_pose(data, model, left_arm.ee_site).rotation()
        phase_c3 = build_phase_c3_left_approach(left_current_rot, p)
        print("  --- phase C3: LEFT approach (target orient = current post-twist EE) ---")
        execute_waypoints(model, data, left_arm, left_mask, phase_c3,
                          viewer=viewer, on_step=on_step)

        # C4: LEFT close gripper tight around block's side faces.
        print(f"  --- phase C4: LEFT close gripper tight (ctrl={LEFT_GRIPPER_TIGHT}) ---")
        gripper_ctrl_before = float(data.ctrl[left_arm.gripper_aid])
        gripper_qpos_before = float(data.qpos[left_arm.gripper_qidx])
        data.ctrl[left_arm.gripper_aid] = LEFT_GRIPPER_TIGHT
        hold_ctrl_for_secs(model, data, 0.8, viewer=viewer, on_step=on_step)
        gripper_ctrl_after = float(data.ctrl[left_arm.gripper_aid])
        gripper_qpos_after = float(data.qpos[left_arm.gripper_qidx])
        print(f"    [LEFT GRIPPER] ctrl before={gripper_ctrl_before:+.4f} -> "
              f"after={gripper_ctrl_after:+.4f}  (target LEFT_GRIPPER_TIGHT={LEFT_GRIPPER_TIGHT})")
        print(f"    [LEFT GRIPPER] qpos before={gripper_qpos_before:+.4f} -> "
              f"after={gripper_qpos_after:+.4f}  "
              f"(residual = ctrl - qpos = {gripper_ctrl_after - gripper_qpos_after:+.4f})")
        p = body_pos(model, data, block_name)
        left_ee_after_pick = data.site_xpos[left_arm.ee_site_id].copy()
        print(f"    [LEFT PICKED] block pos    : {p.round(3)}")
        print(f"    [LEFT PICKED] LEFT  EE pos : {left_ee_after_pick.round(3)}")
        print(f"    [LEFT PICKED] block - EE   : {(p - left_ee_after_pick).round(3)}")
        delta_ee = left_ee_after_pick - right_ee_after_b
        print(f"    [ALIGNMENT]    LEFT EE - RIGHT EE = {delta_ee.round(3)}  "
              f"(x={delta_ee[0]*1000:+.1f}mm, y={delta_ee[1]*1000:+.1f}mm, z={delta_ee[2]*1000:+.1f}mm)")

        print("  --- phase D: RIGHT OPENS gripper (block held by left only) ---")
        data.ctrl[right_arm.gripper_aid] = GRIPPER_OPEN
        hold_ctrl_for_secs(model, data, RIGHT_RELEASE_HOLD_SECS,
                           viewer=viewer, on_step=on_step)
        p = body_pos(model, data, block_name)
        z_dropped = p[2] < 0.90
        print(f"    block after right-open: {p.round(3)}   "
              f"{'FELL' if z_dropped else 'held by left'}")

        print("  --- phase E: RIGHT retracts ---")
        execute_waypoints(model, data, right_arm, right_mask, phase_e,
                          viewer=viewer, on_step=on_step)

        # F: LEFT delivers block to container. RIGHT ramps to rest in
        # parallel during the first ~1.5s of delivery.
        print("  --- phase F: LEFT delivers block; RIGHT ramps to rest in parallel ---")
        parallel_on_step = make_bg_ramp_on_step(
            model, data, right_arm, rest_ctrl_snapshot,
            ramp_secs=1.5, existing_on_step=on_step)
        phase_f = build_phase_f_left_deliver(container_pos_now, left_ee_down)
        execute_waypoints(model, data, left_arm, left_mask, phase_f,
                          viewer=viewer, on_step=parallel_on_step)
        p = body_pos(model, data, block_name)
        print(f"    block after delivery: {p.round(3)}")

        print("  --- phase G: LEFT returns to rest posture ---")
        return_arm_to_rest(model, data, left_arm, rest_ctrl_snapshot,
                           duration=2.5, viewer=viewer, on_step=on_step)

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
        else:
            print("    (episode NOT saved: success=False)")


if __name__ == "__main__":
    main()
