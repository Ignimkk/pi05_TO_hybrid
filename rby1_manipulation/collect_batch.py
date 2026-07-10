"""Collect many scenario-1 episodes into one LeRobot dataset.

Loops over (block, seed) combos, running each rollout headless. Only successful
episodes are appended. All frames go into a single writer so episode_index is
contiguous (LeRobot expects that).

Example:
    python collect_batch.py --root /data/rby1_pickplace \
        --colors red green blue --n-per-color 20

    This will attempt 60 episodes (20 seeds × 3 colors) and keep only the ones
    that succeed. Prints a summary at the end.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import numpy as np
import mujoco

# Force offscreen GL for batch runs.
os.environ.setdefault("MUJOCO_GL", "osmesa")

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from ik_utils import (
    right_arm_handles, left_arm_handles,
    RIGHT_ARM_JOINTS, LEFT_ARM_JOINTS,
    build_dof_mask, site_pose,
    GRIPPER_OPEN,
)
from scene_utils import pick_arm_for_block, randomize_blocks
from episode_logger import LeRobotWriter, Frame, CAMERAS as ALOHA_CAMERAS
from scenario1_single_arm import (
    MODEL_XML, BLOCK_BODIES, CONTAINER_BODY,
    make_waypoints, settle_scene, body_pos, execute_waypoints, check_success,
)


CAM_NAME_MAP = {
    "cam_high": "zed_left",
    "cam_left_wrist": "wrist_cam_l",
    "cam_right_wrist": "wrist_cam_r",
}


def run_one_episode(model, data, writer, *, block_color: str, seed: int, log_fps: int,
                    verbose: bool = False) -> bool:
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "teleop")
    mujoco.mj_resetDataKeyframe(model, data, key)

    # Fresh block layout for this seed.
    rng = np.random.default_rng(seed)
    randomize_blocks(model, data, rng)

    # Hold keyframe pose on ctrl.
    for i in range(model.nu):
        data.ctrl[i] = data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]

    settle_scene(model, data, seconds=1.5)

    block_name = BLOCK_BODIES[block_color]
    arm_side = pick_arm_for_block(body_pos(model, data, block_name))

    if arm_side == "right":
        arm = right_arm_handles(model)
        joints = RIGHT_ARM_JOINTS
    else:
        arm = left_arm_handles(model)
        joints = LEFT_ARM_JOINTS
    arm_mask = build_dof_mask(model, joints)

    ee_down = site_pose(data, model, arm.ee_site).rotation()
    block_pos_now = body_pos(model, data, block_name)
    container_pos_now = body_pos(model, data, CONTAINER_BODY)
    waypoints = make_waypoints(block_pos_now, container_pos_now, ee_down)

    # Dataset logging.
    prompt = f"pick up the {block_color} block and put it in the brown box"
    episode = writer.new_episode(task=prompt)
    log_renderer = mujoco.Renderer(model, height=224, width=224)
    right_h = right_arm_handles(model)
    left_h  = left_arm_handles(model)
    log_state = {"last_t": 0.0, "frame_idx": 0}
    log_interval = 1.0 / log_fps

    def on_step():
        t_sim = data.time
        if t_sim - log_state["last_t"] + 1e-9 < log_interval and log_state["frame_idx"] > 0:
            return
        state = np.array([
            *(data.qpos[q] for q in left_h.qidx[:6]),
            abs(data.qpos[left_h.gripper_qidx]) / abs(GRIPPER_OPEN),
            *(data.qpos[q] for q in right_h.qidx[:6]),
            abs(data.qpos[right_h.gripper_qidx]) / abs(GRIPPER_OPEN),
        ], dtype=np.float32)
        action = np.array([
            *(data.ctrl[a] for a in left_h.aid[:6]),
            abs(data.ctrl[left_h.gripper_aid]) / abs(GRIPPER_OPEN),
            *(data.ctrl[a] for a in right_h.aid[:6]),
            abs(data.ctrl[right_h.gripper_aid]) / abs(GRIPPER_OPEN),
        ], dtype=np.float32)
        imgs = {}
        for cam in ALOHA_CAMERAS:
            log_renderer.update_scene(data, camera=CAM_NAME_MAP[cam])
            imgs[cam] = log_renderer.render()
        episode.append(Frame(state=state, action=action, images=imgs,
                             timestamp=float(t_sim),
                             frame_index=log_state["frame_idx"]))
        log_state["frame_idx"] += 1
        log_state["last_t"] = t_sim

    if verbose:
        print(f"[{block_color:5s} seed={seed:3d} arm={arm_side:5s}]  ", end="", flush=True)

    execute_waypoints(model, data, arm, arm_mask, waypoints, on_step=on_step)
    success = check_success(model, data, block_name, container_pos_now)

    if success:
        writer.save_episode(episode)
    if verbose:
        marker = "PASS" if success else "FAIL"
        print(f"{marker}  frames={len(episode) if success else 0}")
    return success


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root dir (created)")
    ap.add_argument("--colors", nargs="+", choices=["red", "green", "blue"],
                    default=["red", "green", "blue"])
    ap.add_argument("--n-per-color", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--log-fps", type=int, default=15)
    args = ap.parse_args()

    if pathlib.Path(args.root).exists():
        print(f"warning: {args.root} exists; new episodes will be appended-with-index-collision")
    writer = LeRobotWriter(args.root, fps=args.log_fps, image_wh=(224, 224))

    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data  = mujoco.MjData(model)

    results = {c: {"attempts": 0, "passes": 0} for c in args.colors}
    t0 = time.time()

    for color in args.colors:
        for s in range(args.n_per_color):
            seed = args.seed_start + s * len(args.colors) + args.colors.index(color)
            ok = run_one_episode(model, data, writer,
                                 block_color=color, seed=seed,
                                 log_fps=args.log_fps, verbose=True)
            results[color]["attempts"] += 1
            results[color]["passes"] += int(ok)

    writer.finalize()

    dt = time.time() - t0
    total_pass = sum(r["passes"] for r in results.values())
    total_att  = sum(r["attempts"] for r in results.values())
    print("\n=== summary ===")
    for c, r in results.items():
        print(f"  {c:6s}: {r['passes']}/{r['attempts']}")
    print(f"  total : {total_pass}/{total_att}  ({dt:.1f}s, ~{dt/max(1,total_att):.1f}s/episode)")
    print(f"  dataset written to {args.root}")


if __name__ == "__main__":
    main()
