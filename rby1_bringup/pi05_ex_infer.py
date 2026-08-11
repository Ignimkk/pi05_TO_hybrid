"""Experimental pi0.5 inference runner with deterministic RBY1 grid evaluation.

The production-style single-run entry point remains ``pi05_infer.py``.  This
copy adds the deterministic grid loop and consumes coordinates saved by
``rby1_manipulation/preview_block_grid.py``.
"""

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
from datetime import datetime, timezone
import numpy as np

if "--headless" in sys.argv and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "osmesa"

import mujoco
import mujoco.viewer
from PIL import Image

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_XML = str(REPO_ROOT / "rby1_description" / "models" / "rby1a" / "mujoco" / "model.xml")
# The mobile crate-transport scene. Same robot, same cameras, plus the crate /
# shelf / small objects and position actuators on the base (ctrl 26/27/28).
MODEL_XML_TRANSPORT = str(REPO_ROOT / "rby1_description" / "models" / "rby1a"
                          / "mujoco" / "model_transport.xml")
MANIPULATION_DIR = REPO_ROOT / "rby1_manipulation"
if str(MANIPULATION_DIR) not in sys.path:
    sys.path.insert(0, str(MANIPULATION_DIR))

from transport_scene import BASE_ACTS, BASE_JOINTS
from preview_block_grid import (
    BLOCK_BODIES,
    COLORS as GRID_COLORS,
    DEFAULT_GRID_CONFIG,
    body_position,
    load_grid_config,
    reset_and_place_trial,
)

MODELS = {
    "droid": {
        "config": "pi05_droid",
        "checkpoint": "gs://openpi-assets/checkpoints/pi05_droid",
        "obs_format": "droid",
        "action_format": "droid",       # 7 joint vel + 1 gripper
    },
    "libero": {
        "config": "pi05_libero",
        "checkpoint": "gs://openpi-assets/checkpoints/pi05_libero",
        "obs_format": "libero",
        "action_format": "libero",      # 6 dims -> joints[:6] + 1 gripper
    },
    "base": {
        "config": "pi05_droid",
        "checkpoint": "gs://openpi-assets/checkpoints/pi05_base",
        "obs_format": "droid",
        "action_format": "droid",
    },
    "aloha": {
        # For zero-shot dual-arm + 3-camera pipeline test only.
        # Publicly released pi0.5-ALOHA checkpoint doesn't exist, so we use pi0 variant.
        "config": "pi0_aloha_pen_uncap",
        "checkpoint": "gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap",
        "obs_format": "aloha",
        "action_format": "aloha",       # 14 = [L 6 jvel, L grip, R 6 jvel, R grip]
    },
    "rby1": {
        # Our own LoRA fine-tune on rby1_dataset_v1 (see src/openpi/training/config.py).
        "config": "pi05_rby1_lora",
        "checkpoint": "/mnt/dev/work/pi05_TO_hybrid/checkpoints/pi05_rby1_lora/full_run_30k/29999",
        "obs_format": "rby1",
        "action_format": "rby1",        # 14 = [L 6 abs joint, L grip, R 6 abs joint, R grip]
        "model_xml": MODEL_XML,
    },
    "rby1_mobile": {
        # Mobile crate transport. Loads model_transport.xml and predicts the
        # 17-D layout (the rby1 14 plus the planar base pose). Point --checkpoint
        # at a run trained on a 17-D dataset; the block checkpoint will not work.
        "config": "pi05_rby1_mobile_lora",
        "checkpoint": None,
        "obs_format": "rby1_mobile",
        "action_format": "rby1_mobile",  # 17 = rby1 14 + [base_x, base_y, base_yaw]
        "model_xml": MODEL_XML_TRANSPORT,
    },
}

# Single-arm (droid/libero/base) uses right arm.
RIGHT_ARM_JOINTS = [f"right_arm_{i}" for i in range(7)]
RIGHT_ARM_ACTS   = [f"right_arm_{i+1}_act" for i in range(7)]
LEFT_ARM_JOINTS  = [f"left_arm_{i}" for i in range(7)]
LEFT_ARM_ACTS    = [f"left_arm_{i+1}_act" for i in range(7)]
GRIPPER_R_JOINT = "gripper_finger_r1"
GRIPPER_R_ACT   = "gripper_r_act"
GRIPPER_L_JOINT = "gripper_finger_l1"
GRIPPER_L_ACT   = "gripper_l_act"
GRIPPER_OPEN, GRIPPER_CLOSED = -0.05, 0.0
# rby1_dataset_v1 was collected with this exact gripper-open ctrl value (see
# rby1_manipulation/ik_utils.py GRIPPER_OPEN=-0.045); must match for correct
# state/action normalization on the "rby1" model.
RBY1_GRIPPER_OPEN = -0.045

CTRL_HZ = 15
OPEN_LOOP_HORIZON = 8
POLICY_CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")

# 14-D RBY1 layout is [L 6 joints, L grip, R 6 joints, R grip]; motion metrics are
# reported on the 12 arm joints only (grippers are near-binary and would dominate jerk).
# Matches scripts/plot_rby1_jerk_comparison.py.
ARM_DIMS = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)
# The 17-D mobile layout appends the planar base pose. ARM_DIMS is deliberately
# unchanged so BJ/IJ/CD stay comparable with the 14-D block runs; report base
# motion separately rather than folding it into the arm metrics.
BASE_DIMS = np.asarray([14, 15, 16], dtype=np.int64)


def _json_float(value):
    """JSON-safe float: NaN/Inf (e.g. a metric with an empty index set) become null."""
    value = float(value)
    return value if np.isfinite(value) else None


def compute_trial_metrics(executed_actions, measured_states):
    """Per-trial BJ/IJ/CD/AVb on commanded actions and on the measured physical response.

    Uses the paper-exact implementation in ``benchmark/seam_vla/metrics/motion.py`` so grid
    results are directly comparable with the offline RB-Y1 evaluation.
    """
    ws_root = REPO_ROOT.parent  # .../pi0_TO_ws
    if str(ws_root) not in sys.path:
        sys.path.insert(0, str(ws_root))
    from benchmark.seam_vla.metrics.motion import compute_motion_metrics

    metrics = {}
    for label, series in (("action", executed_actions), ("qpos", measured_states)):
        array = np.asarray(series, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] < 3 or array.shape[1] <= int(ARM_DIMS.max()):
            continue
        values = compute_motion_metrics(array[:, ARM_DIMS], OPEN_LOOP_HORIZON)
        metrics[label] = {
            "BJ": _json_float(values["BJ"]),
            "IJ": _json_float(values["IJ"]),
            "CD": _json_float(values["CD"]),
            "AVb": _json_float(values["paper_avb"]),
            "num_boundary": int(values["num_boundary"]),
            "num_interior": int(values["num_interior"]),
            "num_steps": int(values["num_steps"]),
        }
    return metrics


def grid_cell_indices(config, side, grid_index):
    """Map a linear grid index to (column, row) for heatmap binning.

    Column indexes the distinct x values (near -> far from the robot base) and row indexes the
    distinct |y| values, so the left and right grids share one coordinate frame despite the
    right side using negative y.
    """
    positions = config["positions"][side]
    x, y = positions[grid_index]
    xs = sorted({round(float(p[0]), 6) for p in positions})
    ys = sorted({round(float(p[1]), 6) for p in positions}, key=abs)
    return xs.index(round(float(x), 6)), ys.index(round(float(y), 6))


def configure_view_camera(camera, view):
    """Apply a reproducible viewer/recording camera without changing policy input cameras."""
    if view == "free":
        return
    if view != "front":
        raise ValueError(f"unknown view preset: {view}")
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.asarray([0.45, 0.0, 0.85], dtype=np.float64)
    camera.distance = 1.7
    camera.azimuth = 180.0
    camera.elevation = -18.0


def render_cam(model, data, renderer, cam_name, size=224):
    renderer.update_scene(data, camera=cam_name)
    img = renderer.render()
    im = Image.fromarray(img).resize((size, size), Image.BILINEAR)
    return np.asarray(im)


def _hwc_to_chw(img):
    """Convert (H, W, C) uint8 image to (C, H, W) as expected by AlohaInputs."""
    return np.transpose(np.asarray(img), (2, 0, 1))


def capture_policy_input_frames(obs, frame_buffers):
    """Copy the three image arrays from the exact observation sent to the policy."""
    for camera_name in POLICY_CAMERA_NAMES:
        image = np.asarray(obs["images"][camera_name])
        if image.ndim != 3:
            raise ValueError(f"policy input {camera_name!r} has invalid shape {image.shape}")
        # RBY1/ALOHA policy observations store images as CHW; video encoders expect HWC.
        if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
            image = np.transpose(image, (1, 2, 0))
        frame_buffers[camera_name].append(np.ascontiguousarray(image[..., :3]).copy())


def save_policy_input_videos(output_dir, frame_buffers, *, flat=False):
    """Write one MP4 per camera.

    By default each call allocates a fresh ``run_XXXX`` subdirectory so repeated single runs
    never overwrite each other. With ``flat=True`` the files are written straight into
    ``output_dir`` -- used by the grid experiment, where the trial id already makes the path
    unique and the extra nesting only obscures it.
    """
    if not any(frame_buffers[name] for name in POLICY_CAMERA_NAMES):
        print("no policy-input frames to save")
        return

    recording_root = pathlib.Path(output_dir)
    recording_root.mkdir(parents=True, exist_ok=True)
    if flat:
        run_dir = recording_root
    else:
        for run_index in range(1_000_000):
            run_dir = recording_root / f"run_{run_index:04d}"
            try:
                run_dir.mkdir(exist_ok=False)
                break
            except FileExistsError:
                continue
        else:
            raise RuntimeError(f"no available run directory under {recording_root}")

    print(f"policy-input recording directory: {run_dir}")
    input_fps = CTRL_HZ / OPEN_LOOP_HORIZON

    try:
        import imageio.v2 as imageio
    except ImportError:
        imageio = None

    for camera_name in POLICY_CAMERA_NAMES:
        frames = frame_buffers[camera_name]
        if not frames:
            continue
        output_path = run_dir / f"{camera_name}.mp4"
        print(f"saving {len(frames)} policy-input frames -> {output_path}")
        if imageio is not None:
            imageio.mimsave(output_path, frames, fps=input_fps)
            continue

        png_dir = run_dir / camera_name
        png_dir.mkdir(parents=True, exist_ok=True)
        for frame_index, frame in enumerate(frames):
            Image.fromarray(frame).save(png_dir / f"{frame_index:04d}.png")
        print(f"(imageio not installed, saved {camera_name} as a PNG sequence)")


def build_obs(obs_format, m, d, renderer, idx, prompt):
    """Assemble the observation dict expected by the chosen policy transform.

    `idx` is a dict of joint qpos indices computed once in main:
        right_q, right_grip_q, left_q, left_grip_q
    """
    if obs_format in ("droid", "libero"):
        base_img = render_cam(m, d, renderer, "zed_left")
        wrist_img = np.zeros((224, 224, 3), dtype=np.uint8)
        joint_pos = np.array([d.qpos[i] for i in idx["right_q"]], dtype=np.float64)
        grip_norm = float(abs(d.qpos[idx["right_grip_q"]]) / abs(GRIPPER_OPEN))

        if obs_format == "droid":
            return {
                "observation/exterior_image_1_left": base_img,
                "observation/wrist_image_left": wrist_img,
                "observation/joint_position": joint_pos,
                "observation/gripper_position": np.array([grip_norm], dtype=np.float64),
                "prompt": prompt,
            }
        # libero
        state = np.concatenate([joint_pos, np.array([grip_norm], dtype=np.float64)])
        return {
            "observation/state": state,
            "observation/image": base_img,
            "observation/wrist_image": wrist_img,
            "prompt": prompt,
        }

    if obs_format == "aloha":
        # 3 real cameras + 14-dim dual-arm state.
        base_img    = render_cam(m, d, renderer, "zed_left")      # cam_high
        wrist_l_img = render_cam(m, d, renderer, "wrist_cam_l")   # cam_left_wrist
        wrist_r_img = render_cam(m, d, renderer, "wrist_cam_r")   # cam_right_wrist

        left_joint_pos  = np.array([d.qpos[i] for i in idx["left_q"]], dtype=np.float64)
        right_joint_pos = np.array([d.qpos[i] for i in idx["right_q"]], dtype=np.float64)
        left_grip_norm  = float(abs(d.qpos[idx["left_grip_q"]])  / abs(GRIPPER_OPEN))
        right_grip_norm = float(abs(d.qpos[idx["right_grip_q"]]) / abs(GRIPPER_OPEN))

        # ALOHA layout: [left 6 joints, left gripper, right 6 joints, right gripper]
        # RBY1 has 7-DoF arms; drop the last wrist joint (arm_6) for state.
        state = np.concatenate([
            left_joint_pos[:6],
            [left_grip_norm],
            right_joint_pos[:6],
            [right_grip_norm],
        ]).astype(np.float64)

        return {
            "state": state,
            "images": {
                "cam_high":        _hwc_to_chw(base_img),
                "cam_left_wrist":  _hwc_to_chw(wrist_l_img),
                "cam_right_wrist": _hwc_to_chw(wrist_r_img),
            },
            "prompt": prompt,
        }

    if obs_format == "rby1":
        # Same 3-camera / 14-dim dual-arm layout as "aloha", but with the exact
        # gripper-open value the training dataset was collected with.
        base_img    = render_cam(m, d, renderer, "zed_left")      # cam_high
        wrist_l_img = render_cam(m, d, renderer, "wrist_cam_l")   # cam_left_wrist
        wrist_r_img = render_cam(m, d, renderer, "wrist_cam_r")   # cam_right_wrist

        left_joint_pos  = np.array([d.qpos[i] for i in idx["left_q"]], dtype=np.float64)
        right_joint_pos = np.array([d.qpos[i] for i in idx["right_q"]], dtype=np.float64)
        left_grip_norm  = float(abs(d.qpos[idx["left_grip_q"]])  / abs(RBY1_GRIPPER_OPEN))
        right_grip_norm = float(abs(d.qpos[idx["right_grip_q"]]) / abs(RBY1_GRIPPER_OPEN))

        # Dataset layout: [left 6 joints, left gripper, right 6 joints, right gripper].
        # RBY1 arms are 7-DoF; the collection scripts always dropped arm_6 (qidx[:6]).
        state = np.concatenate([
            left_joint_pos[:6],
            [left_grip_norm],
            right_joint_pos[:6],
            [right_grip_norm],
        ]).astype(np.float64)

        return {
            "state": state,
            "images": {
                "cam_high":        _hwc_to_chw(base_img),
                "cam_left_wrist":  _hwc_to_chw(wrist_l_img),
                "cam_right_wrist": _hwc_to_chw(wrist_r_img),
            },
            "prompt": prompt,
        }

    if obs_format == "rby1_mobile":
        # 17-D: the rby1 layout with the planar base pose appended. Cameras and
        # the first 14 entries are identical, so the two share norm statistics
        # for those dims.
        base_obs = build_obs("rby1", m, d, renderer, idx, prompt)
        base_pose = np.array([d.qpos[i] for i in idx["base_q"]], dtype=np.float64)
        base_obs["state"] = np.concatenate([base_obs["state"], base_pose])
        return base_obs

    raise ValueError(f"unknown obs_format: {obs_format}")


def apply_action(action_format, action, d, idx, act):
    """Set d.ctrl in-place based on model action.

    `idx` / `act` are the joint-qpos / actuator-id maps built in main.
    """
    dt = 1.0 / CTRL_HZ
    if action_format == "droid":
        joint_vel = np.clip(action[:7], -1.0, 1.0)
        grip_action = 1.0 if float(action[7]) > 0.5 else 0.0
        for i, aid in enumerate(act["right_a"]):
            d.ctrl[aid] = d.qpos[idx["right_q"][i]] + joint_vel[i] * dt
        d.ctrl[act["right_grip_a"]] = GRIPPER_OPEN * (1.0 - grip_action)
        return

    if action_format == "libero":
        joint_delta = np.clip(action[:6], -0.5, 0.5) * 0.1
        grip_action = 1.0 if float(action[6]) > 0.5 else 0.0
        for i in range(6):
            d.ctrl[act["right_a"][i]] = d.qpos[idx["right_q"][i]] + joint_delta[i]
        d.ctrl[act["right_grip_a"]] = GRIPPER_OPEN * (1.0 - grip_action)
        return

    if action_format == "aloha":
        # (14,) = [L 6 dims, L grip, R 6 dims, R grip]. Apply as joint delta on joints[:6];
        # arm_6 (last wrist joint) held fixed. Scale small since ALOHA units may not match.
        left_delta  = np.clip(action[0:6],  -0.5, 0.5) * 0.1
        left_grip   = 1.0 if float(action[6])  > 0.5 else 0.0
        right_delta = np.clip(action[7:13], -0.5, 0.5) * 0.1
        right_grip  = 1.0 if float(action[13]) > 0.5 else 0.0

        for i in range(6):
            d.ctrl[act["left_a"][i]]  = d.qpos[idx["left_q"][i]]  + left_delta[i]
            d.ctrl[act["right_a"][i]] = d.qpos[idx["right_q"][i]] + right_delta[i]
        d.ctrl[act["left_grip_a"]]  = GRIPPER_OPEN * (1.0 - left_grip)
        d.ctrl[act["right_grip_a"]] = GRIPPER_OPEN * (1.0 - right_grip)
        return

    if action_format == "rby1":
        # (14,) = [L 6 abs joint targets, L grip, R 6 abs joint targets, R grip].
        # The server's data_transforms already convert the model's internal delta-joint
        # prediction back to absolute targets (AbsoluteActions) before returning, so we
        # apply these directly to ctrl -- no local delta/velocity math, unlike droid/libero/aloha.
        left_targets  = action[0:6]
        left_grip     = float(np.clip(action[6], 0.0, 1.0))
        right_targets = action[7:13]
        right_grip    = float(np.clip(action[13], 0.0, 1.0))

        for i in range(6):
            d.ctrl[act["left_a"][i]]  = left_targets[i]
            d.ctrl[act["right_a"][i]] = right_targets[i]
        d.ctrl[act["left_grip_a"]]  = left_grip * RBY1_GRIPPER_OPEN
        d.ctrl[act["right_grip_a"]] = right_grip * RBY1_GRIPPER_OPEN
        return

    if action_format == "rby1_mobile":
        # (17,) = rby1's 14 plus absolute base (x, y, yaw) in world coordinates.
        # Applied straight to the base position actuators, matching how the arm
        # targets are applied.
        apply_action("rby1", action[:14], d, idx, act)
        for i, aid in enumerate(act["base_a"]):
            d.ctrl[aid] = float(action[14 + i])
        return

    raise ValueError(f"unknown action_format: {action_format}")


def load_local_policy(mcfg):
    import jax
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    from openpi.training import config as _config

    print(f"JAX devices: {jax.devices()}")
    t0 = time.time()
    cfg = _config.get_config(mcfg["config"])
    ckpt_dir = download.maybe_download(mcfg["checkpoint"])
    policy = _policy_config.create_trained_policy(cfg, ckpt_dir)
    print(f"Policy loaded in {time.time()-t0:.1f}s")
    return policy


def load_remote_policy(remote):
    from openpi_client import websocket_client_policy as _websocket_client_policy

    host, _, port = remote.partition(":")
    policy = _websocket_client_policy.WebsocketClientPolicy(host=host, port=int(port) if port else None)
    print(f"Connected to remote policy server: {policy.get_server_metadata()}")
    return policy


def build_seam_session(policy, mcfg, seam_config_path):
    """Wrap a locally-loaded policy with SEAM/VLS and return a SeamPolicySession (local mode only)."""
    ws_root = REPO_ROOT.parent  # .../pi0_TO_ws
    if str(ws_root) not in sys.path:
        sys.path.insert(0, str(ws_root))
    from openpi.shared import download
    from openpi.training import config as _config
    from benchmark.seam_vla.config import SeamConfig
    from benchmark.seam_vla.factory import build_seam_policy
    from benchmark.seam_vla.policy.seam_policy import SeamPolicySession

    if seam_config_path is None:
        seam_config_path = str(ws_root / "benchmark/seam_vla/configs/seam_rby1.yaml")
    seam_cfg = SeamConfig.from_yaml(seam_config_path)
    train_config = _config.get_config(mcfg["config"])
    ckpt_dir = download.maybe_download(mcfg["checkpoint"])
    seam_policy = build_seam_policy(policy, train_config, ckpt_dir, seam_cfg)
    print(f"[SEAM] enabled (local): H={seam_policy.H} K={seam_policy.K} L={seam_policy.L} "
          f"M={seam_policy.M} N={seam_policy.N} D={seam_policy.D} "
          f"guided_dims={int(seam_policy._dim_mask.sum())} "
          f"compensation={seam_cfg.seam_delta_base_compensation}")
    return SeamPolicySession(seam_policy)


def append_jsonl(path, record):
    """Append and fsync one trial record so a long evaluation can resume safely."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def completed_grid_trial_ids(path, *, grid_fingerprint, condition):
    path = pathlib.Path(path)
    if not path.exists():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            # Setup errors and interrupted rollouts are retried. Results from a
            # different grid revision or baseline/SEAM condition are independent.
            if (
                record.get("status") not in ("setup_error", "interrupted")
                and record.get("grid_fingerprint") == grid_fingerprint
                and record.get("condition") == condition
                and record.get("trial_id")
            ):
                completed.add(str(record["trial_id"]))
    return completed


def save_video(path, frames):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"saving {len(frames)} frames -> {path}")
    try:
        import imageio
        imageio.mimsave(path, frames, fps=CTRL_HZ)
    except ImportError:
        for index, frame in enumerate(frames):
            Image.fromarray(frame).save(path.with_suffix("").as_posix() + f"_{index:04d}.png")
        print("(imageio not installed, saved as PNG sequence)")


def save_trajectory(
    path,
    *,
    executed_actions,
    measured_states,
    predicted_chunks,
    chunk_start_steps,
    inference_states,
    inference_ms,
    inference_used_vls,
    inference_chunk_indices,
    prompt,
    condition,
):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        executed_actions=np.asarray(executed_actions, dtype=np.float64),
        measured_qpos=np.asarray(measured_states, dtype=np.float64),
        predicted_chunks=np.asarray(predicted_chunks, dtype=np.float64),
        chunk_start_steps=np.asarray(chunk_start_steps, dtype=np.int64),
        inference_states=np.asarray(inference_states, dtype=np.float64),
        inference_ms=np.asarray(inference_ms, dtype=np.float64),
        used_vls=np.asarray(inference_used_vls, dtype=bool),
        chunk_indices=np.asarray(inference_chunk_indices, dtype=np.int64),
        prompt=np.asarray(prompt),
        condition=np.asarray(condition),
        control_hz=np.asarray(CTRL_HZ, dtype=np.int64),
        execution_length=np.asarray(OPEN_LOOP_HORIZON, dtype=np.int64),
    )
    print(f"saving trajectory ({len(executed_actions)} executed steps) -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS.keys()), default="droid",
                    help="which pi0.5 variant to load")
    ap.add_argument("--prompt", default="put the red block in the brown box with your right hand",
                    help="for --model rby1, use one of the 6 exact task strings the model "
                         "was fine-tuned on (see rby1_dataset_v1/meta/tasks.jsonl)")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument(
        "--view",
        choices=("free", "front"),
        default=None,
        help="interactive/recording camera preset; does not alter policy input cameras "
             "(default: 'front' with --grid-experiment, otherwise 'free')",
    )
    ap.add_argument("--record", default=None, help="path to .mp4 for third-person recording")
    ap.add_argument(
        "--trajectory-out",
        default=None,
        metavar="NPZ",
        help="save executed actions, measured RBY1 state, returned chunks, and inference timing "
             "for BJ/IJ/CD/AVb evaluation",
    )
    ap.add_argument("--record-inputs", default=None, metavar="DIR",
                    help="directory for cam_high/cam_left_wrist/cam_right_wrist policy-input MP4s")
    ap.add_argument("--remote", default=None,
                    help="host:port of a running scripts/serve_policy.py server; if set, "
                         "skips local model load and streams obs/actions over websocket instead")
    ap.add_argument("--seam", action="store_true",
                    help="apply SEAM/VLS chunk-boundary smoothing. Local mode: wraps the in-process "
                         "policy. Remote mode: sends a seam_reset flag and assumes the server is "
                         "running serve_seam_policy.py (VLS must run where the model is).")
    ap.add_argument("--seam-config", default=None,
                    help="path to a SEAM yaml (default: benchmark/seam_vla/configs/seam_rby1.yaml)")
    ap.add_argument("--start-delay", type=float, default=2.0,
                    help="seconds to run the simulator and show the viewer before the first "
                         "policy inference request (default: 2.0; use 0 to disable)")
    ap.add_argument("--speed", type=float, default=0,
                    help="playback speed relative to real time: 1.0 paces the sim to wall "
                         "clock (physics run ~17x faster than real time otherwise), 0.5 is "
                         "half speed / slow motion, 0 disables pacing (run as fast as "
                         "possible, e.g. for --headless recording)")
    ap.add_argument(
        "--grid-experiment",
        action="store_true",
        help="run the saved left/right grid for every selected color without per-trial prompts",
    )
    ap.add_argument(
        "--grid-config",
        type=pathlib.Path,
        default=DEFAULT_GRID_CONFIG,
        help="grid JSON written by preview_block_grid.py",
    )
    ap.add_argument(
        "--grid-colors",
        nargs="+",
        choices=GRID_COLORS,
        default=list(GRID_COLORS),
        help="colors to include in --grid-experiment",
    )
    ap.add_argument("--grid-repeats", type=int, default=3)
    ap.add_argument(
        "--trial-max-steps",
        type=int,
        default=600,
        help="maximum policy action steps per grid trial (600 at 15 Hz = 40 s)",
    )
    ap.add_argument(
        "--grid-output-dir",
        type=pathlib.Path,
        default=pathlib.Path("data/rby1_grid_eval"),
        help="results.jsonl and optional per-trial artifacts",
    )
    ap.add_argument(
        "--grid-record",
        action="store_true",
        help="save a third-person MP4 (front view by default) for every grid trial",
    )
    ap.add_argument(
        "--grid-record-inputs",
        action="store_true",
        help="save the three policy-input camera MP4s (cam_high/cam_left_wrist/cam_right_wrist) "
             "for every grid trial, under <grid-output-dir>/policy_inputs/",
    )
    ap.add_argument(
        "--grid-save-trajectories",
        action="store_true",
        help="save the trajectory NPZ for every grid trial",
    )
    ap.add_argument(
        "--grid-record-all",
        action="store_true",
        help="shorthand for --grid-record --grid-record-inputs --grid-save-trajectories",
    )
    ap.add_argument(
        "--no-grid-resume",
        action="store_true",
        help="do not skip trial_ids already present in results.jsonl",
    )
    args = ap.parse_args()

    if args.grid_record_all:
        args.grid_record = True
        args.grid_record_inputs = True
        args.grid_save_trajectories = True
    # The grid evaluation is recorded from a fixed front view so every trial video is
    # comparable; an explicit --view still wins.
    if args.view is None:
        args.view = "front" if args.grid_experiment else "free"

    if args.start_delay < 0:
        ap.error("--start-delay must be non-negative")
    if args.speed < 0:
        ap.error("--speed must be non-negative (0 = unlimited)")
    if args.grid_repeats < 1:
        ap.error("--grid-repeats must be >= 1")
    if args.trial_max_steps < 1:
        ap.error("--trial-max-steps must be >= 1")
    if args.grid_experiment and args.model != "rby1":
        ap.error("--grid-experiment requires --model rby1")
    if args.grid_experiment and args.max_steps > 0:
        ap.error("use --trial-max-steps instead of --max-steps with --grid-experiment")
    if args.grid_experiment and args.record:
        ap.error("use --grid-record instead of --record with --grid-experiment")
    if args.grid_experiment and args.trajectory_out:
        ap.error("use --grid-save-trajectories instead of --trajectory-out with --grid-experiment")
    if args.grid_experiment and args.record_inputs:
        ap.error("use --grid-record-inputs instead of --record-inputs with --grid-experiment")
    for flag, name in (
        (args.grid_record, "--grid-record"),
        (args.grid_record_inputs, "--grid-record-inputs"),
        (args.grid_save_trajectories, "--grid-save-trajectories"),
    ):
        if flag and not args.grid_experiment:
            ap.error(f"{name} requires --grid-experiment")

    mcfg = MODELS[args.model]
    if args.record_inputs and mcfg["obs_format"] not in ("aloha", "rby1"):
        ap.error("--record-inputs requires --model aloha or --model rby1")
    if args.trajectory_out and mcfg["obs_format"] != "rby1":
        ap.error("--trajectory-out currently requires --model rby1")
    print(f"=== Model: {args.model} ===")
    if args.remote:
        print(f"  remote     : {args.remote}  (server must serve obs_format={mcfg['obs_format']!r})")
    else:
        print(f"  config     : {mcfg['config']}")
        print(f"  checkpoint : {mcfg['checkpoint']}")
    print(f"  obs_format : {mcfg['obs_format']}")
    print(f"  act_format : {mcfg['action_format']}")

    # What each rollout has to buffer. In grid mode the trajectory buffers are always
    # collected because BJ/IJ/CD/AVb for results.jsonl are derived from them, even when
    # --grid-save-trajectories is off and the raw NPZ is not written out.
    capture_video = bool(args.record) or (args.grid_experiment and args.grid_record)
    capture_inputs = bool(args.record_inputs) or (args.grid_experiment and args.grid_record_inputs)
    collect_trajectory = bool(args.trajectory_out) or args.grid_experiment

    # The scene follows the model: rby1_mobile needs the transport root, which
    # also supplies the base actuators its action format writes to.
    m = mujoco.MjModel.from_xml_path(mcfg.get("model_xml", MODEL_XML))
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "teleop")
    mujoco.mj_resetDataKeyframe(m, d, key)
    # Without this, body/geom world transforms (xpos/xquat) are stale until the first
    # mj_step -- the very first observation (which drives the first action chunk) would
    # be rendered from a blank/garbage scene.
    mujoco.mj_forward(m, d)

    for i in range(m.nu):
        d.ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]

    # Build joint / actuator index maps once (used by build_obs and apply_action).
    idx = {
        "right_q":       [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
                          for j in RIGHT_ARM_JOINTS],
        "left_q":        [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
                          for j in LEFT_ARM_JOINTS],
        "right_grip_q":  m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_R_JOINT)],
        "left_grip_q":   m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_L_JOINT)],
        # Planar base DoFs. Present in every rby1a model; only the transport root
        # actuates them.
        "base_q":        [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
                          for j in BASE_JOINTS],
    }
    act = {
        "right_a":       [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in RIGHT_ARM_ACTS],
        "left_a":        [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in LEFT_ARM_ACTS],
        "right_grip_a":  mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_R_ACT),
        "left_grip_a":   mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_L_ACT),
        # Empty on model.xml, which has no base actuators; only rby1_mobile uses it.
        "base_a":        [a for a in
                          (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in BASE_ACTS)
                          if a >= 0],
    }

    # Policy-input renderer MUST match the training-data collection resolution exactly
    # (rby1_manipulation/scenario*.py, collect_batch.py render at native 224x224).
    # Rendering at a different aspect ratio (e.g. 640x480) and resizing down distorts
    # the field of view relative to what the model was trained on.
    renderer_pol = mujoco.Renderer(m, height=224, width=224)
    renderer_rec = mujoco.Renderer(m, height=480, width=640)  # third-person --record only
    recording_camera = -1
    if args.view != "free":
        recording_camera = mujoco.MjvCamera()
        configure_view_camera(recording_camera, args.view)

    policy = load_remote_policy(args.remote) if args.remote else load_local_policy(mcfg)
    print(f"Prompt: {args.prompt!r}")

    # SEAM/VLS setup. Local: wrap the in-process policy. Remote: VLS runs server-side (serve_seam_policy);
    # we only send a one-time seam_reset so the server starts a fresh per-episode SEAM state.
    seam_session = None
    seam_remote = False
    if args.seam:
        if args.remote:
            seam_remote = True
            print("[SEAM] remote mode: the server must be running "
                  "benchmark/seam_vla/serving/serve_seam_policy.py; sending seam_reset on first request.")
        else:
            seam_session = build_seam_session(policy, mcfg, args.seam_config)

    steps_per_action = max(1, int(round(1.0 / (CTRL_HZ * m.opt.timestep))))
    print(f"sim dt={m.opt.timestep}s  action interval={1/CTRL_HZ:.3f}s  sim-steps/action={steps_per_action}")

    video_frames = []
    input_frame_buffers = {name: [] for name in POLICY_CAMERA_NAMES}
    executed_actions = []
    measured_states = []
    predicted_chunks = []
    chunk_start_steps = []
    inference_states = []
    inference_ms = []
    inference_used_vls = []
    inference_chunk_indices = []
    ctx = None if args.headless else mujoco.viewer.launch_passive(m, d)
    if ctx is not None and args.view != "free":
        configure_view_camera(ctx.cam, args.view)
        ctx.sync()

    def rby1_state():
        """Current 14-D physical state in the policy/action layout."""
        left = np.asarray([d.qpos[i] for i in idx["left_q"][:6]], dtype=np.float64)
        right = np.asarray([d.qpos[i] for i in idx["right_q"][:6]], dtype=np.float64)
        left_grip = float(abs(d.qpos[idx["left_grip_q"]]) / abs(RBY1_GRIPPER_OPEN))
        right_grip = float(abs(d.qpos[idx["right_grip_q"]]) / abs(RBY1_GRIPPER_OPEN))
        return np.concatenate([left, [left_grip], right, [right_grip]])

    def wait_before_inference():
        """Advance the initial scene in real time before requesting an action."""
        if args.start_delay <= 0:
            return True

        print(f"Showing initial scene for {args.start_delay:.1f}s before inference...")
        deadline = time.monotonic() + args.start_delay
        next_step = time.monotonic()

        while time.monotonic() < deadline:
            if ctx is not None and not ctx.is_running():
                return False

            mujoco.mj_step(m, d)
            if ctx is not None:
                ctx.sync()

            # Keep the warm-up synchronized to wall-clock time so the configured
            # delay corresponds to what the user sees in the interactive viewer.
            next_step += m.opt.timestep
            sleep_time = next_step - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("Starting policy inference.")
        return True

    # Wall-clock seconds one action step should occupy at the requested speed.
    # steps_per_action sim steps advance steps_per_action*timestep of sim time;
    # dividing by --speed lets 0.5 play at half speed, etc. speed<=0 disables pacing.
    sim_seconds_per_action = steps_per_action * m.opt.timestep
    wall_seconds_per_action = (sim_seconds_per_action / args.speed) if args.speed > 0 else 0.0

    def loop_body(*, prompt=None, max_steps=None, stop_check=None):
        policy_prompt = args.prompt if prompt is None else prompt
        step_limit = (
            max_steps
            if max_steps is not None
            else (args.max_steps if args.max_steps > 0 else 10**9)
        )
        chunk = None
        chunk_step = 0
        # Wall-clock anchor for the next action step; reset after each inference so a
        # slow inference call is not "paid back" by sprinting the following steps.
        next_action_deadline = time.monotonic()
        for t_step in range(step_limit):
            if chunk is None or chunk_step >= OPEN_LOOP_HORIZON:
                obs = build_obs(mcfg["obs_format"], m, d, renderer_pol, idx, policy_prompt)
                if capture_inputs:
                    capture_policy_input_frames(obs, input_frame_buffers)
                t_infer = time.time()
                seam_timing = {}
                if seam_session is not None:
                    # Local SEAM: VLS-guided chunk; first chunk is baseline automatically.
                    chunk_arr, diag = seam_session.predict_chunk(obs, want_diagnostics=False)
                    chunk = np.asarray(chunk_arr)
                    seam_timing = {
                        "used_vls": bool(diag.used_vls),
                        "chunk_index": int(diag.chunk_index),
                    }
                else:
                    if seam_remote and t_step == 0:
                        obs["seam_reset"] = True  # tell server-side SEAM to start a fresh episode
                    result = policy.infer(obs)
                    chunk = np.asarray(result["actions"])
                    seam_timing = result.get("seam_timing", {})
                infer_elapsed_ms = (time.time() - t_infer) * 1000.0
                chunk_step = 0

                if collect_trajectory:
                    predicted_chunks.append(chunk.copy())
                    chunk_start_steps.append(t_step)
                    inference_states.append(np.asarray(obs["state"], dtype=np.float64).copy())
                    inference_ms.append(infer_elapsed_ms)
                    inference_used_vls.append(bool(seam_timing.get("used_vls", False)))
                    inference_chunk_indices.append(
                        int(seam_timing.get("chunk_index", len(predicted_chunks) - 1))
                    )

                # brief status line — content varies by embodiment
                if mcfg["obs_format"] in ("aloha", "rby1"):
                    l_pos = np.array([d.qpos[i] for i in idx["left_q"]], dtype=np.float64)
                    r_pos = np.array([d.qpos[i] for i in idx["right_q"]], dtype=np.float64)
                    print(f"[t={t_step:4d}] infer={infer_elapsed_ms/1000.0:.2f}s chunk={chunk.shape}  "
                          f"q_l={l_pos[:3].round(2).tolist()}...  q_r={r_pos[:3].round(2).tolist()}...")
                else:
                    r_pos = np.array([d.qpos[i] for i in idx["right_q"]], dtype=np.float64)
                    print(f"[t={t_step:4d}] infer={infer_elapsed_ms/1000.0:.2f}s chunk={chunk.shape}  "
                          f"q_r={r_pos.round(2).tolist()}")

                # Inference (and the initial ~10s JIT compile) stalls wall clock while
                # sim time is frozen; re-anchor so we resume real-time from here.
                next_action_deadline = time.monotonic()

            action = np.asarray(chunk[chunk_step], dtype=np.float64)
            apply_action(mcfg["action_format"], action, d, idx, act)

            for _ in range(steps_per_action):
                mujoco.mj_step(m, d)
            if ctx is not None:
                ctx.sync()

            # Pace to wall clock so the viewer plays at real time (or --speed x).
            if wall_seconds_per_action > 0:
                next_action_deadline += wall_seconds_per_action
                sleep_time = next_action_deadline - time.monotonic()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # Fell behind (slow render/inference on a weak machine): drop the
                    # backlog instead of accumulating permanent lag.
                    next_action_deadline = time.monotonic()
            if capture_video:
                renderer_rec.update_scene(d, camera=recording_camera)
                video_frames.append(renderer_rec.render())
            if collect_trajectory:
                executed_actions.append(action.copy())
                measured_states.append(rby1_state())

            chunk_step += 1
            if ctx is not None and not ctx.is_running():
                return {"status": "interrupted", "steps": t_step + 1}
            if stop_check is not None:
                stop_status = stop_check(t_step + 1)
                if stop_status is not None:
                    return {"status": stop_status, "steps": t_step + 1}
        return {"status": "timeout", "steps": step_limit}

    def clear_episode_buffers():
        video_frames.clear()
        for frames in input_frame_buffers.values():
            frames.clear()
        executed_actions.clear()
        measured_states.clear()
        predicted_chunks.clear()
        chunk_start_steps.clear()
        inference_states.clear()
        inference_ms.clear()
        inference_used_vls.clear()
        inference_chunk_indices.clear()

    def validate_grid_setup(color, requested, actual):
        errors = []
        target_error_xy = float(np.linalg.norm(actual[color][:2] - requested[color][:2]))
        if target_error_xy > 0.01:
            errors.append(f"target xy shifted {target_error_xy:.4f} m during settle")
        for block_color, position in actual.items():
            if not (0.82 <= position[2] <= 0.90):
                errors.append(f"{block_color} settled at invalid z={position[2]:.4f}")
        actual_positions = list(actual.items())
        for first_index, (first_color, first_pos) in enumerate(actual_positions):
            for second_color, second_pos in actual_positions[first_index + 1:]:
                separation = float(np.linalg.norm(first_pos[:2] - second_pos[:2]))
                if separation < 0.075:
                    errors.append(
                        f"{first_color}/{second_color} separation is only {separation:.4f} m"
                    )
        return errors

    def make_grid_stop_check(color):
        container_position = body_position(m, d, "container")
        block_body_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, BLOCK_BODIES[color]
        )
        block_joint_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_JOINT, f"{color}_block_free"
        )
        block_dof_address = m.jnt_dofadr[block_joint_id]
        stable_steps = 0

        def stop_check(_step):
            nonlocal stable_steps
            position = d.xpos[block_body_id]
            linear_speed = float(
                np.linalg.norm(d.qvel[block_dof_address:block_dof_address + 3])
            )

            # Container interior half-width is 0.10 m and the block half-width is
            # 0.025 m, hence a fully-contained center must be within 0.075 m.
            inside_xy = (
                abs(position[0] - container_position[0]) <= 0.075
                and abs(position[1] - container_position[1]) <= 0.075
            )
            released_height = 0.82 <= position[2] <= 0.92
            stationary = linear_speed <= 0.03
            if inside_xy and released_height and stationary:
                stable_steps += 1
            else:
                stable_steps = 0
            if stable_steps >= 8:
                return "success"

            # Stop early when recovery is no longer plausible.
            if (
                position[2] < 0.72
                or position[0] < 0.30
                or position[0] > 1.00
                or abs(position[1]) > 0.55
            ):
                return "failure"
            return None

        return stop_check

    def run_grid_experiment():
        config = load_grid_config(args.grid_config)
        output_dir = args.grid_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "results.jsonl"
        canonical_grid = json.dumps(
            config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        grid_fingerprint = hashlib.sha256(canonical_grid).hexdigest()[:12]
        snapshot_path = output_dir / f"grid_config_{grid_fingerprint}.json"
        if not snapshot_path.exists():
            with snapshot_path.open("w", encoding="utf-8") as stream:
                json.dump(config, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
        condition = "seam" if args.seam else "baseline"
        completed = (
            set()
            if args.no_grid_resume
            else completed_grid_trial_ids(
                results_path,
                grid_fingerprint=grid_fingerprint,
                condition=condition,
            )
        )
        trial_specs = [
            (color, side, grid_index, repeat)
            for color in args.grid_colors
            for side in ("left", "right")
            for grid_index in range(len(config["positions"][side]))
            for repeat in range(1, args.grid_repeats + 1)
        ]
        pending_count = sum(
            f"{color}_{side}_g{grid_index + 1:02d}_r{repeat}" not in completed
            for color, side, grid_index, repeat in trial_specs
        )
        print(
            "\n=== RBY1 grid experiment ===\n"
            f"  config       : {args.grid_config}\n"
            f"  grid revision: {grid_fingerprint}\n"
            f"  condition    : {condition}\n"
            f"  output       : {output_dir}\n"
            f"  total trials : {len(trial_specs)}\n"
            f"  completed    : {len(trial_specs) - pending_count}\n"
            f"  pending      : {pending_count}\n"
            f"  max steps    : {args.trial_max_steps} "
            f"({args.trial_max_steps / CTRL_HZ:.1f}s simulation time)"
        )

        for ordinal, (color, side, grid_index, repeat) in enumerate(trial_specs, start=1):
            trial_id = f"{color}_{side}_g{grid_index + 1:02d}_r{repeat}"
            if trial_id in completed:
                print(f"[{ordinal:03d}/{len(trial_specs)}] SKIP completed {trial_id}")
                continue
            if ctx is not None and not ctx.is_running():
                print("viewer closed; stopping grid experiment")
                return

            clear_episode_buffers()
            requested = actual = None
            setup_errors = []
            for setup_attempt in range(1, 3):
                requested, actual = reset_and_place_trial(
                    m,
                    d,
                    config,
                    color=color,
                    side=side,
                    grid_index=grid_index,
                    settle_seconds=1.5,
                    on_step=(ctx.sync if ctx is not None else None),
                )
                setup_errors = validate_grid_setup(color, requested, actual)
                if not setup_errors:
                    break
                print(
                    f"[{ordinal:03d}/{len(trial_specs)}] {trial_id} "
                    f"setup attempt {setup_attempt} invalid: {'; '.join(setup_errors)}"
                )

            requested_target = requested[color]
            actual_target = actual[color]
            prompt = f"put the {color} block in the brown box with your {side} hand"
            print(
                f"\n[{ordinal:03d}/{len(trial_specs)}] START {trial_id}\n"
                f"  prompt    : {prompt}\n"
                f"  requested : {requested_target.round(4).tolist()}\n"
                f"  settled   : {actual_target.round(4).tolist()}"
            )

            started_at = datetime.now(timezone.utc)
            if setup_errors:
                record = {
                    "trial_id": trial_id,
                    "color": color,
                    "side": side,
                    "grid_index": grid_index + 1,
                    "repeat": repeat,
                    "requested_xyz": requested_target.tolist(),
                    "settled_xyz": actual_target.tolist(),
                    "prompt": prompt,
                    "condition": condition,
                    "grid_fingerprint": grid_fingerprint,
                    "status": "setup_error",
                    "success": False,
                    "steps": 0,
                    "errors": setup_errors,
                    "started_at": started_at.isoformat(),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                }
                append_jsonl(results_path, record)
                print(f"[{ordinal:03d}/{len(trial_specs)}] SETUP_ERROR {trial_id}")
                continue

            if seam_session is not None:
                seam_session.reset()

            outcome = loop_body(
                prompt=prompt,
                max_steps=args.trial_max_steps,
                stop_check=make_grid_stop_check(color),
            )
            finished_at = datetime.now(timezone.utc)
            status = outcome["status"]
            final_target = body_position(m, d, BLOCK_BODIES[color])
            success = status == "success"

            grid_col, grid_row = grid_cell_indices(config, side, grid_index)
            record = {
                "trial_id": trial_id,
                "color": color,
                "side": side,
                "grid_index": grid_index + 1,
                "grid_col": grid_col,
                "grid_row": grid_row,
                "repeat": repeat,
                "requested_xyz": requested_target.tolist(),
                "settled_xyz": actual_target.tolist(),
                "final_xyz": final_target.tolist(),
                "prompt": prompt,
                "condition": condition,
                "grid_fingerprint": grid_fingerprint,
                "status": status,
                "success": success,
                "steps": int(outcome["steps"]),
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "wall_seconds": (finished_at - started_at).total_seconds(),
                "control_hz": CTRL_HZ,
                "execution_length": OPEN_LOOP_HORIZON,
            }
            # BJ/IJ/CD/AVb on the 12 arm joints, for both the commanded action stream and the
            # measured physical response. Written inline so results.jsonl alone is enough for
            # the quantitative comparison; the NPZ is only needed to re-derive or plot them.
            record["motion_metrics"] = compute_trial_metrics(executed_actions, measured_states)
            if inference_ms:
                record["inference_ms_mean"] = _json_float(np.mean(inference_ms))
                record["inference_ms_max"] = _json_float(np.max(inference_ms))
            record["num_chunks"] = len(predicted_chunks)
            record["num_vls_chunks"] = int(sum(inference_used_vls))

            artifact_stem = f"{condition}_{trial_id}"
            if args.grid_record and video_frames:
                video_path = output_dir / "videos" / f"{artifact_stem}.mp4"
                save_video(video_path, video_frames)
                record["video"] = str(video_path)
            if args.grid_save_trajectories and executed_actions:
                trajectory_path = output_dir / "trajectories" / f"{artifact_stem}.npz"
                save_trajectory(
                    trajectory_path,
                    executed_actions=executed_actions,
                    measured_states=measured_states,
                    predicted_chunks=predicted_chunks,
                    chunk_start_steps=chunk_start_steps,
                    inference_states=inference_states,
                    inference_ms=inference_ms,
                    inference_used_vls=inference_used_vls,
                    inference_chunk_indices=inference_chunk_indices,
                    prompt=prompt,
                    condition=condition,
                )
                record["trajectory"] = str(trajectory_path)
            if capture_inputs and any(input_frame_buffers.values()):
                input_root = pathlib.Path(args.record_inputs or (output_dir / "policy_inputs"))
                input_path = input_root / artifact_stem
                save_policy_input_videos(input_path, input_frame_buffers, flat=True)
                record["policy_inputs"] = str(input_path)

            append_jsonl(results_path, record)
            action_metrics = record["motion_metrics"].get("action", {})
            summary = "  ".join(
                f"{key}={value:.4f}" if isinstance(value, float) else f"{key}=n/a"
                for key, value in (
                    ("BJ", action_metrics.get("BJ")),
                    ("CD", action_metrics.get("CD")),
                )
            )
            print(
                f"[{ordinal:03d}/{len(trial_specs)}] {status.upper()} {trial_id} "
                f"steps={outcome['steps']} final={final_target.round(4).tolist()} {summary}"
            )
            if status == "interrupted":
                return

    try:
        if args.grid_experiment:
            run_grid_experiment()
        elif wait_before_inference():
            loop_body()
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        if ctx is not None:
            ctx.close()
        if not args.grid_experiment and args.record and video_frames:
            save_video(args.record, video_frames)
        if not args.grid_experiment and args.record_inputs:
            save_policy_input_videos(args.record_inputs, input_frame_buffers)
        if not args.grid_experiment and args.trajectory_out and executed_actions:
            save_trajectory(
                args.trajectory_out,
                executed_actions=executed_actions,
                measured_states=measured_states,
                predicted_chunks=predicted_chunks,
                chunk_start_steps=chunk_start_steps,
                inference_states=inference_states,
                inference_ms=inference_ms,
                inference_used_vls=inference_used_vls,
                inference_chunk_indices=inference_chunk_indices,
                prompt=args.prompt,
                condition="seam" if args.seam else "baseline",
            )


if __name__ == "__main__":
    main()
