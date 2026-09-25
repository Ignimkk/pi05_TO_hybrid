import argparse
import json
import os
import pathlib
import sys
import time
import numpy as np

if "--headless" in sys.argv and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "osmesa"

import mujoco
import mujoco.viewer
from PIL import Image

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_XML = str(REPO_ROOT / "rby1_description" / "models" / "rby1a" / "mujoco" / "model.xml")
MODEL_XML_PICK_PLACE_OBSTACLES = str(
    REPO_ROOT / "rby1_description" / "models" / "rby1a" / "mujoco"
    / "model_pick_place_obstacles.xml"
)
MODEL_XML_TRANSPORT = str(
    REPO_ROOT / "rby1_description" / "models" / "rby1a" / "mujoco"
    / "model_transport.xml"
)
MODEL_XML_TRANSPORT_PICK_PLACE_OBSTACLES = str(
    REPO_ROOT / "rby1_description" / "models" / "rby1a" / "mujoco"
    / "model_transport_pick_place_obstacles.xml"
)
MANIPULATION_SRC = REPO_ROOT / "rby1_manipulation" / "src"
if str(MANIPULATION_SRC) not in sys.path:
    sys.path.insert(0, str(MANIPULATION_SRC))

from rby1_manipulation.control.ik import left_arm_handles, right_arm_handles
from rby1_manipulation.control.motion import open_grippers
from rby1_manipulation.data.recording import (
    POLICY_RENDER_REFLECTIONS,
    POLICY_SOURCE_HEIGHT,
    POLICY_SOURCE_WIDTH,
    resize_policy_image,
)
from rby1_manipulation.simulation.fruit_grid import (
    OBJECT_TYPES as FRUIT_TYPES,
    load_fruit_grid_config,
    reset_fruit_grid_scene,
)
from rby1_manipulation.simulation.pick_place_obstacles import (
    PickPlaceObstacleManager,
    load_pick_place_obstacle_config,
)
from rby1_manipulation.simulation.transport_scene import (
    CRATE_JOINT,
    OBJECT_JOINTS,
    load_layout_config,
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
    },
    "rby1_randomized_pick_place_16d": {
        # LoRA fine-tune on local/rby1_randomized_pick_place_16d_v1 (2000 episodes,
        # 0-1599 trained; 1600-1799 validation; 1800-1999 test -- all held out).
        # 16-D schema keeps arm_6, unlike every 14-D model above.
        "config": "pi05_rby1_randomized_pick_place_16d_lora",
        "checkpoint": (
            "/mnt/dev/work/pi05_TO_hybrid/checkpoints/"
            "pi05_rby1_randomized_pick_place_16d_lora/"
            "rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999"
        ),
        "obs_format": "rby1_16d",
        "action_format": "rby1_16d",   # 16 = [L 7 abs joint, L grip, R 7 abs joint, R grip]
        "model_xml": MODEL_XML_TRANSPORT,
    },
    "rby1_transport_14d": {
        "config": "pi05_rby1_lora",
        "checkpoint": None,
        "obs_format": "rby1",
        "action_format": "rby1",
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

DEFAULT_PROMPT = "put the red block in the brown box with your right hand"

# Where the 16-D dataset metadata lives. Only meta/ is read, to reconstruct a
# recorded scene -- the parquet/video shards are not needed. Set
# RBY1_16D_DATASET to point at a different checkout.
RANDOMIZED_16D_DATASET = pathlib.Path(
    os.environ.get(
        "RBY1_16D_DATASET",
        "/mnt/dev/work/pi05_TO_hybrid/data/rby1_randomized_pick_place_16d_v1",
    )
)

CTRL_HZ = 15
OPEN_LOOP_HORIZON = 8
POLICY_CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")


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


def render_cam(model, data, renderer, cam_name, size=224, *, match_rby1_dataset=False):
    renderer.update_scene(data, camera=cam_name)
    if match_rby1_dataset and not POLICY_RENDER_REFLECTIONS:
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    img = renderer.render()
    if match_rby1_dataset:
        return resize_policy_image(img)
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


def save_policy_input_videos(output_dir, frame_buffers):
    """Write one MP4 per camera into a new, non-overwriting run directory."""
    if not any(frame_buffers[name] for name in POLICY_CAMERA_NAMES):
        print("no policy-input frames to save")
        return

    recording_root = pathlib.Path(output_dir)
    recording_root.mkdir(parents=True, exist_ok=True)
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


def load_randomized_episode(episode_index):
    """Read one recorded scene description from the 16-D dataset metadata."""
    path = RANDOMIZED_16D_DATASET / "meta" / "randomized_episodes.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; --model rby1_randomized_pick_place_16d needs the "
            "dataset metadata to reconstruct a scene"
        )
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record["episode_index"] == episode_index:
                return record
    raise ValueError(f"episode_index {episode_index} not present in {path}")


def randomized_split_of(episode_index):
    """Which split an episode belongs to; 'train' means the model memorised it."""
    path = RANDOMIZED_16D_DATASET / "meta" / "randomized_splits.json"
    if not path.exists():
        return None
    splits = json.loads(path.read_text())["splits"]
    for name, span in splits.items():
        if span["start_episode_index"] <= episode_index < span["end_episode_index_exclusive"]:
            return name
    return None


def hold_keyframe_pose(m, d):
    """Point every joint actuator at the pose the keyframe just set.

    `mj_resetDataKeyframe` applies the key's `qpos` **and** its `ctrl`. The
    `teleop` key carries no `ctrl`, so every position actuator is left commanding
    0 while `qpos` holds the teleop pose — and the actuators win as soon as the
    scene settles. The arms survived only because both reset paths overwrite
    their `ctrl` explicitly right afterwards.

    **The head did not.** `head_1` starts at 0.7 rad (looking down at the table)
    and `head_1_act` dragged it to 0 during the settle, so `cam_high` recorded a
    wall instead of the table. Every 16-D record taken before 2026-09-25 has that
    (`run_16d_ep1800`, `20260924_long16d`), and the policy was fed those frames.

    The collection path never had the bug because it holds *every* actuated joint
    at its keyframe qpos (`transport_scene.py`). This is the same thing, so the
    two paths agree. Callers that want a different target still overwrite `ctrl`
    after this returns.
    """
    for actuator in range(m.nu):
        if m.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        joint = m.actuator_trnid[actuator, 0]
        d.ctrl[actuator] = d.qpos[m.jnt_qposadr[joint]]


def reset_randomized_scene(m, d, record, settle_seconds=1.5):
    """Restore the exact initial state the recorded episode started from.

    Sets arm joints, opens both grippers, and places every fruit plus the basket
    at its recorded free-joint pose, then lets the scene settle. This replays a
    scene rather than sampling a new one, so an evaluation is reproducible and can
    be pinned to a held-out episode.
    """
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "teleop")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(m, d, key)
        hold_keyframe_pose(m, d)

    def joint_qposadr(name):
        joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"joint {name!r} missing from the loaded MuJoCo model")
        return int(m.jnt_qposadr[joint_id])

    def actuator_id(name):
        actuator = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator < 0:
            raise ValueError(f"actuator {name!r} missing from the loaded MuJoCo model")
        return int(actuator)

    # Arms: qpos and the matching ctrl target, so the position actuators hold the
    # pose instead of dragging it back to the keyframe.
    for joint_name, value in record["initial_arm_joint_qpos"].items():
        d.qpos[joint_qposadr(joint_name)] = float(value)
    for joints, actuators in ((LEFT_ARM_JOINTS, LEFT_ARM_ACTS),
                              (RIGHT_ARM_JOINTS, RIGHT_ARM_ACTS)):
        for joint_name, actuator_name in zip(joints, actuators):
            d.ctrl[actuator_id(actuator_name)] = float(
                record["initial_arm_joint_qpos"][joint_name]
            )

    # Grippers open, at the exact ctrl value the dataset was collected with.
    for joint_name, actuator_name in ((GRIPPER_L_JOINT, GRIPPER_L_ACT),
                                      (GRIPPER_R_JOINT, GRIPPER_R_ACT)):
        d.qpos[joint_qposadr(joint_name)] = RBY1_GRIPPER_OPEN
        d.ctrl[actuator_id(actuator_name)] = RBY1_GRIPPER_OPEN

    # Free bodies: 7-value [x y z qw qx qy qz] per free joint.
    free_poses = {
        OBJECT_JOINTS[fruit]: pose
        for fruit, pose in record["initial_fruit_poses"].items()
    }
    free_poses[CRATE_JOINT] = record["basket_pose"]
    for joint_name, pose in free_poses.items():
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (7,):
            raise ValueError(f"{joint_name} pose must have 7 values, got {pose.shape}")
        adr = joint_qposadr(joint_name)
        d.qpos[adr:adr + 7] = pose

    d.qvel[:] = 0.0
    d.qacc[:] = 0.0
    mujoco.mj_forward(m, d)
    for _ in range(int(settle_seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
    return record


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

    if obs_format == "rby1_16d":
        # 16-D schema: every arm joint including arm_6, which the 14-D models drop.
        # Mirrors build_state_16() in rby1_manipulation.simulation.transport_scene,
        # which is what recorded observation.state during data collection.
        base_img = render_cam(m, d, renderer, "zed_left", match_rby1_dataset=True)
        wrist_l_img = render_cam(m, d, renderer, "wrist_cam_l", match_rby1_dataset=True)
        wrist_r_img = render_cam(m, d, renderer, "wrist_cam_r", match_rby1_dataset=True)

        left_joint_pos = np.array([d.qpos[i] for i in idx["left_q"]], dtype=np.float64)
        right_joint_pos = np.array([d.qpos[i] for i in idx["right_q"]], dtype=np.float64)
        left_grip_norm = float(abs(d.qpos[idx["left_grip_q"]]) / abs(RBY1_GRIPPER_OPEN))
        right_grip_norm = float(abs(d.qpos[idx["right_grip_q"]]) / abs(RBY1_GRIPPER_OPEN))

        # [left arm_0..arm_6, left gripper, right arm_0..arm_6, right gripper]
        state = np.concatenate([
            left_joint_pos,
            [left_grip_norm],
            right_joint_pos,
            [right_grip_norm],
        ]).astype(np.float64)
        if state.shape != (16,):
            raise ValueError(f"rby1_16d state must be (16,), got {state.shape}")

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
        base_img = render_cam(
            m, d, renderer, "zed_left", match_rby1_dataset=True
        )
        wrist_l_img = render_cam(
            m, d, renderer, "wrist_cam_l", match_rby1_dataset=True
        )
        wrist_r_img = render_cam(
            m, d, renderer, "wrist_cam_r", match_rby1_dataset=True
        )

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

    if action_format == "rby1_16d":
        # (16,) = [L 7 abs joint targets, L grip, R 7 abs joint targets, R grip].
        # Like "rby1" these are absolute targets -- the server's AbsoluteActions
        # transform already undid the model's internal delta-joint prediction.
        # Unlike "rby1", arm_6 IS predicted, so nothing is held at a stale keyframe.
        if len(action) < 16:
            raise ValueError(
                f"rby1_16d expects a 16-D action, got {len(action)}. A 14-D policy "
                "server will silently zero-pad instead of failing -- check --remote."
            )
        left_targets = action[0:7]
        left_grip = float(np.clip(action[7], 0.0, 1.0))
        right_targets = action[8:15]
        right_grip = float(np.clip(action[15], 0.0, 1.0))

        for i in range(7):
            d.ctrl[act["left_a"][i]] = left_targets[i]
            d.ctrl[act["right_a"][i]] = right_targets[i]
        d.ctrl[act["left_grip_a"]] = left_grip * RBY1_GRIPPER_OPEN
        d.ctrl[act["right_grip_a"]] = right_grip * RBY1_GRIPPER_OPEN
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
        # arm_6 is intentionally absent from the 14-D policy. Reassert the
        # teleop-keyframe target on every policy step instead of relying on a
        # stale actuator ctrl value to hold it implicitly.
        d.ctrl[act["left_a"][6]] = act["left_arm6_hold"]
        d.ctrl[act["right_a"][6]] = act["right_arm6_hold"]
        d.ctrl[act["left_grip_a"]]  = left_grip * RBY1_GRIPPER_OPEN
        d.ctrl[act["right_grip_a"]] = right_grip * RBY1_GRIPPER_OPEN
        return

    raise ValueError(f"unknown action_format: {action_format}")


def validate_rby1_observation(obs, *, log=False):
    """Validate the raw 14-D/three-camera interface sent to the policy."""
    state = np.asarray(obs.get("state"))
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError(
            f"RBY1 observation state must be finite shape (14,), got {state.shape}"
        )
    images = obs.get("images", {})
    if set(images) != set(POLICY_CAMERA_NAMES):
        raise ValueError(
            f"RBY1 observation cameras must be {POLICY_CAMERA_NAMES}, got {tuple(images)}"
        )
    camera_summary = {}
    for name in POLICY_CAMERA_NAMES:
        image = np.asarray(images[name])
        if image.shape != (3, 224, 224) or image.dtype != np.uint8:
            raise ValueError(
                f"RBY1 camera {name!r} must be uint8 CHW (3, 224, 224), "
                f"got dtype={image.dtype} shape={image.shape}"
            )
        camera_summary[name] = f"{image.dtype}{tuple(image.shape)}"
    if log:
        print(
            "[first observation] "
            f"state_shape={state.shape} range=[{state.min():+.4f}, {state.max():+.4f}]"
        )
        print(
            "[first observation] state="
            + np.array2string(state, precision=4, separator=", ", suppress_small=True)
        )
        print(f"[first observation] cameras={camera_summary}")


def validate_rby1_action_chunk(chunk, *, log=False):
    """Validate absolute 14-D actions returned after server output transforms."""
    actions = np.asarray(chunk)
    if (
        actions.ndim != 2
        or actions.shape[0] == 0
        or actions.shape[1] != 14
        or not np.isfinite(actions).all()
    ):
        raise ValueError(
            "RBY1 policy actions must be a non-empty finite [horizon, 14] array, "
            f"got {actions.shape}"
        )
    if log:
        joint_actions = actions[:, [*range(6), *range(7, 13)]]
        gripper_actions = actions[:, [6, 13]]
        print(
            "[first action chunk] "
            f"shape={actions.shape} raw_range=[{actions.min():+.4f}, {actions.max():+.4f}] "
            f"joint_range=[{joint_actions.min():+.4f}, {joint_actions.max():+.4f}] "
            f"gripper_range=[{gripper_actions.min():+.4f}, {gripper_actions.max():+.4f}]"
        )
        print(
            "[first action chunk] action[0]="
            + np.array2string(actions[0], precision=4, separator=", ", suppress_small=True)
        )


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS.keys()), default="droid",
                    help="which pi0.5 variant to load")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help="for --model rby1, use one of the 6 exact task strings the model "
                         "was fine-tuned on (see rby1_dataset_v1/meta/tasks.jsonl)")
    ap.add_argument(
        "--episode-index", type=int, default=1800,
        help="for --model rby1_randomized_pick_place_16d: replay this recorded "
             "episode's initial scene. 0-1599 were trained on; 1600-1799 are "
             "validation and 1800-1999 test, so the default is the first held-out "
             "test scene. The episode's own prompt is used unless --prompt is given.")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument(
        "--view",
        choices=("free", "front"),
        default="free",
        help="interactive/recording camera preset; does not alter policy input cameras",
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
    ap.add_argument("--record-ag3s", default=None, metavar="DIR",
                    help="directory for AG3S policy-observation records: one compressed .npz per "
                         "inference step holding qpos, the 14-D state, the three 224x224 policy "
                         "images verbatim, and the returned chunk. Depth and segmentation are NOT "
                         "stored -- qpos regenerates them deterministically via TransportScene")
    ap.add_argument("--safe-remote", action="store_true",
                    help="서버가 π0.5+SEAM+AG3S+TO를 한 프로세스로 돌린다. 로컬은 3카메라 "
                         "관측을 보내고, 서버의 안전 판정이 유효할 때만 앞 8개 action을 "
                         "실행한다. unsafe·timeout·오래된 응답·서버 오류면 현재 관절을 hold")
    ap.add_argument("--safe-timeout", type=float, default=2.0, metavar="SEC",
                    help="서버 응답 제한 시간. 넘으면 hold")
    ap.add_argument("--safe-phase", default="approach",
                    help="AG3S에 주입할 조작 단계. AG3S는 절대 추론하지 않는다")
    ap.add_argument("--safe-manipulators", nargs="*", default=(), metavar="NAME",
                    help="접촉이 허용된 매니퓰레이터")
    ap.add_argument("--trajopt", action="store_true",
                    help="AG3S(attention + ESDF)로 충돌 제약을 만들고 TO로 청크를 정제해 "
                         "**그 결과를 실행**한다. 붙이지 않으면 루프는 예전과 동일하다")
    ap.add_argument("--trajopt-links", choices=("arms", "all"), default="arms",
                    help="충돌 제약을 어느 링크에 걸지. arms는 양팔과 손끝만 — 바퀴·베이스는 "
                         "결정 변수가 아니라 고칠 수 없는 위반을 상수로 깔아 실제 신호를 묻는다")
    ap.add_argument("--trajopt-cameras", nargs="*", default=None, metavar="CAMERA",
                    help="AG3S가 쓸 MuJoCo 카메라. 기본 zed_left wrist_cam_l wrist_cam_r")
    ap.add_argument("--trace", default=None, metavar="DIR",
                    help="제어 루프의 벽시계 타임스탬프를 trace.jsonl로 기록")
    ap.add_argument("--record-constraints", default=None, metavar="DIR",
                    help="AG3S의 제약 생성 중간 산출물(attention·점군·target·ESDF·여유거리)을 "
                         "청크마다 npz로 기록. 시각화가 이것만으로 재현된다")
    ap.add_argument("--record-constraints-esdf", choices=("none", "occupancy", "full"),
                    default="full",
                    help="거리장을 얼마나 저장할지. full은 청크당 약 1.2 MB (20 mm 복셀 실측)")
    ap.add_argument("--record-depth", nargs="*", default=None,
                    metavar="CAMERA",
                    help="with --record-ag3s, also store depth + intrinsics + extrinsics for these "
                         "MuJoCo cameras as uint16 millimetres (the format a real depth camera "
                         "delivers). No argument means all three: zed_left wrist_cam_l wrist_cam_r. "
                         "Adds roughly 1 MB per inference step")
    ap.add_argument("--remote", default=None,
                    help="host:port of a running scripts/serve_policy.py server; if set, "
                         "skips local model load and streams obs/actions over websocket instead")
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="override the selected model's local checkpoint; unnecessary with --remote",
    )
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
        "--obstacle-profile",
        default="clear",
        help="static obstacle profile from pick_place_obstacles.json; use a fruit_* "
             "profile with --model rby1_transport_14d",
    )
    ap.add_argument("--obstacle-config", type=pathlib.Path, default=None)
    ap.add_argument("--obstacle-stop-distance", type=float, default=0.02)
    ap.add_argument(
        "--fruit-layout-index",
        type=int,
        default=None,
        help="reset rby1_transport_14d to one of the 16 training fruit-grid layouts",
    )
    ap.add_argument(
        "--fruit-slot-order",
        nargs=4,
        choices=FRUIT_TYPES,
        default=None,
        metavar=("FRUIT1", "FRUIT2", "FRUIT3", "FRUIT4"),
        help="fruit permutation assigned to the four selected grid slots",
    )
    ap.add_argument(
        "--fruit-preloaded",
        nargs="*",
        choices=FRUIT_TYPES,
        default=None,
        help="fruits initially placed inside the crate; requires --fruit-layout-index",
    )
    ap.add_argument(
        "--fruit-basket-offset",
        type=float,
        default=0.0,
        help="move non-preloaded fruits this many metres radially away from the basket "
             "during a fruit-grid reset (inference-only; requires --fruit-layout-index)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="씬을 결정론적으로 정한다. seed 하나로 fruit layout(2~15) · slot order · "
             "과일 xy jitter(+-12 mm)가 파생된다. **난도는 정하지 않는다** — obstacle "
             "profile 은 --obstacle-profile 로 명시한다 (2026-09-22 판정). crate/shelf/"
             "friction/mass 는 건드리지 않는다: 정책의 파지 성공률이 함께 흔들리면 실패를 "
             "지각 탓인지 정책 탓인지 귀속할 수 없다",
    )
    ap.add_argument(
        "--record-frames",
        default=None,
        metavar="DIR",
        help="manifest.json + frames.jsonl + completeness.json 을 여기 쓴다. "
             "observation / planning / control 프레임마다 한 줄 (T0 의 요구)",
    )
    args = ap.parse_args()

    if args.start_delay < 0:
        ap.error("--start-delay must be non-negative")
    if args.speed < 0:
        ap.error("--speed must be non-negative (0 = unlimited)")
    if args.obstacle_stop_distance < 0:
        ap.error("--obstacle-stop-distance must be non-negative")
    if not np.isfinite(args.fruit_basket_offset) or args.fruit_basket_offset < 0:
        ap.error("--fruit-basket-offset must be a finite non-negative distance")
    if args.fruit_layout_index is not None and args.model != "rby1_transport_14d":
        ap.error("--fruit-layout-index requires --model rby1_transport_14d")
    if args.fruit_layout_index is None and (
        args.fruit_slot_order is not None
        or args.fruit_preloaded is not None
        or args.fruit_basket_offset != 0.0
    ):
        ap.error(
            "--fruit-slot-order/--fruit-preloaded/--fruit-basket-offset require "
            "--fruit-layout-index"
        )
    if args.fruit_slot_order is not None and len(set(args.fruit_slot_order)) != 4:
        ap.error("--fruit-slot-order must contain each fruit exactly once")
    # --seed 는 layout 과 slot order 를 **파생**시킨다. 둘을 함께 주면 어느 것이 이겼는지
    # 기록만 보고 알 수 없으므로 거절한다 — 조용히 한쪽을 무시하면 재현이 깨진다.
    seed_derived = None
    if args.seed is not None:
        if args.model != "rby1_transport_14d":
            ap.error("--seed requires --model rby1_transport_14d")
        if args.fruit_layout_index is not None or args.fruit_slot_order is not None:
            ap.error("--seed derives --fruit-layout-index and --fruit-slot-order; "
                     "pass the seed or the explicit values, not both")
        rng = np.random.default_rng(args.seed)
        # layout 0·1 은 run_0004/run_0005 가 이미 썼다. 프롬프트 공통 원칙 2("기존 seed 와
        # 저장된 결과를 시험 입력으로 재사용하지 않는다")의 보수적 해석이다.
        args.fruit_layout_index = int(rng.integers(2, 16))
        args.fruit_slot_order = [FRUIT_TYPES[i] for i in rng.permutation(len(FRUIT_TYPES))][:4]
        seed_derived = {
            "seed": int(args.seed),
            "fruit_layout_index": args.fruit_layout_index,
            "fruit_slot_order": list(args.fruit_slot_order),
            # 과일 xy jitter. 겹침 검사는 reset_fruit_grid_scene 이 한다.
            "position_jitter_xy_m": 0.012,
            "randomization_spec": "all off — crate/shelf/object_pose/friction/mass 는 "
                                  "건드리지 않는다 (2026-09-22 판정: nuisance 만)",
            "layout_range": [2, 15],
        }
    if args.fruit_preloaded is not None and len(set(args.fruit_preloaded)) != len(
        args.fruit_preloaded
    ):
        ap.error("--fruit-preloaded must not contain duplicates")
    mcfg = MODELS[args.model]
    obstacle_config = load_pick_place_obstacle_config(args.obstacle_config) \
        if args.obstacle_config else load_pick_place_obstacle_config()
    if args.obstacle_profile not in obstacle_config["profiles"]:
        ap.error(
            f"unknown --obstacle-profile {args.obstacle_profile!r}; known: "
            f"{tuple(obstacle_config['profiles'])}"
        )
    expected_obstacle_scene = (
        "fruit"
        if args.model in ("rby1_transport_14d", "rby1_randomized_pick_place_16d")
        else "block"
    )
    profile_scene = obstacle_config["profiles"][args.obstacle_profile]["scene"]
    if profile_scene not in ("any", expected_obstacle_scene):
        ap.error(
            f"--obstacle-profile {args.obstacle_profile!r} is for {profile_scene} scene; "
            f"--model {args.model} uses {expected_obstacle_scene} scene"
        )
    if args.obstacle_profile != "clear" and args.model not in (
        "rby1", "rby1_transport_14d", "rby1_randomized_pick_place_16d"
    ):
        ap.error(
            "static pick-place obstacles require --model rby1, rby1_transport_14d "
            "or rby1_randomized_pick_place_16d"
        )
    if args.checkpoint:
        mcfg = dict(mcfg, checkpoint=args.checkpoint)
    if not args.remote and not mcfg.get("checkpoint"):
        ap.error(f"--model {args.model} requires --remote or --checkpoint")
    if args.record_inputs and mcfg["obs_format"] not in ("aloha", "rby1", "rby1_16d"):
        ap.error("--record-inputs requires an aloha or rby1 3-camera model")
    # AG3S 는 RB-Y1 의 **카메라**에 배선되어 있다 — action 차원이 아니다. 14D 와 16D 는 같은
    # 세 카메라를 쓰므로 둘 다 받는다 (2026-09-24, 16D 전환).
    AG3S_OBS_FORMATS = ("rby1", "rby1_16d")
    if args.trajectory_out and mcfg["obs_format"] not in AG3S_OBS_FORMATS:
        ap.error(f"--trajectory-out requires an RB-Y1 model, got {mcfg['obs_format']!r}")
    if args.record_ag3s and mcfg["obs_format"] not in AG3S_OBS_FORMATS:
        ap.error(f"--record-ag3s requires an RB-Y1 model with the three AG3S cameras "
                 f"(obs_format in {AG3S_OBS_FORMATS}), got {mcfg['obs_format']!r}")
    if args.record_depth is not None and not args.record_ag3s:
        ap.error("--record-depth only does something together with --record-ag3s")
    if args.safe_remote and not args.remote:
        ap.error("--safe-remote requires --remote (the safety layer runs on the server)")
    if args.safe_remote and args.trajopt:
        ap.error("--safe-remote and --trajopt are two places to run the same layer; pick one")
    # 전에는 `obs_format != "rby1"` 로 걸러 16D 를 거절했는데, 그 조건은 카메라가 아니라
    # 차원을 보고 있었다.
    if args.safe_remote and mcfg["obs_format"] not in AG3S_OBS_FORMATS:
        ap.error(f"--safe-remote requires an RB-Y1 model with the three AG3S cameras "
                 f"(obs_format in {AG3S_OBS_FORMATS}), got {mcfg['obs_format']!r}")
    if args.trajopt and mcfg["obs_format"] not in AG3S_OBS_FORMATS:
        ap.error(f"--trajopt requires an RB-Y1 model with the three AG3S cameras "
                 f"(obs_format in {AG3S_OBS_FORMATS}), got {mcfg['obs_format']!r}")
    if args.record_constraints and not args.trajopt:
        ap.error("--record-constraints only does something together with --trajopt")
    # These files are only written once the rollout finishes. Create their
    # directories now so a missing parent does not discard a completed run.
    for output_path in (args.record, args.trajectory_out):
        if output_path:
            parent = pathlib.Path(output_path).expanduser().parent
            if str(parent):
                parent.mkdir(parents=True, exist_ok=True)

    print(f"=== Model: {args.model} ===")
    if args.remote:
        print(f"  remote     : {args.remote}  (server must serve obs_format={mcfg['obs_format']!r})")
    else:
        print(f"  config     : {mcfg['config']}")
        print(f"  checkpoint : {mcfg['checkpoint']}")
    print(f"  obs_format : {mcfg['obs_format']}")
    print(f"  act_format : {mcfg['action_format']}")

    if args.obstacle_profile == "clear":
        model_xml = mcfg.get("model_xml", MODEL_XML)
    elif args.model == "rby1_transport_14d":
        model_xml = MODEL_XML_TRANSPORT_PICK_PLACE_OBSTACLES
    else:
        model_xml = MODEL_XML_PICK_PLACE_OBSTACLES
    m = mujoco.MjModel.from_xml_path(model_xml)
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "teleop")
    mujoco.mj_resetDataKeyframe(m, d, key)
    hold_keyframe_pose(m, d)
    teleop_arm6_targets = {}
    for side, joint_name in (
        ("left", LEFT_ARM_JOINTS[6]),
        ("right", RIGHT_ARM_JOINTS[6]),
    ):
        joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        teleop_arm6_targets[side] = float(d.qpos[m.jnt_qposadr[joint_id]])
    fruit_scene = None
    if args.fruit_layout_index is not None:
        try:
            fruit_scene = reset_fruit_grid_scene(
                m,
                d,
                load_layout_config(),
                load_fruit_grid_config(),
                layout_index=args.fruit_layout_index,
                slot_order=args.fruit_slot_order or FRUIT_TYPES,
                preloaded_objects=args.fruit_preloaded or (),
                settle_seconds=1.5,
                basket_clearance_offset=args.fruit_basket_offset,
                # seed 를 줬으면 과일 xy 를 작게 흔든다. `RandomizationSpec()` 은 손대지
                # 않으므로 crate·shelf·friction·mass 는 그대로다 — seed 는 **지각이 보는
                # 것만** 바꾼다.
                **({"rng": np.random.default_rng(args.seed + 1),
                    "position_jitter_xy": 0.012} if args.seed is not None else {}),
            )
        except ValueError as exc:
            ap.error(str(exc))
        print(
            f"  fruit reset: layout={fruit_scene.layout_index} "
            f"slots={fruit_scene.slot_order} preloaded={fruit_scene.preloaded_objects} "
            f"basket_offset={args.fruit_basket_offset:.3f}m"
        )
        if args.fruit_basket_offset:
            print(
                "  WARNING: shifted fruit positions are outside the training grid; "
                f"{args.fruit_basket_offset:.3f} m may also exceed the empirically "
                "validated arm-reach envelope"
            )

    # Hold every non-gripper actuator at the reset pose before advancing physics.
    # reset_fruit_grid_scene() already does this for fruit-grid runs, but the same
    # initialization is also required by the other RBY1 scenes.
    for i in range(m.nu):
        d.ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
    for side, actuator_name in (
        ("left", LEFT_ARM_ACTS[6]),
        ("right", RIGHT_ARM_ACTS[6]),
    ):
        actuator_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
        )
        d.ctrl[actuator_id] = teleop_arm6_targets[side]
    if args.model == "rby1_transport_14d":
        print(
            "  arm_6 hold : teleop keyframe "
            f"L/R={teleop_arm6_targets['left']:+.3f}/"
            f"{teleop_arm6_targets['right']:+.3f}"
        )

    randomized_episode = None
    if mcfg["obs_format"] == "rby1_16d":
        try:
            randomized_episode = load_randomized_episode(args.episode_index)
        except (FileNotFoundError, ValueError) as exc:
            ap.error(str(exc))
        reset_randomized_scene(m, d, randomized_episode)
        split = randomized_split_of(args.episode_index)
        print(
            f"  scene      : episode {args.episode_index} ({split}) "
            f"target={randomized_episode['target_fruit']} "
            f"arm={randomized_episode['used_arm']} "
            f"layout={randomized_episode['layout_index']}"
        )
        if split == "train":
            print(
                "  WARNING: this episode was in the training set; the model has seen "
                "it. Use --episode-index 1600-1999 for a held-out scene."
            )
        if args.prompt == DEFAULT_PROMPT:
            args.prompt = randomized_episode["prompt"]
            print(f"  prompt     : from episode -- {args.prompt!r}")

    # Match atomic-dataset startup exactly: its recorder starts only after both
    # grippers have been commanded to fraction=1.0 (ctrl=-0.045) and settled for
    # 0.5 s. Starting from the teleop keyframe's closed qpos=0.0 would put the
    # first policy state outside the training distribution.
    if mcfg["obs_format"] == "rby1":
        # rby1_16d is deliberately excluded: reset_randomized_scene() already set
        # both grippers to the dataset's open value as part of replaying the scene.
        reset_right_arm = right_arm_handles(m)
        reset_left_arm = left_arm_handles(m)
        open_grippers(
            m,
            d,
            [reset_right_arm, reset_left_arm],
            secs=0.5,
            opening=1.0,
        )
        left_open_fraction = float(
            abs(d.qpos[reset_left_arm.gripper_qidx]) / abs(RBY1_GRIPPER_OPEN)
        )
        right_open_fraction = float(
            abs(d.qpos[reset_right_arm.gripper_qidx]) / abs(RBY1_GRIPPER_OPEN)
        )
        print(
            f"  gripper init: ctrl={RBY1_GRIPPER_OPEN:.3f} "
            f"state(L/R)={left_open_fraction:.3f}/{right_open_fraction:.3f}"
        )

    # Without this, body/geom world transforms (xpos/xquat) are stale until the first
    # mj_step -- the very first observation (which drives the first action chunk) would
    # be rendered from a blank/garbage scene.
    mujoco.mj_forward(m, d)

    obstacle_manager = None
    if args.obstacle_profile != "clear":
        obstacle_manager = PickPlaceObstacleManager(m, d, obstacle_config)
        obstacle_manager.activate(args.obstacle_profile)
        initial_robot_clearance = obstacle_manager.robot_clearance()
        initial_object_clearance = obstacle_manager.object_clearance()
        if initial_robot_clearance <= args.obstacle_stop_distance:
            ap.error(
                f"obstacle profile starts inside the {args.obstacle_stop_distance:.3f} m "
                f"robot safety margin (clearance={initial_robot_clearance:.3f} m)"
            )
        if initial_object_clearance <= 0.0:
            ap.error(
                "obstacle profile overlaps a movable object at reset "
                f"(clearance={initial_object_clearance:.3f} m)"
            )
        print(
            f"  obstacles  : {args.obstacle_profile} {obstacle_manager.active_slots} "
            f"(initial robot/object clearance {initial_robot_clearance:.3f}/"
            f"{initial_object_clearance:.3f} m)"
        )

    # Build joint / actuator index maps once (used by build_obs and apply_action).
    idx = {
        "right_q":       [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
                          for j in RIGHT_ARM_JOINTS],
        "left_q":        [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
                          for j in LEFT_ARM_JOINTS],
        "right_grip_q":  m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_R_JOINT)],
        "left_grip_q":   m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_L_JOINT)],
    }
    act = {
        "right_a":       [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in RIGHT_ARM_ACTS],
        "left_a":        [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in LEFT_ARM_ACTS],
        "right_grip_a":  mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_R_ACT),
        "left_grip_a":   mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_L_ACT),
        "left_arm6_hold": teleop_arm6_targets["left"],
        "right_arm6_hold": teleop_arm6_targets["right"],
    }

    # RBY1 policy inputs must match EpisodeRecorder: render the physical camera at
    # 299x224 (4:3), disable reflections, then resize to the 224x224 policy tensor.
    # Direct 224x224 rendering changes the horizontal FOV and crops nearby objects.
    if mcfg["obs_format"] in ("rby1", "rby1_16d"):
        renderer_pol = mujoco.Renderer(
            m,
            height=POLICY_SOURCE_HEIGHT,
            width=POLICY_SOURCE_WIDTH,
        )
    else:
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

    safe_client = None
    if args.safe_remote:
        workspace_root = REPO_ROOT.parent
        if str(workspace_root) not in sys.path:
            sys.path.insert(0, str(workspace_root))
        from benchmark.ag3s.experiments.sources.mujoco_source import TransportScene
        from benchmark.trajopt import wire
        from benchmark.trajopt.client import SafeRemoteClient

        # `attach` 는 이 프로세스가 이미 돌리고 있는 `m`/`d` 를 그대로 가리킨다. 두 번째
        # 시뮬레이션이 아니다 — 두 벌이면 로봇이 있는 곳과 서버가 보는 곳이 갈라진다.
        safe_client = SafeRemoteClient(
            policy=policy,
            scene=TransportScene.attach(m, d),
            cameras=tuple(args.trajopt_cameras or wire.DEFAULT_CAMERAS),
            timeout_s=args.safe_timeout,
            phase=args.safe_phase,
            active_manipulators=tuple(args.safe_manipulators),
            trace_dir=args.trace,
        )
        print(f"[safe] server-side AG3S+TO; cameras={safe_client.cameras} "
              f"timeout={args.safe_timeout:.1f}s")

    frame_recorder = None
    if args.record_frames:
        workspace_root = REPO_ROOT.parent
        if str(workspace_root) not in sys.path:
            sys.path.insert(0, str(workspace_root))
        from benchmark.ag3s.runtime.frame_record import FrameRecorder, collect_manifest

        # manifest 는 **불변 설정**이다. 한 번 쓰고 프레임들이 참조한다 (T0).
        manifest = collect_manifest(
            seed=args.seed,
            scene={"model": args.model,
                   # MODELS 항목의 키는 `model_xml` 이다. 없는 항목도 있으므로 get 을 쓴다.
                   "model_xml": str(mcfg.get("model_xml") or ""),
                   "obs_format": mcfg["obs_format"],
                   "action_format": mcfg["action_format"],
                   "policy_config": mcfg.get("config"),
                   # 16D 는 씬을 기록된 에피소드에서 재생한다 — 그것이 seed 의 자리다.
                   "episode_index": getattr(args, "episode_index", None),
                   "episode_split": (randomized_split_of(args.episode_index)
                                     if mcfg["obs_format"] == "rby1_16d" else None),
                   "episode_target_fruit": (randomized_episode or {}).get("target_fruit"),
                   "episode_used_arm": (randomized_episode or {}).get("used_arm"),
                   "episode_prompt": (randomized_episode or {}).get("prompt"),
                   # 배치와 슬롯 순서는 **에피소드가 정한다** (`--fruit-*` 는 안 쓰인다).
                   # 이것이 없으면 "새 씬이었다" 를 기록만으로 되짚을 수 없다 — 아래
                   # `fruit_layout_index` 는 명령줄 인자이고 16D 경로에서는 언제나 None 이다.
                   "episode_layout_index": (randomized_episode or {}).get("layout_index"),
                   "episode_slot_order": list((randomized_episode or {}).get("slot_order")
                                              or ()),
                   "episode_non_target_fruits": list(
                       (randomized_episode or {}).get("non_target_fruits") or ()),
                   "episode_task_type": (randomized_episode or {}).get("task_type"),
                   "obstacle_profile": args.obstacle_profile,
                   "fruit_layout_index": args.fruit_layout_index,
                   "fruit_slot_order": list(args.fruit_slot_order or ()),
                   "fruit_preloaded": list(args.fruit_preloaded or ()),
                   "fruit_basket_offset": args.fruit_basket_offset,
                   "seed_derived": seed_derived,
                   "sim_timestep_s": float(m.opt.timestep)},
            policy={"prompt": args.prompt, "remote": args.remote,
                    "safe_remote": bool(args.safe_remote),
                    "safe_phase": args.safe_phase,
                    "safe_manipulators": list(args.safe_manipulators),
                    "safe_timeout_s": args.safe_timeout},
            # 거리장 설정은 **서버가** 들고 있다. 여기서는 클라이언트가 아는 것만 적고,
            # 서버가 실제로 무엇을 썼는지는 프레임마다 응답의 `field.backend` 가 답한다 —
            # 그쪽이 T0 의 "실제로 사용된 backend provenance" 다.
            esdf={"owner": "server (serve_safe)",
                  "reported_per_frame_in": "field.backend"},
            timing={"ctrl_hz": CTRL_HZ, "open_loop_horizon": OPEN_LOOP_HORIZON,
                    "chunk_period_ms": OPEN_LOOP_HORIZON / CTRL_HZ * 1000.0,
                    "sim_steps_per_action": steps_per_action,
                    "speed": args.speed,
                    # 한도는 서버 설정이다. 클라이언트가 모르면 stale 판정을 못 한다.
                    "max_field_age_sec": None},
            cameras=tuple(args.trajopt_cameras or ("zed_left", "wrist_cam_l",
                                                   "wrist_cam_r")),
            render={"third_person_video": args.record,
                    "recording_camera": args.view,
                    "note": "third-person 프레임은 --record 가 있을 때만 남는다"},
            extra={"argv": sys.argv},
        )
        # 관측 프레임은 **정책 호출당 한 번**이다 — 제어 스텝당 한 번이 아니다. 카메라
        # 캡처와 AG3S 한 바퀴는 아래 `chunk_step >= OPEN_LOOP_HORIZON` 분기 안에서만
        # 일어나고, 그 사이 7 스텝은 이미 받은 청크를 그대로 흘려보낸다. 처음에는 이
        # 기대값을 `max_steps` 로 잡아 10 을 기대했는데, 그러면 실제로 일어나지 않은
        # 관측 8 개가 영구히 "누락" 으로 남아 completeness 가 절대 닫히지 않는다.
        # 기대와 실제가 같은가를 세는 표에서 **기대 쪽이 틀린 경우**다.
        n_policy_calls = (0 if args.max_steps <= 0
                          else -(-args.max_steps // OPEN_LOOP_HORIZON))
        frame_recorder = FrameRecorder(
            args.record_frames, manifest=manifest,
            expected_observation_frames=n_policy_calls)
        print(f"[frames] manifest + frames.jsonl -> {args.record_frames}")

    live_pipeline = None
    if args.trajopt:
        # 벤치마크 패키지는 src/ 한 단계 위에 있다 (`--record-ag3s` 와 같은 규칙).
        workspace_root = REPO_ROOT.parent
        if str(workspace_root) not in sys.path:
            sys.path.insert(0, str(workspace_root))
        from benchmark.ag3s.experiments.sources.mujoco_source import TransportScene, camera_observation
        from benchmark.ag3s.experiments.reports.grounding_report import build_robot_model
        from benchmark.ag3s.runtime.trace import RunTrace
        from benchmark.trajopt.bringup import build_live_pipeline

        trace = RunTrace(args.trace, enabled=bool(args.trace), meta={
            "prompt": args.prompt, "policy_model": args.model, "remote": args.remote,
            "ctrl_hz": CTRL_HZ, "open_loop_horizon": OPEN_LOOP_HORIZON,
            "chunk_period_ms": OPEN_LOOP_HORIZON / CTRL_HZ * 1000.0,
            "trajopt_links": args.trajopt_links,
        })
        constraint_recorder = None
        if args.record_constraints:
            from benchmark.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter

            constraint_recorder = ConstraintRecordWriter(
                args.record_constraints, esdf_mode=args.record_constraints_esdf,
                meta={"prompt": args.prompt, "trajopt_links": args.trajopt_links},
            )
        # AG3S 는 MuJoCo 씬 객체를 통해 카메라를 읽는다. 여기서 만드는 것은 이 프로세스가 이미
        # 들고 있는 `m`/`d` 를 그대로 가리키는 얇은 뷰이지 두 번째 시뮬레이션이 아니다 —
        # 두 벌이면 제약이 설명하는 씬과 로봇이 움직이는 씬이 갈라진다.
        ag3s_scene = TransportScene.attach(m, d)
        ag3s_robot = build_robot_model(ag3s_scene)
        ag3s_cameras = tuple(args.trajopt_cameras or ("zed_left", "wrist_cam_l", "wrist_cam_r"))
        #: 정책 응답이 실어 보내는 attention. 청크마다 갱신된다 (서버 수정 후).
        policy_attention: dict[str, "np.ndarray"] = {}

        # attention 은 정책 응답에서 온다. 서빙이 아직 그것을 싣지 않으면 None 이고, 그러면
        # AG3S 는 target 을 못 잡는다(`no_target`). 그 경우 거리장에서 target 복셀을 파내지
        # 않으므로 제약이 **더 보수적**으로 동작한다 — 안전한 방향의 실패다. 합성 attention 을
        # 몰래 끼워넣지 않는 이유는, 그러면 이 실행이 실측 파이프라인인 척하게 되기 때문이다.
        attention_state = {"warned": False}

        def _capture():
            out = []
            for camera in ag3s_cameras:
                amap = policy_attention.get(camera)
                if amap is None and not attention_state["warned"]:
                    print("[trajopt] 정책 응답에 attention 이 없다 — target 없이 진행한다 "
                          "(거리장이 target 을 파내지 않아 더 보수적). 서빙이 선택 셀을 "
                          "반환하도록 고치면 사라진다")
                    attention_state["warned"] = True
                observation, _frame = camera_observation(
                    ag3s_scene, camera, ag3s_robot, timestamp=time.monotonic(),
                    attention_map=amap,
                )
                out.append(observation)
            return out

        live_pipeline = build_live_pipeline(
            mj_model=m, mj_data=d, capture_fn=_capture,
            state_fn=lambda: ag3s_scene.robot_state(),
            constraint_links=None if args.trajopt_links == "all" else "arms",
            trace=trace, recorder=constraint_recorder,
        )
        print(f"[trajopt] AG3S+TO on; cameras={ag3s_cameras} links={args.trajopt_links} "
              f"constraint spheres={live_pipeline.linearizer.n_spheres} "
              f"chunk budget={OPEN_LOOP_HORIZON / CTRL_HZ * 1000:.0f} ms")

    ag3s_recorder = None
    if args.record_ag3s:
        # The benchmark packages live one level above src/, which is REPO_ROOT here.
        workspace_root = REPO_ROOT.parent
        if str(workspace_root) not in sys.path:
            sys.path.insert(0, str(workspace_root))
        from benchmark.ag3s.experiments.sources.policy_record import PolicyRecordWriter
        depth_cameras = ()
        if args.record_depth is not None:
            depth_cameras = tuple(args.record_depth) or (
                "zed_left", "wrist_cam_l", "wrist_cam_r"
            )
        ag3s_recorder = PolicyRecordWriter(
            args.record_ag3s,
            depth_cameras=depth_cameras,
            model=m,
            model_xml=mcfg.get("model_xml", MODEL_XML),
            prompt=args.prompt,
            extra={
                "policy_model": args.model,
                "remote": args.remote,
                "ctrl_hz": CTRL_HZ,
                "open_loop_horizon": OPEN_LOOP_HORIZON,
                "obstacle_profile": args.obstacle_profile,
                "fruit_layout_index": args.fruit_layout_index,
                "fruit_slot_order": args.fruit_slot_order,
                "argv": sys.argv,
            },
        )
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

    #: 팔당 관절 수. 14-D 는 6 (arm_6 를 고정으로 두고), 16-D 는 7 (arm_6 까지 지령).
    #: **차원에서 파생시킨다** — `[:6]` 을 박아 두면 16-D hold 경로가 14-D 를 내고
    #: `apply_action` 이 거절한다 (2026-09-24 실측: T0 첫 청크가 그렇게 죽었다).
    ARM_JOINT_DIM = 7 if mcfg["action_format"] == "rby1_16d" else 6

    def rby1_state():
        """Current physical state in the policy/action layout.

        레이아웃은 `[왼팔 N, 왼 그리퍼, 오른팔 N, 오른 그리퍼]` 이고 `N = ARM_JOINT_DIM` 이다
        (`build_obs` 의 `rby1_16d` 분기와 같은 순서). hold 청크의 모든 행이 이 값이 되므로
        차원이 틀리면 안전 판정이 아니라 예외로 죽는다.
        """
        n = ARM_JOINT_DIM
        left = np.asarray([d.qpos[i] for i in idx["left_q"][:n]], dtype=np.float64)
        right = np.asarray([d.qpos[i] for i in idx["right_q"][:n]], dtype=np.float64)
        left_grip = float(abs(d.qpos[idx["left_grip_q"]]) / abs(RBY1_GRIPPER_OPEN))
        right_grip = float(abs(d.qpos[idx["right_grip_q"]]) / abs(RBY1_GRIPPER_OPEN))
        out = np.concatenate([left, [left_grip], right, [right_grip]])
        expected = 2 * (n + 1)
        if out.shape != (expected,):
            raise ValueError(
                f"rby1_state 가 {out.shape} 를 냈는데 action_format "
                f"{mcfg['action_format']!r} 은 ({expected},) 를 기대합니다")
        return out

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

    def loop_body():
        chunk = None
        chunk_step = 0
        previous_physical_chunk = None
        # Wall-clock anchor for the next action step; reset after each inference so a
        # slow inference call is not "paid back" by sprinting the following steps.
        next_action_deadline = time.monotonic()
        for t_step in range(0, args.max_steps if args.max_steps > 0 else 10**9):
            if chunk is None or chunk_step >= OPEN_LOOP_HORIZON:
                obs = build_obs(mcfg["obs_format"], m, d, renderer_pol, idx, args.prompt)
                if mcfg["obs_format"] == "rby1":
                    validate_rby1_observation(obs, log=(t_step == 0))
                if args.record_inputs:
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
                    if safe_client is not None:
                        # 서버가 π0.5+SEAM+AG3S+TO 를 다 돌린다. 여기서 받는 것은 청크와
                        # **안전 판정**이고, 실행 여부는 아래에서 로컬이 정한다.
                        result = safe_client.infer(obs, reset=(t_step == 0))
                    else:
                        result = policy.infer(obs)
                    chunk = np.asarray(result["actions"])
                    seam_timing = result.get("seam_timing", {})
                    if live_pipeline is not None:
                        # 서빙이 1단계에서 확정한 (층·헤드·agg·denoise) 셀 하나만 실어 보낸다.
                        # 전체 텐서가 아니라 카메라당 맵 한 장이라 수십 KB다. 키는 MuJoCo 카메라
                        # 이름이어야 `_capture` 가 찾는다.
                        policy_attention.clear()
                        policy_attention.update(result.get("attention", {}) or {})
                if mcfg["action_format"] == "rby1":
                    validate_rby1_action_chunk(chunk, log=(t_step == 0))
                infer_elapsed_ms = (time.time() - t_infer) * 1000.0
                chunk_step = 0

                if frame_recorder is not None:
                    # observation frame = 카메라 캡처 + AG3S 한 바퀴. planning 과 1:1 이지만
                    # **재는 것이 다르다**: 이쪽은 촬영 시각과 지각 판정, 저쪽은 SQP 결과다.
                    #
                    # 촬영 시각은 클라이언트가 실제로 찍은 값(`last_stamps`)이다. 응답의
                    # `observed_at` 은 서버가 고른 가장 최근 한 개뿐이라 카메라 간 시차를
                    # 복원할 수 없다 — 세 대를 순차로 렌더하므로 그 시차가 손목 클라우드의
                    # 번짐으로 그대로 나타난다.
                    #
                    # grounding 상태와 점 개수는 **서버 안에만 있다** (응답에는 안전 판정과
                    # 카메라당 attention 셀 하나뿐이다). 모르는 것을 지어내지 않고
                    # `unavailable-on-client` 로 적는다 — 이 둘을 프레임 카드에 실으려면
                    # 와이어에 필드를 늘려야 하고, 그것은 T1(프레임별 진단 카드)의 일이다.
                    _obs_stamps = (dict(safe_client.last_stamps)
                                   if safe_client is not None else {})
                    _obs_status = (str(safe_client.last_verdict.get("ag3s_status") or "")
                                   if safe_client is not None else "in-process")
                    frame_recorder.observation(
                        t_step=t_step, stamps=_obs_stamps,
                        ag3s_status=_obs_status,
                        grounding_status="unavailable-on-client",
                        validity=("certified" if (safe_client is not None
                                                  and safe_client.last_verdict.get(
                                                      "geometry_certified"))
                                  else "not-certified"),
                        n_points=None,
                        notes=(["grounding 상태·점 개수는 서버 안에만 있다 (T1 에서 와이어에 싣는다)"]
                               if safe_client is not None else
                               ["in-process 경로: 카메라 스탬프가 클라이언트에 없다"]),
                        extra={"cameras": sorted(_obs_stamps),
                               "ipc": (safe_client.last_ipc if safe_client is not None
                                       else "in-process")})

                if frame_recorder is not None:
                    # planning frame = 청크 하나. `field` 는 응답이 실어 온 출처이고,
                    # `ipc` 는 왕복 결과(`ok`/`timeout`/`stale`/`unsafe`/`error`)다 —
                    # 프롬프트 T0 이 프레임마다 요구하는 IPC 기록이 이 둘이다.
                    frame_recorder.planning(
                        seq=(safe_client._seq if safe_client is not None else t_step),
                        t_step=t_step,
                        field=(safe_client.last_field if safe_client is not None else None),
                        verdict=(dict(safe_client.last_verdict)
                                 if safe_client is not None else {}),
                        timing_ms={"policy_infer": infer_elapsed_ms},
                        ipc=(safe_client.last_ipc if safe_client is not None
                             else "in-process"),
                        extra={"safe": (bool(safe_client.last_safe)
                                        if safe_client is not None else None),
                               "hold_reason": (safe_client.last_reason
                                               if safe_client is not None else ""),
                               # **실행된 청크의 실제 폭.** manifest 의 `action_format` 은
                               # 설정 문자열이고, 이것은 그 프레임에 정말 무엇이 왔는지다.
                               # 14D→16D 전환에서 바뀐 값이므로 프레임마다 남긴다.
                               "chunk_shape": list(np.asarray(chunk).shape)})

                if live_pipeline is not None:
                    # closed loop: TO 가 고친 청크를 **실제로 실행한다**. 실패해도 예외를 내지
                    # 않고 정책 청크를 그대로 돌려주는 것이 refiner 의 계약이라, 지각 한 프레임이
                    # 빠져도 제어는 멈추지 않는다.
                    policy_chunk = chunk
                    chunk = np.asarray(
                        live_pipeline.refine(chunk, t_step=t_step,
                                             previous_chunk=previous_physical_chunk),
                        dtype=chunk.dtype,
                    )
                    previous_physical_chunk = chunk
                    to_result = live_pipeline.refiner.last_result
                    if to_result is not None:
                        print(f"[trajopt] t={t_step} {to_result.status.value} "
                              f"iters={to_result.iterations} "
                              f"delta={np.abs(chunk - policy_chunk).max():.4f} rad")

                if ag3s_recorder is not None:
                    ag3s_recorder.record(
                        t_step=t_step, obs=obs, data=d, chunk=chunk,
                        infer_ms=infer_elapsed_ms, model=m,
                    )

                if args.trajectory_out:
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

            if safe_client is not None and not safe_client.last_safe:
                # hold: 현재 관절을 그대로 목표로 준다. 정지가 아니라 **유지**다 — 제어를
                # 끊으면 팔이 중력으로 떨어지고, 그것은 안전 판정이 막으려던 것보다 나쁘다.
                # 앞 `execution_length` 개만 실행한다는 규칙도 여기서 함께 지켜진다:
                # 안전하지 않은 청크는 한 스텝도 실행되지 않는다.
                action = rby1_state()
                if chunk_step == 0:
                    print(f"[safe] t={t_step} HOLD — {safe_client.last_reason}")
            else:
                action = np.asarray(chunk[chunk_step], dtype=np.float64)
            apply_action(mcfg["action_format"], action, d, idx, act)

            if frame_recorder is not None:
                # control frame = 개별 제어 스텝. **여기서 `carried`/`stale` 이 생긴다** —
                # `step_in_chunk > 0` 이면 이 스텝의 기하는 이번 프레임에 갱신된 것이 아니다.
                frame_recorder.control(
                    t_step=t_step, chunk_seq=(safe_client._seq
                                              if safe_client is not None else t_step),
                    step_in_chunk=chunk_step, now=time.monotonic(),
                    field=(safe_client.last_field if safe_client is not None else None),
                    executed=bool(safe_client is None or safe_client.last_safe))

            obstacle_stopped = False
            for sim_step in range(steps_per_action):
                mujoco.mj_step(m, d)
                if obstacle_manager is not None:
                    obstacle_manager.observe_contacts()
                    if sim_step % 5 == 0:
                        clearance = obstacle_manager.robot_clearance()
                        if (
                            obstacle_manager.robot_collision
                            or clearance <= args.obstacle_stop_distance
                        ):
                            obstacle_manager.hold_robot()
                            obstacle_stopped = True
                            break
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
            if args.record:
                renderer_rec.update_scene(d, camera=recording_camera)
                video_frames.append(renderer_rec.render())
            if args.trajectory_out:
                executed_actions.append(action.copy())
                measured_states.append(rby1_state())

            if obstacle_stopped:
                print(
                    "[SAFETY] obstacle stop: "
                    f"clearance={obstacle_manager.min_robot_clearance:.4f} m "
                    f"collision={obstacle_manager.robot_collision}"
                )
                break

            chunk_step += 1
            if ctx is not None and not ctx.is_running():
                break

    try:
        if wait_before_inference():
            loop_body()
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        if obstacle_manager is not None:
            print(f"obstacle safety summary: {obstacle_manager.summary()}")
        if ctx is not None:
            ctx.close()
        if args.record and video_frames:
            print(f"saving {len(video_frames)} frames -> {args.record}")
            try:
                import imageio
                imageio.mimsave(args.record, video_frames, fps=CTRL_HZ)
            except ImportError:
                for i, fr in enumerate(video_frames):
                    Image.fromarray(fr).save(args.record.replace(".mp4", f"_{i:04d}.png"))
                print("(imageio not installed, saved as PNG sequence)")
        if args.record_inputs:
            save_policy_input_videos(args.record_inputs, input_frame_buffers)
        if frame_recorder is not None:
            table = frame_recorder.close(
                ipc_stats=(dict(safe_client.stats) if safe_client is not None else None))
            print("[frames] completeness:")
            for k, v in table.items():
                print(f"    {k}: {v}")
        if safe_client is not None:
            safe_client.close()
        if live_pipeline is not None:
            live_pipeline.close()
        if ag3s_recorder is not None:
            ag3s_recorder.close()
        if args.trajectory_out and executed_actions:
            trajectory_path = pathlib.Path(args.trajectory_out)
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            condition = "seam" if args.seam else "baseline"
            if args.obstacle_profile != "clear":
                condition += f"__obstacle_{args.obstacle_profile}"
            obstacle_summary = obstacle_manager.summary() if obstacle_manager else {}
            np.savez_compressed(
                trajectory_path,
                executed_actions=np.asarray(executed_actions, dtype=np.float64),
                measured_qpos=np.asarray(measured_states, dtype=np.float64),
                predicted_chunks=np.asarray(predicted_chunks, dtype=np.float64),
                chunk_start_steps=np.asarray(chunk_start_steps, dtype=np.int64),
                inference_states=np.asarray(inference_states, dtype=np.float64),
                inference_ms=np.asarray(inference_ms, dtype=np.float64),
                used_vls=np.asarray(inference_used_vls, dtype=bool),
                chunk_indices=np.asarray(inference_chunk_indices, dtype=np.int64),
                prompt=np.asarray(args.prompt),
                condition=np.asarray(condition),
                obstacle_profile=np.asarray(args.obstacle_profile),
                obstacle_safety_json=np.asarray(json.dumps(obstacle_summary)),
                fruit_layout_index=np.asarray(
                    fruit_scene.layout_index if fruit_scene is not None else -1,
                    dtype=np.int64,
                ),
                fruit_slot_order=np.asarray(
                    fruit_scene.slot_order if fruit_scene is not None else (),
                ),
                fruit_preloaded=np.asarray(
                    fruit_scene.preloaded_objects if fruit_scene is not None else (),
                ),
                fruit_basket_offset_m=np.asarray(
                    args.fruit_basket_offset,
                    dtype=np.float64,
                ),
                control_hz=np.asarray(CTRL_HZ, dtype=np.int64),
                execution_length=np.asarray(OPEN_LOOP_HORIZON, dtype=np.int64),
            )
            print(f"saving trajectory ({len(executed_actions)} executed steps) -> {trajectory_path}")


if __name__ == "__main__":
    main()
