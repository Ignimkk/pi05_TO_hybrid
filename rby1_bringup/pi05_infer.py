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
from rby1_manipulation.simulation.transport_scene import load_layout_config

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
    ap.add_argument("--prompt", default="put the red block in the brown box with your right hand",
                    help="for --model rby1, use one of the 6 exact task strings the model "
                         "was fine-tuned on (see rby1_dataset_v1/meta/tasks.jsonl)")
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
        "fruit" if args.model == "rby1_transport_14d" else "block"
    )
    profile_scene = obstacle_config["profiles"][args.obstacle_profile]["scene"]
    if profile_scene not in ("any", expected_obstacle_scene):
        ap.error(
            f"--obstacle-profile {args.obstacle_profile!r} is for {profile_scene} scene; "
            f"--model {args.model} uses {expected_obstacle_scene} scene"
        )
    if args.obstacle_profile != "clear" and args.model not in (
        "rby1", "rby1_transport_14d"
    ):
        ap.error(
            "static pick-place obstacles require --model rby1 or rby1_transport_14d"
        )
    if args.checkpoint:
        mcfg = dict(mcfg, checkpoint=args.checkpoint)
    if not args.remote and not mcfg.get("checkpoint"):
        ap.error(f"--model {args.model} requires --remote or --checkpoint")
    if args.record_inputs and mcfg["obs_format"] not in ("aloha", "rby1"):
        ap.error("--record-inputs requires --model aloha or --model rby1")
    if args.trajectory_out and mcfg["obs_format"] != "rby1":
        ap.error("--trajectory-out currently requires --model rby1")
    if args.record_ag3s and mcfg["obs_format"] != "rby1":
        ap.error("--record-ag3s requires --model rby1 (AG3S is wired to the RB-Y1 cameras)")
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

    # Match atomic-dataset startup exactly: its recorder starts only after both
    # grippers have been commanded to fraction=1.0 (ctrl=-0.045) and settled for
    # 0.5 s. Starting from the teleop keyframe's closed qpos=0.0 would put the
    # first policy state outside the training distribution.
    if mcfg["obs_format"] == "rby1":
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
    if mcfg["obs_format"] == "rby1":
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

    ag3s_recorder = None
    if args.record_ag3s:
        # The benchmark packages live one level above src/, which is REPO_ROOT here.
        workspace_root = REPO_ROOT.parent
        if str(workspace_root) not in sys.path:
            sys.path.insert(0, str(workspace_root))
        from benchmark.ag3s.experiments.policy_record import PolicyRecordWriter
        ag3s_recorder = PolicyRecordWriter(
            args.record_ag3s,
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

    def loop_body():
        chunk = None
        chunk_step = 0
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
                    result = policy.infer(obs)
                    chunk = np.asarray(result["actions"])
                    seam_timing = result.get("seam_timing", {})
                if mcfg["action_format"] == "rby1":
                    validate_rby1_action_chunk(chunk, log=(t_step == 0))
                infer_elapsed_ms = (time.time() - t_infer) * 1000.0
                chunk_step = 0

                if ag3s_recorder is not None:
                    ag3s_recorder.record(
                        t_step=t_step, obs=obs, data=d, chunk=chunk, infer_ms=infer_elapsed_ms
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

            action = np.asarray(chunk[chunk_step], dtype=np.float64)
            apply_action(mcfg["action_format"], action, d, idx, act)

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
