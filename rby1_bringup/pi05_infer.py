"""RBY1 x pi0.5 inference smoke test — DROID / LIBERO / base / aloha / rby1 variants.

Loads chosen pi0.5 checkpoint, feeds cameras + proprio, integrates predicted
actions into position ctrl on RBY1.

Model variants and required cameras/state:
  - droid  : zed_left (+ zero wrist)             , right arm only (7+1)
  - libero : zed_left (+ zero wrist)             , right arm only (7+1)
  - base   : zed_left (+ zero wrist)             , right arm only (7+1) — uses droid transform
  - aloha  : zed_left + wrist_cam_l + wrist_cam_r, BOTH arms (6+1)+(6+1) = 14
             * for zero-shot embodiment test only; do not use "aloha" naming later
             * pi0_aloha_pen_uncap ckpt (dual-arm 7-DoF ViperX; last joint dropped for us)
  - rby1   : zed_left + wrist_cam_l + wrist_cam_r, BOTH arms (6+1)+(6+1) = 14
             * our own pi05_rby1_lora LoRA fine-tune (see checkpoints/pi05_rby1_lora/)
             * actions are already absolute joint targets (server-side AbsoluteActions
               transform undoes the delta-joint training representation), so applied
               directly to ctrl -- no local delta/velocity math needed

Two ways to run:

1. All-local (on a machine with JAX + the openpi package + a GPU to host
   the model — e.g. running directly on the GPU server, headless). Uses
   whatever JAX backend is available (GPU by default if present; pass
   JAX_PLATFORMS=cpu yourself to force CPU). On this container's (virtualized,
   quota-limited) GPU, JAX's default memory preallocation + CUDA graph capture
   are unstable (OOM-retry storms, then `CUDA_ERROR_INVALID_VALUE` on the 2nd+
   inference call) — the env vars below route around it; harmless on a normal GPU:
     XLA_FLAGS="--xla_gpu_enable_command_buffer=" XLA_PYTHON_CLIENT_PREALLOCATE=false \
         python pi05_infer.py --model droid
     python pi05_infer.py --model aloha --headless \
         --max-steps 60 --record /tmp/rby1_aloha.mp4

2. Split mode — MuJoCo (sim + interactive viewer) on your local PC, model
   inference on a remote GPU server. On the server, start the policy once
   (runs on GPU automatically if JAX sees one; add JAX_PLATFORMS=cpu to force CPU;
   same XLA env vars as above needed on this container's virtualized GPU):
     XLA_FLAGS="--xla_gpu_enable_command_buffer=" XLA_PYTHON_CLIENT_PREALLOCATE=false \
         python scripts/serve_policy.py --env DROID --port 8000        # droid/libero via --env
     XLA_FLAGS="--xla_gpu_enable_command_buffer=" XLA_PYTHON_CLIENT_PREALLOCATE=false \
         python scripts/serve_policy.py --port 8000 policy:checkpoint \
         --policy.config pi05_droid \
         --policy.dir gs://openpi-assets/checkpoints/pi05_base          # for "base"
   Then on the local PC (only needs mujoco, pillow, numpy, openpi_client —
   no jax/openpi required):
     python pi05_infer.py --model droid --remote <server-ip>:8000

Notes on interpretation
-----------------------
- DROID  actions: (chunk, 8) = 7 joint velocities (clip [-1,1]) + 1 gripper position [0,1].
- LIBERO actions: (chunk, 7) = 6 EE delta + 1 gripper. Applied naively as joint delta.
- ALOHA  actions: (chunk, 14) = [L 6 joint, L gripper, R 6 joint, R gripper]. Applied as
                  joint delta on joints[:6] of each arm (arm_6 held fixed).
                  Note: pi0_aloha_pen_uncap is trained for ViperX, so raw scale is off.
- RBY1   actions: (chunk, 14) = [L 6 joint, L gripper, R 6 joint, R gripper], same layout
                  as ALOHA but values are absolute joint-position targets (radians) /
                  gripper in [0,1] -- applied directly to ctrl, matching how the training
                  dataset's "action" column was recorded (arm_6 held fixed, dropped from
                  training data same as ALOHA).
- base model has no task-specific norm stats; outputs are expected to be low-quality zero-shot.
"""
import argparse
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


def render_cam(model, data, renderer, cam_name, size=224):
    renderer.update_scene(data, camera=cam_name)
    img = renderer.render()
    im = Image.fromarray(img).resize((size, size), Image.BILINEAR)
    return np.asarray(im)


def _hwc_to_chw(img):
    """Convert (H, W, C) uint8 image to (C, H, W) as expected by AlohaInputs."""
    return np.transpose(np.asarray(img), (2, 0, 1))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS.keys()), default="droid",
                    help="which pi0.5 variant to load")
    ap.add_argument("--prompt", default="put the red block in the brown box with your right hand",
                    help="for --model rby1, use one of the 6 exact task strings the model "
                         "was fine-tuned on (see rby1_dataset_v1/meta/tasks.jsonl)")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--record", default=None, help="path to .mp4 for third-person recording")
    ap.add_argument("--remote", default=None,
                    help="host:port of a running scripts/serve_policy.py server; if set, "
                         "skips local model load and streams obs/actions over websocket instead")
    ap.add_argument("--start-delay", type=float, default=2.0,
                    help="seconds to run the simulator and show the viewer before the first "
                         "policy inference request (default: 2.0; use 0 to disable)")
    args = ap.parse_args()

    if args.start_delay < 0:
        ap.error("--start-delay must be non-negative")

    mcfg = MODELS[args.model]
    print(f"=== Model: {args.model} ===")
    if args.remote:
        print(f"  remote     : {args.remote}  (server must serve obs_format={mcfg['obs_format']!r})")
    else:
        print(f"  config     : {mcfg['config']}")
        print(f"  checkpoint : {mcfg['checkpoint']}")
    print(f"  obs_format : {mcfg['obs_format']}")
    print(f"  act_format : {mcfg['action_format']}")

    m = mujoco.MjModel.from_xml_path(MODEL_XML)
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
    }
    act = {
        "right_a":       [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in RIGHT_ARM_ACTS],
        "left_a":        [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in LEFT_ARM_ACTS],
        "right_grip_a":  mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_R_ACT),
        "left_grip_a":   mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_L_ACT),
    }

    # Policy-input renderer MUST match the training-data collection resolution exactly
    # (rby1_manipulation/scenario*.py, collect_batch.py render at native 224x224).
    # Rendering at a different aspect ratio (e.g. 640x480) and resizing down distorts
    # the field of view relative to what the model was trained on.
    renderer_pol = mujoco.Renderer(m, height=224, width=224)
    renderer_rec = mujoco.Renderer(m, height=480, width=640)  # third-person --record only

    policy = load_remote_policy(args.remote) if args.remote else load_local_policy(mcfg)
    print(f"Prompt: {args.prompt!r}")

    steps_per_action = max(1, int(round(1.0 / (CTRL_HZ * m.opt.timestep))))
    print(f"sim dt={m.opt.timestep}s  action interval={1/CTRL_HZ:.3f}s  sim-steps/action={steps_per_action}")

    video_frames = []
    ctx = None if args.headless else mujoco.viewer.launch_passive(m, d)

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

    def loop_body():
        chunk = None
        chunk_step = 0
        for t_step in range(0, args.max_steps if args.max_steps > 0 else 10**9):
            if chunk is None or chunk_step >= OPEN_LOOP_HORIZON:
                obs = build_obs(mcfg["obs_format"], m, d, renderer_pol, idx, args.prompt)
                t_infer = time.time()
                result = policy.infer(obs)
                chunk = np.asarray(result["actions"])
                chunk_step = 0

                # brief status line — content varies by embodiment
                if mcfg["obs_format"] in ("aloha", "rby1"):
                    l_pos = np.array([d.qpos[i] for i in idx["left_q"]], dtype=np.float64)
                    r_pos = np.array([d.qpos[i] for i in idx["right_q"]], dtype=np.float64)
                    print(f"[t={t_step:4d}] infer={time.time()-t_infer:.2f}s chunk={chunk.shape}  "
                          f"q_l={l_pos[:3].round(2).tolist()}...  q_r={r_pos[:3].round(2).tolist()}...")
                else:
                    r_pos = np.array([d.qpos[i] for i in idx["right_q"]], dtype=np.float64)
                    print(f"[t={t_step:4d}] infer={time.time()-t_infer:.2f}s chunk={chunk.shape}  "
                          f"q_r={r_pos.round(2).tolist()}")

            apply_action(mcfg["action_format"], chunk[chunk_step], d, idx, act)

            for _ in range(steps_per_action):
                mujoco.mj_step(m, d)
            if ctx is not None:
                ctx.sync()
            if args.record:
                renderer_rec.update_scene(d, camera=-1)
                video_frames.append(renderer_rec.render())

            chunk_step += 1
            if ctx is not None and not ctx.is_running():
                break

    try:
        if wait_before_inference():
            loop_body()
    except KeyboardInterrupt:
        print("interrupted")
    finally:
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


if __name__ == "__main__":
    main()
