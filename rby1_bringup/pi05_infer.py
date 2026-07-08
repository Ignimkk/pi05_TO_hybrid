"""RBY1 x pi0.5 inference smoke test — DROID / LIBERO / base variants.

Loads chosen pi0.5 checkpoint, feeds ZED-left image + right-arm proprio,
integrates predicted actions into position ctrl on RBY1's right arm + gripper.

Usage:
  # DROID variant (default) — 7 joint vel + 1 gripper actions
  JAX_PLATFORMS=cpu python pi05_infer.py --model droid

  # LIBERO variant — 6 EE delta + 1 gripper (mapped naively to joints for pipeline test)
  JAX_PLATFORMS=cpu python pi05_infer.py --model libero

  # base model (Fine-Tuning checkpoint) — reuses DROID transform, norm stats may be off
  JAX_PLATFORMS=cpu python pi05_infer.py --model base

  # Headless with recording
  JAX_PLATFORMS=cpu python pi05_infer.py --model droid --headless \
      --max-steps 60 --record /tmp/rby1_pi05_droid.mp4

Notes on interpretation
-----------------------
- DROID  actions: (chunk, 8) = 7 joint velocities (clip [-1,1]) + 1 gripper position [0,1].
- LIBERO actions: (chunk, 7) = 6 EE delta + 1 gripper. We do NOT solve IK for LIBERO here —
  the first 6 dims are applied as joint delta signals for pipeline validation only.
- base model has no task-specific norm stats; outputs are expected to be low-quality zero-shot.
"""
import argparse
import os
import time
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import mujoco
import mujoco.viewer
from PIL import Image

from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi.shared import download

MODEL_XML  = "/home/mk/dev_ws/vla/pi0_TO_ws/src/rby1_description/models/rby1a/mujoco/model.xml"

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
        "action_format": "libero",      # 6 dims applied to joints[:6] + 1 gripper
    },
    "base": {
        "config": "pi05_droid",         # reuse DROID transform (base has no dedicated inference config)
        "checkpoint": "gs://openpi-assets/checkpoints/pi05_base",
        "obs_format": "droid",
        "action_format": "droid",
    },
}

RIGHT_ARM_JOINTS = [f"right_arm_{i}" for i in range(7)]
RIGHT_ARM_ACTS   = [f"right_arm_{i+1}_act" for i in range(7)]
GRIPPER_JOINT = "gripper_finger_r1"
GRIPPER_ACT   = "gripper_r_act"
GRIPPER_OPEN, GRIPPER_CLOSED = -0.05, 0.0

CTRL_HZ = 15
OPEN_LOOP_HORIZON = 8


def render_cam(model, data, renderer, cam_name, size=224):
    renderer.update_scene(data, camera=cam_name)
    img = renderer.render()
    im = Image.fromarray(img).resize((size, size), Image.BILINEAR)
    return np.asarray(im)


def build_obs(obs_format, base_img, wrist_img, joint_pos, grip_norm, prompt):
    if obs_format == "droid":
        return {
            "observation/exterior_image_1_left": base_img,
            "observation/wrist_image_left": wrist_img,
            "observation/joint_position": joint_pos,
            "observation/gripper_position": np.array([grip_norm], dtype=np.float64),
            "prompt": prompt,
        }
    if obs_format == "libero":
        state = np.concatenate([joint_pos, np.array([grip_norm], dtype=np.float64)])
        return {
            "observation/state": state,
            "observation/image": base_img,
            "observation/wrist_image": wrist_img,
            "prompt": prompt,
        }
    raise ValueError(f"unknown obs_format: {obs_format}")


def apply_action(action_format, action, d, right_qidx, right_aid, grip_aid):
    """Set d.ctrl in-place based on model action."""
    dt = 1.0 / CTRL_HZ
    if action_format == "droid":
        # 7 joint velocities + 1 gripper position
        joint_vel = np.clip(action[:7], -1.0, 1.0)
        grip_action = 1.0 if float(action[7]) > 0.5 else 0.0
        for i, aid in enumerate(right_aid):
            d.ctrl[aid] = d.qpos[right_qidx[i]] + joint_vel[i] * dt
        d.ctrl[grip_aid] = GRIPPER_OPEN * (1.0 - grip_action)
        return

    if action_format == "libero":
        # 6 EE delta (nonsense on our joints, applied as delta) + 1 gripper
        # First 6 dims -> joint delta on joints[:6]; joint 7 held fixed.
        joint_delta = np.clip(action[:6], -0.5, 0.5) * 0.1
        grip_action = 1.0 if float(action[6]) > 0.5 else 0.0
        for i in range(6):
            d.ctrl[right_aid[i]] = d.qpos[right_qidx[i]] + joint_delta[i]
        d.ctrl[grip_aid] = GRIPPER_OPEN * (1.0 - grip_action)
        return

    raise ValueError(f"unknown action_format: {action_format}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS.keys()), default="droid",
                    help="which pi0.5 variant to load")
    ap.add_argument("--prompt", default="pick up the red block and put it in the brown box")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--record", default=None, help="path to .mp4 for third-person recording")
    args = ap.parse_args()

    mcfg = MODELS[args.model]
    print(f"=== Model: {args.model} ===")
    print(f"  config     : {mcfg['config']}")
    print(f"  checkpoint : {mcfg['checkpoint']}")
    print(f"  obs_format : {mcfg['obs_format']}")
    print(f"  act_format : {mcfg['action_format']}")

    m = mujoco.MjModel.from_xml_path(MODEL_XML)
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "teleop")
    mujoco.mj_resetDataKeyframe(m, d, key)

    for i in range(m.nu):
        d.ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]

    right_qidx = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
                  for j in RIGHT_ARM_JOINTS]
    right_aid  = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
                  for a in RIGHT_ARM_ACTS]
    grip_qidx  = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_JOINT)]
    grip_aid   = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACT)

    renderer_pol = mujoco.Renderer(m, height=480, width=640)
    renderer_rec = mujoco.Renderer(m, height=480, width=640)

    import jax
    print(f"JAX devices: {jax.devices()}")
    t0 = time.time()
    cfg = _config.get_config(mcfg["config"])
    ckpt_dir = download.maybe_download(mcfg["checkpoint"])
    policy = _policy_config.create_trained_policy(cfg, ckpt_dir)
    print(f"Policy loaded in {time.time()-t0:.1f}s")
    print(f"Prompt: {args.prompt!r}")

    steps_per_action = max(1, int(round(1.0 / (CTRL_HZ * m.opt.timestep))))
    print(f"sim dt={m.opt.timestep}s  action interval={1/CTRL_HZ:.3f}s  sim-steps/action={steps_per_action}")

    video_frames = []
    ctx = None if args.headless else mujoco.viewer.launch_passive(m, d)

    def loop_body():
        chunk = None
        chunk_step = 0
        for t_step in range(0, args.max_steps if args.max_steps > 0 else 10**9):
            if chunk is None or chunk_step >= OPEN_LOOP_HORIZON:
                base_img = render_cam(m, d, renderer_pol, "zed_left")
                wrist_img = np.zeros((224, 224, 3), dtype=np.uint8)
                joint_pos = np.array([d.qpos[i] for i in right_qidx], dtype=np.float64)
                grip_norm = float(abs(d.qpos[grip_qidx]) / abs(GRIPPER_OPEN))

                obs = build_obs(mcfg["obs_format"], base_img, wrist_img, joint_pos, grip_norm, args.prompt)
                t_infer = time.time()
                result = policy.infer(obs)
                chunk = np.asarray(result["actions"])
                chunk_step = 0
                print(f"[t={t_step:4d}] infer={time.time()-t_infer:.2f}s  chunk={chunk.shape}  "
                      f"q_r={joint_pos.round(2).tolist()}  grip={grip_norm:.2f}")

            apply_action(mcfg["action_format"], chunk[chunk_step], d,
                         right_qidx, right_aid, grip_aid)

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
