"""Live MuJoCo camera preview matching the transport dataset by default.

The default uses ``model_transport.xml`` and the same 4:3 render followed by a
224x224 resize as ``EpisodeRecorder``. This makes a bare invocation show what
the policy receives instead of the block scene at a different aspect ratio.
"""
from __future__ import annotations

import argparse
import pathlib
import time

import mujoco
import mujoco.viewer
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "rby1_description" / "models" / "rby1a" / "mujoco"
POLICY_SIZE = 224
POLICY_SOURCE_WIDTH = round(POLICY_SIZE * 4 / 3)
POLICY_SOURCE_HEIGHT = POLICY_SIZE
NATIVE_SIZE = (640, 480)
CAM_HZ = 30
CAMERAS = (
    ("cam_high", "zed_left"),
    ("cam_left_wrist", "wrist_cam_l"),
    ("cam_right_wrist", "wrist_cam_r"),
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        choices=("transport", "blocks"),
        default="transport",
        help="scene to preview (default: fruit/basket transport scene)",
    )
    parser.add_argument(
        "--native-view",
        action="store_true",
        help="show native 640x480 instead of the exact 224x224 policy tensor",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    model_name = "model_transport.xml" if args.model == "transport" else "model.xml"
    model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / model_name))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "teleop")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    for actuator_id in range(model.nu):
        joint_id = model.actuator_trnid[actuator_id, 0]
        data.ctrl[actuator_id] = data.qpos[model.jnt_qposadr[joint_id]]

    if args.native_view:
        render_width, render_height = NATIVE_SIZE
        display_size = NATIVE_SIZE
    else:
        render_width, render_height = POLICY_SOURCE_WIDTH, POLICY_SOURCE_HEIGHT
        display_size = (POLICY_SIZE, POLICY_SIZE)
    renderer = mujoco.Renderer(model, height=render_height, width=render_width)

    root = tk.Tk()
    mode = "native 640x480" if args.native_view else "policy 224x224"
    root.title(f"RBY1 cameras - {args.model} - {mode}")
    labels: dict[str, tk.Label] = {}
    for column, (logical_name, camera_name) in enumerate(CAMERAS):
        label = tk.Label(root)
        label.grid(row=0, column=column, padx=4, pady=4)
        tk.Label(root, text=f"{logical_name}\n({camera_name})").grid(
            row=1, column=column
        )
        labels[logical_name] = label

    def render_camera(camera_name: str) -> np.ndarray:
        renderer.update_scene(data, camera=camera_name)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        image = Image.fromarray(renderer.render())
        if image.size != display_size:
            image = image.resize(display_size, resample=Image.Resampling.BILINEAR)
        return np.asarray(image)

    references = []
    for logical_name, camera_name in CAMERAS:
        photo = ImageTk.PhotoImage(Image.fromarray(render_camera(camera_name)))
        labels[logical_name].configure(image=photo)
        references.append(photo)

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_render = 0.0
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            if time.time() - last_render > 1.0 / CAM_HZ:
                for logical_name, camera_name in CAMERAS:
                    photo = ImageTk.PhotoImage(
                        Image.fromarray(render_camera(camera_name))
                    )
                    labels[logical_name].configure(image=photo)
                    labels[logical_name].image = photo
                last_render = time.time()
            try:
                root.update_idletasks()
                root.update()
            except tk.TclError:
                break
            delay = model.opt.timestep - (time.time() - step_start)
            if delay > 0:
                time.sleep(delay)

    renderer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
