"""Live view: MuJoCo main viewer + ZED-left + both wrist cameras.

Controls:
  - Main viewer: drag mocap targets, use standard mujoco.viewer controls
  - Close any window to quit
"""
import time
import numpy as np
import mujoco
import mujoco.viewer
from PIL import Image
import tkinter as tk
from PIL import ImageTk

MODEL = "/home/mk/dev_ws/vla/pi0_TO_ws/src/rby1_description/models/rby1a/mujoco/model.xml"
CAM_W, CAM_H = 640, 480
CAM_HZ = 30

CAMERAS = ["zed_left", "wrist_cam_r", "wrist_cam_l"]

model = mujoco.MjModel.from_xml_path(MODEL)
data = mujoco.MjData(model)

key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "teleop")
mujoco.mj_resetDataKeyframe(model, data, key_id)

for i in range(model.nu):
    jid = model.actuator_trnid[i, 0]
    data.ctrl[i] = data.qpos[model.jnt_qposadr[jid]]

renderer = mujoco.Renderer(model, height=CAM_H, width=CAM_W)

root = tk.Tk()
root.title("RBY1 cameras")

# One column per camera, image on top, label below.
labels = {}
for col, name in enumerate(CAMERAS):
    lbl = tk.Label(root)
    lbl.grid(row=0, column=col, padx=4, pady=4)
    tk.Label(root, text=name).grid(row=1, column=col)
    labels[name] = lbl

def render_cam(name):
    renderer.update_scene(data, camera=name)
    return renderer.render()

# Prime images (keep references to avoid GC)
_refs = []
for name in CAMERAS:
    ph = ImageTk.PhotoImage(Image.fromarray(render_cam(name)))
    labels[name].configure(image=ph)
    _refs.append(ph)

def on_close():
    root.destroy()
root.protocol("WM_DELETE_WINDOW", on_close)

with mujoco.viewer.launch_passive(model, data) as viewer:
    last_render = 0.0
    while viewer.is_running():
        try:
            root.winfo_exists()
        except tk.TclError:
            break

        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()

        if time.time() - last_render > 1.0 / CAM_HZ:
            for name in CAMERAS:
                img = render_cam(name)
                ph = ImageTk.PhotoImage(Image.fromarray(img))
                labels[name].configure(image=ph)
                labels[name].image = ph
            last_render = time.time()

        try:
            root.update_idletasks()
            root.update()
        except tk.TclError:
            break

        dt = model.opt.timestep - (time.time() - step_start)
        if dt > 0:
            time.sleep(dt)
