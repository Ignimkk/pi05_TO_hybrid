import mujoco
import mujoco.viewer

MODEL = "/home/mk/dev_ws/vla/pi0_TO_ws/src/rby1_description/models/rby1a/mujoco/model.xml"

model = mujoco.MjModel.from_xml_path(MODEL)
data = mujoco.MjData(model)

key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "teleop")
if key_id >= 0:
    mujoco.mj_resetDataKeyframe(model, data, key_id)

mujoco.viewer.launch(model, data)
