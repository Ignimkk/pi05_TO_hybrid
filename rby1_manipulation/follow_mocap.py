"""Mink IK: right (and optionally left) arm follows a mocap target.

Open the viewer, drag `link_right_arm_6_target` (and `link_left_arm_6_target` if
--dual) mocap sphere with the mouse, and the corresponding EE tracks it.

Key implementation notes learned during smoke testing:
1. `mink.Configuration` uses ALL DoFs of the model. On RBY1 that includes the
   base planar joints (joint_x/y/th) and 6-DoF torso. If left free, the IK
   "solves" the target by displacing those DoFs even though they are actuated
   by wheels/torso motors, not directly settable. Result: kinematic IK claims
   convergence, but the actual sim EE barely moves.
   Fix: mask solver velocities to the arm DoFs only before integrating.

2. RBY1 joints have damping=50 in the default class, so PD position tracking
   is intentionally slow (~1-2 s to settle a joint delta). For scripted demo
   collection this is fine; for real-time teleop the arm lags the mouse.

3. Running the IK solver a few times per control tick lets `config.q` converge
   toward the target BEFORE it becomes the actuator setpoint, so the ctrl
   trajectory is smoother.

Usage:
    python follow_mocap.py            # right arm only
    python follow_mocap.py --dual     # both arms

Controls:
    - Right-click and drag the small red spheres (mocap targets) to move them.
    - Close the viewer window to quit.
"""
import argparse
import pathlib
import time

import mujoco
import mujoco.viewer
import mink
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_XML = str(REPO_ROOT / "rby1_description" / "models" / "rby1a" / "mujoco" / "model.xml")

RIGHT_EE_SITE = "right_ee"
LEFT_EE_SITE  = "left_ee"
RIGHT_TARGET_BODY = "link_right_arm_6_target"
LEFT_TARGET_BODY  = "link_left_arm_6_target"

RIGHT_ARM_JOINTS = [f"right_arm_{i}" for i in range(7)]
LEFT_ARM_JOINTS  = [f"left_arm_{i}"  for i in range(7)]
RIGHT_ARM_ACTS   = [f"right_arm_{i+1}_act" for i in range(7)]
LEFT_ARM_ACTS    = [f"left_arm_{i+1}_act"  for i in range(7)]

CTRL_HZ = 60
IK_INNER_ITERS = 10        # inner solves per control tick
IK_INNER_DT = 1e-2         # dt for inner integration


def mocap_se3(model, data, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    mocap_id = model.body_mocapid[body_id]
    return mink.SE3.from_rotation_and_translation(
        mink.SO3(np.asarray(data.mocap_quat[mocap_id]).copy()),
        np.asarray(data.mocap_pos[mocap_id]).copy(),
    )


def snap_mocap_to_site(model, data, mocap_body, site_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mocap_body)
    mocap_id = model.body_mocapid[body_id]
    site_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    data.mocap_pos[mocap_id] = data.site_xpos[site_id]
    mat = data.site_xmat[site_id].reshape(3, 3)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat.reshape(-1))
    data.mocap_quat[mocap_id] = q


def build_dof_mask(model, joint_names):
    """A boolean mask over model.nv that is True at the given joints' DoFs."""
    mask = np.zeros(model.nv, dtype=bool)
    for jn in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        mask[model.jnt_dofadr[jid]] = True
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dual", action="store_true", help="drive both arms")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data  = mujoco.MjData(model)

    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "teleop")
    mujoco.mj_resetDataKeyframe(model, data, key)
    for i in range(model.nu):
        data.ctrl[i] = data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]
    mujoco.mj_forward(model, data)

    # Seed the mocap sphere(s) at the current EE pose so nothing snaps at startup.
    snap_mocap_to_site(model, data, RIGHT_TARGET_BODY, RIGHT_EE_SITE)
    if args.dual:
        snap_mocap_to_site(model, data, LEFT_TARGET_BODY, LEFT_EE_SITE)

    # Actuator + qpos maps.
    right_aid  = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in RIGHT_ARM_ACTS]
    right_qidx = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
                  for j in RIGHT_ARM_JOINTS]
    left_aid   = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in LEFT_ARM_ACTS]
    left_qidx  = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
                  for j in LEFT_ARM_JOINTS]

    # DoF masks: which joints IK is allowed to move.
    if args.dual:
        arm_mask = build_dof_mask(model, RIGHT_ARM_JOINTS + LEFT_ARM_JOINTS)
    else:
        arm_mask = build_dof_mask(model, RIGHT_ARM_JOINTS)

    # ---- mink setup ----
    config = mink.Configuration(model)
    config.update(data.qpos)

    tasks = []
    right_task = mink.FrameTask(RIGHT_EE_SITE, "site",
                                position_cost=100.0, orientation_cost=10.0, lm_damping=0.001)
    tasks.append(right_task)
    if args.dual:
        left_task = mink.FrameTask(LEFT_EE_SITE, "site",
                                   position_cost=100.0, orientation_cost=10.0, lm_damping=0.001)
        tasks.append(left_task)

    posture = mink.PostureTask(model, cost=1e-4)
    posture.set_target_from_configuration(config)
    tasks.append(posture)

    dt = 1.0 / CTRL_HZ
    sim_steps_per_ctrl = max(1, int(round(dt / model.opt.timestep)))
    print(f"IK @ {CTRL_HZ} Hz  (inner iters={IK_INNER_ITERS})  ->  {sim_steps_per_ctrl} sim steps/ctrl tick")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Force the mocap markers visible for this script (they are hidden
        # elsewhere via group=3 to keep camera renders clean). Group 4 shows
        # the little site dots too.
        viewer.opt.geomgroup[3] = 1
        viewer.opt.sitegroup[4] = 1

        print("[follow_mocap] hidden markers turned back on for this session.")
        print("[follow_mocap] Interact with mocap:")
        print("    double-click on the red target sphere/box to select it,")
        print("    then Ctrl + right-click + drag  -> translate")
        print("           Ctrl + left-click  + drag  -> rotate")

        while viewer.is_running():
            step_start = time.time()

            # 1. Update task targets from the user-controlled mocap sphere(s).
            right_task.set_target(mocap_se3(model, data, RIGHT_TARGET_BODY))
            if args.dual:
                left_task.set_target(mocap_se3(model, data, LEFT_TARGET_BODY))

            # 2. Sync config to the sim state, then run several IK inner iters
            #    so the joint targets settle before we command them to actuators.
            config.update(data.qpos)
            for _ in range(IK_INNER_ITERS):
                vel = mink.solve_ik(config, tasks, IK_INNER_DT,
                                    solver="daqp", damping=1e-4)
                vel = vel * arm_mask                        # arm-only movement
                config.integrate_inplace(vel, IK_INNER_DT)

            # 3. Push the (arm-only) joint targets into the position actuators.
            for i, aid in enumerate(right_aid):
                data.ctrl[aid] = config.q[right_qidx[i]]
            if args.dual:
                for i, aid in enumerate(left_aid):
                    data.ctrl[aid] = config.q[left_qidx[i]]

            # 4. Advance the sim.
            for _ in range(sim_steps_per_ctrl):
                mujoco.mj_step(model, data)

            viewer.sync()

            # 5. Real-time pacing.
            wait = dt - (time.time() - step_start)
            if wait > 0:
                time.sleep(wait)


if __name__ == "__main__":
    main()
