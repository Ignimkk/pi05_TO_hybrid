"""Contracts for static obstacle evaluation in block and fruit scenes."""
from __future__ import annotations

import unittest

import mujoco
import numpy as np

from rby1_manipulation.paths import (
    BLOCK_MODEL_XML,
    PICK_PLACE_OBSTACLE_MODEL_XML,
    TRANSPORT_MODEL_XML,
    TRANSPORT_PICK_PLACE_OBSTACLE_MODEL_XML,
)
from rby1_manipulation.simulation.pick_place_obstacles import (
    PickPlaceObstacleManager,
    load_pick_place_obstacle_config,
    profile_names,
)


class PickPlaceObstacleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_pick_place_obstacle_config()

    def make_scene(self, model_path=PICK_PLACE_OBSTACLE_MODEL_XML):
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "teleop")
        mujoco.mj_resetDataKeyframe(model, data, key)
        for actuator_id in range(model.nu):
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            data.ctrl[actuator_id] = data.qpos[model.jnt_qposadr[joint_id]]
        mujoco.mj_forward(model, data)
        return model, data, PickPlaceObstacleManager(model, data, self.config)

    def test_variant_preserves_policy_interface(self) -> None:
        for clear_path, obstacle_path in (
            (BLOCK_MODEL_XML, PICK_PLACE_OBSTACLE_MODEL_XML),
            (TRANSPORT_MODEL_XML, TRANSPORT_PICK_PLACE_OBSTACLE_MODEL_XML),
        ):
            clear_model = mujoco.MjModel.from_xml_path(str(clear_path))
            obstacle_model = mujoco.MjModel.from_xml_path(str(obstacle_path))
            self.assertEqual(
                (obstacle_model.nq, obstacle_model.nv, obstacle_model.nu),
                (clear_model.nq, clear_model.nv, clear_model.nu),
            )
            clear_actuators = [
                mujoco.mj_id2name(clear_model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
                for index in range(clear_model.nu)
            ]
            obstacle_actuators = [
                mujoco.mj_id2name(obstacle_model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
                for index in range(obstacle_model.nu)
            ]
            self.assertEqual(obstacle_actuators, clear_actuators)

    def test_profiles_are_static_and_start_without_overlap(self) -> None:
        self.assertIn("right_bollard", profile_names(self.config))
        self.assertIn("fruit_right_bollard", profile_names(self.config))
        profile_scenes = (
            ("block", PICK_PLACE_OBSTACLE_MODEL_XML),
            ("fruit", TRANSPORT_PICK_PLACE_OBSTACLE_MODEL_XML),
        )
        for scene, model_path in profile_scenes:
            for profile in profile_names(self.config):
                if self.config["profiles"][profile]["scene"] not in ("any", scene):
                    continue
                model, data, obstacles = self.make_scene(model_path)
                obstacles.activate(profile)
                for _ in range(int(round(0.5 / model.opt.timestep))):
                    mujoco.mj_step(model, data)
                    obstacles.observe_contacts()
                if profile == "clear":
                    self.assertEqual(obstacles.active_slots, ())
                    self.assertTrue(np.isinf(obstacles.robot_clearance()))
                else:
                    self.assertGreater(obstacles.robot_clearance(), 0.02, profile)
                    self.assertGreater(obstacles.object_clearance(), 0.005, profile)
                    self.assertFalse(obstacles.robot_collision, profile)

    def test_hold_robot_copies_measured_joint_positions(self) -> None:
        model, data, obstacles = self.make_scene()
        data.ctrl[:] = 0.0
        obstacles.hold_robot()
        for actuator_id in range(model.nu):
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            qpos_id = int(model.jnt_qposadr[joint_id])
            self.assertAlmostEqual(data.ctrl[actuator_id], data.qpos[qpos_id])

    def test_proximity_stop_triggers_before_contact_is_required(self) -> None:
        model, data, obstacles = self.make_scene()
        obstacles.activate("right_bollard")
        target_geom = obstacles.robot_geom_ids[-1]
        # Put the bollard collision-cylinder centre on a real arm collision geom.
        # This bypasses profile validation only to exercise the runtime safety gate.
        mocap_id = obstacles._mocap_ids["bollard_0"]
        data.mocap_pos[mocap_id] = data.geom_xpos[target_geom] - np.array([0.0, 0.0, 0.13])
        mujoco.mj_forward(model, data)
        self.assertTrue(obstacles.unsafe(0.02))
        self.assertLessEqual(obstacles.min_robot_clearance, 0.02)


if __name__ == "__main__":
    unittest.main()
