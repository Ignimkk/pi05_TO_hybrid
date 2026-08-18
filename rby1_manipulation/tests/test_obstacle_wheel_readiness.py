"""Contracts for evaluation obstacles and the wheel-drive interface."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from rby1_manipulation.control.mobile_base import (
    unicycle_to_wheels,
    wheel_preflight,
)
from rby1_manipulation.paths import TRANSPORT_MODEL_XML, TRANSPORT_WHEEL_MODEL_XML
from rby1_manipulation.simulation.obstacles import (
    ObstacleCollisionMonitor,
    TransportObstacleManager,
    load_obstacle_config,
    profile_names,
)


class ObstacleWheelReadinessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_obstacle_config()

    def test_profiles_and_slots_compile_without_changing_control_dimensions(self) -> None:
        self.assertEqual(
            profile_names(self.config),
            ("clear", "static_offset", "static_blocked", "dynamic_crossing", "mixed"),
        )
        for path, dimensions in (
            (TRANSPORT_MODEL_XML, (66, 61, 29)),
            (TRANSPORT_WHEEL_MODEL_XML, (66, 61, 26)),
        ):
            model = mujoco.MjModel.from_xml_path(str(path))
            self.assertEqual((model.nq, model.nv, model.nu), dimensions)
            for slot in (
                "static_box_0",
                "static_box_1",
                "static_column_0",
                "dynamic_human_0",
                "dynamic_cart_0",
            ):
                body_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle_{slot}"
                )
                self.assertGreaterEqual(body_id, 0)
                self.assertGreaterEqual(int(model.body_mocapid[body_id]), 0)

    def test_dynamic_profile_is_deterministic_and_clear_profile_is_empty(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(TRANSPORT_MODEL_XML))
        data = mujoco.MjData(model)
        obstacles = TransportObstacleManager(model, data, self.config)

        obstacles.activate("clear")
        self.assertEqual(obstacles.active, {})
        self.assertTrue(np.isinf(obstacles.planar_clearance([0.0, 0.0])))

        obstacles.activate("dynamic_crossing")
        np.testing.assert_allclose(
            obstacles.positions()["dynamic_human_0"], [-0.75, -0.58, 0.72]
        )
        obstacles.update(2.0)
        np.testing.assert_allclose(
            obstacles.positions()["dynamic_human_0"], [-0.075, -0.58, 0.72],
            atol=1e-12,
        )

    def test_collision_monitor_records_planar_clearance(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(TRANSPORT_MODEL_XML))
        data = mujoco.MjData(model)
        obstacles = TransportObstacleManager(model, data, self.config)
        obstacles.activate("static_blocked")
        monitor = ObstacleCollisionMonitor(model, data, obstacles)
        clearance = monitor.observe_base_clearance([0.0, 0.0])
        self.assertLess(clearance, 0.05)
        self.assertEqual(monitor.min_planar_clearance, clearance)

        for profile in ("static_offset", "dynamic_crossing", "mixed"):
            obstacles.activate(profile)
            self.assertGreater(
                obstacles.planar_clearance([0.0, 0.0]),
                0.05,
                f"{profile} must not trigger the safety stop at reset",
            )

    def test_wheel_model_preflight_and_command_signs(self) -> None:
        wheel_model = mujoco.MjModel.from_xml_path(str(TRANSPORT_WHEEL_MODEL_XML))
        report = wheel_preflight(wheel_model)
        self.assertTrue(report.ready, report.issues)
        self.assertEqual(set(report.actuator_ids), {"left", "right"})

        kinematic_model = mujoco.MjModel.from_xml_path(str(TRANSPORT_MODEL_XML))
        self.assertFalse(wheel_preflight(kinematic_model).ready)

        left, right = unicycle_to_wheels(0.2, 0.0)
        self.assertLess(left, 0.0)
        self.assertAlmostEqual(left, right)
        left, right = unicycle_to_wheels(0.0, 0.5)
        self.assertGreater(left, 0.0)
        self.assertLess(right, 0.0)


if __name__ == "__main__":
    unittest.main()
