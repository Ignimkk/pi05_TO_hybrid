"""Static contracts for the experimental wheel-drive interface."""
from __future__ import annotations

import unittest

import mujoco

from rby1_manipulation.control.mobile_base import unicycle_to_wheels, wheel_preflight
from rby1_manipulation.paths import TRANSPORT_MODEL_XML, TRANSPORT_WHEEL_MODEL_XML


class WheelReadinessTest(unittest.TestCase):
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
