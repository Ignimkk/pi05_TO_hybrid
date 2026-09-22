"""Fast contracts for the manipulation package refactor."""

from __future__ import annotations

import json
import tempfile
import unittest

from pathlib import Path

import mujoco
import numpy as np

from rby1_manipulation.control.ik import left_arm_handles, right_arm_handles
from rby1_manipulation.data.episode import Frame, LeRobotWriter
from rby1_manipulation.paths import (
    BLOCK_GRID_CONFIG,
    BLOCK_MODEL_XML,
    FRUIT_GRID_CONFIG,
    PICK_PLACE_OBSTACLE_CONFIG,
    PICK_PLACE_OBSTACLE_MODEL_XML,
    TRANSPORT_LAYOUT_CONFIG,
    TRANSPORT_MODEL_XML,
    TRANSPORT_PICK_PLACE_OBSTACLE_MODEL_XML,
    TRANSPORT_WHEEL_MODEL_XML,
)
from rby1_manipulation.planning.transport import object_grasp_dz
from rby1_manipulation.simulation.transport_scene import (
    build_action_14,
    build_action_17,
    build_state_14,
    build_state_17,
    base_handles,
)


class PackageContractsTest(unittest.TestCase):
    def test_packaged_configs_are_valid_json(self) -> None:
        for path in (
            BLOCK_GRID_CONFIG,
            FRUIT_GRID_CONFIG,
            TRANSPORT_LAYOUT_CONFIG,
            PICK_PLACE_OBSTACLE_CONFIG,
        ):
            self.assertTrue(path.is_file(), path)
            self.assertIsInstance(json.loads(path.read_text(encoding="utf-8")), dict)

    def test_mujoco_model_dimensions_are_unchanged(self) -> None:
        expected = {
            BLOCK_MODEL_XML: (52, 49, 26),
            PICK_PLACE_OBSTACLE_MODEL_XML: (52, 49, 26),
            TRANSPORT_MODEL_XML: (66, 61, 29),
            TRANSPORT_PICK_PLACE_OBSTACLE_MODEL_XML: (66, 61, 29),
            TRANSPORT_WHEEL_MODEL_XML: (66, 61, 26),
        }
        for path, dimensions in expected.items():
            model = mujoco.MjModel.from_xml_path(str(path))
            self.assertEqual((model.nq, model.nv, model.nu), dimensions, path.name)

    def test_table_long_axis_is_one_meter_and_symmetric(self) -> None:
        for path in (BLOCK_MODEL_XML, TRANSPORT_MODEL_XML):
            model = mujoco.MjModel.from_xml_path(str(path))
            table_top = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, "table_top"
            )
            np.testing.assert_allclose(
                model.geom_size[table_top], [0.25, 0.50, 0.02]
            )
            leg_y = {
                name: float(model.geom_pos[mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_GEOM, name
                ), 1])
                for name in (
                    "table_leg_fl", "table_leg_fr", "table_leg_bl", "table_leg_br"
                )
            }
            self.assertAlmostEqual(leg_y["table_leg_fl"], 0.48)
            self.assertAlmostEqual(leg_y["table_leg_bl"], 0.48)
            self.assertAlmostEqual(leg_y["table_leg_fr"], -0.48)
            self.assertAlmostEqual(leg_y["table_leg_br"], -0.48)

    def test_14d_transport_contract_matches_mobile_prefix(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(TRANSPORT_MODEL_XML))
        data = mujoco.MjData(model)
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "teleop")
        mujoco.mj_resetDataKeyframe(model, data, key)
        for actuator_id in range(model.nu):
            joint_id = model.actuator_trnid[actuator_id, 0]
            data.ctrl[actuator_id] = data.qpos[model.jnt_qposadr[joint_id]]
        left, right, base = left_arm_handles(model), right_arm_handles(model), base_handles(model)
        np.testing.assert_array_equal(
            build_state_14(data, left, right),
            build_state_17(model, data, left, right, base)[:14],
        )
        np.testing.assert_array_equal(
            build_action_14(data, left, right),
            build_action_17(model, data, left, right, base)[:14],
        )

    def test_banana_grasp_height_has_table_clearance_floor(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(TRANSPORT_MODEL_XML))
        self.assertAlmostEqual(object_grasp_dz(model, "banana"), 0.030)
        self.assertAlmostEqual(object_grasp_dz(model, "pear"), 0.015)

    def test_episode_1000_uses_second_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            writer = LeRobotWriter(temporary, fps=15, image_wh=(2, 2), schema="rby1_14")
            episode = writer.new_episode("test", episode_index=1000)
            episode.append(Frame(
                state=np.zeros(14, dtype=np.float32),
                action=np.zeros(14, dtype=np.float32),
                images={},
                timestamp=0.0,
                frame_index=0,
            ))
            writer._write_parquet(episode)
            expected = Path(temporary) / "data" / "chunk-001" / "episode_001000.parquet"
            self.assertTrue(expected.is_file())


if __name__ == "__main__":
    unittest.main()
