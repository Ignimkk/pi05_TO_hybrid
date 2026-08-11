"""Fast contracts for the manipulation package refactor."""

from __future__ import annotations

import json
import unittest

import mujoco

from rby1_manipulation.paths import (
    BLOCK_GRID_CONFIG,
    BLOCK_MODEL_XML,
    TRANSPORT_LAYOUT_CONFIG,
    TRANSPORT_MODEL_XML,
    TRANSPORT_WHEEL_MODEL_XML,
)


class PackageContractsTest(unittest.TestCase):
    def test_packaged_configs_are_valid_json(self) -> None:
        for path in (BLOCK_GRID_CONFIG, TRANSPORT_LAYOUT_CONFIG):
            self.assertTrue(path.is_file(), path)
            self.assertIsInstance(json.loads(path.read_text(encoding="utf-8")), dict)

    def test_mujoco_model_dimensions_are_unchanged(self) -> None:
        expected = {
            BLOCK_MODEL_XML: (52, 49, 26),
            TRANSPORT_MODEL_XML: (66, 61, 29),
            TRANSPORT_WHEEL_MODEL_XML: (66, 61, 26),
        }
        for path, dimensions in expected.items():
            model = mujoco.MjModel.from_xml_path(str(path))
            self.assertEqual((model.nq, model.nv, model.nu), dimensions, path.name)


if __name__ == "__main__":
    unittest.main()
