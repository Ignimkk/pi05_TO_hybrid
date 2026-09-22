import dataclasses
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import mujoco
import numpy as np

from rby1_manipulation.data.collect_randomized_pick_place_dataset import (
    FRUITS,
    build_schedule,
    schedule_summary,
    smoke_schedule,
)
from rby1_manipulation.data.episode import SCHEMAS
from rby1_manipulation.simulation.fruit_grid import load_fruit_grid_config, table_placements
from rby1_manipulation.simulation.randomized_pick_place import (
    ToggleRange,
    _candidate_values,
    _precheck,
    load_default_model,
    load_randomization_config,
    randomization_config_fingerprint,
    ready_arm_joint_qpos,
    sample_valid_scene,
)
from rby1_manipulation.simulation.transport_scene import load_layout_config
from rby1_manipulation.tasks.transport_atomic import _check_randomized_dataset_schema


class RandomizedPickPlaceCollectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_randomization_config()
        cls.schedule = build_schedule(config=cls.config)

    def test_schedule_is_balanced_and_split_contiguous(self):
        summary = schedule_summary(self.schedule)
        self.assertEqual(summary["total"], 2000)
        self.assertEqual(summary["splits"], {"test": 200, "train": 1600, "validation": 200})
        self.assertEqual(summary["targets"], {fruit: 500 for fruit in FRUITS})
        self.assertEqual(summary["layouts"], {index: 125 for index in range(16)})
        self.assertEqual(
            Counter("left" if spec.target_slot < 2 else "right" for spec in self.schedule),
            {"left": 1000, "right": 1000},
        )
        for fruit in FRUITS:
            members = [spec for spec in self.schedule if spec.target_fruit == fruit]
            self.assertEqual(Counter(spec.target_slot for spec in members), {index: 125 for index in range(4)})
            self.assertEqual(Counter(spec.split for spec in members), {
                "train": 400, "validation": 50, "test": 50,
            })
            for split in ("train", "validation", "test"):
                counts = Counter(
                    spec.prompt for spec in members if spec.split == split
                )
                self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_smoke_schedule_has_one_of_each_target(self):
        smoke = smoke_schedule(self.schedule, 1)
        self.assertEqual([spec.target_fruit for spec in smoke], list(FRUITS))
        self.assertEqual([spec.plan_index for spec in smoke], list(range(4)))
        self.assertTrue(all(spec.split == "train" for spec in smoke))

    def test_rby1_16_schema_order(self):
        schema = SCHEMAS["rby1_16"]
        self.assertEqual(schema["dim"], 16)
        self.assertEqual(schema["names"][0:8], [
            "left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3",
            "left_arm_4", "left_arm_5", "left_arm_6", "left_gripper",
        ])
        self.assertEqual(schema["names"][8:16], [
            "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3",
            "right_arm_4", "right_arm_5", "right_arm_6", "right_gripper",
        ])

    def test_disabled_randomization_returns_nominal_values(self):
        model, _ = load_default_model()
        layout = load_layout_config()
        grid = load_fruit_grid_config()
        disabled = dataclasses.replace(
            self.config,
            target_position=ToggleRange(False, self.config.target_position.values),
            goal_position=ToggleRange(False, self.config.goal_position.values),
            target_orientation=ToggleRange(False, self.config.target_orientation.values),
            robot_initial_configuration=ToggleRange(
                False, self.config.robot_initial_configuration.values
            ),
        )
        first = _candidate_values(
            model, disabled, layout, grid, "apple", 0, FRUITS, np.random.default_rng(1)
        )
        second = _candidate_values(
            model, disabled, layout, grid, "apple", 0, FRUITS, np.random.default_rng(99)
        )
        nominal = table_placements(grid, layout, layout_index=0, slot_order=FRUITS)["apple"]
        self.assertTrue(np.array_equal(first[0], nominal))
        self.assertTrue(np.array_equal(first[1], np.asarray(layout["crate"]["xy"])))
        self.assertEqual(first[2], 0.0)
        self.assertEqual(first[3], second[3])

    def test_ready_pose_spreads_both_arms_outward(self):
        model, _ = load_default_model()
        ready = ready_arm_joint_qpos(model, self.config)
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "teleop")
        key_qpos = np.asarray(model.key_qpos[key_id], dtype=float)
        for side, sign in (("left", 1.0), ("right", -1.0)):
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_arm_1"
            )
            qidx = int(model.jnt_qposadr[joint_id])
            self.assertAlmostEqual(
                ready[f"{side}_arm_1"] - float(key_qpos[qidx]),
                sign * np.deg2rad(5.0),
            )

    def test_sampling_is_seed_reproducible_and_valid(self):
        layout = load_layout_config()
        grid = load_fruit_grid_config()
        scenes = []
        for _ in range(2):
            model, data = load_default_model()
            scenes.append(sample_valid_scene(
                model, data, layout, grid, self.config,
                target="orange", layout_index=3, slot_order=FRUITS,
                rng=np.random.default_rng(123),
            ))
        self.assertTrue(np.allclose(scenes[0].requested_target_pose, scenes[1].requested_target_pose))
        self.assertTrue(np.allclose(scenes[0].requested_goal_pose, scenes[1].requested_goal_pose))
        self.assertTrue(scenes[0].validity["valid"])
        self.assertEqual(scenes[0].initial_robot_state.shape, (16,))
        self.assertTrue(np.allclose(scenes[0].initial_robot_state[[7, 15]], 1.0, atol=1e-3))

        other_model, other_data = load_default_model()
        different = sample_valid_scene(
            other_model, other_data, layout, grid, self.config,
            target="orange", layout_index=3, slot_order=FRUITS,
            rng=np.random.default_rng(124),
        )
        self.assertFalse(np.allclose(
            scenes[0].requested_target_pose,
            different.requested_target_pose,
        ))
        self.assertFalse(np.allclose(
            scenes[0].initial_robot_state,
            different.initial_robot_state,
        ))

    def test_workspace_overlap_and_joint_limit_are_rejected(self):
        model, _ = load_default_model()
        layout = load_layout_config()
        grid = load_fruit_grid_config()
        _, goal_xy, _, initial_joints = _candidate_values(
            model, self.config, layout, grid, "apple", 0, FRUITS,
            np.random.default_rng(5),
        )
        nominal = table_placements(grid, layout, layout_index=0, slot_order=FRUITS)

        reasons = _precheck(
            model, self.config, layout, grid, "apple", 0, FRUITS,
            np.array([0.0, 0.0, nominal["apple"][2]]), goal_xy, initial_joints,
        )
        self.assertIn("target_outside_workspace", reasons)

        reasons = _precheck(
            model, self.config, layout, grid, "apple", 0, FRUITS,
            nominal["banana"].copy(), goal_xy, initial_joints,
        )
        self.assertIn("target_distractor_overlap", reasons)

        bad_joints = dict(initial_joints)
        joint_name = next(iter(bad_joints))
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        bad_joints[joint_name] = float(model.jnt_range[joint_id][1]) + 1.0
        reasons = _precheck(
            model, self.config, layout, grid, "apple", 0, FRUITS,
            nominal["apple"].copy(), goal_xy, bad_joints,
        )
        self.assertIn("joint_limit", reasons)

    def test_initial_collision_is_rejected(self):
        model, data = load_default_model()
        disabled = dataclasses.replace(
            self.config,
            max_sampling_attempts=1,
            target_position=ToggleRange(False, self.config.target_position.values),
            goal_position=ToggleRange(False, self.config.goal_position.values),
            target_orientation=ToggleRange(False, self.config.target_orientation.values),
            robot_initial_configuration=ToggleRange(
                False, self.config.robot_initial_configuration.values
            ),
        )
        with mock.patch(
            "rby1_manipulation.simulation.randomized_pick_place._contact_pairs",
            return_value=([["arm", "table"]], []),
        ):
            with self.assertRaisesRegex(RuntimeError, "initial_arm_collision"):
                sample_valid_scene(
                    model, data, load_layout_config(), load_fruit_grid_config(), disabled,
                    target="apple", layout_index=0, slot_order=FRUITS,
                    rng=np.random.default_rng(7),
                )

    def test_config_fingerprint_is_stable(self):
        self.assertEqual(
            randomization_config_fingerprint(self.config),
            randomization_config_fingerprint(load_randomization_config()),
        )

    def test_resume_rejects_config_fingerprint_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meta").mkdir()
            (root / "meta" / "randomized_schema.json").write_text(json.dumps({
                "version": 1,
                "randomization_config_fingerprint": "different",
            }), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "different randomization config"):
                _check_randomized_dataset_schema(
                    root,
                    config_fingerprint=randomization_config_fingerprint(self.config),
                )


if __name__ == "__main__":
    unittest.main()
