import itertools
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import mujoco
import pyarrow.parquet as pq

from rby1_manipulation.data.collect_atomic_transport_dataset import (
    FAMILY_SPLITS,
    FAMILY_TOTALS,
    FRUITS,
    PARAPHRASES,
    RECOVERY_TYPES,
    SPLIT_TOTALS,
    build_schedule,
    schedule_summary,
)
from rby1_manipulation.data.episode import EpisodeBuffer, Frame, LeRobotWriter
from rby1_manipulation.data.finalize_atomic_dataset import split_ranges
from rby1_manipulation.data.recording import (
    POLICY_IMAGE_SIZE,
    POLICY_SOURCE_HEIGHT,
    POLICY_SOURCE_WIDTH,
    resize_policy_image,
)
from rby1_manipulation.evaluation.atomic_transport import aggregate_metrics
from rby1_manipulation.simulation.transport_scene import MODEL_XML
from rby1_manipulation.tasks.transport_atomic import (
    ATOMIC_SCHEMA_VERSION,
    _check_dataset_schema,
)


class AtomicTransportCollectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schedule = build_schedule()

    def test_2000_episode_family_split_and_target_balance(self):
        summary = schedule_summary(self.schedule)
        self.assertEqual(summary["total"], 2000)
        self.assertEqual(summary["families"], dict(sorted(FAMILY_TOTALS.items())))
        self.assertEqual(summary["splits"], dict(sorted(SPLIT_TOTALS.items())))
        self.assertEqual(summary["target_fruits"], {fruit: 480 for fruit in FRUITS})
        self.assertEqual(summary["target_slots"], {
            fruit: [120, 120, 120, 120] for fruit in FRUITS
        })
        self.assertEqual(summary["target_arms"], {
            fruit: {"left": 240, "right": 240} for fruit in FRUITS
        })
        self.assertLessEqual(
            max(summary["layouts"].values()) - min(summary["layouts"].values()), 2
        )

    def test_each_family_has_the_designed_split(self):
        actual = defaultdict(Counter)
        for spec in self.schedule:
            actual[spec.scenario_family][spec.split] += 1
        for family, expected in FAMILY_SPLITS.items():
            self.assertEqual(dict(actual[family]), expected)
        self.assertEqual(
            [spec.split for spec in self.schedule],
            ["train"] * 1600 + ["validation"] * 200 + ["test"] * 200,
        )

    def test_every_episode_is_atomic_and_linked_groups_do_not_leak(self):
        groups = defaultdict(list)
        for spec in self.schedule:
            if spec.task == "place_one":
                self.assertIsNotNone(spec.target_fruit)
                self.assertNotIn(spec.target_fruit, spec.preloaded)
                self.assertEqual(
                    spec.task_prompt,
                    f"put the {spec.target_fruit} in the basket",
                )
            else:
                self.assertEqual(spec.task, "lift_basket")
                self.assertIsNone(spec.target_fruit)
                self.assertEqual(spec.task_prompt, "lift the basket")
            if spec.sequence_group_id:
                groups[spec.sequence_group_id].append(spec)

        self.assertEqual(len(groups), 140)
        self.assertEqual(Counter(len(members) for members in groups.values()), {2: 100, 3: 40})
        for members in groups.values():
            members.sort(key=lambda spec: spec.sequence_step)
            self.assertEqual(len({spec.split for spec in members}), 1)
            self.assertEqual(len({spec.layout_index for spec in members}), 1)
            self.assertEqual(len({spec.slot_order for spec in members}), 1)
            prior = []
            for spec in members:
                self.assertEqual(list(spec.preloaded), prior)
                prior.append(spec.target_fruit)

    def test_preloaded_recovery_and_lift_balance(self):
        recovery = [spec for spec in self.schedule if spec.scenario_family == "recovery"]
        self.assertEqual(Counter(spec.recovery_type for spec in recovery), {
            recovery_type: 40 for recovery_type in RECOVERY_TYPES
        })
        lift = [spec for spec in self.schedule if spec.task == "lift_basket"]
        self.assertEqual(Counter(len(spec.preloaded) for spec in lift), {
            count: 16 for count in range(5)
        })
        self.assertEqual(Counter(
            fruit for spec in lift for fruit in spec.preloaded
        ), {fruit: 40 for fruit in FRUITS})
        preloaded = [
            spec for spec in self.schedule
            if spec.scenario_family == "preloaded_single"
        ]
        for fruit in FRUITS:
            selected = [spec for spec in preloaded if spec.target_fruit == fruit]
            self.assertEqual(Counter(len(spec.preloaded) for spec in selected), {
                1: 20, 2: 20, 3: 20
            })

    def test_paraphrases_are_an_optional_balanced_extension(self):
        schedule = build_schedule(include_paraphrases=True)
        extension = [spec for spec in schedule if spec.is_paraphrase]
        self.assertEqual(len(schedule), 2240)
        self.assertEqual(len(extension), 240)
        counts = Counter((spec.target_fruit, spec.task_prompt) for spec in extension)
        for fruit in FRUITS:
            for template in PARAPHRASES:
                self.assertEqual(counts[(fruit, template.format(fruit=fruit))], 20)

    def test_partial_dataset_split_ranges_follow_saved_episode_order(self):
        episodes = [
            {"episode_index": index, "split": split}
            for index, split in enumerate(
                ["train"] * 1591 + ["validation"] * 199 + ["test"] * 199
            )
        ]
        ranges, details = split_ranges(episodes)
        self.assertEqual(ranges, {
            "train": "0:1591",
            "validation": "1591:1790",
            "test": "1790:1989",
        })
        self.assertEqual(details["validation"]["episodes"], 199)

    def test_split_ranges_reject_reappearing_split(self):
        episodes = [
            {"episode_index": 0, "split": "train"},
            {"episode_index": 1, "split": "validation"},
            {"episode_index": 2, "split": "train"},
        ]
        with self.assertRaisesRegex(ValueError, "contiguous"):
            split_ranges(episodes)

    def test_frame_phase_metadata_is_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = LeRobotWriter(
                directory, fps=15, image_wh=(4, 4), schema="rby1_14",
                frame_metadata=True,
            )
            episode = writer.new_episode(task="put the apple in the basket")
            episode.append(Frame(
                state=np.zeros(14, dtype=np.float32),
                action=np.zeros(14, dtype=np.float32),
                images={},
                timestamp=0.0,
                frame_index=0,
                phase_index=12,
                prompt_timestamp=0.0,
            ))
            writer._write_parquet(episode)
            table = pq.read_table(Path(directory) / "data/chunk-000/episode_000000.parquet")
            self.assertIn("phase_index", table.column_names)
            self.assertIn("prompt_timestamp", table.column_names)
            self.assertEqual(table["phase_index"].to_pylist(), [12])
            self.assertIn("phase_index", writer._features_dict())

    def test_policy_camera_preserves_4_3_fov_before_square_resize(self):
        self.assertEqual(POLICY_SOURCE_HEIGHT, POLICY_IMAGE_SIZE)
        self.assertEqual(POLICY_SOURCE_WIDTH, 299)
        source = np.zeros((POLICY_SOURCE_HEIGHT, POLICY_SOURCE_WIDTH, 3), dtype=np.uint8)
        self.assertEqual(resize_policy_image(source).shape, (224, 224, 3))

    def test_transport_near_plane_does_not_clip_wrist_grasp(self):
        model = mujoco.MjModel.from_xml_path(MODEL_XML)
        self.assertLessEqual(model.stat.extent * model.vis.map.znear, 0.03)

    def test_old_atomic_schema_cannot_be_mixed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meta").mkdir()
            (root / "meta" / "atomic_schema.json").write_text(
                '{"version": 1}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, f"v{ATOMIC_SCHEMA_VERSION}"):
                _check_dataset_schema(root)

    def test_prompt_compliance_metrics_count_extra_fruit(self):
        good_validation = {
            "target_newly_inside": True,
            "fully_released": True,
            "non_target_unchanged": True,
            "non_target_inserted": False,
            "wrong_target_grasped": False,
            "safe_retreat": True,
            "returned_to_ready": True,
            "wrist_camera_view_valid": True,
            "terminal_hold_valid": True,
        }
        bad_validation = {**good_validation, "non_target_inserted": True}
        records = [
            {
                "task_type": "place_one", "success": True,
                "canonical_prompt": "put the apple in the basket",
                "validation": good_validation,
            },
            {
                "task_type": "place_one", "success": True,
                "canonical_prompt": "put the apple in the basket",
                "validation": bad_validation,
            },
        ]
        metrics = aggregate_metrics(records)
        self.assertEqual(metrics["prompt_compliance_rate"], 0.5)
        self.assertEqual(metrics["unnecessary_fruit_insertion_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
