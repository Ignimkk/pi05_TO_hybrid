"""Contracts for the balanced fixed-base transport dataset."""

from __future__ import annotations

import unittest

from rby1_manipulation.control.bimanual import BiWaypoint
from rby1_manipulation.data.collect_transport_dataset import (
    build_schedule,
    reduce_reference_schedule,
    schedule_summary,
)
from rby1_manipulation.simulation.fruit_grid import (
    layout_count,
    load_fruit_grid_config,
)
from rby1_manipulation.tasks.transport_pack_lift import (
    DEFAULT_SPEED_SCALE,
    scaled_waypoints,
)


class TransportCollectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schedule = build_schedule()
        cls.summary = schedule_summary(cls.schedule)

    def test_1200_episode_family_and_count_balance(self) -> None:
        self.assertEqual(len(self.schedule), 1200)
        self.assertEqual(
            self.summary["families"],
            {"pack_only": 400, "lift_only": 400, "pack_and_lift": 400},
        )
        self.assertEqual(
            self.summary["counts"]["pack_only"],
            {"1": 100, "2": 100, "3": 100, "4": 100},
        )
        self.assertEqual(
            self.summary["counts"]["lift_only"],
            {"0": 80, "1": 80, "2": 80, "3": 80, "4": 80},
        )
        self.assertEqual(
            self.summary["counts"]["pack_and_lift"],
            {"1": 100, "2": 100, "3": 100, "4": 100},
        )

    def test_fruit_and_grid_exposure_is_balanced(self) -> None:
        self.assertEqual(
            self.summary["fruit_exposure"]["pack_only"],
            {"apple": 250, "banana": 250, "orange": 250, "pear": 250},
        )
        self.assertEqual(
            self.summary["fruit_exposure"]["lift_only"],
            {"apple": 200, "banana": 200, "orange": 200, "pear": 200},
        )
        self.assertEqual(
            self.summary["fruit_exposure"]["pack_and_lift"],
            {"apple": 250, "banana": 250, "orange": 250, "pear": 250},
        )
        self.assertTrue(all(value == 75 for value in self.summary["layouts"].values()))
        self.assertTrue(all(
            counts == [300, 300, 300, 300]
            for counts in self.summary["slot_exposure"].values()
        ))

    def test_reference_recollection_halves_only_lift_family(self) -> None:
        reduced = reduce_reference_schedule(self.schedule, lift_only_episodes=200)
        summary = schedule_summary(reduced)
        self.assertEqual(len(reduced), 1000)
        self.assertEqual(
            summary["families"],
            {"pack_only": 400, "lift_only": 200, "pack_and_lift": 400},
        )
        self.assertEqual(
            summary["counts"]["lift_only"],
            {"0": 40, "1": 40, "2": 40, "3": 40, "4": 40},
        )
        self.assertEqual(
            summary["fruit_exposure"]["lift_only"],
            {"apple": 100, "banana": 100, "orange": 100, "pear": 100},
        )
        self.assertEqual(
            sorted(summary["layouts"].values()),
            [62] * 8 + [63] * 8,
        )
        self.assertTrue(all(
            counts == [250, 250, 250, 250]
            for counts in summary["slot_exposure"].values()
        ))

    def test_fruit_grid_has_16_safe_layouts(self) -> None:
        self.assertEqual(layout_count(load_fruit_grid_config()), 16)

    def test_default_timing_is_accelerated_without_mutating_input(self) -> None:
        original = BiWaypoint("test", duration=2.0, wait_after=0.5)
        accelerated = scaled_waypoints([original], DEFAULT_SPEED_SCALE)[0]
        self.assertEqual(DEFAULT_SPEED_SCALE, 1.25)
        self.assertAlmostEqual(accelerated.duration, 1.6)
        self.assertAlmostEqual(accelerated.wait_after, 0.4)
        self.assertEqual(original.duration, 2.0)
        self.assertEqual(original.wait_after, 0.5)

    def test_object_close_starts_without_a_long_post_approach_pause(self) -> None:
        original = BiWaypoint("obj_descend_trim", duration=1.0, wait_after=0.4)
        accelerated = scaled_waypoints(
            [original],
            DEFAULT_SPEED_SCALE,
            object_pre_close_hold_secs=0.25,
        )[0]
        self.assertAlmostEqual(accelerated.duration, 0.8)
        self.assertAlmostEqual(accelerated.wait_after, 0.25)


if __name__ == "__main__":
    unittest.main()
