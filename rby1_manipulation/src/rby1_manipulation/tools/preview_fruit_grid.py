"""Preview one deterministic fruit-grid scene without executing robot motion."""

from __future__ import annotations

import argparse
import time

import mujoco

from rby1_manipulation.paths import TRANSPORT_MODEL_XML
from rby1_manipulation.simulation.fruit_grid import (
    DEFAULT_FRUIT_GRID_CONFIG,
    fruit_grid_fingerprint,
    layout_count,
    load_fruit_grid_config,
    reset_fruit_grid_scene,
)
from rby1_manipulation.simulation.transport_scene import OBJECT_TYPES, load_layout_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-index", type=int, default=0)
    parser.add_argument("--slot-order", nargs=4, default=list(OBJECT_TYPES), choices=OBJECT_TYPES)
    parser.add_argument("--preloaded", nargs="*", default=[], choices=OBJECT_TYPES)
    parser.add_argument("--fruit-grid", default=str(DEFAULT_FRUIT_GRID_CONFIG))
    parser.add_argument("--config", default=None)
    parser.add_argument("--headless", action="store_true",
                        help="print settled positions and exit without a viewer")
    args = parser.parse_args()

    grid = load_fruit_grid_config(args.fruit_grid)
    layout = load_layout_config(args.config) if args.config else load_layout_config()
    if not 0 <= args.layout_index < layout_count(grid):
        parser.error(f"--layout-index must be in [0, {layout_count(grid) - 1}]")

    model = mujoco.MjModel.from_xml_path(str(TRANSPORT_MODEL_XML))
    data = mujoco.MjData(model)
    scene = reset_fruit_grid_scene(
        model,
        data,
        layout,
        grid,
        layout_index=args.layout_index,
        slot_order=args.slot_order,
        preloaded_objects=args.preloaded,
    )

    print(f"fruit grid {fruit_grid_fingerprint(grid)} layout={args.layout_index}")
    print(f"slot order={list(scene.slot_order)} preloaded={list(scene.preloaded_objects)}")
    for fruit in OBJECT_TYPES:
        requested = scene.requested_positions[fruit].round(3).tolist()
        actual = scene.actual_positions[fruit].round(3).tolist()
        print(f"  {fruit:7s} requested={requested} actual={actual}")

    if args.headless:
        return 0

    from mujoco import viewer as mj_viewer
    with mj_viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
