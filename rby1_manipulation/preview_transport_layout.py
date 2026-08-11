"""Interactive layout tuner for the transport scene.

The counterpart of preview_block_grid.py: opens the scene in a passive viewer,
prints what the geometry actually resolves to, and lets you nudge the crate,
shelf and dock pose and write the result back to transport_layout.json.

    python preview_transport_layout.py                 # viewer + report
    python preview_transport_layout.py --report-only   # no viewer

Commands (typed at the prompt while the viewer is open):
    crate X Y      move the crate
    shelf X Y      move the shelf
    dock X Y YAW   move the shelf dock pose
    level N        change the target shelf level
    reset          reload the scene from the config
    save           write the current config back to transport_layout.json
    quit
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from transport_scene import (
    BASE_FRONT_OVERHANG,
    CRATE_BODY,
    CRATE_HALF,
    DEFAULT_LAYOUT_CONFIG,
    MODEL_XML,
    SHELF_LEVEL_SITES,
    base_handles,
    body_position,
    handle_positions,
    load_layout_config,
    reset_transport_scene,
    save_layout_config,
    shelf_target_position,
)


def report(model, data, config, state) -> None:
    print("\n--- resolved layout ---")
    requested = np.array([*config["crate"]["xy"], config["crate"]["spawn_z"]])
    settled = body_position(model, data, CRATE_BODY)
    print(f"  crate requested {requested.round(4).tolist()}")
    print(f"  crate settled   {settled.round(4).tolist()}  "
          f"(dz {settled[2] - requested[2]:+.4f})  mass {state.crate_mass:.3f} kg")
    for side, pos in handle_positions(model, data).items():
        print(f"  handle {side:5s}     {pos.round(4).tolist()}")
    for name in state.object_poses:
        print(f"  {name:12s}    {body_position(model, data, name).round(4).tolist()}")

    print(f"  shelf body      {state.shelf_pos.round(4).tolist()}")
    for i, site in enumerate(SHELF_LEVEL_SITES):
        marker = " <- target" if i == state.target_level else ""
        print(f"    level {i}       {shelf_target_position(model, data, i).round(4).tolist()}{marker}")

    dock = state.dock_pose
    shelf_front_y = state.shelf_pos[1] + config["shelf"]["half_depth"]
    base_front_y = dock[1] - BASE_FRONT_OVERHANG
    print(f"  dock            x={dock[0]:.3f} y={dock[1]:.3f} yaw={dock[2]:.4f}")
    print(f"  base front y    {base_front_y:.3f}  vs shelf front {shelf_front_y:.3f}  "
          f"-> clearance {base_front_y - shelf_front_y:.3f} m")

    target = shelf_target_position(model, data, state.target_level)
    centre = target + np.array([0.0, 0.0, CRATE_HALF[2]])
    reach = float(np.hypot(centre[0] - dock[0], centre[1] - dock[1]))
    print(f"  crate centre on shelf {centre.round(4).tolist()}  "
          f"-> {reach:.3f} m from the dock at z={centre[2]:.3f}")
    print("  (run `transport_scene.py --reach-report` to confirm the arms can "
          "actually hit these poses)")


def apply_command(config: dict, line: str) -> bool:
    """Mutate `config` in place. Returns True if the scene must be rebuilt."""
    parts = line.split()
    if not parts:
        return False
    cmd, args = parts[0], parts[1:]
    if cmd == "crate" and len(args) == 2:
        config["crate"]["xy"] = [float(args[0]), float(args[1])]
    elif cmd == "shelf" and len(args) == 2:
        config["shelf"]["xy"] = [float(args[0]), float(args[1])]
    elif cmd == "dock" and len(args) == 3:
        config["dock"]["shelf"] = [float(a) for a in args]
    elif cmd == "level" and len(args) == 1:
        config["shelf"]["target_level"] = int(args[0])
    elif cmd == "reset":
        pass
    else:
        print(f"  ? {line!r} - see the module docstring for the command list")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(DEFAULT_LAYOUT_CONFIG))
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    config = load_layout_config(args.config)
    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data = mujoco.MjData(model)
    state = reset_transport_scene(model, data, config)
    report(model, data, config, state)

    if args.report_only:
        return 0

    from mujoco import viewer as mj_viewer
    with mj_viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        while viewer.is_running():
            try:
                line = input("layout> ").strip()
            except EOFError:
                break
            if line in ("quit", "q", "exit"):
                break
            if line == "save":
                try:
                    path = save_layout_config(config, args.config)
                except ValueError as exc:
                    print(f"  refused: {exc}")
                else:
                    print(f"  saved -> {path}")
                continue
            if apply_command(config, line):
                try:
                    state = reset_transport_scene(model, data, config)
                except (ValueError, KeyError) as exc:
                    print(f"  invalid layout: {exc}")
                    continue
                report(model, data, config, state)
                viewer.sync()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
