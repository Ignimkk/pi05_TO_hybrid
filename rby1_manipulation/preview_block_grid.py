"""Preview, adjust, and persist the RBY1 block-placement evaluation grid.

This tool never loads or calls a policy.  It places a real MuJoCo block at one
grid point at a time, lets physics settle, and keeps a passive viewer open while
the operator inspects or edits the coordinates.

The saved JSON file is also consumed by ``rby1_bringup/pi05_infer.py
--grid-experiment``, so a coordinate edited here is the coordinate used by the
actual policy evaluation.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import tempfile
from typing import Callable

import mujoco
import mujoco.viewer
import numpy as np

from scene_utils import set_block_pose


SRC_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_XML = SRC_ROOT / "rby1_description" / "models" / "rby1a" / "mujoco" / "model.xml"
DEFAULT_GRID_CONFIG = pathlib.Path(__file__).with_name("block_grid.json")

COLORS = ("red", "green", "blue")
SIDES = ("left", "right")
BLOCK_BODIES = {color: f"{color}_block" for color in COLORS}
BLOCK_JOINTS = {color: f"{color}_block_free" for color in COLORS}

DEFAULT_GRID = {
    "version": 1,
    "spawn_z": 0.87,
    "positions": {
        "left": [
            [0.475, 0.178],
            [0.525, 0.178],
            [0.575, 0.178],
            [0.625, 0.178],
            [0.475, 0.235],
            [0.525, 0.235],
            [0.575, 0.235],
            [0.625, 0.235],
            [0.475, 0.292],
            [0.525, 0.292],
            [0.575, 0.292],
            [0.625, 0.292],
        ],
        "right": [
            [0.475, -0.178],
            [0.525, -0.178],
            [0.575, -0.178],
            [0.625, -0.178],
            [0.475, -0.235],
            [0.525, -0.235],
            [0.575, -0.235],
            [0.625, -0.235],
            [0.475, -0.292],
            [0.525, -0.292],
            [0.575, -0.292],
            [0.625, -0.292],
        ],
    },
    # The two non-target colors are placed on the side opposite the target.
    # Their fixed 10 cm center separation prevents contact between 5 cm cubes.
    "distractors": {
        "left": [[0.475, -0.292], [0.575, -0.292]],
        "right": [[0.475, 0.292], [0.575, 0.292]],
    },
}


def _validate_xy(xy, *, label: str, expected_side: str | None = None) -> list[float]:
    if not isinstance(xy, (list, tuple)) or len(xy) != 2:
        raise ValueError(f"{label} must be [x, y], got {xy!r}")
    x, y = (float(xy[0]), float(xy[1]))
    if not np.isfinite([x, y]).all():
        raise ValueError(f"{label} contains a non-finite coordinate")
    if not 0.40 <= x <= 0.65:
        raise ValueError(f"{label} x={x:.3f} is outside the safe preview range [0.40, 0.65]")
    if not -0.35 <= y <= 0.35:
        raise ValueError(f"{label} y={y:.3f} is outside the measured arm range [-0.35, 0.35]")
    if expected_side == "left" and y <= 0.05:
        raise ValueError(f"{label} must have positive y for the left side")
    if expected_side == "right" and y >= -0.05:
        raise ValueError(f"{label} must have negative y for the right side")
    return [x, y]


def validate_grid_config(raw: dict) -> dict:
    """Return a normalized copy of a grid config, raising on unsafe values."""
    if not isinstance(raw, dict):
        raise ValueError("grid config must be a JSON object")
    spawn_z = float(raw.get("spawn_z", 0.87))
    if not np.isfinite(spawn_z) or not 0.84 <= spawn_z <= 1.00:
        raise ValueError(f"spawn_z={spawn_z!r} must be in [0.84, 1.00]")

    positions_raw = raw.get("positions")
    distractors_raw = raw.get("distractors")
    if not isinstance(positions_raw, dict) or not isinstance(distractors_raw, dict):
        raise ValueError("grid config requires 'positions' and 'distractors' objects")

    positions: dict[str, list[list[float]]] = {}
    distractors: dict[str, list[list[float]]] = {}
    for side in SIDES:
        side_positions = positions_raw.get(side)
        if not isinstance(side_positions, list) or not side_positions:
            raise ValueError(f"positions.{side} must contain at least one [x, y] coordinate")
        positions[side] = [
            _validate_xy(xy, label=f"positions.{side}[{i}]", expected_side=side)
            for i, xy in enumerate(side_positions)
        ]

        side_distractors = distractors_raw.get(side)
        if not isinstance(side_distractors, list) or len(side_distractors) != 2:
            raise ValueError(f"distractors.{side} must contain exactly 2 [x, y] coordinates")
        # Distractors intentionally sit on the side opposite the target.
        opposite = "right" if side == "left" else "left"
        distractors[side] = [
            _validate_xy(xy, label=f"distractors.{side}[{i}]", expected_side=opposite)
            for i, xy in enumerate(side_distractors)
        ]
        separation = np.linalg.norm(
            np.asarray(distractors[side][0]) - np.asarray(distractors[side][1])
        )
        if separation < 0.075:
            raise ValueError(
                f"distractors.{side} are only {separation:.3f} m apart; require >= 0.075 m"
            )

    return {
        "version": int(raw.get("version", 1)),
        "spawn_z": spawn_z,
        "positions": positions,
        "distractors": distractors,
    }


def load_grid_config(path: str | pathlib.Path = DEFAULT_GRID_CONFIG) -> dict:
    path = pathlib.Path(path)
    if not path.exists():
        return validate_grid_config(copy.deepcopy(DEFAULT_GRID))
    with path.open(encoding="utf-8") as stream:
        return validate_grid_config(json.load(stream))


def save_grid_config(config: dict, path: str | pathlib.Path = DEFAULT_GRID_CONFIG) -> pathlib.Path:
    """Validate and atomically save a grid config."""
    path = pathlib.Path(path)
    normalized = validate_grid_config(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(normalized, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return path


def requested_xyz(config: dict, side: str, grid_index: int) -> np.ndarray:
    xy = config["positions"][side][grid_index]
    return np.asarray([xy[0], xy[1], config["spawn_z"]], dtype=np.float64)


def trial_block_placements(
    config: dict,
    *,
    color: str,
    side: str,
    grid_index: int,
) -> dict[str, np.ndarray]:
    """Return deterministic target and distractor xyz positions for one trial."""
    if color not in COLORS:
        raise ValueError(f"unknown color {color!r}")
    if side not in SIDES:
        raise ValueError(f"unknown side {side!r}")
    grid_size = len(config["positions"][side])
    if not 0 <= grid_index < grid_size:
        raise ValueError(f"grid_index must be in [0, {grid_size - 1}], got {grid_index}")

    placements = {color: requested_xyz(config, side, grid_index)}
    other_colors = [candidate for candidate in COLORS if candidate != color]
    for other_color, xy in zip(other_colors, config["distractors"][side]):
        placements[other_color] = np.asarray(
            [xy[0], xy[1], config["spawn_z"]],
            dtype=np.float64,
        )
    return placements


def body_position(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xpos[body_id].copy()


def reset_and_place_trial(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: dict,
    *,
    color: str,
    side: str,
    grid_index: int,
    settle_seconds: float = 1.5,
    on_step: Callable[[], None] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Reset teleop, place all blocks, settle, and return requested/actual xyz."""
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "teleop")
    if key_id < 0:
        raise RuntimeError("model has no 'teleop' keyframe")
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    placements = trial_block_placements(
        config,
        color=color,
        side=side,
        grid_index=grid_index,
    )
    for block_color, xyz in placements.items():
        set_block_pose(model, data, BLOCK_JOINTS[block_color], xyz)
    mujoco.mj_forward(model, data)

    # Hold every actuated joint at the teleop qpos during the settle period.
    for actuator_id in range(model.nu):
        joint_id = model.actuator_trnid[actuator_id, 0]
        qpos_id = model.jnt_qposadr[joint_id]
        data.ctrl[actuator_id] = data.qpos[qpos_id]

    settle_steps = int(round(settle_seconds / model.opt.timestep))
    for step in range(settle_steps):
        mujoco.mj_step(model, data)
        if on_step is not None and (step % 10 == 0 or step == settle_steps - 1):
            on_step()

    actual = {
        block_color: body_position(model, data, BLOCK_BODIES[block_color])
        for block_color in COLORS
    }
    return placements, actual


def print_grid(config: dict) -> None:
    print("\nConfigured grid coordinates (spawn x, y in meters)")
    for side in SIDES:
        print(f"\n{side.upper()}")
        for index, xy in enumerate(config["positions"][side], start=1):
            print(f"  g{index:02d}: x={xy[0]:.3f}, y={xy[1]:+.3f}")
    print(f"\nspawn_z={config['spawn_z']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_GRID_CONFIG)
    parser.add_argument("--color", choices=COLORS, default="red")
    parser.add_argument("--side", choices=SIDES, default="left")
    parser.add_argument("--grid-index", type=int, default=1)
    parser.add_argument("--settle-seconds", type=float, default=1.5)
    args = parser.parse_args()

    config = load_grid_config(args.config)
    grid_size = len(config["positions"][args.side])
    if not 1 <= args.grid_index <= grid_size:
        parser.error(f"--grid-index must be in [1, {grid_size}] for side {args.side}")
    color = args.color
    side = args.side
    grid_index = args.grid_index - 1
    dirty = False

    print_grid(config)
    print(
        "\nCommands:\n"
        "  Enter/n       next grid point       p              previous grid point\n"
        "  side l|r      select side           color r|g|b    select block color\n"
        "  set X Y       replace selected xy   move DX DY     offset selected xy\n"
        "  save          persist coordinates   print          print all coordinates\n"
        "  q             quit\n"
    )

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            requested, actual = reset_and_place_trial(
                model,
                data,
                config,
                color=color,
                side=side,
                grid_index=grid_index,
                settle_seconds=args.settle_seconds,
                on_step=viewer.sync,
            )
            requested_target = requested[color]
            actual_target = actual[color]
            print(
                f"\n[{color} {side} g{grid_index + 1:02d}]"
                f"{'  *unsaved*' if dirty else ''}\n"
                f"  requested: {requested_target.round(4).tolist()}\n"
                f"  settled  : {actual_target.round(4).tolist()}"
            )

            try:
                command = input("grid> ").strip()
            except (EOFError, KeyboardInterrupt):
                command = "q"
                print()
            parts = command.split()
            verb = parts[0].lower() if parts else "n"

            try:
                if verb in ("n", "next"):
                    grid_index = (grid_index + 1) % len(config["positions"][side])
                elif verb in ("p", "prev"):
                    grid_index = (grid_index - 1) % len(config["positions"][side])
                elif verb == "side" and len(parts) == 2:
                    side_token = parts[1].lower()
                    side = {"l": "left", "r": "right"}.get(side_token, side_token)
                    if side not in SIDES:
                        raise ValueError("side must be l/left or r/right")
                elif verb == "color" and len(parts) == 2:
                    color_token = parts[1].lower()
                    color = {"r": "red", "g": "green", "b": "blue"}.get(
                        color_token, color_token
                    )
                    if color not in COLORS:
                        raise ValueError("color must be r/red, g/green, or b/blue")
                elif verb in ("set", "move") and len(parts) == 3:
                    first, second = float(parts[1]), float(parts[2])
                    current = config["positions"][side][grid_index]
                    candidate = (
                        [first, second]
                        if verb == "set"
                        else [current[0] + first, current[1] + second]
                    )
                    config["positions"][side][grid_index] = _validate_xy(
                        candidate,
                        label=f"positions.{side}[{grid_index}]",
                        expected_side=side,
                    )
                    dirty = True
                elif verb == "save":
                    saved_path = save_grid_config(config, args.config)
                    dirty = False
                    print(f"saved grid config -> {saved_path}")
                elif verb == "print":
                    print_grid(config)
                elif verb in ("q", "quit", "exit"):
                    if dirty:
                        answer = input("Unsaved coordinate changes. Save before exit? [y/N] ").strip().lower()
                        if answer in ("y", "yes"):
                            saved_path = save_grid_config(config, args.config)
                            print(f"saved grid config -> {saved_path}")
                    break
                else:
                    print("unknown command; use n, p, side, color, set, move, save, print, or q")
            except ValueError as exc:
                print(f"invalid command: {exc}")


if __name__ == "__main__":
    main()
