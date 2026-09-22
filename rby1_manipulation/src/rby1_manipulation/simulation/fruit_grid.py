"""Deterministic fruit-grid placement for 14-D packing/lifting datasets."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import mujoco
import numpy as np

from rby1_manipulation.paths import FRUIT_GRID_CONFIG
from rby1_manipulation.simulation.common import TRANSPORT_SMALL_OBJ_REACH, set_block_pose
from rby1_manipulation.simulation.transport_scene import (
    OBJECT_JOINTS,
    OBJECT_TYPES,
    RandomizationSpec,
    SceneState,
    body_position,
    body_rotation,
    free_body_pose,
    reset_transport_scene,
)


DEFAULT_FRUIT_GRID_CONFIG = FRUIT_GRID_CONFIG
SIDES = ("left", "right")
MIN_TABLE_SEPARATION = 0.075
MIN_CRATE_SLOT_SEPARATION = 0.070


@dataclass
class FruitScene:
    state: SceneState
    requested_positions: dict[str, np.ndarray]
    actual_positions: dict[str, np.ndarray]
    preloaded_objects: tuple[str, ...]
    layout_index: int
    slot_order: tuple[str, ...]


def _xy(value, label: str) -> list[float]:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (2,) or not np.isfinite(arr).all():
        raise ValueError(f"{label} must be a finite [x, y] coordinate")
    return arr.tolist()


def validate_fruit_grid_config(raw: Mapping) -> dict:
    """Validate grid coordinates and return a normalized independent copy."""
    cfg = copy.deepcopy(dict(raw))
    if int(cfg.get("version", 1)) != 1:
        raise ValueError(f"unsupported fruit-grid version {cfg.get('version')!r}")

    positions: dict[str, list[list[float]]] = {}
    for side in SIDES:
        values = cfg.get("positions", {}).get(side)
        if not isinstance(values, list) or len(values) != 4:
            raise ValueError(f"positions.{side} must contain exactly four slots")
        positions[side] = []
        for index, value in enumerate(values):
            x, y = _xy(value, f"positions.{side}[{index}]")
            x_lo, x_hi = TRANSPORT_SMALL_OBJ_REACH["x"]
            y_lo, y_hi = TRANSPORT_SMALL_OBJ_REACH["y"]
            if not x_lo <= x <= x_hi:
                raise ValueError(f"positions.{side}[{index}] x={x:.3f} outside {(x_lo, x_hi)}")
            if not y_lo <= abs(y) <= y_hi:
                raise ValueError(f"positions.{side}[{index}] |y|={abs(y):.3f} outside {(y_lo, y_hi)}")
            if (side == "left" and y <= 0.0) or (side == "right" and y >= 0.0):
                raise ValueError(f"positions.{side}[{index}] has the wrong y sign")
            positions[side].append([x, y])

    pairs = cfg.get("slot_pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("slot_pairs must contain index pairs")
    normalized_pairs: list[list[int]] = []
    for pair_index, pair in enumerate(pairs):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"slot_pairs[{pair_index}] must be [i, j]")
        first, second = int(pair[0]), int(pair[1])
        if first == second or not 0 <= first < 4 or not 0 <= second < 4:
            raise ValueError(f"slot_pairs[{pair_index}] has invalid indices")
        for side in SIDES:
            distance = np.linalg.norm(
                np.asarray(positions[side][first]) - np.asarray(positions[side][second])
            )
            if distance < MIN_TABLE_SEPARATION:
                raise ValueError(
                    f"slot_pairs[{pair_index}] is only {distance:.3f} m apart on {side}"
                )
        normalized_pairs.append([first, second])

    crate_slots_raw = cfg.get("crate_slots")
    if not isinstance(crate_slots_raw, list) or len(crate_slots_raw) != 4:
        raise ValueError("crate_slots must contain exactly four local [x, y] slots")
    crate_slots = [_xy(value, f"crate_slots[{i}]") for i, value in enumerate(crate_slots_raw)]
    for index, (x, y) in enumerate(crate_slots):
        if abs(x) > 0.045 or abs(y) > 0.080:
            raise ValueError(f"crate_slots[{index}]={[x, y]} is too close to the crate wall")
    for i in range(4):
        for j in range(i + 1, 4):
            distance = np.linalg.norm(np.asarray(crate_slots[i]) - np.asarray(crate_slots[j]))
            if distance < MIN_CRATE_SLOT_SEPARATION:
                raise ValueError(f"crate_slots[{i}] and [{j}] are only {distance:.3f} m apart")

    return {
        "version": 1,
        "positions": positions,
        "slot_pairs": normalized_pairs,
        "crate_slots": crate_slots,
    }


def load_fruit_grid_config(path: str | Path = DEFAULT_FRUIT_GRID_CONFIG) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return validate_fruit_grid_config(json.load(stream))


def fruit_grid_fingerprint(config: Mapping) -> str:
    normalized = validate_fruit_grid_config(config)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def layout_count(config: Mapping) -> int:
    pairs = config["slot_pairs"]
    return len(pairs) * len(pairs)


def layout_positions(config: Mapping, layout_index: int) -> list[np.ndarray]:
    """Return two left slots followed by two right slots for one layout."""
    count = layout_count(config)
    if not 0 <= layout_index < count:
        raise ValueError(f"layout_index must be in [0, {count - 1}]")
    pair_count = len(config["slot_pairs"])
    left_pair = config["slot_pairs"][layout_index // pair_count]
    right_pair = config["slot_pairs"][layout_index % pair_count]
    return [
        *(np.asarray(config["positions"]["left"][index], dtype=float) for index in left_pair),
        *(np.asarray(config["positions"]["right"][index], dtype=float) for index in right_pair),
    ]


def table_placements(
    grid_config: Mapping,
    layout_config: Mapping,
    *,
    layout_index: int,
    slot_order: Sequence[str],
) -> dict[str, np.ndarray]:
    order = tuple(slot_order)
    if len(order) != len(OBJECT_TYPES) or set(order) != set(OBJECT_TYPES):
        raise ValueError(f"slot_order must be a permutation of {OBJECT_TYPES}")
    slots = layout_positions(grid_config, layout_index)
    return {
        fruit: np.array([*xy, float(layout_config["objects"][fruit]["spawn_z"])])
        for fruit, xy in zip(order, slots)
    }


def offset_table_placements_from_basket(
    placements: Mapping[str, np.ndarray],
    basket_xy: Sequence[float],
    *,
    offset_m: float,
    excluded_objects: Sequence[str] = (),
) -> dict[str, np.ndarray]:
    """Move table objects radially away from the basket by ``offset_m``.

    A new mapping and copied position arrays are returned so the configured training
    grid is never mutated. Objects already preloaded in the basket can be excluded.
    """
    if not np.isfinite(offset_m) or offset_m < 0.0:
        raise ValueError("offset_m must be a finite non-negative distance")
    center = np.asarray(basket_xy, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("basket_xy must be a finite [x, y] coordinate")

    excluded = set(excluded_objects)
    shifted: dict[str, np.ndarray] = {}
    for name, value in placements.items():
        position = np.asarray(value, dtype=float).copy()
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError(f"placement {name!r} must be a finite [x, y, z] coordinate")
        if offset_m and name not in excluded:
            direction = position[:2] - center
            distance = float(np.linalg.norm(direction))
            if distance <= np.finfo(float).eps:
                raise ValueError(f"cannot offset {name!r}: it is at the basket center")
            position[:2] += direction * (offset_m / distance)
        shifted[name] = position
    return shifted


def reset_fruit_grid_scene(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    layout_config: Mapping,
    grid_config: Mapping,
    *,
    layout_index: int,
    slot_order: Sequence[str],
    preloaded_objects: Sequence[str] = (),
    rng: np.random.Generator | None = None,
    randomize: RandomizationSpec | None = None,
    settle_seconds: float = 1.5,
    position_jitter_xy: float = 0.0,
    basket_clearance_offset: float = 0.0,
    object_position_overrides: Mapping[str, Sequence[float]] | None = None,
    object_yaw_overrides: Mapping[str, float] | None = None,
    initial_joint_qpos: Mapping[str, float] | None = None,
) -> FruitScene:
    """Reset, place four fruits on the grid/in crate, settle, and report poses."""
    order = tuple(slot_order)
    preloaded = tuple(preloaded_objects)
    if len(set(preloaded)) != len(preloaded) or not set(preloaded) <= set(OBJECT_TYPES):
        raise ValueError(f"preloaded_objects must be unique members of {OBJECT_TYPES}")

    if position_jitter_xy < 0.0:
        raise ValueError("position_jitter_xy must be non-negative")
    if not np.isfinite(basket_clearance_offset) or basket_clearance_offset < 0.0:
        raise ValueError("basket_clearance_offset must be a finite non-negative distance")
    rng = rng if rng is not None else np.random.default_rng()
    spec = randomize if randomize is not None else RandomizationSpec()
    spec.object_types = OBJECT_TYPES
    spec.object_pose = False
    state = reset_transport_scene(
        model,
        data,
        layout_config,
        rng=rng,
        randomize=spec,
        settle_seconds=0.0,
    )

    requested = table_placements(
        grid_config,
        layout_config,
        layout_index=layout_index,
        slot_order=order,
    )
    for fruit, value in (object_position_overrides or {}).items():
        if fruit not in OBJECT_TYPES:
            raise ValueError(f"unknown object position override {fruit!r}")
        position = np.asarray(value, dtype=float)
        if position.shape == (2,):
            requested[fruit][:2] = position
        elif position.shape == (3,):
            requested[fruit] = position.copy()
        else:
            raise ValueError(f"position override for {fruit} must be [x, y] or [x, y, z]")
    if position_jitter_xy:
        placed: list[np.ndarray] = []
        for fruit in order:
            nominal = requested[fruit].copy()
            candidate = nominal.copy()
            for _ in range(100):
                candidate[:2] = nominal[:2] + rng.uniform(
                    -position_jitter_xy, position_jitter_xy, size=2
                )
                x_lo, x_hi = TRANSPORT_SMALL_OBJ_REACH["x"]
                y_lo, y_hi = TRANSPORT_SMALL_OBJ_REACH["y"]
                candidate[0] = float(np.clip(candidate[0], x_lo, x_hi))
                sign = 1.0 if nominal[1] > 0.0 else -1.0
                candidate[1] = sign * float(np.clip(abs(candidate[1]), y_lo, y_hi))
                if fruit in preloaded or all(
                    np.linalg.norm(candidate[:2] - other[:2]) >= MIN_TABLE_SEPARATION
                    for other in placed
                ):
                    break
            else:
                raise RuntimeError("could not jitter fruit grid without overlap")
            requested[fruit] = candidate.copy()
            if fruit not in preloaded:
                placed.append(candidate.copy())
    crate_pos = body_position(model, data, "crate")
    crate_rot = body_rotation(model, data, "crate")
    crate_pose = free_body_pose(model, data, "crate_free")
    nominal_crate_z = float(layout_config["crate"]["spawn_z"])

    requested = offset_table_placements_from_basket(
        requested,
        crate_pos[:2],
        offset_m=basket_clearance_offset,
        excluded_objects=preloaded,
    )

    for preload_index, fruit in enumerate(preloaded):
        local_xy = np.asarray(grid_config["crate_slots"][preload_index], dtype=float)
        local_z = float(layout_config["objects"][fruit]["spawn_z"]) - nominal_crate_z
        requested[fruit] = crate_pos + crate_rot @ np.array([*local_xy, local_z])

    yaw_overrides = dict(object_yaw_overrides or {})
    unknown_yaws = set(yaw_overrides) - set(OBJECT_TYPES)
    if unknown_yaws:
        raise ValueError(f"unknown object yaw overrides: {sorted(unknown_yaws)}")
    for fruit in OBJECT_TYPES:
        if fruit in preloaded:
            quat = tuple(crate_pose[3:7])
        else:
            yaw = float(yaw_overrides.get(fruit, 0.0))
            quat = (float(np.cos(yaw / 2.0)), 0.0, 0.0, float(np.sin(yaw / 2.0)))
        set_block_pose(model, data, OBJECT_JOINTS[fruit], requested[fruit], quat)

    for joint_name, value in (initial_joint_qpos or {}).items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"unknown initial joint {joint_name!r}")
        if model.jnt_type[joint_id] not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            raise ValueError(f"initial joint override {joint_name!r} is not scalar")
        data.qpos[int(model.jnt_qposadr[joint_id])] = float(value)

    mujoco.mj_forward(model, data)
    for actuator_id in range(model.nu):
        joint_id = model.actuator_trnid[actuator_id, 0]
        data.ctrl[actuator_id] = data.qpos[model.jnt_qposadr[joint_id]]
    for _ in range(int(round(settle_seconds / model.opt.timestep))):
        mujoco.mj_step(model, data)

    actual = {fruit: body_position(model, data, fruit) for fruit in OBJECT_TYPES}
    state.crate_pose = free_body_pose(model, data, "crate_free")
    state.object_poses = {
        fruit: free_body_pose(model, data, OBJECT_JOINTS[fruit]) for fruit in OBJECT_TYPES
    }
    state.active_objects = OBJECT_TYPES
    return FruitScene(
        state=state,
        requested_positions=requested,
        actual_positions=actual,
        preloaded_objects=preloaded,
        layout_index=layout_index,
        slot_order=order,
    )
