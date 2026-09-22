"""Config-driven randomized fruit-to-basket scenes for 16-D demonstrations."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import mujoco
import numpy as np

from rby1_manipulation.control.bimanual import grasp_site_target, plan_error
from rby1_manipulation.control.ik import (
    GRIPPER_L_JOINT,
    GRIPPER_OPEN,
    GRIPPER_R_JOINT,
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    build_dof_mask,
    left_arm_handles,
    right_arm_handles,
)
from rby1_manipulation.paths import RANDOMIZED_PICK_PLACE_CONFIG
from rby1_manipulation.planning.transport import (
    OBJECT_RELEASE_DZ,
    capture_object_grasp_frames,
    object_grasp_dz,
)
from rby1_manipulation.simulation.common import TRANSPORT_SMALL_OBJ_REACH, pick_arm_for_block
from rby1_manipulation.simulation.fruit_grid import (
    reset_fruit_grid_scene,
    table_placements,
)
from rby1_manipulation.simulation.transport_scene import (
    CRATE_BODY,
    CRATE_HALF,
    CRATE_JOINT,
    MODEL_XML,
    OBJECT_BODIES,
    OBJECT_JOINTS,
    OBJECT_TYPES,
    RandomizationSpec,
    TABLE_X_RANGE,
    TABLE_Y_RANGE,
    build_state_16,
    free_body_pose,
)


@dataclass(frozen=True)
class ToggleRange:
    enabled: bool
    values: tuple[float, float]


@dataclass(frozen=True)
class LanguageConfig:
    enabled: bool
    templates: tuple[str, ...]


@dataclass(frozen=True)
class RandomizedPickPlaceConfig:
    version: int
    max_sampling_attempts: int
    settle_seconds: float
    min_object_clearance_m: float
    max_settle_speed_mps: float
    max_ik_position_error_m: float
    ready_joint_offsets: dict[str, tuple[float, ...]]
    target_position: ToggleRange
    goal_position: ToggleRange
    target_orientation: ToggleRange
    robot_initial_configuration: ToggleRange
    language_instruction: LanguageConfig


@dataclass
class RandomizedScene:
    target_fruit: str
    used_arm: str
    requested_target_pose: np.ndarray
    settled_target_pose: np.ndarray
    requested_goal_pose: np.ndarray
    settled_goal_pose: np.ndarray
    initial_robot_state: np.ndarray
    initial_arm_joint_qpos: dict[str, float]
    target_yaw_rad: float
    sampling_attempts: int
    rejected_reasons: dict[str, int]
    validity: dict

    def to_dict(self) -> dict:
        value = asdict(self)
        for key in (
            "requested_target_pose",
            "settled_target_pose",
            "requested_goal_pose",
            "settled_goal_pose",
            "initial_robot_state",
        ):
            value[key] = np.asarray(value[key]).round(8).tolist()
        return value


def _toggle(raw: Mapping, key: str, range_key: str, *, degrees: bool = False) -> ToggleRange:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    values = np.asarray(value.get(range_key), dtype=float)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError(f"{key}.{range_key} must contain two finite values")
    if range_key.endswith("jitter_xy_m") and np.any(values < 0.0):
        raise ValueError(f"{key}.{range_key} must be non-negative")
    if degrees:
        values = np.deg2rad(values)
    if values[0] > values[1] and not range_key.endswith("jitter_xy_m"):
        raise ValueError(f"{key}.{range_key} must be ordered")
    return ToggleRange(bool(value.get("enabled", False)), tuple(float(x) for x in values))


def load_randomization_config(
    path: str | Path = RANDOMIZED_PICK_PLACE_CONFIG,
) -> RandomizedPickPlaceConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(raw.get("version", -1)) != 1:
        raise ValueError(f"unsupported randomized pick-place config version {raw.get('version')!r}")
    language = raw.get("language_instruction")
    if not isinstance(language, Mapping):
        raise ValueError("language_instruction must be an object")
    templates = tuple(str(value) for value in language.get("templates", ()))
    if not templates or any("{target}" not in value for value in templates):
        raise ValueError("every language template must contain {target}")
    ready = raw.get("ready_configuration", {})
    if not isinstance(ready, Mapping):
        raise ValueError("ready_configuration must be an object")
    ready_joint_offsets: dict[str, tuple[float, ...]] = {}
    for side in ("left", "right"):
        key = f"{side}_arm_joint_offsets_deg"
        values = np.asarray(ready.get(key, [0.0] * 7), dtype=float)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError(f"ready_configuration.{key} must contain seven finite values")
        ready_joint_offsets[side] = tuple(float(value) for value in np.deg2rad(values))

    config = RandomizedPickPlaceConfig(
        version=1,
        max_sampling_attempts=int(raw.get("max_sampling_attempts", 100)),
        settle_seconds=float(raw.get("settle_seconds", 1.5)),
        min_object_clearance_m=float(raw.get("min_object_clearance_m", 0.075)),
        max_settle_speed_mps=float(raw.get("max_settle_speed_mps", 0.03)),
        max_ik_position_error_m=float(raw.get("max_ik_position_error_m", 0.015)),
        ready_joint_offsets=ready_joint_offsets,
        target_position=_toggle(raw, "target_position", "jitter_xy_m"),
        goal_position=_toggle(raw, "goal_position", "jitter_xy_m"),
        target_orientation=_toggle(raw, "target_orientation", "yaw_range_deg", degrees=True),
        robot_initial_configuration=_toggle(
            raw, "robot_initial_configuration", "joint_noise_deg", degrees=True
        ),
        language_instruction=LanguageConfig(
            bool(language.get("enabled", False)), templates
        ),
    )
    if config.max_sampling_attempts <= 0:
        raise ValueError("max_sampling_attempts must be positive")
    for name in (
        "settle_seconds", "min_object_clearance_m", "max_settle_speed_mps",
        "max_ik_position_error_m",
    ):
        if getattr(config, name) <= 0.0:
            raise ValueError(f"{name} must be positive")
    return config


def randomization_config_fingerprint(config: RandomizedPickPlaceConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def canonical_prompt(target: str) -> str:
    return f"Put the {target} in the basket."


def prompts_for_target(config: RandomizedPickPlaceConfig, target: str) -> tuple[str, ...]:
    if not config.language_instruction.enabled:
        return (canonical_prompt(target),)
    return tuple(template.format(target=target) for template in config.language_instruction.templates)


def ready_arm_joint_qpos(
    model: mujoco.MjModel,
    config: RandomizedPickPlaceConfig,
) -> dict[str, float]:
    """Return the collector-specific nominal ready pose before episode noise."""
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "teleop")
    if key_id < 0:
        raise RuntimeError("model has no 'teleop' keyframe")
    key_qpos = np.asarray(model.key_qpos[key_id], dtype=float)
    values: dict[str, float] = {}
    for side, joint_names in (("left", LEFT_ARM_JOINTS), ("right", RIGHT_ARM_JOINTS)):
        offsets = config.ready_joint_offsets[side]
        for index, joint_name in enumerate(joint_names):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qidx = int(model.jnt_qposadr[joint_id])
            values[joint_name] = float(key_qpos[qidx] + offsets[index])
    return values


def _subtree(model: mujoco.MjModel, root_name: str) -> set[int]:
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
    if root < 0:
        raise KeyError(root_name)
    result = {root}
    for body_id in range(root + 1, model.nbody):
        if int(model.body_parentid[body_id]) in result:
            result.add(body_id)
    return result


def _contact_pairs(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[list[list[str]], list[list[str]]]:
    arm_bodies = _subtree(model, "link_left_arm_0") | _subtree(model, "link_right_arm_0")
    prop_roots = {CRATE_BODY: _subtree(model, CRATE_BODY)}
    prop_roots.update({name: _subtree(model, body) for name, body in OBJECT_BODIES.items()})
    arm_pairs: set[tuple[str, str]] = set()
    prop_pairs: set[tuple[str, str]] = set()
    for index in range(data.ncon):
        contact = data.contact[index]
        geom_ids = (int(contact.geom1), int(contact.geom2))
        body_ids = tuple(int(model.geom_bodyid[geom]) for geom in geom_ids)
        names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or str(geom)
            for geom in geom_ids
        )
        if body_ids[0] in arm_bodies or body_ids[1] in arm_bodies:
            arm_pairs.add(tuple(sorted(names)))
        labels = []
        for body_id in body_ids:
            labels.append(next((name for name, bodies in prop_roots.items() if body_id in bodies), None))
        if labels[0] is not None and labels[1] is not None and labels[0] != labels[1]:
            prop_pairs.add(tuple(sorted((labels[0], labels[1]))))
    return [list(pair) for pair in sorted(arm_pairs)], [list(pair) for pair in sorted(prop_pairs)]


def _free_speed(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str) -> float:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    dof = int(model.jnt_dofadr[joint_id])
    return float(np.linalg.norm(data.qvel[dof:dof + 3]))


def _ik_errors(model: mujoco.MjModel, data: mujoco.MjData, target: str, grid: Mapping) -> dict[str, float]:
    target_body = OBJECT_BODIES[target]
    target_pos = np.asarray(data.xpos[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body)
    ], dtype=float)
    crate_pos = np.asarray(data.xpos[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CRATE_BODY)
    ], dtype=float)
    arm_side = pick_arm_for_block(target_pos)
    arm = left_arm_handles(model) if arm_side == "left" else right_arm_handles(model)
    joints = LEFT_ARM_JOINTS if arm_side == "left" else RIGHT_ARM_JOINTS
    mask = build_dof_mask(model, joints)
    frames = capture_object_grasp_frames(model, data, target_body)
    rotation = frames.left if arm_side == "left" else frames.right
    grasp_dz = object_grasp_dz(model, target_body)
    slot = np.asarray(grid["crate_slots"][0], dtype=float)
    rim_z = crate_pos[2] + CRATE_HALF[2]
    points = {
        "pre_grasp": grasp_site_target(target_pos + np.array([0.0, 0.0, 0.10]), rotation),
        "grasp": grasp_site_target(target_pos + np.array([0.0, 0.0, grasp_dz]), rotation),
        "lift": grasp_site_target(target_pos + np.array([0.0, 0.0, 0.24]), rotation),
        "pre_place": grasp_site_target(
            np.array([crate_pos[0] + slot[0], crate_pos[1] + slot[1], rim_z + 0.20]), rotation
        ),
        "place": grasp_site_target(
            np.array([crate_pos[0] + slot[0], crate_pos[1] + slot[1], rim_z + OBJECT_RELEASE_DZ]),
            rotation,
        ),
        "retreat": grasp_site_target(
            np.array([crate_pos[0] + slot[0], crate_pos[1] + slot[1], rim_z + 0.21]), rotation
        ),
    }
    return {
        label: plan_error(model, data, arm, point, mask, rotation)
        for label, point in points.items()
    }


def _candidate_values(
    model: mujoco.MjModel,
    config: RandomizedPickPlaceConfig,
    layout: Mapping,
    grid: Mapping,
    target: str,
    layout_index: int,
    slot_order: Sequence[str],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
    nominal = table_placements(grid, layout, layout_index=layout_index, slot_order=slot_order)
    target_xyz = nominal[target].copy()
    if config.target_position.enabled:
        target_xyz[:2] += rng.uniform(
            -np.asarray(config.target_position.values), np.asarray(config.target_position.values)
        )
    goal_xy = np.asarray(layout["crate"]["xy"], dtype=float).copy()
    if config.goal_position.enabled:
        goal_xy += rng.uniform(
            -np.asarray(config.goal_position.values), np.asarray(config.goal_position.values)
        )
    yaw = 0.0
    if config.target_orientation.enabled:
        yaw = float(rng.uniform(*config.target_orientation.values))

    initial_joints = ready_arm_joint_qpos(model, config)
    for joint_name in initial_joints:
        value = initial_joints[joint_name]
        if config.robot_initial_configuration.enabled:
            value += float(rng.uniform(*config.robot_initial_configuration.values))
        initial_joints[joint_name] = value
    return target_xyz, goal_xy, yaw, initial_joints


def _precheck(
    model: mujoco.MjModel,
    config: RandomizedPickPlaceConfig,
    layout: Mapping,
    grid: Mapping,
    target: str,
    layout_index: int,
    slot_order: Sequence[str],
    target_xyz: np.ndarray,
    goal_xy: np.ndarray,
    initial_joints: Mapping[str, float],
) -> list[str]:
    reasons: list[str] = []
    x_bounds, y_bounds = TRANSPORT_SMALL_OBJ_REACH["x"], TRANSPORT_SMALL_OBJ_REACH["y"]
    if not x_bounds[0] <= target_xyz[0] <= x_bounds[1] or not y_bounds[0] <= abs(target_xyz[1]) <= y_bounds[1]:
        reasons.append("target_outside_workspace")
    if not 0.44 <= goal_xy[0] <= 0.56 or abs(goal_xy[1]) > 0.05:
        reasons.append("goal_outside_workspace")
    if (
        goal_xy[0] - CRATE_HALF[0] < TABLE_X_RANGE[0]
        or goal_xy[0] + CRATE_HALF[0] > TABLE_X_RANGE[1]
        or goal_xy[1] - CRATE_HALF[1] < TABLE_Y_RANGE[0]
        or goal_xy[1] + CRATE_HALF[1] > TABLE_Y_RANGE[1]
    ):
        reasons.append("goal_outside_table")
    nominal = table_placements(grid, layout, layout_index=layout_index, slot_order=slot_order)
    for fruit, position in nominal.items():
        if fruit != target and np.linalg.norm(target_xyz[:2] - position[:2]) < config.min_object_clearance_m:
            reasons.append("target_distractor_overlap")
            break
    nearest_x = max(abs(float(target_xyz[0] - goal_xy[0])) - CRATE_HALF[0], 0.0)
    nearest_y = max(abs(float(target_xyz[1] - goal_xy[1])) - CRATE_HALF[1], 0.0)
    if math.hypot(nearest_x, nearest_y) < config.min_object_clearance_m / 2.0:
        reasons.append("target_goal_overlap")
    for joint_name, value in initial_joints.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if model.jnt_limited[joint_id]:
            lo, hi = model.jnt_range[joint_id]
            if not float(lo) <= value <= float(hi):
                reasons.append("joint_limit")
                break
    return sorted(set(reasons))


def sample_valid_scene(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    layout: Mapping,
    grid: Mapping,
    config: RandomizedPickPlaceConfig,
    *,
    target: str,
    layout_index: int,
    slot_order: Sequence[str],
    rng: np.random.Generator,
) -> RandomizedScene:
    if target not in OBJECT_TYPES:
        raise ValueError(f"unknown target fruit {target!r}")
    rejected: Counter[str] = Counter()
    left, right = left_arm_handles(model), right_arm_handles(model)
    for attempt in range(1, config.max_sampling_attempts + 1):
        target_xyz, goal_xy, yaw, initial_joints = _candidate_values(
            model, config, layout, grid, target, layout_index, slot_order, rng
        )
        reasons = _precheck(
            model, config, layout, grid, target, layout_index, slot_order,
            target_xyz, goal_xy, initial_joints,
        )
        if reasons:
            rejected.update(reasons)
            continue

        sampled_layout = copy.deepcopy(dict(layout))
        sampled_layout["crate"] = dict(sampled_layout["crate"])
        sampled_layout["crate"]["xy"] = goal_xy.tolist()
        reset_fruit_grid_scene(
            model,
            data,
            sampled_layout,
            grid,
            layout_index=layout_index,
            slot_order=slot_order,
            rng=rng,
            randomize=RandomizationSpec(),
            settle_seconds=config.settle_seconds,
            object_position_overrides={target: target_xyz},
            object_yaw_overrides={target: yaw},
            initial_joint_qpos={
                **initial_joints,
                GRIPPER_L_JOINT: GRIPPER_OPEN,
                GRIPPER_R_JOINT: GRIPPER_OPEN,
            },
        )
        arm_pairs, prop_pairs = _contact_pairs(model, data)
        target_speed = _free_speed(model, data, OBJECT_JOINTS[target])
        goal_speed = _free_speed(model, data, CRATE_JOINT)
        if arm_pairs:
            rejected["initial_arm_collision"] += 1
            continue
        if prop_pairs:
            rejected["initial_prop_collision"] += 1
            continue
        if max(target_speed, goal_speed) > config.max_settle_speed_mps:
            rejected["scene_not_settled"] += 1
            continue
        ik_errors = _ik_errors(model, data, target, grid)
        max_ik_error = max(ik_errors.values())
        if max_ik_error > config.max_ik_position_error_m:
            rejected["ik_unreachable"] += 1
            continue

        target_pose = free_body_pose(model, data, OBJECT_JOINTS[target])
        goal_pose = free_body_pose(model, data, CRATE_JOINT)
        requested_target = np.array([
            *target_xyz,
            math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0),
        ])
        requested_goal = np.array([
            goal_xy[0], goal_xy[1], float(layout["crate"]["spawn_z"]),
            1.0, 0.0, 0.0, 0.0,
        ])
        used_arm = pick_arm_for_block(target_pose[:3])
        return RandomizedScene(
            target_fruit=target,
            used_arm=used_arm,
            requested_target_pose=requested_target,
            settled_target_pose=target_pose,
            requested_goal_pose=requested_goal,
            settled_goal_pose=goal_pose,
            initial_robot_state=build_state_16(data, left, right),
            initial_arm_joint_qpos={
                name: float(data.qpos[int(model.jnt_qposadr[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ])])
                for name in (*LEFT_ARM_JOINTS, *RIGHT_ARM_JOINTS)
            },
            target_yaw_rad=yaw,
            sampling_attempts=attempt,
            rejected_reasons=dict(sorted(rejected.items())),
            validity={
                "valid": True,
                "arm_collision_pairs": arm_pairs,
                "prop_collision_pairs": prop_pairs,
                "target_speed_mps": target_speed,
                "goal_speed_mps": goal_speed,
                "ik_position_errors_m": ik_errors,
                "max_ik_position_error_m": max_ik_error,
            },
        )
    raise RuntimeError(
        "could not sample a valid randomized scene after "
        f"{config.max_sampling_attempts} attempts: {dict(rejected)}"
    )


def load_default_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    return model, mujoco.MjData(model)


__all__ = [
    "RandomizedPickPlaceConfig",
    "RandomizedScene",
    "canonical_prompt",
    "load_default_model",
    "load_randomization_config",
    "prompts_for_target",
    "ready_arm_joint_qpos",
    "randomization_config_fingerprint",
    "sample_valid_scene",
]
