"""Static obstacle profiles and inference-time safety for pick-and-place."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np

from rby1_manipulation.paths import PICK_PLACE_OBSTACLE_CONFIG
DEFAULT_CONFIG = PICK_PLACE_OBSTACLE_CONFIG
SLOTS = ("bollard_0", "bollard_1", "divider_0")
MOVABLE_OBJECT_BODIES = (
    "red_block", "green_block", "blue_block",
    "apple", "banana", "orange", "pear", "crate",
)
PARKING_POSITIONS = {
    "bollard_0": np.array([8.0, 8.0, 2.0]),
    "bollard_1": np.array([9.0, 8.0, 2.0]),
    "divider_0": np.array([10.0, 8.0, 2.0]),
}


def validate_pick_place_obstacle_config(raw: dict) -> dict:
    """Normalize a profile config and reject unsafe or ambiguous placements."""
    if not isinstance(raw, dict):
        raise ValueError("pick-place obstacle config must be a JSON object")
    config = copy.deepcopy(raw)
    if int(config.get("version", 1)) != 1:
        raise ValueError(f"unsupported obstacle config version {config.get('version')!r}")
    table_z = float(config.get("table_surface_z", 0.82))
    if not np.isfinite(table_z) or not 0.75 <= table_z <= 0.90:
        raise ValueError("table_surface_z must be finite and in [0.75, 0.90]")
    config["table_surface_z"] = table_z

    profiles = config.get("profiles")
    if not isinstance(profiles, dict) or "clear" not in profiles:
        raise ValueError("profiles must be a mapping containing 'clear'")
    if profiles["clear"].get("obstacles", []) != []:
        raise ValueError("the 'clear' profile must not contain obstacles")

    for profile_name, profile in profiles.items():
        scene = profile.get("scene", "any" if profile_name == "clear" else None)
        if scene not in ("any", "block", "fruit"):
            raise ValueError(
                f"profiles.{profile_name}.scene must be any, block, or fruit"
            )
        profile["scene"] = scene
        obstacles = profile.get("obstacles", [])
        if not isinstance(obstacles, list):
            raise ValueError(f"profiles.{profile_name}.obstacles must be a list")
        used: set[str] = set()
        for index, obstacle in enumerate(obstacles):
            label = f"profiles.{profile_name}.obstacles[{index}]"
            slot = obstacle.get("slot")
            if slot not in SLOTS:
                raise ValueError(f"{label}.slot={slot!r}; known slots: {SLOTS}")
            if slot in used:
                raise ValueError(f"{label}: slot {slot!r} is used twice")
            used.add(slot)
            position = np.asarray(obstacle.get("position"), dtype=float)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError(f"{label}.position must contain 3 finite values")
            if not 0.38 <= position[0] <= 0.85 or not -0.35 <= position[1] <= 0.35:
                raise ValueError(f"{label}.position lies outside the tabletop workspace")
            if abs(float(position[2]) - table_z) > 0.02:
                raise ValueError(f"{label}.position z must be the table surface {table_z:.3f}")
            obstacle["position"] = position.tolist()
            yaw = float(obstacle.get("yaw", 0.0))
            if not np.isfinite(yaw):
                raise ValueError(f"{label}.yaw must be finite")
            obstacle["yaw"] = yaw
    return config


def load_pick_place_obstacle_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return validate_pick_place_obstacle_config(json.load(stream))


def profile_names(config: dict | None = None) -> tuple[str, ...]:
    if config is None:
        config = load_pick_place_obstacle_config()
    return tuple(config["profiles"])


def _is_descendant(model: mujoco.MjModel, body_id: int, ancestor_id: int) -> bool:
    while body_id > 0:
        if body_id == ancestor_id:
            return True
        body_id = int(model.body_parentid[body_id])
    return False


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom#{geom_id}"


class PickPlaceObstacleManager:
    """Place fixed obstacle props and measure clearance from the RBY1 body."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: dict):
        self.model = model
        self.data = data
        self.config = validate_pick_place_obstacle_config(config)
        self._mocap_ids: dict[str, int] = {}
        self._collision_geoms: dict[str, tuple[int, ...]] = {}
        for slot in SLOTS:
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"pick_obstacle_{slot}"
            )
            geom_ids = tuple(
                geom_id
                for geom_id in range(model.ngeom)
                if int(model.geom_bodyid[geom_id]) == body_id
                and (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
            )
            if body_id < 0 or not geom_ids:
                raise RuntimeError(
                    "model is missing static pick-place obstacle slots; load "
                    "model_pick_place_obstacles.xml"
                )
            mocap_id = int(model.body_mocapid[body_id])
            if mocap_id < 0:
                raise RuntimeError(f"pick obstacle slot {slot!r} is not a mocap body")
            self._mocap_ids[slot] = mocap_id
            self._collision_geoms[slot] = geom_ids

        arm_root_ids = (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_left_arm_0"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_right_arm_0"),
        )
        self.robot_geom_ids = tuple(
            geom_id for geom_id in range(model.ngeom)
            if any(
                _is_descendant(model, int(model.geom_bodyid[geom_id]), root_id)
                for root_id in arm_root_ids
            )
            and (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
        )
        payload_body_ids = {
            body_id
            for body_name in MOVABLE_OBJECT_BODIES
            if (body_id := mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )) >= 0
        }
        self.payload_geom_ids = tuple(
            geom_id for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in payload_body_ids
            and (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id])
        )
        self.active_slots: tuple[str, ...] = ()
        self.profile_name = "clear"
        self.reset_metrics()
        self.park_all()

    def reset_metrics(self) -> None:
        self.contact_steps = 0
        self.robot_collision = False
        self.payload_collision = False
        self.min_robot_clearance = float("inf")
        self.contact_pairs: set[tuple[str, str]] = set()

    def park_all(self) -> None:
        self.active_slots = ()
        for slot, position in PARKING_POSITIONS.items():
            mocap_id = self._mocap_ids[slot]
            self.data.mocap_pos[mocap_id] = position
            self.data.mocap_quat[mocap_id] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(self.model, self.data)

    def activate(self, profile_name: str) -> None:
        profiles = self.config["profiles"]
        if profile_name not in profiles:
            raise ValueError(f"unknown obstacle profile {profile_name!r}; known: {tuple(profiles)}")
        self.park_all()
        self.reset_metrics()
        active: list[str] = []
        for obstacle in profiles[profile_name].get("obstacles", []):
            slot = obstacle["slot"]
            mocap_id = self._mocap_ids[slot]
            self.data.mocap_pos[mocap_id] = obstacle["position"]
            yaw = float(obstacle["yaw"])
            self.data.mocap_quat[mocap_id] = (
                math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)
            )
            active.append(slot)
        self.active_slots = tuple(active)
        self.profile_name = profile_name
        mujoco.mj_forward(self.model, self.data)

    @property
    def collision_geom_ids(self) -> set[int]:
        return {
            geom_id
            for slot in self.active_slots
            for geom_id in self._collision_geoms[slot]
        }

    def positions(self) -> dict[str, np.ndarray]:
        return {
            slot: self.data.mocap_pos[self._mocap_ids[slot]].copy()
            for slot in self.active_slots
        }

    def _minimum_distance(self, other_geom_ids: Iterable[int], max_distance: float) -> float:
        if not self.active_slots:
            return float("inf")
        minimum = float(max_distance)
        from_to = np.zeros(6, dtype=np.float64)
        for obstacle_geom in self.collision_geom_ids:
            for other_geom in other_geom_ids:
                distance = float(mujoco.mj_geomDistance(
                    self.model,
                    self.data,
                    obstacle_geom,
                    int(other_geom),
                    max_distance,
                    from_to,
                ))
                minimum = min(minimum, distance)
        return minimum

    def robot_clearance(self, max_distance: float = 0.25) -> float:
        clearance = self._minimum_distance(self.robot_geom_ids, max_distance)
        self.min_robot_clearance = min(self.min_robot_clearance, clearance)
        return clearance

    def object_clearance(self, max_distance: float = 0.25) -> float:
        return self._minimum_distance(self.payload_geom_ids, max_distance)

    def observe_contacts(self) -> bool:
        obstacle_geoms = self.collision_geom_ids
        if not obstacle_geoms:
            return False
        robot_geoms = set(self.robot_geom_ids)
        payload_geoms = set(self.payload_geom_ids)
        contacted = False
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if not pair & obstacle_geoms:
                continue
            if pair & robot_geoms:
                self.robot_collision = True
            elif pair & payload_geoms:
                self.payload_collision = True
            else:
                continue
            contacted = True
            self.contact_pairs.add(tuple(sorted(_geom_name(self.model, geom) for geom in pair)))
        if contacted:
            self.contact_steps += 1
        return contacted

    def unsafe(self, stop_distance: float) -> bool:
        self.observe_contacts()
        clearance = self.robot_clearance(max(0.25, stop_distance + 0.05))
        return self.robot_collision or clearance <= stop_distance

    def hold_robot(self) -> None:
        """Freeze every joint-position actuator at its measured joint position."""
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id >= 0:
                self.data.ctrl[actuator_id] = self.data.qpos[self.model.jnt_qposadr[joint_id]]

    def summary(self) -> dict:
        return {
            "profile": self.profile_name,
            "active_slots": list(self.active_slots),
            "robot_collision": self.robot_collision,
            "payload_collision": self.payload_collision,
            "contact_steps": self.contact_steps,
            "min_robot_clearance_m": self.min_robot_clearance,
            "contact_pairs": [list(pair) for pair in sorted(self.contact_pairs)],
        }
