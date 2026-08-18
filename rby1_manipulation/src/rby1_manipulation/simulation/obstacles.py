"""Evaluation-only static and dynamic obstacles for the transport scene.

Obstacle bodies are precompiled mocap slots.  Activating a profile changes only
their pose, primitive size, and color; it never changes nq/nu or the 14-D
training schema.  Dynamic motion is deterministic as a function of elapsed
simulation time, which makes collision-avoidance comparisons reproducible.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import mujoco
import numpy as np

from rby1_manipulation.paths import TRANSPORT_OBSTACLE_CONFIG
from rby1_manipulation.simulation.transport_scene import CRATE_BODY, OBJECT_BODIES

DEFAULT_OBSTACLE_CONFIG = TRANSPORT_OBSTACLE_CONFIG
PARKING_POSITIONS = {
    "static_box_0": np.array([8.0, 8.0, 2.0]),
    "static_box_1": np.array([9.0, 8.0, 2.0]),
    "static_column_0": np.array([10.0, 8.0, 2.0]),
    "dynamic_human_0": np.array([11.0, 8.0, 2.0]),
    "dynamic_cart_0": np.array([12.0, 8.0, 2.0]),
}
SLOT_SPECS = {
    "static_box_0": ("box", np.array([0.40, 0.40, 0.50])),
    "static_box_1": ("box", np.array([0.40, 0.40, 0.50])),
    "static_column_0": ("cylinder", np.array([0.35, 0.80])),
    "dynamic_human_0": ("capsule", np.array([0.25, 0.75])),
    "dynamic_cart_0": ("box", np.array([0.50, 0.35, 0.50])),
}


def _vector(value, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{label}: expected {size} finite values, got {value!r}")
    return result


def validate_obstacle_config(raw: dict) -> dict:
    config = copy.deepcopy(raw)
    if int(config.get("version", 1)) != 1:
        raise ValueError(f"unsupported obstacle config version {config.get('version')!r}")
    profiles = config.get("profiles")
    if not isinstance(profiles, dict) or "clear" not in profiles:
        raise ValueError("obstacle config needs a profiles mapping containing 'clear'")

    for profile_name, profile in profiles.items():
        obstacles = profile.get("obstacles", [])
        used_slots: set[str] = set()
        for index, obstacle in enumerate(obstacles):
            label = f"profiles.{profile_name}.obstacles[{index}]"
            slot = obstacle.get("slot")
            if slot not in SLOT_SPECS:
                raise ValueError(f"{label}.slot={slot!r}; known slots: {sorted(SLOT_SPECS)}")
            if slot in used_slots:
                raise ValueError(f"{label}: slot {slot!r} is used twice")
            used_slots.add(slot)

            expected_kind = "dynamic" if slot.startswith("dynamic_") else "static"
            if obstacle.get("kind") != expected_kind:
                raise ValueError(f"{label}.kind must be {expected_kind!r}")
            obstacle["position"] = _vector(obstacle["position"], 3, f"{label}.position").tolist()
            shape, max_size = SLOT_SPECS[slot]
            size = _vector(obstacle["size"], len(max_size), f"{label}.size")
            if np.any(size <= 0.0) or np.any(size > max_size):
                raise ValueError(
                    f"{label}.size must be positive and <= compiled {shape} maximum "
                    f"{max_size.tolist()}"
                )
            obstacle["size"] = size.tolist()
            obstacle["yaw"] = float(obstacle.get("yaw", 0.0))
            rgba = _vector(obstacle.get("rgba", [0.8, 0.3, 0.1, 1.0]), 4, f"{label}.rgba")
            if np.any(rgba < 0.0) or np.any(rgba > 1.0):
                raise ValueError(f"{label}.rgba must stay in [0, 1]")
            obstacle["rgba"] = rgba.tolist()

            motion = obstacle.get("motion")
            if expected_kind == "static" and motion is not None:
                raise ValueError(f"{label}: static obstacle cannot have motion")
            if expected_kind == "dynamic":
                if not isinstance(motion, dict) or motion.get("type") != "line":
                    raise ValueError(f"{label}.motion must be a line motion")
                if motion.get("mode", "ping_pong") not in ("ping_pong", "loop"):
                    raise ValueError(f"{label}.motion.mode must be ping_pong or loop")
                motion["start"] = _vector(motion["start"], 3, f"{label}.motion.start").tolist()
                motion["end"] = _vector(motion["end"], 3, f"{label}.motion.end").tolist()
                if np.linalg.norm(np.subtract(motion["end"], motion["start"])) < 1e-6:
                    raise ValueError(f"{label}.motion path has zero length")
                motion["speed"] = float(motion["speed"])
                motion["start_delay"] = float(motion.get("start_delay", 0.0))
                motion["phase"] = float(motion.get("phase", 0.0)) % 1.0
                if motion["speed"] <= 0.0 or motion["start_delay"] < 0.0:
                    raise ValueError(f"{label}.motion speed/delay is invalid")
    return config


def load_obstacle_config(path: str | Path = DEFAULT_OBSTACLE_CONFIG) -> dict:
    with open(path, encoding="utf-8") as stream:
        return validate_obstacle_config(json.load(stream))


def profile_names(config: dict) -> tuple[str, ...]:
    return tuple(config["profiles"])


@dataclass(frozen=True)
class ActiveObstacle:
    slot: str
    kind: str
    shape: str
    size: np.ndarray
    motion: dict | None

    @property
    def planar_radius(self) -> float:
        if self.shape == "box":
            return float(np.hypot(self.size[0], self.size[1]))
        return float(self.size[0])


class TransportObstacleManager:
    """Place and animate precompiled obstacle slots."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: dict):
        self.model = model
        self.data = data
        self.config = validate_obstacle_config(config)
        self._mocap_ids: Dict[str, int] = {}
        self._geom_ids: Dict[str, tuple[int, int]] = {}
        for slot in SLOT_SPECS:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle_{slot}")
            if body_id < 0:
                raise RuntimeError(f"compiled model is missing obstacle slot {slot!r}")
            mocap_id = int(model.body_mocapid[body_id])
            if mocap_id < 0:
                raise RuntimeError(f"obstacle slot {slot!r} is not a mocap body")
            collision = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle_{slot}_collision"
            )
            visual = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle_{slot}_visual"
            )
            self._mocap_ids[slot] = mocap_id
            self._geom_ids[slot] = (collision, visual)
        self.active: Dict[str, ActiveObstacle] = {}
        self.profile_name = "clear"
        self.park_all()

    def park_all(self) -> None:
        self.active.clear()
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
        self.profile_name = profile_name
        for obstacle in profiles[profile_name].get("obstacles", []):
            slot = obstacle["slot"]
            shape, _ = SLOT_SPECS[slot]
            size = np.asarray(obstacle["size"], dtype=float)
            collision, visual = self._geom_ids[slot]
            self.model.geom_size[collision, :len(size)] = size
            self.model.geom_size[visual, :len(size)] = size
            rgba = np.asarray(obstacle["rgba"], dtype=float)
            self.model.geom_rgba[collision] = rgba
            self.model.geom_rgba[visual] = rgba
            mocap_id = self._mocap_ids[slot]
            self.data.mocap_pos[mocap_id] = obstacle["position"]
            yaw = float(obstacle["yaw"])
            self.data.mocap_quat[mocap_id] = (
                math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)
            )
            self.active[slot] = ActiveObstacle(
                slot=slot,
                kind=obstacle["kind"],
                shape=shape,
                size=size,
                motion=copy.deepcopy(obstacle.get("motion")),
            )
        self.update(0.0)
        mujoco.mj_forward(self.model, self.data)

    def update(self, elapsed_seconds: float) -> None:
        for slot, obstacle in self.active.items():
            if obstacle.motion is None:
                continue
            motion = obstacle.motion
            start = np.asarray(motion["start"], dtype=float)
            end = np.asarray(motion["end"], dtype=float)
            path_length = float(np.linalg.norm(end - start))
            elapsed = max(0.0, float(elapsed_seconds) - motion["start_delay"])
            progress = elapsed * motion["speed"] / path_length + motion["phase"]
            if motion.get("mode", "ping_pong") == "loop":
                alpha = progress % 1.0
            else:
                folded = progress % 2.0
                alpha = folded if folded <= 1.0 else 2.0 - folded
            self.data.mocap_pos[self._mocap_ids[slot]] = (1.0 - alpha) * start + alpha * end

    def positions(self) -> Dict[str, np.ndarray]:
        return {
            slot: self.data.mocap_pos[self._mocap_ids[slot]].copy()
            for slot in self.active
        }

    def planar_clearance(self, base_xy: Iterable[float], base_radius: float = 0.42) -> float:
        """Conservative 2-D clearance from the base footprint to active obstacles."""
        base_xy = np.asarray(tuple(base_xy), dtype=float)
        if not self.active:
            return float("inf")
        return min(
            float(np.linalg.norm(position[:2] - base_xy))
            - base_radius
            - self.active[slot].planar_radius
            for slot, position in self.positions().items()
        )

    @property
    def collision_geom_ids(self) -> set[int]:
        return {self._geom_ids[slot][0] for slot in self.active}


def _is_descendant(model: mujoco.MjModel, body_id: int, ancestor_id: int) -> bool:
    while body_id > 0:
        if body_id == ancestor_id:
            return True
        body_id = int(model.body_parentid[body_id])
    return False


class ObstacleCollisionMonitor:
    """Track obstacle contacts with the robot, crate, or fruit payloads."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 obstacles: TransportObstacleManager):
        self.model = model
        self.data = data
        self.obstacles = obstacles
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        payload_ids = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CRATE_BODY),
            *(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
              for name in OBJECT_BODIES.values()),
        }
        self.protected_geom_ids = {
            geom_id for geom_id in range(model.ngeom)
            if _is_descendant(model, int(model.geom_bodyid[geom_id]), base_id)
            or int(model.geom_bodyid[geom_id]) in payload_ids
        }
        self.contact_steps = 0
        self.min_contact_distance = float("inf")
        self.min_planar_clearance = float("inf")
        self.geom_pairs: set[tuple[str, str]] = set()

    @property
    def collided(self) -> bool:
        return self.contact_steps > 0

    def observe_base_clearance(self, base_xy: Iterable[float]) -> float:
        """Record and return conservative base-to-obstacle planar clearance."""
        clearance = self.obstacles.planar_clearance(base_xy)
        self.min_planar_clearance = min(self.min_planar_clearance, clearance)
        return clearance

    def observe(self) -> bool:
        obstacle_geoms = self.obstacles.collision_geom_ids
        contacted = False
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if not (pair & obstacle_geoms) or not (pair & self.protected_geom_ids):
                continue
            contacted = True
            self.min_contact_distance = min(self.min_contact_distance, float(contact.dist))
            names = tuple(sorted(
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom)
                or f"geom#{geom}"
                for geom in pair
            ))
            self.geom_pairs.add(names)
        if contacted:
            self.contact_steps += 1
        return contacted
