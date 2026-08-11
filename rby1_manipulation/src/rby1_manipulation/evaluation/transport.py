"""Measurable success conditions for the transport scenarios.

Three predicates, all usable both by the scripted collectors (to label an
episode) and by policy evaluation (to stop a rollout):

    check_grasp      - is an arm actually holding the thing it was told to hold
    object_in_crate  - is a small object inside the crate
    crate_on_shelf   - is the crate seated on the goal board, upright and still

The two frame-relative tests are expressed in the *object's own* frame rather
than in world coordinates, so they keep working while the crate is being carried
and tilted, and while the shelf pose is randomized.

`StopCheck` wraps any predicate with the same 8-consecutive-steps debounce that
pi05_ex_infer.make_grid_stop_check uses for the block task, so success criteria
stay comparable across scenarios.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

import mujoco
import numpy as np

from rby1_manipulation.control.ik import ArmHandles
from rby1_manipulation.simulation.transport_scene import (
    CRATE_BODY,
    CRATE_HALF,
    CRATE_INTERIOR_HALF,
    CRATE_JOINT,
    OBJECT_BODIES,
    SHELF_BODY,
    SHELF_LEVEL_SITES,
    TABLE_TOP_Z,
)

# Finger bodies per arm, used to detect a real grasp contact.
FINGER_BODIES = {
    "right": ("ee_finger_r1", "ee_finger_r2"),
    "left": ("ee_finger_l1", "ee_finger_l2"),
}


# ---------- small helpers ----------

def _body_id(model: mujoco.MjModel, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise KeyError(f"no body named {name!r}")
    return bid


def _geoms_of(model: mujoco.MjModel, body_name: str) -> set:
    bid = _body_id(model, body_name)
    return {g for g in range(model.ngeom) if model.geom_bodyid[g] == bid}


def _free_joint_vel(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str) -> np.ndarray:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    adr = int(model.jnt_dofadr[jid])
    return data.qvel[adr:adr + 6]


def _object_half_extent(model: mujoco.MjModel, data: mujoco.MjData,
                        body_name: str, frame_R: np.ndarray) -> np.ndarray:
    """Union half-extents of a body's collision geoms along `frame_R`'s axes.

    Using one overall bounding sphere would be far too strict for an elongated
    object: the roughly 79 mm banana would report about 44 mm in every direction and be judged
    outside a crate it is comfortably sitting in.  Mesh vertices are therefore
    transformed exactly, which also supports the banana's convex decomposition.
    """
    bid = _body_id(model, body_name)
    body_pos = data.xpos[bid]
    points: list[np.ndarray] = []

    for g in range(model.ngeom):
        if model.geom_bodyid[g] != bid or not model.geom_contype[g]:
            continue
        size = model.geom_size[g]
        geom_R = data.geom_xmat[g].reshape(3, 3)
        R_rel = frame_R.T @ geom_R
        center = frame_R.T @ (data.geom_xpos[g] - body_pos)
        gtype = model.geom_type[g]
        if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            extent = np.full(3, float(size[0]))
            points.extend((center - extent, center + extent))
        elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            # radius everywhere, plus the half-length projected onto each axis
            extent = float(size[0]) + np.abs(R_rel[:, 2]) * float(size[1])
            points.extend((center - extent, center + extent))
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            extent = np.abs(R_rel) @ np.asarray(size[:3], dtype=float)
            points.extend((center - extent, center + extent))
        elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = int(model.geom_dataid[g])
            adr = int(model.mesh_vertadr[mesh_id])
            num = int(model.mesh_vertnum[mesh_id])
            vertices = np.asarray(model.mesh_vert[adr:adr + num], dtype=float)

            # MuJoCo recentres mesh vertices into an inertia frame at compile
            # time and folds the inverse transform into geom_pos/geom_quat.
            # data.geom_xpos/xmat therefore already map the stored vertices to
            # world coordinates; applying mesh_pos/quat again would double it.
            world_vertices = vertices @ geom_R.T + data.geom_xpos[g]
            points.extend((world_vertices - body_pos) @ frame_R)
        else:
            extent = np.full(3, float(model.geom_rbound[g]))
            points.extend((center - extent, center + extent))

    if not points:
        return np.zeros(3)
    all_points = np.vstack(points)
    return np.maximum(np.abs(all_points.min(axis=0)), np.abs(all_points.max(axis=0)))


# ---------- grasp ----------

@dataclass
class GraspResult:
    ok: bool
    per_arm: Dict[str, bool] = field(default_factory=dict)
    gripper_qpos: Dict[str, float] = field(default_factory=dict)
    contacts: Dict[str, int] = field(default_factory=dict)
    lifted: bool = False


def check_grasp(model: mujoco.MjModel, data: mujoco.MjData,
                arms: Dict[str, ArmHandles], target_body: str, *,
                min_contacts_per_arm: int = 1,
                gripper_open_eps: float = 0.004,
                require_lift: bool = False,
                lift_margin: float = 0.02) -> GraspResult:
    """True per arm when that arm's fingers touch `target_body` and are not shut.

    `arms` maps 'right'/'left' -> ArmHandles; only the listed arms are judged, so
    the same function covers the one-hand object pick and the two-hand crate grasp.

    The two conditions are complementary: contacts alone can fire on a graze,
    while a gripper that closed past `gripper_open_eps` means the pads met each
    other and there is nothing in between.
    """
    target_geoms = _geoms_of(model, target_body)
    result = GraspResult(ok=False)

    for side, arm in arms.items():
        finger_geoms: set = set()
        for body in FINGER_BODIES[side]:
            finger_geoms |= _geoms_of(model, body)

        n = 0
        for c in range(data.ncon):
            con = data.contact[c]
            g1, g2 = con.geom1, con.geom2
            if (g1 in finger_geoms and g2 in target_geoms) or \
               (g2 in finger_geoms and g1 in target_geoms):
                n += 1

        q = abs(float(data.qpos[arm.gripper_qidx]))
        result.contacts[side] = n
        result.gripper_qpos[side] = q
        result.per_arm[side] = (n >= min_contacts_per_arm) and (q > gripper_open_eps)

    z = data.xpos[_body_id(model, target_body)][2]
    result.lifted = bool(z > TABLE_TOP_Z + lift_margin)
    result.ok = all(result.per_arm.values()) and (result.lifted or not require_lift)
    return result


# ---------- object inside the crate ----------

def object_in_crate(model: mujoco.MjModel, data: mujoco.MjData, object_body: str, *,
                    crate_body: str = CRATE_BODY,
                    margin: float = 0.005,
                    z_range: Tuple[float, float] = (-0.052, 0.075)) -> bool:
    """Is `object_body` inside the crate's interior volume?

    Evaluated in the crate's local frame, so a crate that is lifted, carried or
    tilted still gives the right answer.

    The test is: the object's CENTRE is over the crate floor, and its height is
    within the wall span. An earlier version demanded full containment - interior
    half-extent minus the object's own half-extent minus a margin - which is
    unsatisfiable for anything resting against an inner wall: a 32 mm apple
    touching the x wall sits at |x| = 0.050 while that rule allowed only 0.045,
    so a correctly packed apple was reported as outside. The z window does the
    work of rejecting an object merely perched on the rim (an apple balanced
    there would be at z = 0.092, past the 0.075 bound).
    """
    crate_bid = _body_id(model, crate_body)
    obj_bid = _body_id(model, object_body)

    R = data.xmat[crate_bid].reshape(3, 3)
    local = R.T @ (data.xpos[obj_bid] - data.xpos[crate_bid])

    half = _object_half_extent(model, data, object_body, R)
    if np.any(CRATE_INTERIOR_HALF[:2] - half[:2] <= 0):
        raise ValueError(f"{object_body} is too large to fit inside the crate")
    limit = CRATE_INTERIOR_HALF[:2] - margin
    return bool(np.all(np.abs(local[:2]) < limit) and z_range[0] < local[2] < z_range[1])


def objects_in_crate(model: mujoco.MjModel, data: mujoco.MjData,
                     object_bodies: Sequence[str] = tuple(OBJECT_BODIES)) -> Dict[str, bool]:
    return {name: object_in_crate(model, data, name) for name in object_bodies}


# ---------- crate on the shelf ----------

@dataclass
class ShelfResult:
    ok: bool
    xy_err: float
    z_err: float
    tilt_deg: float
    speed: float


def crate_on_shelf(model: mujoco.MjModel, data: mujoco.MjData, *,
                   level: int = len(SHELF_LEVEL_SITES) - 1,
                   shelf_body: str = SHELF_BODY,
                   crate_body: str = CRATE_BODY,
                   crate_joint: str = CRATE_JOINT,
                   crate_half_z: float = float(CRATE_HALF[2]),
                   xy_tol: float = 0.10,
                   z_tol: float = 0.03,
                   tilt_tol_deg: float = 20.0,
                   vel_tol: float = 0.03) -> ShelfResult:
    """Is the crate seated on the goal board, roughly upright, and at rest?

    Position is compared in the shelf's own frame so a randomized shelf pose
    needs no change here. Tolerances:
      xy_tol  0.10 - about half the slack between the 0.18x0.30 crate and the
                     0.60x0.30 board, so the crate must be genuinely on it
      z_tol   0.03 - covers the ~10 mm IK residual at this height plus settling
      tilt    20 deg - the bimanual carry pitches the crate by ~11 deg
      vel_tol 0.03 - matches the block task's stationarity threshold
    """
    shelf_bid = _body_id(model, shelf_body)
    crate_bid = _body_id(model, crate_body)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SHELF_LEVEL_SITES[level])

    R_shelf = data.xmat[shelf_bid].reshape(3, 3)
    local = R_shelf.T @ (data.xpos[crate_bid] - data.site_xpos[site_id])

    xy_err = float(np.max(np.abs(local[:2])))
    z_err = float(abs(local[2] - crate_half_z))

    up = data.xmat[crate_bid].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
    tilt_deg = float(math.degrees(math.acos(np.clip(up[2], -1.0, 1.0))))

    speed = float(np.abs(_free_joint_vel(model, data, crate_joint)).max())

    ok = (xy_err < xy_tol and z_err < z_tol
          and tilt_deg < tilt_tol_deg and speed <= vel_tol)
    return ShelfResult(ok=ok, xy_err=xy_err, z_err=z_err, tilt_deg=tilt_deg, speed=speed)


# ---------- debounce ----------

class StopCheck:
    """Fire only after a predicate has held for `hold_steps` consecutive calls.

    Same debounce as the block task's grid stop check, so a momentary bounce off
    the shelf board is not reported as a success.
    """

    def __init__(self, predicate: Callable[[mujoco.MjModel, mujoco.MjData], bool],
                 hold_steps: int = 8):
        self.predicate = predicate
        self.hold_steps = hold_steps
        self.streak = 0

    def reset(self) -> None:
        self.streak = 0

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        self.streak = self.streak + 1 if self.predicate(model, data) else 0
        return self.streak >= self.hold_steps


def make_transport_stop_check(*, scenario: str, level: int,
                              object_body: Optional[str] = None,
                              hold_steps: int = 8) -> StopCheck:
    """Build the stop condition for a scenario.

    scenario='crate'  -> crate seated on the goal board
    scenario='pack'   -> the object is inside the crate AND the crate is seated
    """
    if scenario == "crate":
        def predicate(model, data) -> bool:
            return crate_on_shelf(model, data, level=level).ok
    elif scenario == "pack":
        if object_body is None:
            raise ValueError("scenario='pack' needs object_body")

        def predicate(model, data) -> bool:
            return (object_in_crate(model, data, object_body)
                    and crate_on_shelf(model, data, level=level).ok)
    else:
        raise ValueError(f"unknown scenario {scenario!r}")
    return StopCheck(predicate, hold_steps=hold_steps)
