"""Waypoint builders shared by both transport scenarios.

Scenario 2 is scenario 1 with a packing phase in front, so the crate phases live
here rather than in either script.

Two rules encoded below, both learned the hard way:

  * PRE-grasp targets are callables that re-read the handle sites, so a crate that
    settled or got nudged is tracked. POST-grasp targets are frozen at build time
    - once the grippers hold the handles, a "current handle position" target moves
    with the arm and the lift would never make progress.

  * Grasp orientation is never hand-authored. It is captured from the live EE
    site rotation, because the IK only converges near the wrist orientation the
    arms already hold; a strict top-down pose leaves 200+ mm of residual on the
    left arm. Capturing it again at the dock pose is what makes the place phase
    work after the base has yawed - the arms rotate with the base, so their
    current EE rotation is by definition the right one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import mink
import mujoco
import numpy as np

from rby1_manipulation.control.bimanual import BiWaypoint, approach_target, grasp_site_target
from rby1_manipulation.control.ik import site_pose
from rby1_manipulation.simulation.transport_scene import (
    CRATE_BODY,
    CRATE_HALF,
    CRATE_HANDLE_LOCAL_Z,
    HANDLE_SITES,
    body_position,
    body_rotation,
    handle_positions,
    shelf_target_position,
    site_position,
)

# Standoff along the approach axis for the pre-grasp hover.
CRATE_APPROACH_STANDOFF = 0.10
# How far the crate is lifted off the table before driving.
CRATE_LIFT_DZ = 0.14
# Hover height above the target shelf board before lowering.
SHELF_APPROACH_DZ = 0.09
# How far the arms back off after releasing on the shelf.
SHELF_RETRACT_DZ = 0.12
# Release height above the crate rim when dropping a small object inside.
OBJECT_RELEASE_DZ = 0.03
# The banana is low and curved: its mesh-derived 60% height (about 13 mm)
# places the long fingertips through the tabletop. This floor keeps the finger
# tips clear while still pinching below the banana's crown.
OBJECT_GRASP_DZ_MIN = {"banana": 0.022}
# A mesh-height fraction puts the pear grasp about 26 mm above its centre, on
# the narrow neck. The pads initially touch but lose it as soon as the arm
# lifts. Grasp the wider body instead; the tall pear still leaves ample table
# clearance at this height.
OBJECT_GRASP_DZ_OVERRIDE = {"pear": 0.015}


@dataclass
class GraspFrames:
    """Live EE orientations, captured from the model rather than authored."""
    right: mink.SO3
    left: mink.SO3


def capture_grasp_frames(model, data) -> GraspFrames:
    return GraspFrames(
        right=site_pose(data, model, "right_ee").rotation(),
        left=site_pose(data, model, "left_ee").rotation(),
    )


def _live_handle_target(side: str, rot: mink.SO3, standoff: float):
    """Callable target that re-reads the handle site at plan time."""
    def target(model, data) -> np.ndarray:
        return approach_target(site_position(model, data, HANDLE_SITES[side]), rot, standoff)
    return target


# ---------- crate phases ----------

def crate_approach_waypoints(frames: GraspFrames, *,
                             standoff: float = CRATE_APPROACH_STANDOFF,
                             trim: bool = False) -> list[BiWaypoint]:
    """Hover over both handles, then descend onto them with the grippers open."""
    waypoints = [
        BiWaypoint(
            "crate_hover",
            right_pos=_live_handle_target("right", frames.right, standoff),
            left_pos=_live_handle_target("left", frames.left, standoff),
            right_quat=frames.right, left_quat=frames.left,
            gripper="open", duration=2.0, wait_after=0.4,
        ),
        BiWaypoint(
            "crate_descend",
            right_pos=_live_handle_target("right", frames.right, 0.0),
            left_pos=_live_handle_target("left", frames.left, 0.0),
            right_quat=frames.right, left_quat=frames.left,
            gripper="open", duration=1.6, wait_after=0.3,
        ),
    ]
    if trim:
        waypoints.append(BiWaypoint(
            "crate_grasp_trim",
            right_pos=_live_handle_target("right", frames.right, 0.0),
            left_pos=_live_handle_target("left", frames.left, 0.0),
            right_quat=frames.right, left_quat=frames.left,
            gripper="open", duration=1.0, wait_after=0.4,
        ))
    return waypoints


def crate_lift_waypoints(model, data, frames: GraspFrames, *,
                         clear_z: Optional[float] = None,
                         lift_dz: float = CRATE_LIFT_DZ,
                         duration: float = 2.5) -> list[BiWaypoint]:
    """Straight-up lift. Targets are frozen: the handles now move with the arms.

    `clear_z` is the ABSOLUTE world height the crate centre must reach, and is
    what the scenarios pass (config carry.clear_z). It has to clear the shelf's
    top board, because the drive phase carries the crate straight at the shelf -
    lifting by a fixed delta instead leaves the crate at chest height, where it
    rams the shelf mid-drive and is torn out of the grippers.

    `lift_dz` is the fallback relative lift when no clearance is specified.

    Emitted as two waypoints. Both targets are callables of the form
    "current handle + (clear_z - current crate z)", which makes the second one a
    closed-loop trim: the arms sag ~70 mm under the crate's weight, and a single
    open-loop lift lands the crate below the board it is supposed to fly over.
    """
    def lift_target(side: str, rot: mink.SO3):
        def target(model_, data_) -> np.ndarray:
            handle = site_position(model_, data_, HANDLE_SITES[side])
            crate_z = body_position(model_, data_, CRATE_BODY)[2]
            dz = max(0.0, clear_z - crate_z) if clear_z is not None else lift_dz
            return grasp_site_target(handle + np.array([0.0, 0.0, dz]), rot)
        return target

    return [
        BiWaypoint(
            "crate_lift",
            right_pos=lift_target("right", frames.right),
            left_pos=lift_target("left", frames.left),
            right_quat=frames.right, left_quat=frames.left,
            gripper="hold", duration=duration, wait_after=0.6,
        ),
        BiWaypoint(
            "crate_lift_trim",
            right_pos=lift_target("right", frames.right),
            left_pos=lift_target("left", frames.left),
            right_quat=frames.right, left_quat=frames.left,
            gripper="hold", duration=1.5, wait_after=0.5,
        ),
    ]


def drive_waypoints(config: dict, *, turn_secs: float = 3.0,
                    drive_secs: float = 4.0) -> list[BiWaypoint]:
    """Yaw toward the shelf, then translate. Arms hold, so the crate rides along.

    Split into two moves rather than one diagonal so the swept volume stays
    predictable and the turn happens away from the table.
    """
    dock = np.asarray(config["dock"]["shelf"], dtype=float)
    start = np.asarray(config["dock"]["table"], dtype=float)
    return [
        BiWaypoint("base_turn", base=[start[0], start[1], dock[2]],
                   gripper="hold", duration=turn_secs, wait_after=0.5),
        BiWaypoint("base_drive", base=dock.tolist(),
                   gripper="hold", duration=drive_secs, wait_after=0.8),
    ]


def crate_place_waypoints(model, data, frames: GraspFrames, level: int, *,
                          approach_dz: float = SHELF_APPROACH_DZ) -> list[BiWaypoint]:
    """Carry the crate over the target board and lower it onto the surface.

    Handle offsets come from the crate's *current* orientation, so this works
    regardless of how far the base yawed on the way over. The release is left to
    the caller (see crate_retract_waypoints).
    """
    crate_R = body_rotation(model, data, CRATE_BODY)
    board = shelf_target_position(model, data, level)
    centre = board + np.array([0.0, 0.0, CRATE_HALF[2]])
    offsets = {
        "left": crate_R @ np.array([0.0, +0.200, CRATE_HANDLE_LOCAL_Z]),
        "right": crate_R @ np.array([0.0, -0.200, CRATE_HANDLE_LOCAL_Z]),
    }
    hover = {s: centre + o + np.array([0.0, 0.0, approach_dz]) for s, o in offsets.items()}
    seat = {s: centre + o for s, o in offsets.items()}

    return [
        BiWaypoint(
            "shelf_hover",
            right_pos=grasp_site_target(hover["right"], frames.right),
            left_pos=grasp_site_target(hover["left"], frames.left),
            right_quat=frames.right, left_quat=frames.left,
            gripper="hold", duration=2.5, wait_after=0.5,
        ),
        BiWaypoint(
            "shelf_seat",
            right_pos=grasp_site_target(seat["right"], frames.right),
            left_pos=grasp_site_target(seat["left"], frames.left),
            right_quat=frames.right, left_quat=frames.left,
            gripper="hold", duration=1.8, wait_after=0.6,
        ),
    ]


def crate_retract_waypoints(model, data, frames: GraspFrames, *,
                            retract_dz: float = SHELF_RETRACT_DZ) -> list[BiWaypoint]:
    """Back both arms straight up from wherever they are, grippers open.

    Called after the release so the crate is left standing on the board.
    """
    up = np.array([0.0, 0.0, retract_dz])
    return [
        BiWaypoint(
            "shelf_retract",
            right_pos=site_position(model, data, "right_ee") + up,
            left_pos=site_position(model, data, "left_ee") + up,
            right_quat=frames.right, left_quat=frames.left,
            gripper="hold", duration=1.8, wait_after=0.4,
        ),
    ]


# ---------- small-object phases (single arm) ----------

def object_grasp_dz(model, object_body: str, fraction: float = 0.6) -> float:
    """How far above an object's centre to put the finger pads.

    Grasping exactly at the centre drives the fingertips into the table: the pads
    reach ~30 mm below the EE site and a 32 mm apple only sits 32 mm off the
    surface. Biting at 60% of the object's vertical radius keeps the fingers clear
    while still closing on a wide part of the object - the same trick as
    scenario1_single_arm.GRASP_HEIGHT_ABOVE_CENTER, scaled per object instead of
    hard-coded for a 5 cm cube.

    The relevant dimension is the object's VERTICAL half-extent: the radius for a
    sphere or a horizontal capsule, but size[2] for a box, whose size[0] is its
    long axis and would send the pads far above the object.
    """
    if object_body in OBJECT_GRASP_DZ_OVERRIDE:
        return OBJECT_GRASP_DZ_OVERRIDE[object_body]

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    vertical_halves: list[float] = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != bid or not model.geom_contype[g]:
            continue
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            half = float(model.geom_size[g][2])
        elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            # geom_aabb = [centre xyz, half-size xyz] in the compiled geom
            # frame. Include its centre offset because a decomposed mesh such
            # as the banana has several vertically offset convex pieces.
            half = abs(float(model.geom_aabb[g][2])) + float(model.geom_aabb[g][5])
        else:
            half = float(model.geom_size[g][0])
        vertical_halves.append(half)
    if not vertical_halves:
        raise KeyError(f"{object_body} has no collision geom")
    mesh_height = fraction * max(vertical_halves)
    return max(mesh_height, OBJECT_GRASP_DZ_MIN.get(object_body, mesh_height))


def object_pick_waypoints(model, data, arm_side: str, frames: GraspFrames,
                          object_body: str, *, approach_dz: float = 0.10,
                          traverse_clearance: float = 0.14,
                          grasp_dz: Optional[float] = None) -> list[BiWaypoint]:
    """Fly over the crate handles, hover over the object, then descend onto it.

    The first waypoint matters: the arms ramp in joint space, so a direct move
    from the rest pose to a hover 100 mm above the object sweeps the forearm
    straight through the crate handle (top at crate z + 97 mm) and tips the crate
    up ~50 mm before the object is even touched. Going over the handle height
    first keeps the crate untouched.
    """
    rot = frames.right if arm_side == "right" else frames.left
    if grasp_dz is None:
        grasp_dz = object_grasp_dz(model, object_body)

    crate_z = body_position(model, data, CRATE_BODY)[2]
    traverse_z = crate_z + CRATE_HANDLE_LOCAL_Z + 0.012 + traverse_clearance

    def at(dz: float):
        def target(model_, data_) -> np.ndarray:
            pos = body_position(model_, data_, object_body)
            return grasp_site_target(pos + np.array([0.0, 0.0, dz]), rot)
        return target

    def above():
        def target(model_, data_) -> np.ndarray:
            pos = body_position(model_, data_, object_body)
            return grasp_site_target(np.array([pos[0], pos[1], traverse_z]), rot)
        return target

    key = "right_pos" if arm_side == "right" else "left_pos"
    quat_key = "right_quat" if arm_side == "right" else "left_quat"
    return [
        BiWaypoint("obj_over", **{key: above(), quat_key: rot},
                   gripper="open", duration=2.0, wait_after=0.3),
        BiWaypoint("obj_hover", **{key: at(approach_dz), quat_key: rot},
                   gripper="open", duration=1.8, wait_after=0.3),
        BiWaypoint("obj_descend", **{key: at(grasp_dz), quat_key: rot},
                   gripper="open", duration=1.4, wait_after=0.2),
        # Trim pass. A single descend leaves ~26 mm of servo tracking error, and
        # the objects are only 30-50 mm wide at the grasp height, so one pad
        # reaches the object first and simply sweeps it across the table.
        # Re-solving from the settled pose brings this down to a few mm.
        BiWaypoint("obj_descend_trim", **{key: at(grasp_dz), quat_key: rot},
                   gripper="open", duration=1.0, wait_after=0.4),
    ]


def object_into_crate_waypoints(model, data, arm_side: str, frames: GraspFrames, *,
                                traverse_clearance: float = 0.14,
                                release_dz: float = OBJECT_RELEASE_DZ,
                                release_offset_xy: Sequence[float] = (0.0, 0.0),
                                ) -> list[BiWaypoint]:
    """Lift the object, fly it over the handles, and stop just inside the rim.

    The traverse height is measured from the HANDLE TOP, not the crate rim. The
    handles stand 97 mm above the crate origin while the rim is only 60 mm up, so
    a path planned off the rim drags the object straight through the near handle
    and knocks it out of the gripper halfway across. The 140 mm clearance is set
    by the FINGERS, not the object: they hang ~40 mm below the pad centre, so a
    80 mm clearance leaves the fingertips only 6 mm over the handle and they
    clip it on the way past.

    The release happens outside this list so the caller can open only the arm that
    is holding something. Release height is kept low - the crate is a free body,
    and dropping into it from higher up shoves it sideways.
    """
    rot = frames.right if arm_side == "right" else frames.left
    key = "right_pos" if arm_side == "right" else "left_pos"
    quat_key = "right_quat" if arm_side == "right" else "left_quat"

    crate_pos = body_position(model, data, CRATE_BODY)
    offset_xy = np.asarray(release_offset_xy, dtype=float)
    if offset_xy.shape != (2,):
        raise ValueError("release_offset_xy must contain exactly [x, y]")
    crate_R = body_rotation(model, data, CRATE_BODY)
    release_offset_world = crate_R @ np.array([offset_xy[0], offset_xy[1], 0.0])
    ee_site = "right_ee" if arm_side == "right" else "left_ee"
    here = site_position(model, data, ee_site)

    handle_top_z = crate_pos[2] + CRATE_HANDLE_LOCAL_Z + 0.012
    rim_z = crate_pos[2] + CRATE_HALF[2]
    traverse_z = handle_top_z + traverse_clearance
    over = np.array([crate_pos[0], crate_pos[1], traverse_z]) + release_offset_world
    drop = np.array([crate_pos[0], crate_pos[1], rim_z + release_dz]) + release_offset_world
    return [
        BiWaypoint("obj_lift",
                   **{key: np.array([here[0], here[1], traverse_z]), quat_key: rot},
                   gripper="hold", duration=1.8, wait_after=0.4),
        BiWaypoint("obj_carry", **{key: grasp_site_target(over, rot), quat_key: rot},
                   gripper="hold", duration=2.2, wait_after=0.5),
        BiWaypoint("obj_lower", **{key: grasp_site_target(drop, rot), quat_key: rot},
                   gripper="hold", duration=1.4, wait_after=0.4),
    ]


def arm_retract_waypoint(arm_side: str, frames: GraspFrames, model, data, *,
                         dz: float = 0.18) -> BiWaypoint:
    rot = frames.right if arm_side == "right" else frames.left
    key = "right_pos" if arm_side == "right" else "left_pos"
    quat_key = "right_quat" if arm_side == "right" else "left_quat"
    ee_site = "right_ee" if arm_side == "right" else "left_ee"
    here = site_position(model, data, ee_site)
    return BiWaypoint("obj_retract", **{key: here + np.array([0.0, 0.0, dz]), quat_key: rot},
                      gripper="hold", duration=1.5, wait_after=0.3)
