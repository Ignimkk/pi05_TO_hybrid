"""Transport scene: config, reset/randomization, base handles, 17-D state/action.

This is the `preview_block_grid.py` of the mobile crate-transport scenarios: it
owns the JSON layout config, the deterministic reset, and the runtime
randomization knobs, so the scenario scripts stay pure motion code.

The scene lives in model_transport.xml (a separate root from the block scenario's
model.xml, which is left untouched). Its extras over the block model:

  * a `crate` free body with two graspable handle bars,
  * `apple` / `banana` / `orange` / `pear` free bodies for one-hand grasping,
  * a 3-level `shelf` with one goal site per level,
  * position actuators on the base's joint_x / joint_y / joint_th (ctrl 26/27/28),
    which makes the 17-D state/action layout below possible.

Run `python transport_scene.py --self-check` to validate the compiled model
against every invariant the scenarios depend on.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

import mujoco
import numpy as np

from rby1_manipulation.control.ik import (
    ArmHandles,
    GRIPPER_OPEN,
    left_arm_handles,
    right_arm_handles,
)
from rby1_manipulation.paths import (
    MUJOCO_MODEL_DIR,
    TRANSPORT_LAYOUT_CONFIG,
    TRANSPORT_MODEL_XML,
    TRANSPORT_WHEEL_MODEL_XML,
)
from rby1_manipulation.simulation.common import (
    TRANSPORT_SMALL_OBJ_REACH,
    set_block_pose,
)

_MUJOCO_DIR = MUJOCO_MODEL_DIR
# Always absolute: loading these models through a relative path trips MuJoCo's
# duplicate-include check on rby1.xml's twice-included WHEEL/geoms.xml.
MODEL_XML = str(TRANSPORT_MODEL_XML)
MODEL_XML_WHEELS = str(TRANSPORT_WHEEL_MODEL_XML)

DEFAULT_LAYOUT_CONFIG = TRANSPORT_LAYOUT_CONFIG

# ---------- scene naming (mirrors scenes/scene_transport.xml) ----------

CRATE_BODY = "crate"
CRATE_JOINT = "crate_free"
HANDLE_SITES = {"left": "handle_l_site", "right": "handle_r_site"}

OBJECT_BODIES = {
    "apple": "apple",
    "banana": "banana",
    "orange": "orange",
    "pear": "pear",
}
OBJECT_JOINTS = {name: f"{name}_free" for name in OBJECT_BODIES}
OBJECT_TYPES = tuple(OBJECT_BODIES)

SHELF_BODY = "shelf"
SHELF_LEVEL_SITES = ("shelf_level_0_site", "shelf_level_1_site", "shelf_level_2_site")

TABLE_BODY = "table"
TABLE_TOP_Z = 0.82
TABLE_X_RANGE = (0.40, 0.90)

# Crate half-extents (outer) and interior clearance, from scene_transport.xml.
CRATE_HALF = np.array([0.090, 0.150, 0.060])
CRATE_INTERIOR_HALF = np.array([0.082, 0.142, 0.056])
CRATE_HANDLE_LOCAL_Z = 0.085

# Where an inactive object type is parked. It is *not* removed from the model:
# a changing nq would invalidate the keyframe and every hard-coded qpos index.
PARKING_XYZ = np.array([5.0, 5.0, 0.1])

# Base footprint (mesh AABB), used to check the dock pose clears the shelf.
BASE_FRONT_OVERHANG = 0.33

# ---------- base handles / 17-D interface ----------

BASE_ACTS = ("base_x_act", "base_y_act", "base_yaw_act")
BASE_JOINTS = ("joint_x", "joint_y", "joint_th")


@dataclass
class BaseHandles:
    """Cached indices for the three planar base DoFs."""
    joint_names: Sequence[str] = BASE_JOINTS
    act_names: Sequence[str] = BASE_ACTS
    qidx: list = field(default_factory=list)
    dofidx: list = field(default_factory=list)
    aid: list = field(default_factory=list)


def base_handles(model: mujoco.MjModel, *, require_actuators: bool = True) -> BaseHandles:
    """Resolve the base joint/actuator indices.

    `require_actuators=False` is for the experimental wheel-drive model, which
    deliberately has no base actuators; `aid` comes back empty there.
    """
    h = BaseHandles()
    h.qidx = [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)])
              for j in h.joint_names]
    h.dofidx = [int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)])
                for j in h.joint_names]
    aid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in h.act_names]
    if any(i < 0 for i in aid):
        if require_actuators:
            raise RuntimeError(
                "model has no base actuators; load model_transport.xml, not model.xml")
        aid = []
    h.aid = aid
    return h


def build_state_17(model: mujoco.MjModel, data: mujoco.MjData,
                   left: ArmHandles, right: ArmHandles,
                   base: BaseHandles) -> np.ndarray:
    """[L arm 0..5, L grip, R arm 0..5, R grip, base_x, base_y, base_yaw].

    The first 14 entries are byte-identical to the existing block-scenario
    layout (same joint subset, same abs(qpos)/0.045 gripper normalization), so a
    17-D episode can be truncated back to the 14-D schema.
    """
    return np.array([
        *(data.qpos[q] for q in left.qidx[:6]),
        abs(data.qpos[left.gripper_qidx]) / abs(GRIPPER_OPEN),
        *(data.qpos[q] for q in right.qidx[:6]),
        abs(data.qpos[right.gripper_qidx]) / abs(GRIPPER_OPEN),
        data.qpos[base.qidx[0]], data.qpos[base.qidx[1]], data.qpos[base.qidx[2]],
    ], dtype=np.float32)


def build_action_17(model: mujoco.MjModel, data: mujoco.MjData,
                    left: ArmHandles, right: ArmHandles,
                    base: BaseHandles) -> np.ndarray:
    """Same layout as build_state_17 but read from ctrl (the commanded target).

    In wheel-drive mode there are no base actuators to read a command from, so
    the base entries fall back to the measured pose. That makes the action a
    partial copy of the state for those three dims - another reason wheel mode is
    not used for data collection.
    """
    base_cmd = ([data.ctrl[a] for a in base.aid] if base.aid
                else [data.qpos[q] for q in base.qidx])
    return np.array([
        *(data.ctrl[a] for a in left.aid[:6]),
        abs(data.ctrl[left.gripper_aid]) / abs(GRIPPER_OPEN),
        *(data.ctrl[a] for a in right.aid[:6]),
        abs(data.ctrl[right.gripper_aid]) / abs(GRIPPER_OPEN),
        *base_cmd,
    ], dtype=np.float32)


# ---------- config ----------

def _as_xy(value, label: str) -> list:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (2,):
        raise ValueError(f"{label}: expected [x, y], got {value!r}")
    return arr.tolist()


def validate_layout_config(raw: dict) -> dict:
    """Return a normalized copy, raising on geometry the robot cannot execute.

    Every bound here comes from a measurement on the compiled model, not a guess;
    the comments name what breaks when the bound is crossed.
    """
    cfg = copy.deepcopy(raw)
    if int(cfg.get("version", 1)) != 1:
        raise ValueError(f"unsupported layout config version {cfg.get('version')!r}")

    crate = cfg["crate"]
    crate["xy"] = _as_xy(crate["xy"], "crate.xy")
    crate["spawn_z"] = float(crate["spawn_z"])
    crate["handle_half_y"] = float(crate["handle_half_y"])
    cx, cy = crate["xy"]
    # Arm reach at grasp height tops out near x = 0.58; past that IK residual
    # grows from a few mm to tens of mm and the grasp misses the handle.
    if not 0.44 <= cx <= 0.56:
        raise ValueError(f"crate.xy[0]={cx:.3f} outside the reachable band [0.44, 0.56]")
    if cx - CRATE_HALF[0] < TABLE_X_RANGE[0] or cx + CRATE_HALF[0] > TABLE_X_RANGE[1]:
        raise ValueError(f"crate footprint x leaves the table top ({TABLE_X_RANGE})")
    if abs(cy) > 0.05:
        raise ValueError(f"crate.xy[1]={cy:.3f}: keep the crate centred so both arms reach")
    if not 0.15 <= crate["handle_half_y"] <= 0.26:
        raise ValueError("crate.handle_half_y must stay within the arms' |y| envelope")

    objects = cfg["objects"]
    missing_objects = set(OBJECT_TYPES) - set(objects)
    if missing_objects:
        raise ValueError(f"objects is missing required types: {sorted(missing_objects)}")
    for name, spec in objects.items():
        if name not in OBJECT_BODIES:
            raise ValueError(f"unknown object type {name!r}; known: {OBJECT_TYPES}")
        spec["xy"] = _as_xy(spec["xy"], f"objects.{name}.xy")
        spec["spawn_z"] = float(spec["spawn_z"])
        ox, oy = spec["xy"]
        lo_x, hi_x = TRANSPORT_SMALL_OBJ_REACH["x"]
        lo_y, hi_y = TRANSPORT_SMALL_OBJ_REACH["y"]
        if not lo_x <= ox <= hi_x:
            raise ValueError(f"objects.{name}.xy[0]={ox:.3f} outside {(lo_x, hi_x)}")
        # |y| >= 0.25 keeps the spawn clear of the handle outer faces at |y| = 0.21.
        if not lo_y <= abs(oy) <= hi_y:
            raise ValueError(f"objects.{name}: |y|={abs(oy):.3f} outside {(lo_y, hi_y)}")

    shelf = cfg["shelf"]
    shelf["xy"] = _as_xy(shelf["xy"], "shelf.xy")
    shelf["half_depth"] = float(shelf["half_depth"])
    shelf["levels"] = [float(z) for z in shelf["levels"]]
    shelf["target_level"] = int(shelf["target_level"])
    if not 2 <= len(shelf["levels"]) <= 3:
        raise ValueError("shelf.levels must have 2 or 3 entries")
    if not 0 <= shelf["target_level"] < len(shelf["levels"]):
        raise ValueError("shelf.target_level out of range")
    # The gripper + wrist stick ~0.25 m above the crate top, so a level with a
    # board above it needs ~0.45 m of clearance to insert into. Only the open top
    # level is placeable; the lower boards are distractors.
    if shelf["target_level"] != len(shelf["levels"]) - 1 and not cfg.get("allow_unreachable_level"):
        raise ValueError(
            "shelf.target_level must be the top (open) level; set "
            "allow_unreachable_level=true to override")

    dock = cfg["dock"]
    for key in ("table", "shelf"):
        arr = np.asarray(dock[key], dtype=float)
        if arr.shape != (3,):
            raise ValueError(f"dock.{key}: expected [x, y, yaw]")
        dock[key] = arr.tolist()
    shelf_front_y = shelf["xy"][1] + shelf["half_depth"]
    base_front_y = dock["shelf"][1] - BASE_FRONT_OVERHANG
    if base_front_y - shelf_front_y <= 0.08:
        raise ValueError(
            f"dock.shelf leaves only {base_front_y - shelf_front_y:.3f} m between the "
            "base front and the shelf; need > 0.08 m")

    carry = cfg["carry"]
    carry["lift_z"] = float(carry["lift_z"])
    carry["clear_z"] = float(carry["clear_z"])
    top_board_z = shelf["xy"] and shelf["levels"][shelf["target_level"]]
    if carry["clear_z"] < top_board_z + CRATE_HALF[2] - 0.02:
        raise ValueError("carry.clear_z is below the target board + crate half height")
    return cfg


def load_layout_config(path: pathlib.Path | str = DEFAULT_LAYOUT_CONFIG) -> dict:
    with open(path) as f:
        return validate_layout_config(json.load(f))


def save_layout_config(config: dict, path: pathlib.Path | str = DEFAULT_LAYOUT_CONFIG) -> pathlib.Path:
    path = pathlib.Path(path)
    validate_layout_config(config)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    tmp.replace(path)
    return path


# ---------- randomization ----------

@dataclass
class RandomizationSpec:
    """Everything the scene can vary between episodes. All defaults are off."""
    crate_xy_jitter: float = 0.0            # m, uniform box around the nominal xy
    crate_yaw_jitter: float = 0.0           # rad
    crate_mass_range: Optional[Tuple[float, float]] = None   # kg, total crate mass
    object_types: Tuple[str, ...] = OBJECT_TYPES             # which objects are active
    object_pose: bool = False               # resample xy in TRANSPORT_SMALL_OBJ_REACH
    friction_range: Optional[Tuple[float, float]] = None     # multiplier on sliding friction
    shelf_y_jitter: float = 0.0             # m (applied to body_pos, not a freejoint)
    target_level: Optional[int] = None      # override config's shelf.target_level


@dataclass
class SceneState:
    """What a reset actually produced. Scenario scripts plan from this."""
    crate_pose: np.ndarray                  # (7,) x y z qw qx qy qz
    object_poses: Dict[str, np.ndarray]     # active objects only, (7,) each
    active_objects: Tuple[str, ...]
    shelf_pos: np.ndarray                   # (3,) world
    target_level: int
    target_site: str
    dock_pose: np.ndarray                   # (3,) base x, y, yaw at the shelf
    crate_mass: float


# Nominal mass / friction per model, captured before the first perturbation so
# repeated resets scale from the original values instead of compounding.
_NOMINAL_CACHE: Dict[int, Tuple[mujoco.MjModel, dict]] = {}


def _nominals(model: mujoco.MjModel) -> dict:
    entry = _NOMINAL_CACHE.get(id(model))
    if entry is not None and entry[0] is model:
        return entry[1]
    crate_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CRATE_BODY)
    crate_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == crate_bid]
    nom = {
        "crate_bid": crate_bid,
        "crate_geoms": crate_geoms,
        "crate_mass": float(model.body_mass[crate_bid]),
        "crate_inertia": model.body_inertia[crate_bid].copy(),
        "geom_friction": model.geom_friction.copy(),
        "shelf_bid": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, SHELF_BODY),
        "shelf_pos": model.body_pos[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, SHELF_BODY)].copy(),
    }
    _NOMINAL_CACHE[id(model)] = (model, nom)
    return nom


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def _yaw_matrix(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def reset_transport_scene(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: dict,
    *,
    rng: Optional[np.random.Generator] = None,
    randomize: Optional[RandomizationSpec] = None,
    settle_seconds: float = 1.5,
    on_step: Optional[Callable[[], None]] = None,
) -> SceneState:
    """Reset to the teleop keyframe, place every object, settle, report the result."""
    rng = rng if rng is not None else np.random.default_rng()
    spec = randomize if randomize is not None else RandomizationSpec()
    nom = _nominals(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "teleop")
    if key_id < 0:
        raise RuntimeError("model has no 'teleop' keyframe")
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    # --- crate ---
    cx, cy = config["crate"]["xy"]
    if spec.crate_xy_jitter:
        cx += float(rng.uniform(-spec.crate_xy_jitter, spec.crate_xy_jitter))
        cy += float(rng.uniform(-spec.crate_xy_jitter, spec.crate_xy_jitter))
    yaw = float(rng.uniform(-spec.crate_yaw_jitter, spec.crate_yaw_jitter)) if spec.crate_yaw_jitter else 0.0
    crate_xyz = np.array([cx, cy, config["crate"]["spawn_z"]])
    set_block_pose(model, data, CRATE_JOINT, crate_xyz, tuple(_yaw_quat(yaw)))

    # --- small objects: active ones placed, the rest parked out of the way ---
    active = tuple(t for t in spec.object_types if t in OBJECT_BODIES)
    for object_index, name in enumerate(OBJECT_TYPES):
        if name not in active:
            # Give each inactive free body its own parking bay; stacking several
            # objects at one pose creates violent contacts before the first step.
            parking = PARKING_XYZ + np.array([0.25 * object_index, 0.0, 0.0])
            set_block_pose(model, data, OBJECT_JOINTS[name], parking)
            continue
        ox, oy = config["objects"][name]["xy"]
        oz = config["objects"][name]["spawn_z"]
        if spec.object_pose:
            side = 1.0 if oy >= 0 else -1.0
            # Two objects share each arm's narrow y band. Independent sampling
            # across the full x envelope can leave no valid second placement and
            # used to return an overlapping last candidate after 50 attempts.
            # Jitter around the two configured x slots instead: their nominal
            # 90 mm separation remains at least 78 mm, enough for these meshes.
            ox = float(np.clip(ox + rng.uniform(-0.006, 0.006),
                               *TRANSPORT_SMALL_OBJ_REACH["x"]))
            y_lo, y_hi = TRANSPORT_SMALL_OBJ_REACH["y"]
            oy = float(side * rng.uniform(y_lo, y_hi))
        xyz = np.array([ox, oy, oz])
        # Identity for every fruit: the banana mesh's long axis must stay along
        # world x,
        # across the gripper's closing direction, or the pads meet its tapered
        # ends and it is flicked away instead of grasped.
        set_block_pose(model, data, OBJECT_JOINTS[name], xyz)

    # --- crate mass / friction (runtime model edits, no recompile) ---
    crate_mass = nom["crate_mass"]
    if spec.crate_mass_range is not None:
        crate_mass = float(rng.uniform(*spec.crate_mass_range))
        scale = crate_mass / nom["crate_mass"]
        model.body_mass[nom["crate_bid"]] = crate_mass
        model.body_inertia[nom["crate_bid"]] = nom["crate_inertia"] * scale
    else:
        model.body_mass[nom["crate_bid"]] = nom["crate_mass"]
        model.body_inertia[nom["crate_bid"]] = nom["crate_inertia"]

    model.geom_friction[:] = nom["geom_friction"]
    if spec.friction_range is not None:
        mult = float(rng.uniform(*spec.friction_range))
        for g in nom["crate_geoms"]:
            model.geom_friction[g, 0] = nom["geom_friction"][g, 0] * mult

    # --- shelf pose (static body: move body_pos, not a joint) ---
    shelf_pos = nom["shelf_pos"].copy()
    if spec.shelf_y_jitter:
        shelf_pos[1] += float(rng.uniform(-spec.shelf_y_jitter, spec.shelf_y_jitter))
    model.body_pos[nom["shelf_bid"]] = shelf_pos

    mujoco.mj_forward(model, data)

    # Hold every actuated joint at its keyframe qpos while things settle.
    for aid in range(model.nu):
        data.ctrl[aid] = data.qpos[model.jnt_qposadr[model.actuator_trnid[aid, 0]]]

    steps = int(round(settle_seconds / model.opt.timestep))
    for i in range(steps):
        mujoco.mj_step(model, data)
        if on_step is not None and (i % 10 == 0 or i == steps - 1):
            on_step()

    target_level = spec.target_level if spec.target_level is not None else config["shelf"]["target_level"]
    return SceneState(
        crate_pose=free_body_pose(model, data, CRATE_JOINT),
        object_poses={n: free_body_pose(model, data, OBJECT_JOINTS[n]) for n in active},
        active_objects=active,
        shelf_pos=shelf_pos.copy(),
        target_level=target_level,
        target_site=SHELF_LEVEL_SITES[target_level],
        dock_pose=np.asarray(config["dock"]["shelf"], dtype=float),
        crate_mass=crate_mass,
    )


# ---------- pose queries ----------

def body_position(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    return data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)].copy()


def body_rotation(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xmat[bid].reshape(3, 3).copy()


def site_position(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> np.ndarray:
    return data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)].copy()


def free_body_pose(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str) -> np.ndarray:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    adr = int(model.jnt_qposadr[jid])
    return data.qpos[adr:adr + 7].copy()


def handle_positions(model: mujoco.MjModel, data: mujoco.MjData) -> Dict[str, np.ndarray]:
    """Current world positions of both crate handles, keyed 'left'/'right'.

    Read these fresh before every waypoint: the crate shifts a few cm during
    contact, and planning against a stale pose lets the error run away.
    """
    return {side: site_position(model, data, name) for side, name in HANDLE_SITES.items()}


def shelf_target_position(model: mujoco.MjModel, data: mujoco.MjData, level: int) -> np.ndarray:
    return site_position(model, data, SHELF_LEVEL_SITES[level])


# ---------- self check ----------

def _self_check(config_path: pathlib.Path, verbose: bool = True) -> int:
    """Assert every invariant the scenarios rely on. Returns a process exit code."""
    failures: list[str] = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        status = "ok  " if ok else "FAIL"
        if verbose:
            print(f"  [{status}] {label}{('  ' + detail) if detail else ''}")
        if not ok:
            failures.append(label)

    print(f"config: {config_path}")
    config = load_layout_config(config_path)
    print("  [ok  ] layout config validates")

    print(f"model: {MODEL_XML}")
    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data = mujoco.MjData(model)

    free = [(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j), int(model.jnt_qposadr[j]))
            for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    print("\nfreejoint map (declaration order defines the keyframe layout):")
    for name, adr in free:
        print(f"    {name:14s} qpos[{adr}:{adr + 7}]")
    expected_free = [CRATE_JOINT, *(OBJECT_JOINTS[name] for name in OBJECT_TYPES)]
    check([n for n, _ in free] == expected_free, "freejoint order", str(expected_free))
    check(model.nq == 31 + 7 * len(free), "nq == 31 + 7 * n_free", f"nq={model.nq}")
    check(model.key_qpos.shape[1] == model.nq, "keyframe length == nq",
          f"{model.key_qpos.shape[1]}")

    print("\nactuators:")
    acts = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    block_model = mujoco.MjModel.from_xml_path(str(_MUJOCO_DIR / "model.xml"))
    block_acts = [mujoco.mj_id2name(block_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                  for i in range(block_model.nu)]
    check(acts[:len(block_acts)] == block_acts, "ctrl 0..25 match model.xml")
    check(tuple(acts[len(block_acts):]) == BASE_ACTS, "ctrl 26..28 are the base actuators",
          str(acts[len(block_acts):]))

    # Loading two models in one process is exactly what the scene_utils cache bug
    # broke; exercising it here keeps the fix honest.
    check(free_body_pose(model, data, CRATE_JOINT).shape == (7,),
          "free_body_pose works with two models loaded")

    print("\nscene prop collision geoms:")
    prop_bodies = {CRATE_BODY, SHELF_BODY, TABLE_BODY, *OBJECT_BODIES.values()}
    bad = []
    for g in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
        if body in prop_bodies and model.geom_group[g] == 3 and model.geom_conaffinity[g] != 1:
            bad.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g))
    check(not bad, "every group-3 prop geom has conaffinity=1", str(bad))

    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    collision_counts = {"apple": 1, "banana": 5, "orange": 1, "pear": 1}
    for body, expected_collisions in collision_counts.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        collision_geoms = [
            g for g in range(model.ngeom)
            if model.geom_bodyid[g] == bid and model.geom_contype[g]
        ]
        visual_geoms = [
            g for g in range(model.ngeom)
            if model.geom_bodyid[g] == bid and not model.geom_contype[g]
        ]
        check(len(collision_geoms) == expected_collisions,
              f"{body} collision decomposition has {expected_collisions} mesh(es)",
              f"count={len(collision_geoms)}")
        check(all(int(model.geom_type[g]) == mesh_type for g in collision_geoms + visual_geoms),
              f"{body} visual/collision geoms are meshes")

    state = reset_transport_scene(model, data, config, settle_seconds=1.5)

    print("\nsettled scene:")
    origin_hits = [n for n, _ in free
                   if np.linalg.norm(free_body_pose(model, data, n)[:3]) < 0.05]
    check(not origin_hits, "no free body spawned at the world origin", str(origin_hits))

    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    total_normal, touching = 0.0, set()
    f6 = np.zeros(6)
    for c in range(data.ncon):
        con = data.contact[c]
        if floor in (con.geom1, con.geom2):
            mujoco.mj_contactForce(model, data, c, f6)
            total_normal += abs(f6[0])
            for g in (con.geom1, con.geom2):
                if g != floor:
                    touching.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                                   model.geom_bodyid[g]))
    check(not ({"base", "wheel_r", "wheel_l"} & touching),
          "base/wheels do not touch the floor", str(sorted(touching)))
    check(total_normal < 100.0, "floor normal force < 100 N", f"{total_normal:.2f} N")

    crate_z = body_position(model, data, CRATE_BODY)[2]
    check(abs(crate_z - (TABLE_TOP_Z + CRATE_HALF[2])) < 0.002,
          "crate rests on the table", f"z={crate_z:.4f}")
    dofadr = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, CRATE_JOINT)]
    crate_vel = float(np.abs(data.qvel[dofadr:dofadr + 6]).max())
    check(crate_vel < 1e-3, "crate is at rest", f"|qvel|max={crate_vel:.2e}")
    check(np.isfinite(data.qpos).all(), "no NaN in qpos")
    print(f"    crate mass {state.crate_mass:.3f} kg")

    print("\nkey poses:")
    for side, pos in handle_positions(model, data).items():
        print(f"    handle {side:5s} {pos.round(4)}")
    print(f"    shelf target (level {state.target_level}) "
          f"{shelf_target_position(model, data, state.target_level).round(4)}")

    left, right, base = left_arm_handles(model), right_arm_handles(model), base_handles(model)
    s17 = build_state_17(model, data, left, right, base)
    a17 = build_action_17(model, data, left, right, base)
    check(s17.shape == (17,) and a17.shape == (17,), "state/action are 17-D")
    print(f"    state17  {np.round(s17, 3).tolist()}")

    print()
    if failures:
        print(f"SELF-CHECK FAILED ({len(failures)}): {failures}")
        return 1
    print("SELF-CHECK PASSED")
    return 0


def _reach_report(config_path: pathlib.Path, tol_mm: float = 10.0) -> int:
    """Dry-run the IK for every planned grasp/place pose and print the residual.

    This is the fast way to re-validate after editing transport_layout.json:
    it costs a few seconds instead of a full scenario run, and it catches a crate
    or shelf that has been nudged outside the arms' workspace.

    Imported lazily so that --self-check does not need mink.
    """
    from rby1_manipulation.control.bimanual import grasp_site_target, plan_error
    from rby1_manipulation.control.ik import (
        LEFT_ARM_JOINTS,
        RIGHT_ARM_JOINTS,
        build_dof_mask,
    )
    from rby1_manipulation.control.motion import ramp_base
    from rby1_manipulation.planning.transport import (
        CRATE_APPROACH_STANDOFF,
        CRATE_LIFT_DZ,
        SHELF_APPROACH_DZ,
        capture_grasp_frames,
    )

    config = load_layout_config(config_path)
    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data = mujoco.MjData(model)
    state = reset_transport_scene(model, data, config, settle_seconds=1.5)

    left, right = left_arm_handles(model), right_arm_handles(model)
    lmask = build_dof_mask(model, LEFT_ARM_JOINTS)
    rmask = build_dof_mask(model, RIGHT_ARM_JOINTS)
    frames = capture_grasp_frames(model, data)

    rows: list[tuple[str, str, float]] = []

    def probe(label: str, targets: Dict[str, np.ndarray], f) -> None:
        """`f` must be the frames the targets were built from - mixing the table
        wrist orientation with a shelf target silently costs tens of mm."""
        for side, pos in targets.items():
            arm, mask, rot = ((right, rmask, f.right) if side == "right"
                              else (left, lmask, f.left))
            rows.append((label, side, plan_error(model, data, arm, pos, mask, rot)))

    def handle_targets(source: Dict[str, np.ndarray], f, extra=np.zeros(3)) -> Dict[str, np.ndarray]:
        return {s: grasp_site_target(p + extra, f.right if s == "right" else f.left)
                for s, p in source.items()}

    handles = handle_positions(model, data)
    probe("crate_hover", {s: p + CRATE_APPROACH_STANDOFF * (frames.right if s == "right"
                                                            else frames.left).as_matrix()[:, 2]
                          for s, p in handle_targets(handles, frames).items()}, frames)
    probe("crate_descend", handle_targets(handles, frames), frames)
    probe("crate_lift", handle_targets(handles, frames, np.array([0.0, 0.0, CRATE_LIFT_DZ])), frames)

    # Shelf poses must be probed from the dock pose: the arms rotate with the
    # base, so both the reachable set and the EE orientation change.
    base = base_handles(model)
    dock = state.dock_pose
    ramp_base(model, data, base, [0.0, 0.0, dock[2]], 2.5)
    ramp_base(model, data, base, dock.tolist(), 3.5)
    for _ in range(600):
        mujoco.mj_step(model, data)
    dock_frames = capture_grasp_frames(model, data)
    # The crate is still sitting on the table in this dry run, so its live xmat
    # is NOT what it will have on arrival. Once grasped it is rigidly held by the
    # arms, so its yaw tracks the base's - predict with the dock yaw instead.
    crate_R = _yaw_matrix(float(dock[2]))
    centre = shelf_target_position(model, data, state.target_level) + np.array([0, 0, CRATE_HALF[2]])
    offsets = {"left": crate_R @ np.array([0.0, +0.200, CRATE_HANDLE_LOCAL_Z]),
               "right": crate_R @ np.array([0.0, -0.200, CRATE_HANDLE_LOCAL_Z])}
    seat = {s: centre + o for s, o in offsets.items()}
    hover = {s: p + np.array([0.0, 0.0, SHELF_APPROACH_DZ]) for s, p in seat.items()}
    probe("shelf_hover", handle_targets(hover, dock_frames), dock_frames)
    probe("shelf_seat", handle_targets(seat, dock_frames), dock_frames)

    print(f"reach report (tolerance {tol_mm:.0f} mm)\n")
    worst = 0.0
    for label, side, err in rows:
        mm = err * 1000.0
        worst = max(worst, mm)
        print(f"  [{'ok  ' if mm <= tol_mm else 'FAIL'}] {label:14s} {side:5s} {mm:6.1f} mm")
    print(f"\nworst residual {worst:.1f} mm")
    if worst > tol_mm:
        print("REACH REPORT FAILED")
        return 1
    print("REACH REPORT PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(DEFAULT_LAYOUT_CONFIG))
    ap.add_argument("--self-check", action="store_true",
                    help="validate the compiled model against every scenario invariant")
    ap.add_argument("--reach-report", action="store_true",
                    help="dry-run IK on every planned pose and print the residuals")
    ap.add_argument("--tol-mm", type=float, default=10.0)
    args = ap.parse_args()
    if args.self_check:
        return _self_check(pathlib.Path(args.config))
    if args.reach_report:
        return _reach_report(pathlib.Path(args.config), tol_mm=args.tol_mm)
    print(json.dumps(load_layout_config(args.config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
