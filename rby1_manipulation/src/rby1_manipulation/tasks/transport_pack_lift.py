"""Collect fixed-base 14-D fruit packing and crate lifting demonstrations.

The task is intentionally compositional:

* ``pack_only`` puts one to four instructed fruits in the crate and stops.
* ``lift_only`` lifts a crate that starts with zero to four fruits inside.
* ``pack_and_lift`` packs the instructed fruits, then lifts the crate.

No mode drives the base or moves toward the shelf.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from typing import Sequence

if "--headless" in sys.argv and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "osmesa"

import mujoco
import numpy as np

from rby1_manipulation.control.bimanual import execute_bimanual_waypoints
from rby1_manipulation.control.ik import (
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    build_dof_mask,
    left_arm_handles,
    right_arm_handles,
)
from rby1_manipulation.control.motion import (
    CRATE_SQUEEZE,
    SMALL_OBJ_SQUEEZE,
    adaptive_close,
    hold_ctrl_for_secs,
    open_grippers,
    return_arms_to_rest,
    snapshot_rest_ctrl,
)
from rby1_manipulation.evaluation.transport import check_grasp, object_in_crate
from rby1_manipulation.planning.transport import (
    arm_retract_waypoint,
    capture_grasp_frames,
    crate_approach_waypoints,
    crate_lift_waypoints,
    object_into_crate_waypoints,
    object_pick_waypoints,
)
from rby1_manipulation.simulation.common import pick_arm_for_block
from rby1_manipulation.simulation.fruit_grid import (
    DEFAULT_FRUIT_GRID_CONFIG,
    load_fruit_grid_config,
    reset_fruit_grid_scene,
)
from rby1_manipulation.simulation.transport_scene import (
    CRATE_BODY,
    MODEL_XML,
    OBJECT_BODIES,
    OBJECT_TYPES,
    RandomizationSpec,
    TABLE_BODY,
    base_handles,
    body_position,
    build_action_14,
    build_state_14,
    load_layout_config,
)


TASKS = ("pack_only", "lift_only", "pack_and_lift")
DEFAULT_SPEED_SCALE = 1.25
OBJECT_CLOSE_PRESS_SECS = 0.70
OBJECT_CLOSE_SETTLE_SECS = 0.40
# After releasing and retracting, return the used arm to its initial joint pose
# only when the next fruit belongs to the opposite arm. This clears the shared
# space above the crate without adding redundant home motions for same-arm runs.
ARM_SWITCH_RETURN_SECS = 1.5
CONTACT_MONITOR_HZ = 100.0
CAM_NAME_MAP = {
    "cam_high": "zed_left",
    "cam_left_wrist": "wrist_cam_l",
    "cam_right_wrist": "wrist_cam_r",
}


class GripperTableContactMonitor:
    """Track forbidden finger/table contacts during an object grasp."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.table_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, TABLE_BODY
        )
        self.finger_body_ids = {
            side: {
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"ee_finger_{tag}1"),
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"ee_finger_{tag}2"),
            }
            for side, tag in (("right", "r"), ("left", "l"))
        }
        self.finger_geom_masks = {
            side: np.isin(model.geom_bodyid, tuple(body_ids))
            for side, body_ids in self.finger_body_ids.items()
        }
        self.table_geom_mask = model.geom_bodyid == self.table_body_id
        self.active_side: str | None = None
        self.contact_steps = 0
        self.min_distance = float("inf")
        self.geom_pairs: set[tuple[str, str]] = set()

    def start(self, arm_side: str) -> None:
        self.active_side = arm_side
        self.contact_steps = 0
        self.min_distance = float("inf")
        self.geom_pairs.clear()

    def observe(self) -> None:
        if self.active_side is None or self.data.ncon == 0:
            return
        geom1 = np.asarray(self.data.contact.geom1, dtype=int)
        geom2 = np.asarray(self.data.contact.geom2, dtype=int)
        fingers = self.finger_geom_masks[self.active_side]
        matched = (
            (self.table_geom_mask[geom1] & fingers[geom2])
            | (self.table_geom_mask[geom2] & fingers[geom1])
        )
        matched_indices = np.flatnonzero(matched)
        if matched_indices.size:
            distances = np.asarray(self.data.contact.dist, dtype=float)[matched]
            self.min_distance = min(self.min_distance, float(np.min(distances)))
        for index in matched_indices:
            first, second = int(geom1[index]), int(geom2[index])
            name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, first)
            name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, second)
            self.geom_pairs.add((name1 or str(first), name2 or str(second)))
        if matched_indices.size:
            self.contact_steps += 1

    def stop(self) -> tuple[int, float, tuple[tuple[str, str], ...]]:
        result = self.contact_steps, self.min_distance, tuple(sorted(self.geom_pairs))
        self.active_side = None
        return result


class InterArmContactMonitor:
    """Track contacts between the left- and right-arm body subtrees."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.right_body_ids = self._subtree("link_right_arm_0")
        self.left_body_ids = self._subtree("link_left_arm_0")
        body_side = np.zeros(model.nbody, dtype=np.int8)
        body_side[list(self.right_body_ids)] = 1
        body_side[list(self.left_body_ids)] = 2
        self.geom_side = body_side[model.geom_bodyid]
        self.contact_steps = 0
        self.min_distance = float("inf")
        self.geom_pairs: set[tuple[str, str]] = set()

    def _subtree(self, root_name: str) -> set[int]:
        root_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, root_name
        )
        if root_id < 0:
            raise KeyError(f"missing arm root body: {root_name}")
        body_ids = {root_id}
        for body_id in range(root_id + 1, self.model.nbody):
            if int(self.model.body_parentid[body_id]) in body_ids:
                body_ids.add(body_id)
        return body_ids

    def observe(self) -> None:
        if self.data.ncon == 0:
            return
        geom1 = np.asarray(self.data.contact.geom1, dtype=int)
        geom2 = np.asarray(self.data.contact.geom2, dtype=int)
        side1 = self.geom_side[geom1]
        side2 = self.geom_side[geom2]
        matched = (
            ((side1 == 1) & (side2 == 2))
            | ((side1 == 2) & (side2 == 1))
        )
        matched_indices = np.flatnonzero(matched)
        if matched_indices.size:
            distances = np.asarray(self.data.contact.dist, dtype=float)[matched]
            self.min_distance = min(self.min_distance, float(np.min(distances)))
        for index in matched_indices:
            first, second = int(geom1[index]), int(geom2[index])
            name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, first)
            name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, second)
            self.geom_pairs.add((name1 or str(first), name2 or str(second)))
        if matched_indices.size:
            self.contact_steps += 1


def scaled_waypoints(
    waypoints,
    speed_scale: float,
    object_pre_close_hold_secs: float | None = None,
):
    """Shorten trajectory and settling times without changing the path."""
    if speed_scale <= 0.0:
        raise ValueError("speed_scale must be greater than zero")
    accelerated = []
    for waypoint in waypoints:
        wait_after = waypoint.wait_after / speed_scale
        if waypoint.label == "obj_descend_trim" and object_pre_close_hold_secs is not None:
            wait_after = min(wait_after, object_pre_close_hold_secs)
        accelerated.append(replace(
            waypoint,
            duration=waypoint.duration / speed_scale,
            wait_after=wait_after,
        ))
    return accelerated


def _unique_fruits(values: Sequence[str], label: str, *, allow_empty: bool) -> tuple[str, ...]:
    fruits = tuple(values)
    if not allow_empty and not fruits:
        raise ValueError(f"{label} requires at least one fruit")
    if len(fruits) > len(OBJECT_TYPES) or len(set(fruits)) != len(fruits):
        raise ValueError(f"{label} must contain zero to four unique fruits")
    unknown = set(fruits) - set(OBJECT_TYPES)
    if unknown:
        raise ValueError(f"{label} contains unknown fruits: {sorted(unknown)}")
    return fruits


def default_prompt(task: str, objects: Sequence[str]) -> str:
    canonical = [fruit for fruit in OBJECT_TYPES if fruit in objects]
    if task == "lift_only":
        return "lift the crate"
    if len(canonical) == len(OBJECT_TYPES):
        object_phrase = "all four fruits"
    elif len(canonical) == 1:
        object_phrase = f"the {canonical[0]}"
    else:
        object_phrase = "the " + ", ".join(canonical[:-1]) + f" and {canonical[-1]}"
    if task == "pack_only":
        return f"put {object_phrase} in the crate"
    return f"put {object_phrase} in the crate and lift the crate"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--objects", nargs="*", default=[] , choices=OBJECT_TYPES,
                        help="fruits to pack, in expert execution order")
    parser.add_argument("--preloaded", nargs="*", default=[], choices=OBJECT_TYPES,
                        help="fruits initially inside the crate for lift_only")
    parser.add_argument("--slot-order", nargs=4, default=list(OBJECT_TYPES), choices=OBJECT_TYPES,
                        help="fruit permutation assigned to the four table grid slots")
    parser.add_argument("--layout-index", type=int, default=0)
    parser.add_argument("--fruit-grid", default=str(DEFAULT_FRUIT_GRID_CONFIG))
    parser.add_argument("--config", default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--record", default=None)
    parser.add_argument("--log-dataset", default=None)
    parser.add_argument("--log-fps", type=int, default=15)
    parser.add_argument("--task-prompt", default=None)
    parser.add_argument("--save-failed", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-scene", action="store_true")
    parser.add_argument("--crate-jitter", type=float, default=0.005)
    parser.add_argument("--crate-yaw-jitter", type=float, default=0.0,
                        help="kept at zero for collection; nonzero yaw reduces handle contact")
    parser.add_argument("--mass-min", type=float, default=0.65)
    parser.add_argument("--mass-max", type=float, default=1.0)
    parser.add_argument("--friction-min", type=float, default=0.9)
    parser.add_argument("--friction-max", type=float, default=1.2)
    parser.add_argument("--grip-open", type=float, default=1.0)
    parser.add_argument(
        "--speed-scale",
        type=float,
        default=DEFAULT_SPEED_SCALE,
        help="motion speed multiplier; 1.0 restores the original timing",
    )
    parser.add_argument(
        "--object-pre-close-hold",
        type=float,
        default=None,
        help="optional maximum pause in seconds between final object approach and close",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.speed_scale <= 0.0:
        raise SystemExit("--speed-scale must be greater than zero")
    if args.object_pre_close_hold is not None and args.object_pre_close_hold < 0.0:
        raise SystemExit("--object-pre-close-hold must be non-negative")
    try:
        objects = _unique_fruits(args.objects, "--objects", allow_empty=False) \
            if args.task != "lift_only" else ()
        preloaded = _unique_fruits(args.preloaded, "--preloaded", allow_empty=True) \
            if args.task == "lift_only" else ()
        slot_order = _unique_fruits(args.slot_order, "--slot-order", allow_empty=False)
        if len(slot_order) != 4:
            raise ValueError("--slot-order must contain all four fruits")
        if args.task == "lift_only" and args.objects:
            raise ValueError("lift_only uses --preloaded, not --objects")
        if args.task != "lift_only" and args.preloaded:
            raise ValueError(f"{args.task} does not accept --preloaded")
    except ValueError as error:
        raise SystemExit(str(error)) from error

    layout = load_layout_config(args.config) if args.config else load_layout_config()
    fruit_grid = load_fruit_grid_config(args.fruit_grid)
    rng = np.random.default_rng(args.seed)
    randomization = RandomizationSpec()
    if args.random_scene:
        randomization = RandomizationSpec(
            crate_xy_jitter=args.crate_jitter,
            crate_yaw_jitter=args.crate_yaw_jitter,
            crate_mass_range=(args.mass_min, args.mass_max),
            friction_range=(args.friction_min, args.friction_max),
        )

    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data = mujoco.MjData(model)
    scene = reset_fruit_grid_scene(
        model,
        data,
        layout,
        fruit_grid,
        layout_index=args.layout_index,
        slot_order=slot_order,
        preloaded_objects=preloaded,
        rng=rng,
        randomize=randomization,
    )

    right, left = right_arm_handles(model), left_arm_handles(model)
    rmask = build_dof_mask(model, RIGHT_ARM_JOINTS)
    lmask = build_dof_mask(model, LEFT_ARM_JOINTS)
    arms = {"right": right, "left": left}
    rest_ctrl = snapshot_rest_ctrl(data, [right, left])
    base = base_handles(model)
    base_start = np.asarray([data.qpos[index] for index in base.qidx], dtype=float)
    prompt_objects = preloaded if args.task == "lift_only" else objects
    prompt = args.task_prompt or default_prompt(args.task, prompt_objects)

    print(f"=== transport dataset task: {args.task} ===")
    print(f"  prompt       : {prompt}")
    print(f"  pack order   : {list(objects)}")
    print(f"  preloaded    : {list(preloaded)}")
    print(f"  grid layout  : {args.layout_index}")
    print(f"  slot order   : {list(slot_order)}")
    print(f"  schema       : rby1_14")
    print(f"  speed scale  : {args.speed_scale:.2f}x")

    recorder = None
    logger = None
    if args.record or args.log_dataset:
        from rby1_manipulation.data.recording import EpisodeRecorder
        recorder = EpisodeRecorder(
            model,
            data,
            record_path=args.record,
            dataset_root=args.log_dataset,
            fps=args.log_fps,
            cam_name_map=CAM_NAME_MAP,
            task=prompt,
            schema="rby1_14",
            defer_policy_rendering=bool(args.log_dataset),
            model_xml_path=MODEL_XML,
            state_fn=lambda: build_state_14(data, left, right),
            action_fn=lambda: build_action_14(data, left, right),
        )
        logger = recorder.on_step

    table_contact_monitor = GripperTableContactMonitor(model, data)
    inter_arm_contact_monitor = InterArmContactMonitor(model, data)
    contact_monitor_period = max(
        1, int(round(1.0 / (CONTACT_MONITOR_HZ * model.opt.timestep)))
    )
    contact_monitor_step = 0

    def on_step() -> None:
        nonlocal contact_monitor_step
        contact_monitor_step += 1
        if contact_monitor_step % contact_monitor_period == 0:
            table_contact_monitor.observe()
            inter_arm_contact_monitor.observe()
        if logger is not None:
            logger()

    def run(viewer=None) -> bool:
        kwargs = {"viewer": viewer, "on_step": on_step}

        def execute(waypoints) -> None:
            execute_bimanual_waypoints(
                model,
                data,
                right_arm=right,
                left_arm=left,
                right_mask=rmask,
                left_mask=lmask,
                waypoints=scaled_waypoints(
                    waypoints,
                    args.speed_scale,
                    object_pre_close_hold_secs=args.object_pre_close_hold,
                ),
                **kwargs,
            )

        def secs(value: float) -> float:
            return value / args.speed_scale

        packed_objects: list[str] = list(preloaded)
        if args.task != "lift_only":
            for pack_index, fruit in enumerate(objects):
                object_body = OBJECT_BODIES[fruit]
                object_pos = body_position(model, data, object_body)
                arm_side = pick_arm_for_block(object_pos)
                frames = capture_grasp_frames(model, data)
                print(f"\n--- pack {pack_index + 1}/{len(objects)}: {fruit} with {arm_side} ---")
                table_contact_monitor.start(arm_side)
                execute(object_pick_waypoints(model, data, arm_side, frames, object_body))
                adaptive_close(
                    model,
                    data,
                    [arms[arm_side]],
                    squeeze=SMALL_OBJ_SQUEEZE,
                    press_secs=secs(OBJECT_CLOSE_PRESS_SECS),
                    settle_secs=secs(OBJECT_CLOSE_SETTLE_SECS),
                    **kwargs,
                )
                table_contact_monitor.observe()
                contact_steps, min_distance, geom_pairs = table_contact_monitor.stop()
                table_clear = contact_steps == 0
                print(
                    f"    gripper-table clear={table_clear} contact-steps={contact_steps} "
                    f"min-dist={min_distance if not table_clear else float('nan'):.5f} "
                    f"pairs={geom_pairs}"
                )
                if not table_clear:
                    return False
                grasp = check_grasp(model, data, {arm_side: arms[arm_side]}, object_body)
                print(f"    grasp ok={grasp.ok} contacts={grasp.contacts}")
                if not grasp.ok:
                    return False

                execute(object_into_crate_waypoints(
                    model,
                    data,
                    arm_side,
                    frames,
                    release_offset_xy=fruit_grid["crate_slots"][pack_index],
                ))
                open_grippers(model, data, [arms[arm_side]], secs=secs(0.8),
                              opening=args.grip_open, **kwargs)
                execute([arm_retract_waypoint(arm_side, frames, model, data)])
                hold_ctrl_for_secs(model, data, secs(0.8), **kwargs)
                inside = object_in_crate(model, data, object_body)
                print(f"    {fruit} in crate={inside} pos={body_position(model, data, object_body).round(3)}")
                if not inside:
                    return False
                packed_objects.append(fruit)

                if pack_index + 1 < len(objects):
                    next_fruit = objects[pack_index + 1]
                    next_pos = body_position(model, data, OBJECT_BODIES[next_fruit])
                    next_arm_side = pick_arm_for_block(next_pos)
                    if next_arm_side != arm_side:
                        print(
                            f"    arm switch {arm_side} -> {next_arm_side}: "
                            f"return {arm_side} arm to rest"
                        )
                        return_arms_to_rest(
                            model,
                            data,
                            [arms[arm_side]],
                            rest_ctrl,
                            duration=secs(ARM_SWITCH_RETURN_SECS),
                            **kwargs,
                        )

        if args.task in ("lift_only", "pack_and_lift"):
            print("\n--- lift crate (fixed base; no shelf transport) ---")
            if args.task == "pack_and_lift":
                print("    return both arms to the initial rest pose")
                return_arms_to_rest(
                    model, data, [right, left], rest_ctrl, duration=secs(2.0), **kwargs
                )
            hold_ctrl_for_secs(model, data, secs(1.5), **kwargs)
            crate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CRATE_BODY)
            crate_R = data.xmat[crate_id].reshape(3, 3)
            crate_yaw = float(np.arctan2(crate_R[1, 0], crate_R[0, 0]))
            print(f"    settled crate yaw={crate_yaw:+.4f} rad")
            frames = capture_grasp_frames(model, data)
            execute(crate_approach_waypoints(frames, trim=True))
            print(f"    loaded fruits={len(packed_objects)} squeeze={CRATE_SQUEEZE:.3f}")
            adaptive_close(
                model,
                data,
                [right, left],
                squeeze=CRATE_SQUEEZE,
                press_secs=secs(0.7),
                settle_secs=secs(0.4),
                **kwargs,
            )
            grasp = check_grasp(model, data, {"right": right, "left": left}, CRATE_BODY)
            print(f"    crate grasp ok={grasp.ok} contacts={grasp.contacts}")
            if not grasp.ok:
                return False
            execute(crate_lift_waypoints(
                model,
                data,
                frames,
                clear_z=float(layout["carry"]["lift_z"]),
            ))
            hold_ctrl_for_secs(model, data, secs(1.0), **kwargs)
            lifted_grasp = check_grasp(
                model,
                data,
                {"right": right, "left": left},
                CRATE_BODY,
                require_lift=True,
            )
            crate_z = float(body_position(model, data, CRATE_BODY)[2])
            height_ok = crate_z >= float(layout["carry"]["lift_z"]) - 0.04
            print(f"    crate z={crate_z:.3f} grasp-held={lifted_grasp.ok} height-ok={height_ok}")
            if not lifted_grasp.ok or not height_ok:
                return False
        else:
            hold_ctrl_for_secs(model, data, secs(1.5), **kwargs)

        packed_ok = all(
            object_in_crate(model, data, OBJECT_BODIES[fruit]) for fruit in packed_objects
        )
        base_now = np.asarray([data.qpos[index] for index in base.qidx], dtype=float)
        base_drift = float(np.max(np.abs(base_now - base_start)))
        base_ok = base_drift < 0.01
        arms_clear = inter_arm_contact_monitor.contact_steps == 0
        print(f"    packed objects remain={packed_ok}")
        print(f"    base drift max={base_drift:.5f} m/rad")
        print(
            f"    inter-arm clear={arms_clear} "
            f"contact-steps={inter_arm_contact_monitor.contact_steps} "
            f"pairs={tuple(sorted(inter_arm_contact_monitor.geom_pairs))}"
        )
        return packed_ok and base_ok and arms_clear

    if args.headless:
        success = run(None)
    else:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as viewer:
            success = run(viewer)

    print(f"\n>>> SUCCESS = {success}")
    if recorder is not None:
        recorder.finish(success=success, save_failed=args.save_failed)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
