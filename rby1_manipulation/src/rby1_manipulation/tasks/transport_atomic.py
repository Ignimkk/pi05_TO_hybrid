"""Atomic RBY1 fruit-to-basket and basket-lift demonstrations.

One invocation records exactly one language instruction.  Multi-instruction
experiments are represented by several episodes sharing ``sequence_group_id``;
this module never packs more than one newly instructed fruit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, replace
from enum import IntEnum
from pathlib import Path
from typing import Callable, Sequence

if "--headless" in sys.argv and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "osmesa"

import mujoco
import numpy as np

from rby1_manipulation.control.bimanual import (
    BiWaypoint,
    execute_bimanual_waypoints,
    resolve_target,
)
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
    wait_arms_at_rest,
)
from rby1_manipulation.data.recording import EpisodeRecorder
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
    CRATE_HALF,
    CRATE_JOINT,
    MODEL_XML,
    OBJECT_BODIES,
    OBJECT_JOINTS,
    OBJECT_TYPES,
    RandomizationSpec,
    TABLE_TOP_Z,
    TABLE_X_RANGE,
    TABLE_Y_RANGE,
    base_handles,
    body_position,
    build_action_14,
    build_state_14,
    free_body_pose,
    load_layout_config,
    site_position,
)
from rby1_manipulation.tasks.transport_pack_lift import (
    ARM_REST_SETTLE_TIMEOUT_SECS,
    ARM_REST_TOLERANCE_RAD,
    ARM_SWITCH_RETURN_SECS,
    CAM_NAME_MAP,
    CONTACT_MONITOR_HZ,
    DEFAULT_SPEED_SCALE,
    GripperTableContactMonitor,
    InterArmContactMonitor,
    OBJECT_CLOSE_PRESS_SECS,
    OBJECT_CLOSE_SETTLE_SECS,
    scaled_waypoints,
)


class Phase(IntEnum):
    INITIAL_HOLD = 0
    APPROACH = 1
    PREGRASP_ALIGN = 2
    GRIPPER_CLOSE = 3
    GRASP_VERIFY = 4
    LIFT_FROM_TABLE = 5
    TRANSPORT = 6
    LOWER_INTO_BASKET = 7
    RELEASE = 8
    RELEASE_VERIFY = 9
    RETREAT = 10
    TARGET_SETTLE = 11
    RETURN_TO_READY = 12
    TERMINAL_HOLD = 13
    RECOVERY_REOPEN = 20
    RECOVERY_RETREAT = 21
    RECOVERY_REAPPROACH = 22
    BASKET_APPROACH = 30
    BASKET_GRASP = 31
    BASKET_LIFT = 32


PHASE_NAMES = {int(phase): phase.name.lower() for phase in Phase}
RECOVERY_TYPES = (
    "empty_close",
    "early_close",
    "occlusion_reobserve",
    "unsafe_path_replan",
)
FAILURE_REASONS = (
    "target_initially_inside",
    "target_not_grasped",
    "wrong_target_grasped",
    "dropped_before_basket",
    "target_outside_basket",
    "incomplete_release",
    "target_unstable",
    "non_target_moved",
    "non_target_inserted",
    "preloaded_fruit_ejected",
    "unsafe_retreat",
    "arm_return_failed",
    "wrist_camera_view_invalid",
    "table_collision",
    "inter_arm_collision",
    "basket_not_grasped",
    "basket_not_lifted",
    "base_drift",
    "recovery_setup_invalid",
)

INITIAL_HOLD_SECS = 1.0
TARGET_SETTLE_SECS = 1.5
TERMINAL_HOLD_SECS = 2.0
ATOMIC_SCHEMA_VERSION = 2
NON_TARGET_DISPLACEMENT_TOLERANCE_M = 0.020
TARGET_LINEAR_SPEED_TOLERANCE_MPS = 0.030
RETREAT_TARGET_CLEARANCE_M = 0.100
RETREAT_ABOVE_RIM_M = 0.100
LIFT_HEIGHT_M = 0.150
WRIST_CAMERA_MIN_TABLE_CLEARANCE_M = 0.04
WRIST_CAMERA_TABLE_MARGIN_M = 0.01
WRIST_CAMERA_MAX_NEAR_PLANE_M = 0.03


def wrist_camera_table_view(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm_side: str,
) -> tuple[float, bool, list[float] | None]:
    """Return lens clearance and whether its centre ray lands on the tabletop."""
    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, f"wrist_cam_{arm_side[0]}"
    )
    if camera_id < 0:
        raise KeyError(f"missing wrist camera for {arm_side}")
    position = np.asarray(data.cam_xpos[camera_id], dtype=float)
    rotation = np.asarray(data.cam_xmat[camera_id], dtype=float).reshape(3, 3)
    forward = -rotation[:, 2]
    clearance = float(position[2] - TABLE_TOP_Z)
    if forward[2] >= -1e-9:
        return clearance, False, None
    distance = (TABLE_TOP_Z - position[2]) / forward[2]
    if distance <= 0.0:
        return clearance, False, None
    intersection = position + distance * forward
    margin = WRIST_CAMERA_TABLE_MARGIN_M
    on_table = (
        TABLE_X_RANGE[0] + margin <= intersection[0] <= TABLE_X_RANGE[1] - margin
        and TABLE_Y_RANGE[0] + margin <= intersection[1] <= TABLE_Y_RANGE[1] - margin
    )
    return clearance, bool(on_table), intersection.round(6).tolist()


def canonical_prompt(task: str, target_fruit: str | None = None) -> str:
    if task == "lift_basket":
        return "lift the basket"
    if target_fruit not in OBJECT_TYPES:
        raise ValueError("place_one requires a target fruit")
    return f"put the {target_fruit} in the basket"


@dataclass
class PhaseTracker:
    recorder: EpisodeRecorder | None = None
    phase: Phase = Phase.INITIAL_HOLD
    events: dict[str, int] = field(default_factory=dict)

    def frame(self) -> int:
        if self.recorder is None or self.recorder.episode is None:
            return 0
        return len(self.recorder.episode)

    def set(self, phase: Phase, event: str | None = None) -> None:
        self.phase = phase
        self.events.setdefault(event or f"{PHASE_NAMES[int(phase)]}_start_frame", self.frame())


@dataclass
class RuntimeMetrics:
    terminal_target_max_speed: float = 0.0
    table_contact_steps: int = 0
    table_contact_pairs: tuple = ()
    inter_arm_contact_steps: int = 0
    inter_arm_contact_pairs: tuple = ()


def _append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def _write_atomic_schema(root: Path) -> None:
    path = root / "meta" / "atomic_schema.json"
    if path.exists():
        return
    value = {
        "version": ATOMIC_SCHEMA_VERSION,
        "container_semantic": "basket",
        "internal_container_body": CRATE_BODY,
        "control_schema": "rby1_14_with_scripted_ik_arm_joint_6",
        "camera_preprocess": "render_4_3_then_resize_224_square",
        "phase_labels": {str(key): value for key, value in PHASE_NAMES.items()},
        "failure_reasons": list(FAILURE_REASONS),
        "recovery_types": list(RECOVERY_TYPES),
    }
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _check_dataset_schema(root: Path) -> None:
    """Refuse to append v2 episodes to the incompatible pilot dataset."""
    schema_path = root / "meta" / "atomic_schema.json"
    episodes_path = root / "meta" / "episodes.jsonl"
    if not schema_path.exists():
        if episodes_path.exists():
            raise RuntimeError(
                f"{root} already has episodes but no atomic schema; use a new dataset root"
            )
        return
    value = json.loads(schema_path.read_text(encoding="utf-8"))
    version = int(value.get("version", -1))
    if version != ATOMIC_SCHEMA_VERSION:
        raise RuntimeError(
            f"{root} uses atomic schema v{version}; the corrected collector uses "
            f"v{ATOMIC_SCHEMA_VERSION}. Use a new dataset root instead of mixing episodes."
        )


def _free_body_speed(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str) -> float:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    dof = int(model.jnt_dofadr[joint_id])
    return float(np.linalg.norm(data.qvel[dof:dof + 3]))


def _offset_waypoint(wp: BiWaypoint, arm_side: str, offset: Sequence[float]) -> BiWaypoint:
    key = "right_pos" if arm_side == "right" else "left_pos"
    original = getattr(wp, key)
    delta = np.asarray(offset, dtype=float)

    def shifted(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        return resolve_target(original, model, data) + delta

    return replace(wp, **{key: shifted})


def _pose_dict(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, list[float]]:
    return {
        fruit: free_body_pose(model, data, OBJECT_JOINTS[fruit]).round(7).tolist()
        for fruit in OBJECT_TYPES
    }


def _basket_membership(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, bool]:
    return {
        fruit: object_in_crate(model, data, OBJECT_BODIES[fruit])
        for fruit in OBJECT_TYPES
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=("place_one", "lift_basket"), required=True)
    parser.add_argument("--target-fruit", choices=OBJECT_TYPES, default=None)
    parser.add_argument("--preloaded", nargs="*", choices=OBJECT_TYPES, default=[])
    parser.add_argument("--slot-order", nargs=4, choices=OBJECT_TYPES,
                        default=list(OBJECT_TYPES))
    parser.add_argument("--layout-index", type=int, default=0)
    parser.add_argument("--fruit-grid", default=str(DEFAULT_FRUIT_GRID_CONFIG))
    parser.add_argument("--config", default=None)
    parser.add_argument("--task-prompt", default=None)
    parser.add_argument("--canonical-prompt", default=None)
    parser.add_argument("--is-paraphrase", action="store_true")
    parser.add_argument("--scenario-family", default="clean_single")
    parser.add_argument("--recovery-type", choices=RECOVERY_TYPES, default=None)
    parser.add_argument("--sequence-group-id", default=None)
    parser.add_argument("--sequence-step", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=1)
    parser.add_argument("--plan-index", type=int, default=-1)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--record", default=None)
    parser.add_argument("--log-dataset", default=None)
    parser.add_argument("--log-fps", type=int, default=15)
    parser.add_argument("--save-failed", action="store_true")
    parser.add_argument("--random-scene", action="store_true")
    parser.add_argument("--basket-jitter", type=float, default=0.010)
    parser.add_argument("--basket-yaw-jitter", type=float, default=0.05236)
    parser.add_argument("--fruit-jitter", type=float, default=0.006)
    parser.add_argument("--mass-min", type=float, default=0.65)
    parser.add_argument("--mass-max", type=float, default=1.0)
    parser.add_argument("--friction-min", type=float, default=0.9)
    parser.add_argument("--friction-max", type=float, default=1.2)
    parser.add_argument("--speed-scale", type=float, default=DEFAULT_SPEED_SCALE)
    parser.add_argument("--initial-hold", type=float, default=INITIAL_HOLD_SECS)
    parser.add_argument("--terminal-hold", type=float, default=TERMINAL_HOLD_SECS)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    target = args.target_fruit
    preloaded = tuple(args.preloaded)
    slot_order = tuple(args.slot_order)
    if args.task == "place_one" and target is None:
        raise SystemExit("place_one requires --target-fruit")
    if args.task == "lift_basket" and target is not None:
        raise SystemExit("lift_basket does not accept --target-fruit")
    if len(set(preloaded)) != len(preloaded):
        raise SystemExit("--preloaded fruits must be unique")
    if target in preloaded:
        raise SystemExit("target fruit cannot already be in the basket")
    if len(slot_order) != 4 or set(slot_order) != set(OBJECT_TYPES):
        raise SystemExit("--slot-order must be a permutation of all four fruits")
    if args.initial_hold < 1.0 or args.terminal_hold < 1.0:
        raise SystemExit("initial and terminal hold must both be at least 1.0 second")
    if args.speed_scale <= 0.0:
        raise SystemExit("--speed-scale must be positive")
    if args.log_dataset:
        _check_dataset_schema(Path(args.log_dataset))

    prompt_canonical = args.canonical_prompt or canonical_prompt(args.task, target)
    prompt = args.task_prompt or prompt_canonical
    layout = load_layout_config(args.config) if args.config else load_layout_config()
    grid = load_fruit_grid_config(args.fruit_grid)
    rng = np.random.default_rng(args.seed)
    randomization = RandomizationSpec()
    if args.random_scene:
        randomization = RandomizationSpec(
            crate_xy_jitter=args.basket_jitter,
            crate_yaw_jitter=args.basket_yaw_jitter,
            crate_mass_range=(args.mass_min, args.mass_max),
            friction_range=(args.friction_min, args.friction_max),
        )

    model = mujoco.MjModel.from_xml_path(MODEL_XML)
    data = mujoco.MjData(model)
    render_near_plane_m = float(model.stat.extent * model.vis.map.znear)
    reset_fruit_grid_scene(
        model,
        data,
        layout,
        grid,
        layout_index=args.layout_index,
        slot_order=slot_order,
        preloaded_objects=preloaded,
        rng=rng,
        randomize=randomization,
        position_jitter_xy=args.fruit_jitter if args.random_scene else 0.0,
    )

    right, left = right_arm_handles(model), left_arm_handles(model)
    arms = {"right": right, "left": left}
    rmask = build_dof_mask(model, RIGHT_ARM_JOINTS)
    lmask = build_dof_mask(model, LEFT_ARM_JOINTS)
    open_grippers(model, data, [right, left], secs=0.5, opening=1.0)
    rest_ctrl = snapshot_rest_ctrl(data, [right, left])
    unobserved_arm_6_start = {
        side: float(data.qpos[arm.qidx[6]]) for side, arm in arms.items()
    }
    base = base_handles(model)
    base_start = np.asarray([data.qpos[index] for index in base.qidx], dtype=float)
    initial_membership = _basket_membership(model, data)
    initial_poses = _pose_dict(model, data)
    initial_basket_pose = free_body_pose(model, data, CRATE_JOINT).copy()

    tracker = PhaseTracker()
    recorder: EpisodeRecorder | None = None
    if args.record or args.log_dataset:
        recorder = EpisodeRecorder(
            model,
            data,
            state_fn=lambda: build_state_14(data, left, right),
            action_fn=lambda: build_action_14(data, left, right),
            cam_name_map=CAM_NAME_MAP,
            task=prompt,
            record_path=args.record,
            dataset_root=args.log_dataset,
            fps=args.log_fps,
            schema="rby1_14",
            defer_policy_rendering=bool(args.log_dataset),
            model_xml_path=MODEL_XML,
            phase_fn=lambda: int(tracker.phase),
            prompt_timestamp=0.0,
        )
        tracker.recorder = recorder

    table_monitor = GripperTableContactMonitor(model, data)
    inter_arm_monitor = InterArmContactMonitor(model, data)
    contact_period = max(1, int(round(1.0 / (CONTACT_MONITOR_HZ * model.opt.timestep))))
    semantic_period = max(1, int(round(1.0 / (args.log_fps * model.opt.timestep))))
    monitor_step = 0
    terminal_target_speeds: list[float] = []
    terminal_target_positions: list[np.ndarray] = []
    active_arm_side: str | None = None
    non_target_max_displacement = {
        fruit: 0.0 for fruit in OBJECT_TYPES
        if fruit != target and not initial_membership[fruit]
    }
    non_target_ever_inserted = False
    preloaded_ever_ejected = False
    wrong_target_grasped = False
    unobserved_arm_6_max_motion = {"left": 0.0, "right": 0.0}
    wrist_camera_monitor_active = False
    wrist_camera_samples = 0
    wrist_camera_off_table_steps = 0
    wrist_camera_low_steps = 0
    wrist_camera_min_clearance = float("inf")
    wrist_camera_last_intersection: list[float] | None = None

    def on_step() -> None:
        nonlocal monitor_step, non_target_ever_inserted
        nonlocal preloaded_ever_ejected, wrong_target_grasped
        nonlocal wrist_camera_samples, wrist_camera_off_table_steps
        nonlocal wrist_camera_low_steps, wrist_camera_min_clearance
        nonlocal wrist_camera_last_intersection
        monitor_step += 1
        if monitor_step % contact_period == 0:
            table_monitor.observe()
            inter_arm_monitor.observe()
            for side, arm_handle in arms.items():
                motion = abs(
                    float(data.qpos[arm_handle.qidx[6]])
                    - unobserved_arm_6_start[side]
                )
                unobserved_arm_6_max_motion[side] = max(
                    unobserved_arm_6_max_motion[side], motion
                )
            if wrist_camera_monitor_active and active_arm_side is not None:
                clearance, on_table, intersection = wrist_camera_table_view(
                    model, data, active_arm_side
                )
                wrist_camera_samples += 1
                wrist_camera_min_clearance = min(
                    wrist_camera_min_clearance, clearance
                )
                wrist_camera_last_intersection = intersection
                wrist_camera_off_table_steps += int(not on_table)
                wrist_camera_low_steps += int(
                    clearance < WRIST_CAMERA_MIN_TABLE_CLEARANCE_M
                )
            if (
                target is not None
                and active_arm_side is not None
                and tracker.phase in {
                    Phase.GRIPPER_CLOSE,
                    Phase.GRASP_VERIFY,
                    Phase.LIFT_FROM_TABLE,
                    Phase.TRANSPORT,
                    Phase.LOWER_INTO_BASKET,
                }
            ):
                for fruit in OBJECT_TYPES:
                    if fruit == target:
                        continue
                    wrong_target_grasped |= check_grasp(
                        model,
                        data,
                        {active_arm_side: arms[active_arm_side]},
                        OBJECT_BODIES[fruit],
                    ).ok
            if tracker.phase == Phase.TERMINAL_HOLD and target is not None:
                terminal_target_speeds.append(
                    _free_body_speed(model, data, OBJECT_JOINTS[target])
                )
                terminal_target_positions.append(
                    body_position(model, data, OBJECT_BODIES[target])
                )
        if target is not None and monitor_step % semantic_period == 0:
            for fruit in OBJECT_TYPES:
                if fruit == target:
                    continue
                membership_now = object_in_crate(model, data, OBJECT_BODIES[fruit])
                if initial_membership[fruit]:
                    preloaded_ever_ejected |= not membership_now
                else:
                    non_target_ever_inserted |= membership_now
                    displacement = float(np.linalg.norm(
                        body_position(model, data, OBJECT_BODIES[fruit])
                        - np.asarray(initial_poses[fruit][:3])
                    ))
                    non_target_max_displacement[fruit] = max(
                        non_target_max_displacement[fruit], displacement
                    )
        if recorder is not None:
            recorder.on_step()

    def run(viewer=None) -> tuple[bool, str | None, dict, str | None, int, int]:
        nonlocal active_arm_side, wrist_camera_monitor_active
        kwargs = {"viewer": viewer, "on_step": on_step}

        def execute(phase: Phase, waypoints: Sequence[BiWaypoint], event: str | None = None) -> None:
            tracker.set(phase, event)
            execute_bimanual_waypoints(
                model,
                data,
                right_arm=right,
                left_arm=left,
                right_mask=rmask,
                left_mask=lmask,
                waypoints=scaled_waypoints(waypoints, args.speed_scale),
                **kwargs,
            )

        tracker.set(Phase.INITIAL_HOLD, "initial_hold_start_frame")
        hold_ctrl_for_secs(model, data, args.initial_hold, **kwargs)
        tracker.events["initial_hold_end_frame"] = tracker.frame()

        if args.task == "lift_basket":
            frames = capture_grasp_frames(model, data)
            execute(Phase.BASKET_APPROACH, crate_approach_waypoints(frames, trim=True))
            tracker.set(Phase.BASKET_GRASP, "grasp_close_start_frame")
            adaptive_close(
                model,
                data,
                [right, left],
                squeeze=CRATE_SQUEEZE,
                press_secs=0.7 / args.speed_scale,
                settle_secs=0.4 / args.speed_scale,
                **kwargs,
            )
            grasp = check_grasp(model, data, {"right": right, "left": left}, CRATE_BODY)
            hold_ctrl_for_secs(model, data, 0.2, **kwargs)
            tracker.events["grasp_verified_frame"] = tracker.frame()
            if not grasp.ok:
                return False, "basket_not_grasped", {"basket_grasp": False}, "both", 1, 0
            execute(
                Phase.BASKET_LIFT,
                crate_lift_waypoints(model, data, frames, clear_z=float(layout["carry"]["lift_z"])),
            )
            tracker.events["success_frame"] = tracker.frame()
            tracker.set(Phase.TERMINAL_HOLD, "terminal_hold_start_frame")
            hold_ctrl_for_secs(model, data, args.terminal_hold, **kwargs)
            tracker.events["terminal_hold_end_frame"] = tracker.frame()
            basket_z_delta = float(body_position(model, data, CRATE_BODY)[2] - initial_basket_pose[2])
            held = check_grasp(
                model, data, {"right": right, "left": left}, CRATE_BODY, require_lift=True
            ).ok
            membership = _basket_membership(model, data)
            contents_ok = all(membership[fruit] for fruit in preloaded)
            base_drift = float(np.max(np.abs(
                np.asarray([data.qpos[index] for index in base.qidx]) - base_start
            )))
            checks = {
                "basket_grasp_held": held,
                "basket_lift_height_m": basket_z_delta,
                "preloaded_fruits_remain": contents_ok,
                "base_drift": base_drift,
                "inter_arm_clear": inter_arm_monitor.contact_steps == 0,
                "unobserved_arm_6_max_motion_rad": unobserved_arm_6_max_motion,
                "terminal_hold_valid": (
                    (
                        tracker.events["terminal_hold_end_frame"]
                        - tracker.events["terminal_hold_start_frame"]
                    ) >= args.log_fps
                    if recorder is not None else args.terminal_hold >= 1.0
                ),
            }
            if not held or basket_z_delta < LIFT_HEIGHT_M:
                return False, "basket_not_lifted", checks, "both", 1, 0
            if not contents_ok:
                return False, "preloaded_fruit_ejected", checks, "both", 1, 0
            if base_drift >= 0.01:
                return False, "base_drift", checks, "both", 1, 0
            if inter_arm_monitor.contact_steps:
                return False, "inter_arm_collision", checks, "both", 1, 0
            return True, None, checks, "both", 1, 0

        assert target is not None
        if initial_membership[target]:
            return False, "target_initially_inside", {}, None, 0, 0
        object_body = OBJECT_BODIES[target]
        arm_side = pick_arm_for_block(body_position(model, data, object_body))
        active_arm_side = arm_side
        arm = arms[arm_side]
        frames = capture_grasp_frames(model, data)
        grasp_attempts = 0
        grasp_retries = 0

        def reopen_and_retract() -> None:
            tracker.set(Phase.RECOVERY_REOPEN)
            open_grippers(model, data, [arm], secs=0.5, opening=1.0, **kwargs)
            execute(Phase.RECOVERY_RETREAT, [arm_retract_waypoint(
                arm_side, frames, model, data, dz=0.08
            )])

        # Deterministic recovery prefixes teach the policy to stop, open and
        # re-approach without ever making an unsafe collision part of the expert.
        if args.recovery_type:
            pick = object_pick_waypoints(model, data, arm_side, frames, object_body)
            if args.recovery_type in ("empty_close", "early_close"):
                prefix = pick[:2] if args.recovery_type == "empty_close" else pick[:1]
                execute(Phase.APPROACH, prefix)
                high = _offset_waypoint(pick[-1], arm_side, (0.0, 0.0, 0.08))
                execute(Phase.PREGRASP_ALIGN, [high])
                tracker.set(Phase.GRIPPER_CLOSE, "recovery_close_start_frame")
                grasp_attempts += 1
                adaptive_close(
                    model, data, [arm], squeeze=SMALL_OBJ_SQUEEZE,
                    press_secs=OBJECT_CLOSE_PRESS_SECS / args.speed_scale,
                    settle_secs=OBJECT_CLOSE_SETTLE_SECS / args.speed_scale,
                    **kwargs,
                )
                false_grasp = check_grasp(model, data, {arm_side: arm}, object_body)
                if false_grasp.ok:
                    return False, "recovery_setup_invalid", {}, arm_side, grasp_attempts, 0
                grasp_retries += 1
                reopen_and_retract()
            elif args.recovery_type == "occlusion_reobserve":
                execute(Phase.APPROACH, pick[:2])
                reopen_and_retract()
            elif args.recovery_type == "unsafe_path_replan":
                execute(Phase.APPROACH, pick[:1])
                reopen_and_retract()

        tracker.set(Phase.RECOVERY_REAPPROACH if args.recovery_type else Phase.APPROACH)
        pick = object_pick_waypoints(model, data, arm_side, frames, object_body)
        execute(Phase.APPROACH, pick[:2], "approach_start_frame")
        table_monitor.start(arm_side)
        wrist_camera_monitor_active = True
        execute(Phase.PREGRASP_ALIGN, pick[2:], "pregrasp_start_frame")
        tracker.set(Phase.GRIPPER_CLOSE, "grasp_close_start_frame")
        grasp_attempts += 1
        adaptive_close(
            model,
            data,
            [arm],
            squeeze=SMALL_OBJ_SQUEEZE,
            press_secs=OBJECT_CLOSE_PRESS_SECS / args.speed_scale,
            settle_secs=OBJECT_CLOSE_SETTLE_SECS / args.speed_scale,
            **kwargs,
        )
        table_steps, _, table_pairs = table_monitor.stop()
        wrist_camera_monitor_active = False
        tracker.set(Phase.GRASP_VERIFY, "grasp_verify_start_frame")
        grasp = check_grasp(model, data, {arm_side: arm}, object_body)
        hold_ctrl_for_secs(model, data, 0.2, **kwargs)
        tracker.events["grasp_verified_frame"] = tracker.frame()
        if table_steps:
            return False, "table_collision", {"table_contact_pairs": table_pairs}, arm_side, grasp_attempts, grasp_retries
        camera_view_valid = (
            wrist_camera_samples > 0
            and wrist_camera_off_table_steps == 0
            and wrist_camera_low_steps == 0
            and render_near_plane_m <= WRIST_CAMERA_MAX_NEAR_PLANE_M
        )
        camera_checks = {
            "wrist_camera_view_valid": camera_view_valid,
            "wrist_camera_samples": wrist_camera_samples,
            "wrist_camera_off_table_steps": wrist_camera_off_table_steps,
            "wrist_camera_low_steps": wrist_camera_low_steps,
            "wrist_camera_min_table_clearance_m": wrist_camera_min_clearance,
            "render_near_plane_m": render_near_plane_m,
            "wrist_camera_last_table_intersection": wrist_camera_last_intersection,
        }
        if not camera_view_valid:
            return False, "wrist_camera_view_invalid", camera_checks, arm_side, grasp_attempts, grasp_retries
        if not grasp.ok:
            return False, "target_not_grasped", {
                **camera_checks, "grasp_contacts": grasp.contacts
            }, arm_side, grasp_attempts, grasp_retries

        transport = object_into_crate_waypoints(
            model,
            data,
            arm_side,
            frames,
            release_offset_xy=grid["crate_slots"][len(preloaded)],
        )
        execute(Phase.LIFT_FROM_TABLE, transport[:1])
        execute(Phase.TRANSPORT, transport[1:2])
        execute(Phase.LOWER_INTO_BASKET, transport[2:])
        tracker.set(Phase.RELEASE, "release_start_frame")
        open_grippers(model, data, [arm], secs=0.8 / args.speed_scale,
                      opening=1.0, **kwargs)
        tracker.events["release_end_frame"] = tracker.frame()
        tracker.set(Phase.RELEASE_VERIFY)
        released = not check_grasp(model, data, {arm_side: arm}, object_body).ok
        hold_ctrl_for_secs(model, data, 0.2, **kwargs)
        tracker.set(Phase.RETREAT, "retreat_start_frame")
        execute_bimanual_waypoints(
            model,
            data,
            right_arm=right,
            left_arm=left,
            right_mask=rmask,
            left_mask=lmask,
            waypoints=scaled_waypoints(
                [arm_retract_waypoint(arm_side, frames, model, data)], args.speed_scale
            ),
            **kwargs,
        )
        tracker.events["retreat_end_frame"] = tracker.frame()
        tracker.set(Phase.TARGET_SETTLE, "target_settle_start_frame")
        hold_ctrl_for_secs(model, data, TARGET_SETTLE_SECS, **kwargs)

        membership = _basket_membership(model, data)
        speed = _free_body_speed(model, data, OBJECT_JOINTS[target])
        ee_pos = site_position(model, data, arm.ee_site)
        target_pos = body_position(model, data, object_body)
        basket_pos = body_position(model, data, CRATE_BODY)
        target_clearance = float(np.linalg.norm(ee_pos - target_pos))
        above_rim = float(ee_pos[2] - (basket_pos[2] + CRATE_HALF[2]))
        non_targets = [fruit for fruit in OBJECT_TYPES if fruit != target]
        final_poses_now = _pose_dict(model, data)
        table_displacements = {
            fruit: float(np.linalg.norm(
                np.asarray(final_poses_now[fruit][:3]) - np.asarray(initial_poses[fruit][:3])
            ))
            for fruit in non_targets if not initial_membership[fruit]
        }
        non_target_inserted = non_target_ever_inserted or any(
            membership[fruit] and not initial_membership[fruit] for fruit in non_targets
        )
        preloaded_remain = (
            not preloaded_ever_ejected
            and all(membership[fruit] for fruit in preloaded)
        )
        non_target_stationary = all(
            value <= NON_TARGET_DISPLACEMENT_TOLERANCE_M
            for value in non_target_max_displacement.values()
        )
        safe_retreat = (
            target_clearance >= RETREAT_TARGET_CLEARANCE_M
            and above_rim >= RETREAT_ABOVE_RIM_M
        )
        pre_terminal_checks = {
            "target_newly_inside": membership[target] and not initial_membership[target],
            "fully_released": released,
            "target_speed_mps": speed,
            "target_stable": speed <= TARGET_LINEAR_SPEED_TOLERANCE_MPS,
            "non_target_inserted": non_target_inserted,
            "non_target_final_displacement_m": table_displacements,
            "non_target_max_displacement_m": non_target_max_displacement,
            "non_target_unchanged": non_target_stationary,
            "wrong_target_grasped": wrong_target_grasped,
            "preloaded_fruits_remain": preloaded_remain,
            "retreat_target_clearance_m": target_clearance,
            "retreat_above_rim_m": above_rim,
            "safe_retreat": safe_retreat,
            **camera_checks,
            "unobserved_arm_6_max_motion_rad": unobserved_arm_6_max_motion,
        }
        if not membership[target]:
            return False, "target_outside_basket", pre_terminal_checks, arm_side, grasp_attempts, grasp_retries
        if not released:
            return False, "incomplete_release", pre_terminal_checks, arm_side, grasp_attempts, grasp_retries
        if speed > TARGET_LINEAR_SPEED_TOLERANCE_MPS:
            return False, "target_unstable", pre_terminal_checks, arm_side, grasp_attempts, grasp_retries
        if non_target_inserted:
            return False, "non_target_inserted", pre_terminal_checks, arm_side, grasp_attempts, grasp_retries
        if wrong_target_grasped:
            return False, "wrong_target_grasped", pre_terminal_checks, arm_side, grasp_attempts, grasp_retries
        if not non_target_stationary:
            return False, "non_target_moved", pre_terminal_checks, arm_side, grasp_attempts, grasp_retries
        if not preloaded_remain:
            return False, "preloaded_fruit_ejected", pre_terminal_checks, arm_side, grasp_attempts, grasp_retries
        if not safe_retreat:
            return False, "unsafe_retreat", pre_terminal_checks, arm_side, grasp_attempts, grasp_retries
        tracker.set(Phase.RETURN_TO_READY, "return_to_ready_start_frame")
        return_arms_to_rest(
            model,
            data,
            [arm],
            rest_ctrl,
            duration=ARM_SWITCH_RETURN_SECS / args.speed_scale,
            **kwargs,
        )
        try:
            ready_error = wait_arms_at_rest(
                model,
                data,
                [arm],
                rest_ctrl,
                tolerance=ARM_REST_TOLERANCE_RAD,
                timeout=ARM_REST_SETTLE_TIMEOUT_SECS,
                **kwargs,
            )
        except RuntimeError as error:
            return False, "arm_return_failed", {
                **pre_terminal_checks,
                "returned_to_ready": False,
                "arm_return_error": str(error),
            }, arm_side, grasp_attempts, grasp_retries
        tracker.events["return_to_ready_end_frame"] = tracker.frame()
        tracker.events["success_frame"] = tracker.frame()
        tracker.set(Phase.TERMINAL_HOLD, "terminal_hold_start_frame")
        hold_ctrl_for_secs(model, data, args.terminal_hold, **kwargs)
        tracker.events["terminal_hold_end_frame"] = tracker.frame()
        final_membership = _basket_membership(model, data)
        tail_samples = max(1, int(round(0.5 * CONTACT_MONITOR_HZ)))
        terminal_tail_max_speed = max(terminal_target_speeds[-tail_samples:], default=0.0)
        terminal_final_speed = _free_body_speed(model, data, OBJECT_JOINTS[target])
        tail_positions = terminal_target_positions[-tail_samples:]
        terminal_tail_displacement = max(
            (float(np.linalg.norm(position - tail_positions[-1])) for position in tail_positions),
            default=0.0,
        )
        base_drift = float(np.max(np.abs(
            np.asarray([data.qpos[index] for index in base.qidx]) - base_start
        )))
        final_ready_error = max(
            abs(float(data.qpos[qidx]) - float(rest_ctrl[aid]))
            for aid, qidx in zip(arm.aid, arm.qidx)
        )
        checks = {
            **pre_terminal_checks,
            "used_arm_ready_error_rad": final_ready_error,
            "returned_to_ready": final_ready_error <= ARM_REST_TOLERANCE_RAD,
            "target_inside_after_hold": final_membership[target],
            "terminal_target_max_speed_mps": terminal_tail_max_speed,
            "terminal_target_final_speed_mps": terminal_final_speed,
            "terminal_target_tail_displacement_m": terminal_tail_displacement,
            "terminal_hold_valid": (
                (
                    tracker.events["terminal_hold_end_frame"]
                    - tracker.events["terminal_hold_start_frame"]
                ) >= args.log_fps
                if recorder is not None else args.terminal_hold >= 1.0
            ),
            "base_drift": base_drift,
            "inter_arm_clear": inter_arm_monitor.contact_steps == 0,
        }
        if (
            not final_membership[target]
            or terminal_final_speed > TARGET_LINEAR_SPEED_TOLERANCE_MPS
            or terminal_tail_displacement > 0.010
        ):
            return False, "target_unstable", checks, arm_side, grasp_attempts, grasp_retries
        if base_drift >= 0.01:
            return False, "base_drift", checks, arm_side, grasp_attempts, grasp_retries
        if not checks["returned_to_ready"]:
            return False, "arm_return_failed", checks, arm_side, grasp_attempts, grasp_retries
        if inter_arm_monitor.contact_steps:
            return False, "inter_arm_collision", checks, arm_side, grasp_attempts, grasp_retries
        return True, None, checks, arm_side, grasp_attempts, grasp_retries

    print("=== atomic transport task ===")
    print(f"  prompt       : {prompt}")
    print(f"  task         : {args.task}")
    print(f"  target       : {target}")
    print(f"  preloaded    : {list(preloaded)}")
    print(f"  family       : {args.scenario_family}")
    print(f"  recovery     : {args.recovery_type}")

    if args.headless:
        success, failure_reason, checks, used_arm, grasp_attempts, grasp_retries = run(None)
    else:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as viewer:
            success, failure_reason, checks, used_arm, grasp_attempts, grasp_retries = run(viewer)

    print(f"\n>>> SUCCESS = {success}")
    print(f">>> FAILURE_REASON = {failure_reason}")
    print(f">>> VALIDATION = {json.dumps(checks, sort_keys=True)}")
    dataset_episode_index = None
    if recorder is not None:
        dataset_episode_index = recorder.finish(success=success, save_failed=args.save_failed)

    if dataset_episode_index is not None and args.log_dataset:
        final_poses = _pose_dict(model, data)
        metadata = {
            "schema_version": ATOMIC_SCHEMA_VERSION,
            "episode_index": dataset_episode_index,
            "plan_index": args.plan_index,
            "split": args.split,
            "scenario_family": args.scenario_family,
            "sequence_group_id": args.sequence_group_id,
            "sequence_step": args.sequence_step,
            "sequence_length": args.sequence_length,
            "task_type": args.task,
            "canonical_prompt": prompt_canonical,
            "prompt": prompt,
            "is_paraphrase": bool(args.is_paraphrase),
            "target_fruit": target,
            "non_target_fruits": [fruit for fruit in OBJECT_TYPES if fruit != target],
            "preloaded_fruits": list(preloaded),
            "container_semantic": "basket",
            "internal_container_body": CRATE_BODY,
            "layout_index": args.layout_index,
            "slot_order": list(slot_order),
            "basket_pose": initial_basket_pose.round(7).tolist(),
            "initial_fruit_poses": initial_poses,
            "final_fruit_poses": final_poses,
            "used_arm": used_arm,
            "grasp_attempts": grasp_attempts,
            "grasp_retries": grasp_retries,
            "recovery_type": args.recovery_type,
            "events": tracker.events,
            "success_frame": tracker.events.get("success_frame"),
            "success_time": (
                tracker.events["success_frame"] / args.log_fps
                if "success_frame" in tracker.events else None
            ),
            "release_frame": tracker.events.get("release_end_frame"),
            "retreat_start_frame": tracker.events.get("retreat_start_frame"),
            "retreat_end_frame": tracker.events.get("retreat_end_frame"),
            "terminal_hold_seconds": (
                (
                    tracker.events.get("terminal_hold_end_frame", 0)
                    - tracker.events.get("terminal_hold_start_frame", 0)
                ) / args.log_fps
            ),
            "episode_length": len(recorder.episode) if recorder.episode is not None else 0,
            "fps": args.log_fps,
            "state_dim": 14,
            "action_dim": 14,
            "success": success,
            "failure_reason": failure_reason,
            "validation": checks,
            "seed": args.seed,
        }
        root = Path(args.log_dataset)
        _write_atomic_schema(root)
        _append_jsonl(root / "meta" / "atomic_episodes.jsonl", metadata)
        print(f">>> ATOMIC_METADATA_EPISODE = {dataset_episode_index}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
