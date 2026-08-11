"""Replay a recorded RB-Y1 grid trial in MuJoCo from its saved actions — no policy needed.

The scene is reset exactly as the original trial (same grid config, colour, side, grid index, so the
block spawns at the identical pose), then the saved actions are pushed into ``d.ctrl`` with the same
cadence the live run used. Because no policy is queried, this isolates *what the recorded actions do*
from *what the policy would now predict*.

Two modes:

``--mode executed`` (default)
    Replay ``executed_actions`` — the exact stream the robot ran. If the outcome reproduces, the
    failure is fully determined by the action sequence; if it does not, contact/solver stochasticity
    is contributing.

``--mode chunk --chunk-index N``
    Replay executed actions up to chunk *N*'s start step, then execute that chunk's **full H steps**
    instead of only the K that were actually used. This answers "what did the policy actually plan
    at the moment of failure, and would it have worked if left to run?"

    src/openpi/.venv/bin/python scripts/replay_trial.py \
        --results data/rby1_grid_eval_seam/results.jsonl \
        --trial-id blue_left_g08_r1 --record /tmp/replay.mp4
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
for extra in (SRC / "rby1_manipulation" / "src", SRC / "rby1_bringup"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import mujoco  # noqa: E402
from rby1_manipulation.tools.preview_block_grid import (  # noqa: E402
    BLOCK_BODIES, body_position, load_grid_config, reset_and_place_trial,
)

MODEL_XML = str(SRC / "rby1_description" / "models" / "rby1a" / "mujoco" / "model.xml")
CTRL_HZ = 15
RBY1_GRIPPER_OPEN = -0.045
LEFT_ARM_ACTS = [f"left_arm_{i+1}_act" for i in range(7)]
RIGHT_ARM_ACTS = [f"right_arm_{i+1}_act" for i in range(7)]


def build_actuator_map(m):
    name2id = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)  # noqa: E731
    return {
        "left_a": [name2id(a) for a in LEFT_ARM_ACTS],
        "right_a": [name2id(a) for a in RIGHT_ARM_ACTS],
        "left_grip_a": name2id("gripper_l_act"),
        "right_grip_a": name2id("gripper_r_act"),
    }


def apply_rby1_action(action, d, act):
    """Same mapping as pi05_ex_infer.apply_action for action_format 'rby1' (absolute targets)."""
    for i in range(6):
        d.ctrl[act["left_a"][i]] = action[i]
        d.ctrl[act["right_a"][i]] = action[7 + i]
    d.ctrl[act["left_grip_a"]] = float(np.clip(action[6], 0.0, 1.0)) * RBY1_GRIPPER_OPEN
    d.ctrl[act["right_grip_a"]] = float(np.clip(action[13], 0.0, 1.0)) * RBY1_GRIPPER_OPEN


def find_record(results_path, trial_id):
    for line in pathlib.Path(results_path).open(encoding="utf-8"):
        if line.strip() and json.loads(line)["trial_id"] == trial_id:
            return json.loads(line)
    raise SystemExit(f"trial {trial_id!r} not found in {results_path}")


def success_check(m, d, colour):
    """Mirror the live stop_check success test (container-relative, settled, released)."""
    container = body_position(m, d, "container")
    pos = body_position(m, d, BLOCK_BODIES[colour])
    return (abs(pos[0] - container[0]) <= 0.075
            and abs(pos[1] - container[1]) <= 0.075
            and 0.82 <= pos[2] <= 0.92)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True, help="results.jsonl of the run to replay")
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--mode", choices=("executed", "chunk"), default="executed")
    parser.add_argument("--chunk-index", type=int, default=None,
                        help="chunk mode: which chunk to run in full (0-based)")
    parser.add_argument("--grid-config", type=pathlib.Path, default=None,
                        help="defaults to the grid_config_<fingerprint>.json beside results.jsonl")
    parser.add_argument("--record", type=pathlib.Path, default=None, help="write an MP4")
    parser.add_argument("--view", choices=("front", "free"), default="front")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    record = find_record(args.results, args.trial_id)
    results_dir = pathlib.Path(args.results).parent
    grid_path = args.grid_config or (results_dir / f"grid_config_{record['grid_fingerprint']}.json")
    config = load_grid_config(grid_path)

    data = np.load(record["trajectory"], allow_pickle=True)
    executed = np.asarray(data["executed_actions"], dtype=np.float64)
    chunks = np.asarray(data["predicted_chunks"], dtype=np.float64)
    starts = np.asarray(data["chunk_start_steps"], dtype=np.int64)

    if args.mode == "chunk":
        if args.chunk_index is None:
            parser.error("--mode chunk requires --chunk-index")
        if not 0 <= args.chunk_index < chunks.shape[0]:
            parser.error(f"--chunk-index must be in [0, {chunks.shape[0]-1}]")
        prefix = executed[: int(starts[args.chunk_index])]
        stream = np.concatenate([prefix, chunks[args.chunk_index]], axis=0)
        print(f"chunk mode: {len(prefix)} executed steps, then chunk "
              f"{args.chunk_index} in full ({chunks.shape[1]} steps)")
    else:
        stream = executed
    if args.max_steps:
        stream = stream[: args.max_steps]

    m = mujoco.MjModel.from_xml_path(MODEL_XML)
    d = mujoco.MjData(m)
    act = build_actuator_map(m)

    # Identical scene setup to the live run: same grid point, same settle.
    requested, settled = reset_and_place_trial(
        m, d, config, color=record["color"], side=record["side"],
        grid_index=record["grid_index"] - 1, settle_seconds=1.5,
    )
    print(f"trial      : {record['trial_id']}  ({record['condition']})")
    print(f"recorded   : status={record['status']} success={record['success']} "
          f"steps={record['steps']}")
    print(f"spawn      : requested={np.round(requested[record['color']],5).tolist()}  "
          f"settled={np.round(settled[record['color']],5).tolist()}")
    print(f"  (original settled was {np.round(record['settled_xyz'],5).tolist()})")

    renderer = camera = None
    frames = []
    if args.record:
        renderer = mujoco.Renderer(m, height=480, width=640)
        if args.view == "front":
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = np.asarray([0.45, 0.0, 0.85])
            camera.distance, camera.azimuth, camera.elevation = 1.7, 180.0, -18.0
        else:
            camera = -1

    steps_per_action = max(1, int(round(1.0 / (CTRL_HZ * m.opt.timestep))))
    stable = 0
    replay_success = False
    for t, action in enumerate(stream):
        apply_rby1_action(action, d, act)
        for _ in range(steps_per_action):
            mujoco.mj_step(m, d)
        if renderer is not None:
            renderer.update_scene(d, camera=camera)
            frames.append(renderer.render())
        if success_check(m, d, record["color"]):
            stable += 1
            if stable >= 8:
                replay_success = True
                print(f"replay     : SUCCESS at step {t+1}")
                break
        else:
            stable = 0

    final = body_position(m, d, BLOCK_BODIES[record["color"]])
    print(f"replay     : success={replay_success}  steps_run={t+1}")
    print(f"final block: replay={np.round(final,4).tolist()}  "
          f"original={np.round(record['final_xyz'],4).tolist()}")
    print(f"  block position difference: {np.linalg.norm(final - np.array(record['final_xyz'])):.4f} m")
    verdict = "재현됨" if replay_success == record["success"] else "재현 안 됨 (결과가 뒤집힘)"
    print(f"reproduced : {verdict}")

    if frames:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        import imageio
        imageio.mimsave(args.record, frames, fps=CTRL_HZ)
        print(f"video      : {args.record} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
