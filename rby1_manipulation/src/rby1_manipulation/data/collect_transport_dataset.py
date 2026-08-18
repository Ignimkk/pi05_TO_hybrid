"""Balanced 1,200-episode collector for fixed-base fruit packing and crate lift."""

from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from rby1_manipulation.simulation.fruit_grid import (
    fruit_grid_fingerprint,
    layout_count,
    load_fruit_grid_config,
)
from rby1_manipulation.simulation.transport_scene import OBJECT_TYPES
from rby1_manipulation.tasks.transport_pack_lift import DEFAULT_SPEED_SCALE, default_prompt


DEFAULT_EPISODES = 1200
FAMILIES = ("pack_only", "lift_only", "pack_and_lift")
FRUITS = tuple(OBJECT_TYPES)
FRUIT_PERMUTATIONS = tuple(itertools.permutations(FRUITS))


@dataclass(frozen=True)
class EpisodeSpec:
    plan_index: int
    task: str
    object_count: int
    objects: tuple[str, ...]
    preloaded: tuple[str, ...]
    layout_index: int
    slot_order: tuple[str, ...]
    seed: int
    task_prompt: str


def split_even(total: int, buckets: int) -> list[int]:
    quotient, remainder = divmod(total, buckets)
    return [quotient + (index < remainder) for index in range(buckets)]


def balanced_prefix_orders(total: int, prefix_length: int, *, phase: int = 0) -> list[tuple[str, ...]]:
    """Return nearly uniform ordered fruit prefixes; exact for the 1,200 plan."""
    if not 0 <= prefix_length <= len(FRUITS):
        raise ValueError("prefix_length must be in [0, 4]")
    if prefix_length == 0:
        return [()] * total

    full_cycles, remainder = divmod(total, len(FRUIT_PERMUTATIONS))
    orders = [permutation[:prefix_length]
              for _ in range(full_cycles) for permutation in FRUIT_PERMUTATIONS]

    group = 0
    while remainder:
        base = list(FRUITS)
        if group % 2:
            base.reverse()
        shift = (phase + group) % len(base)
        base = base[shift:] + base[:shift]
        rotations = [tuple(base[index:] + base[:index]) for index in range(len(base))]
        take = min(remainder, len(rotations))
        orders.extend(rotation[:prefix_length] for rotation in rotations[:take])
        remainder -= take
        group += 1
    return orders


def build_schedule(total_episodes: int = DEFAULT_EPISODES, *, base_seed: int = 1000,
                   shuffle_seed: int = 20260811) -> list[EpisodeSpec]:
    if total_episodes <= 0:
        raise ValueError("total_episodes must be positive")
    family_budgets = dict(zip(FAMILIES, split_even(total_episodes, len(FAMILIES))))
    raw: list[tuple[str, int, tuple[str, ...], tuple[str, ...]]] = []

    for family_index, family in enumerate(FAMILIES):
        counts = range(0, 5) if family == "lift_only" else range(1, 5)
        per_count = split_even(family_budgets[family], len(counts))
        for count_index, (count, budget) in enumerate(zip(counts, per_count)):
            orders = balanced_prefix_orders(
                budget,
                count,
                phase=family_index + count_index,
            )
            for order in orders:
                if family == "lift_only":
                    raw.append((family, count, (), order))
                else:
                    raw.append((family, count, order, ()))

    rng = np.random.default_rng(shuffle_seed)
    grid = load_fruit_grid_config()
    n_layouts = layout_count(grid)
    layouts = [index % n_layouts for index in range(total_episodes)]
    slot_orders = [FRUIT_PERMUTATIONS[index % len(FRUIT_PERMUTATIONS)]
                   for index in range(total_episodes)]
    rng.shuffle(layouts)
    rng.shuffle(slot_orders)

    assigned = []
    for index, ((family, count, objects, preloaded), layout_index, slot_order) in enumerate(
        zip(raw, layouts, slot_orders)
    ):
        prompt_objects = preloaded if family == "lift_only" else objects
        assigned.append(EpisodeSpec(
            plan_index=index,
            task=family,
            object_count=count,
            objects=objects,
            preloaded=preloaded,
            layout_index=layout_index,
            slot_order=tuple(slot_order),
            seed=base_seed + index,
            task_prompt=default_prompt(family, prompt_objects),
        ))

    permutation = rng.permutation(len(assigned))
    shuffled: list[EpisodeSpec] = []
    for execution_index, source_index in enumerate(permutation):
        spec = assigned[int(source_index)]
        shuffled.append(EpisodeSpec(
            **{**asdict(spec), "plan_index": execution_index}
        ))
    return shuffled


def schedule_summary(schedule: Sequence[EpisodeSpec]) -> dict:
    summary = {
        "total": len(schedule),
        "families": {family: 0 for family in FAMILIES},
        "counts": {family: {} for family in FAMILIES},
        "fruit_exposure": {family: {fruit: 0 for fruit in FRUITS} for family in FAMILIES},
        "layouts": {},
        "slot_exposure": {fruit: [0, 0, 0, 0] for fruit in FRUITS},
    }
    for spec in schedule:
        summary["families"][spec.task] += 1
        count_key = str(spec.object_count)
        summary["counts"][spec.task][count_key] = \
            summary["counts"][spec.task].get(count_key, 0) + 1
        selected = spec.preloaded if spec.task == "lift_only" else spec.objects
        for fruit in selected:
            summary["fruit_exposure"][spec.task][fruit] += 1
        layout_key = str(spec.layout_index)
        summary["layouts"][layout_key] = summary["layouts"].get(layout_key, 0) + 1
        for slot_index, fruit in enumerate(spec.slot_order):
            summary["slot_exposure"][fruit][slot_index] += 1
    return summary


def _command(spec: EpisodeSpec, output_dir: Path, *, log_fps: int,
             random_scene: bool, speed_scale: float,
             object_pre_close_hold: float | None) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "rby1_manipulation.tasks.transport_pack_lift",
        "--task", spec.task,
        "--layout-index", str(spec.layout_index),
        "--slot-order", *spec.slot_order,
        "--seed", str(spec.seed),
        "--headless",
        "--log-dataset", str(output_dir),
        "--log-fps", str(log_fps),
        "--speed-scale", str(speed_scale),
        "--task-prompt", spec.task_prompt,
    ]
    if spec.objects:
        command.extend(("--objects", *spec.objects))
    if spec.preloaded:
        command.extend(("--preloaded", *spec.preloaded))
    if random_scene:
        command.append("--random-scene")
    if object_pre_close_hold is not None:
        command.extend(("--object-pre-close-hold", str(object_pre_close_hold)))
    return command


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_manifest(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def _validate_existing_dataset(output_dir: Path) -> None:
    info_path = output_dir / "meta" / "info.json"
    if not info_path.exists():
        return
    if not (output_dir / "transport_collection_plan.json").exists():
        raise RuntimeError(
            "output directory already contains a dataset not created by this collector"
        )
    info = json.loads(info_path.read_text(encoding="utf-8"))
    state_shape = info.get("features", {}).get("observation.state", {}).get("shape")
    action_shape = info.get("features", {}).get("action", {}).get("shape")
    if state_shape != [14] or action_shape != [14]:
        raise RuntimeError(
            f"existing dataset is not 14-D: state={state_shape}, action={action_shape}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--shuffle-seed", type=int, default=20260811)
    parser.add_argument("--log-fps", type=int, default=15)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--speed-scale", type=float, default=DEFAULT_SPEED_SCALE)
    parser.add_argument("--object-pre-close-hold", type=float, default=None)
    parser.add_argument("--no-random-scene", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-limit", type=int, default=12)
    parser.add_argument("--only", choices=FAMILIES, default=None)
    args = parser.parse_args()
    if args.speed_scale <= 0.0:
        raise SystemExit("--speed-scale must be greater than zero")
    if args.object_pre_close_hold is not None and args.object_pre_close_hold < 0.0:
        raise SystemExit("--object-pre-close-hold must be non-negative")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_existing_dataset(output_dir)

    schedule = build_schedule(
        args.episodes,
        base_seed=args.base_seed,
        shuffle_seed=args.shuffle_seed,
    )
    if args.only:
        schedule = [spec for spec in schedule if spec.task == args.only]
    summary = schedule_summary(schedule)
    grid = load_fruit_grid_config()
    plan_metadata = {
        "version": 2,
        "schema": "rby1_14",
        "obstacles": False,
        "base_motion": False,
        "requested_episodes": args.episodes,
        "only": args.only,
        "selected_episodes": len(schedule),
        "base_seed": args.base_seed,
        "shuffle_seed": args.shuffle_seed,
        "speed_scale": args.speed_scale,
        "object_pre_close_hold": args.object_pre_close_hold,
        "fruit_grid_fingerprint": fruit_grid_fingerprint(grid),
        "summary": summary,
        "episodes": [asdict(spec) for spec in schedule],
    }
    plan_path = output_dir / "transport_collection_plan.json"
    existing_plan = _load_json(plan_path, None)
    if existing_plan is not None:
        comparable_keys = (
            "requested_episodes", "only", "base_seed", "shuffle_seed",
            "speed_scale", "object_pre_close_hold", "fruit_grid_fingerprint",
        )
        if any(existing_plan.get(key) != plan_metadata.get(key) for key in comparable_keys):
            raise RuntimeError("output directory contains a different collection plan")
    elif not args.dry_run:
        _write_json(plan_path, plan_metadata)

    print("=== balanced transport dataset collection ===")
    print(f"  schema       : rby1_14")
    print(f"  obstacles    : disabled")
    print(f"  base motion  : disabled")
    print(f"  speed scale  : {args.speed_scale:.2f}x")
    print(f"  pre-close    : {args.object_pre_close_hold!r} s override")
    print(f"  target       : {len(schedule)} successful episodes")
    print(f"  families     : {summary['families']}")
    print(f"  counts       : {summary['counts']}")
    print(f"  grid         : {len(summary['layouts'])} layouts, fingerprint "
          f"{plan_metadata['fruit_grid_fingerprint']}")

    if args.dry_run:
        for spec in schedule[:args.dry_run_limit]:
            print(f"  [{spec.plan_index:04d}] " + " ".join(_command(
                spec, output_dir, log_fps=args.log_fps,
                random_scene=not args.no_random_scene,
                speed_scale=args.speed_scale,
                object_pre_close_hold=args.object_pre_close_hold,
            )))
        return 0

    stats_path = output_dir / "transport_collection_stats.json"
    manifest_path = output_dir / "transport_episode_manifest.jsonl"
    stats = _load_json(stats_path, {"version": 1, "episodes": {}})
    completed = sum(1 for value in stats["episodes"].values() if value.get("success"))
    info_path = output_dir / "meta" / "info.json"
    if info_path.exists():
        dataset_episodes = int(json.loads(info_path.read_text(encoding="utf-8"))["total_episodes"])
        if dataset_episodes != completed:
            raise RuntimeError(
                "resume state is inconsistent: dataset contains "
                f"{dataset_episodes} episodes but stats records {completed} successes"
            )
    started = time.time()

    for sequence_index, spec in enumerate(schedule, start=1):
        key = str(spec.plan_index)
        record = stats["episodes"].setdefault(key, {"attempts": 0, "success": False})
        if record.get("success"):
            continue

        for attempt in range(int(record["attempts"]), args.max_attempts):
            attempt_spec = EpisodeSpec(
                **{**asdict(spec), "seed": spec.seed + attempt * args.episodes}
            )
            command = _command(
                attempt_spec,
                output_dir,
                log_fps=args.log_fps,
                random_scene=not args.no_random_scene,
                speed_scale=args.speed_scale,
                object_pre_close_hold=args.object_pre_close_hold,
            )
            episode_start = time.time()
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                )
                stdout = result.stdout or ""
                success = result.returncode == 0 and ">>> SUCCESS = True" in stdout
                message = "ok" if success else " | ".join(stdout.splitlines()[-4:])
            except subprocess.TimeoutExpired:
                stdout = ""
                success = False
                message = "TIMEOUT"

            record["attempts"] = attempt + 1
            record["last_seed"] = attempt_spec.seed
            record["last_message"] = message[-500:]
            record["success"] = success
            if success:
                match = re.search(r"episode (\d+) \(", stdout)
                dataset_index = int(match.group(1)) if match else None
                record["dataset_episode_index"] = dataset_index
                completed += 1
                _append_manifest(manifest_path, {
                    **asdict(attempt_spec),
                    "dataset_episode_index": dataset_index,
                    "attempt": attempt + 1,
                    "elapsed_seconds": round(time.time() - episode_start, 3),
                })
            _write_json(stats_path, stats)
            status = "OK" if success else f"FAIL {message[:100]}"
            print(f"  [{sequence_index:04d}/{len(schedule):04d}] plan={spec.plan_index:04d} "
                  f"attempt={attempt + 1} completed={completed:04d} {status}")
            if success:
                break

    elapsed = time.time() - started
    failed = [key for key, value in stats["episodes"].items() if not value.get("success")]
    print(f"=== complete: {completed}/{len(schedule)} successes in {elapsed / 3600:.2f} h ===")
    if failed:
        print(f"  exhausted plans: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
