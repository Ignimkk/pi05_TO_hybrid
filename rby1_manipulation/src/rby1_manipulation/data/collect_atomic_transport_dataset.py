"""Build and collect the balanced 2,000-episode atomic basket dataset."""
from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from rby1_manipulation.simulation.fruit_grid import (
    fruit_grid_fingerprint,
    layout_count,
    load_fruit_grid_config,
)
from rby1_manipulation.simulation.transport_scene import OBJECT_TYPES
from rby1_manipulation.data.collect_transport_dataset import balanced_prefix_orders
from rby1_manipulation.tasks.transport_atomic import (
    ATOMIC_SCHEMA_VERSION,
    PHASE_NAMES,
    RECOVERY_TYPES,
    _check_dataset_schema,
    canonical_prompt,
)


FRUITS = tuple(OBJECT_TYPES)
SPLITS = ("train", "validation", "test")
SPLIT_TOTALS = {"train": 1600, "validation": 200, "test": 200}
FAMILY_TOTALS = {
    "clean_single": 1200,
    "linked_atomic": 320,
    "preloaded_single": 240,
    "recovery": 160,
    "lift_basket": 80,
}
FAMILY_SPLITS = {
    "clean_single": {"train": 960, "validation": 120, "test": 120},
    "linked_atomic": {"train": 256, "validation": 32, "test": 32},
    "preloaded_single": {"train": 192, "validation": 24, "test": 24},
    "recovery": {"train": 128, "validation": 16, "test": 16},
    "lift_basket": {"train": 64, "validation": 8, "test": 8},
}
PARAPHRASES = (
    "place the {fruit} inside the basket",
    "move the {fruit} into the basket",
    "pick up the {fruit} and put it in the basket",
)


@dataclass(frozen=True)
class AtomicEpisodeSpec:
    plan_index: int
    scenario_family: str
    task: str
    target_fruit: str | None
    preloaded: tuple[str, ...]
    layout_index: int
    slot_order: tuple[str, ...]
    seed: int
    task_prompt: str
    canonical_prompt: str
    is_paraphrase: bool
    recovery_type: str | None
    sequence_group_id: str | None
    sequence_step: int
    sequence_length: int
    split: str


def _pair_groups(count: int) -> list[tuple[str, str]]:
    a, b, c, d = FRUITS
    cycle = [(a, b), (c, d), (a, c), (b, d), (a, d), (b, c)]
    groups = [cycle[index % len(cycle)] for index in range(count)]
    return [group if index % 2 == 0 else tuple(reversed(group))
            for index, group in enumerate(groups)]


def _triple_groups(count: int) -> list[tuple[str, str, str]]:
    cycle = [tuple(fruit for fruit in FRUITS if fruit != missing) for missing in FRUITS]
    groups = [cycle[index % len(cycle)] for index in range(count)]
    return [group[index % 3:] + group[:index % 3] for index, group in enumerate(groups)]


def _blank_spec(*, family: str, task: str, target: str | None, preloaded=(),
                split: str, prompt: str | None = None, paraphrase: bool = False,
                recovery: str | None = None, group: str | None = None,
                step: int = 0, length: int = 1) -> AtomicEpisodeSpec:
    canonical = canonical_prompt(task, target)
    return AtomicEpisodeSpec(
        plan_index=-1,
        scenario_family=family,
        task=task,
        target_fruit=target,
        preloaded=tuple(preloaded),
        layout_index=-1,
        slot_order=FRUITS,
        seed=-1,
        task_prompt=prompt or canonical,
        canonical_prompt=canonical,
        is_paraphrase=paraphrase,
        recovery_type=recovery,
        sequence_group_id=group,
        sequence_step=step,
        sequence_length=length,
        split=split,
    )


def _build_raw_schedule(include_paraphrases: bool) -> list[AtomicEpisodeSpec]:
    specs: list[AtomicEpisodeSpec] = []
    per_fruit_clean = {"train": 240, "validation": 30, "test": 30}
    for split in SPLITS:
        for fruit in FRUITS:
            specs.extend(
                _blank_spec(family="clean_single", task="place_one", target=fruit,
                            split=split)
                for _ in range(per_fruit_clean[split])
            )

    pair_counts = {"train": 80, "validation": 10, "test": 10}
    triple_counts = {"train": 32, "validation": 4, "test": 4}
    group_number = 0
    for split in SPLITS:
        groups: list[tuple[str, ...]] = [
            *_pair_groups(pair_counts[split]),
            *_triple_groups(triple_counts[split]),
        ]
        for targets in groups:
            group_id = f"sequence_{group_number:04d}"
            group_number += 1
            for step, fruit in enumerate(targets):
                specs.append(_blank_spec(
                    family="linked_atomic",
                    task="place_one",
                    target=fruit,
                    preloaded=targets[:step],
                    split=split,
                    group=group_id,
                    step=step,
                    length=len(targets),
                ))

    per_fruit_preloaded = {"train": 48, "validation": 6, "test": 6}
    for split in SPLITS:
        for target in FRUITS:
            others = tuple(fruit for fruit in FRUITS if fruit != target)
            permutations = tuple(itertools.permutations(others))
            per_count = per_fruit_preloaded[split] // 3
            for count in (1, 2, 3):
                for index in range(per_count):
                    order = permutations[index % len(permutations)]
                    specs.append(_blank_spec(
                        family="preloaded_single",
                        task="place_one",
                        target=target,
                        preloaded=order[:count],
                        split=split,
                    ))

    per_fruit_recovery = {"train": 32, "validation": 4, "test": 4}
    for split in SPLITS:
        for target in FRUITS:
            for index in range(per_fruit_recovery[split]):
                specs.append(_blank_spec(
                    family="recovery",
                    task="place_one",
                    target=target,
                    split=split,
                    recovery=RECOVERY_TYPES[index % len(RECOVERY_TYPES)],
                ))

    # The overall lift distribution is exactly 16 episodes for each initial
    # basket load count and 40 preloaded appearances for every fruit.
    lift_split_by_count = {
        0: (13, 2, 1),
        1: (13, 2, 1),
        2: (13, 2, 1),
        3: (13, 1, 2),
        4: (12, 1, 3),
    }
    for count in range(5):
        orders = balanced_prefix_orders(16, count, phase=count)
        cursor = 0
        for split, amount in zip(SPLITS, lift_split_by_count[count]):
            for preloaded in orders[cursor:cursor + amount]:
                specs.append(_blank_spec(
                    family="lift_basket", task="lift_basket", target=None,
                    preloaded=preloaded, split=split,
                ))
            cursor += amount

    if include_paraphrases:
        per_template_split = {"train": 16, "validation": 2, "test": 2}
        for split in SPLITS:
            for fruit in FRUITS:
                for template in PARAPHRASES:
                    specs.extend(
                        _blank_spec(
                            family="paraphrase_extension",
                            task="place_one",
                            target=fruit,
                            split=split,
                            prompt=template.format(fruit=fruit),
                            paraphrase=True,
                        )
                        for _ in range(per_template_split[split])
                    )
    return specs


def _unit_key(spec: AtomicEpisodeSpec, unique_index: int) -> str:
    return spec.sequence_group_id or f"episode_{unique_index:06d}"


def _assign_scenes(specs: Sequence[AtomicEpisodeSpec], *, seed: int) -> list[AtomicEpisodeSpec]:
    rng = np.random.default_rng(seed)
    units: dict[str, list[AtomicEpisodeSpec]] = {}
    unit_order: list[str] = []
    for index, spec in enumerate(specs):
        key = _unit_key(spec, index)
        if key not in units:
            units[key] = []
            unit_order.append(key)
        units[key].append(spec)
    rng.shuffle(unit_order)

    grid = load_fruit_grid_config()
    layouts = np.zeros(layout_count(grid), dtype=int)
    target_slots = {fruit: np.zeros(4, dtype=int) for fruit in FRUITS}
    permutations = tuple(itertools.permutations(FRUITS))
    assigned_units: list[list[AtomicEpisodeSpec]] = []
    for unit_index, key in enumerate(unit_order):
        group = units[key]
        weight = len(group)
        candidate_layouts = np.flatnonzero(layouts == layouts.min())
        layout = int(candidate_layouts[unit_index % len(candidate_layouts)])
        layouts[layout] += weight

        targets = tuple(spec.target_fruit for spec in group if spec.target_fruit is not None)
        if targets:
            scored = []
            for permutation in permutations:
                score = 0
                for fruit in targets:
                    slot = permutation.index(fruit)
                    score += int(target_slots[fruit][slot])
                scored.append(score)
            minimum = min(scored)
            candidates = [p for p, score in zip(permutations, scored) if score == minimum]
            slot_order = candidates[unit_index % len(candidates)]
            for fruit in targets:
                target_slots[fruit][slot_order.index(fruit)] += 1
        else:
            slot_order = permutations[unit_index % len(permutations)]

        scene_seed = 1000 + unit_index
        assigned_units.append([
            replace(spec, layout_index=layout, slot_order=tuple(slot_order), seed=scene_seed)
            for spec in group
        ])

    flattened: list[AtomicEpisodeSpec] = []
    # Keep each dataset split in a contiguous episode-index range so LeRobot's
    # standard ``info.json`` range syntax can expose train/validation/test.
    for split in SPLITS:
        execution_order = [
            index for index, group in enumerate(assigned_units)
            if group[0].split == split
        ]
        rng.shuffle(execution_order)
        for unit_index in execution_order:
            flattened.extend(sorted(
                assigned_units[unit_index], key=lambda spec: spec.sequence_step
            ))
    return [replace(spec, plan_index=index) for index, spec in enumerate(flattened)]


def build_schedule(*, seed: int = 20260820,
                   include_paraphrases: bool = False) -> list[AtomicEpisodeSpec]:
    schedule = _assign_scenes(_build_raw_schedule(include_paraphrases), seed=seed)
    validate_schedule(schedule, include_paraphrases=include_paraphrases)
    return schedule


def schedule_summary(schedule: Sequence[AtomicEpisodeSpec]) -> dict:
    families = Counter(spec.scenario_family for spec in schedule)
    splits = Counter(spec.split for spec in schedule)
    targets = Counter(spec.target_fruit for spec in schedule if spec.target_fruit)
    target_arms = Counter()
    target_slots = {fruit: [0, 0, 0, 0] for fruit in FRUITS}
    for spec in schedule:
        if spec.target_fruit:
            slot = spec.slot_order.index(spec.target_fruit)
            target_slots[spec.target_fruit][slot] += 1
            target_arms[(spec.target_fruit, "left" if slot < 2 else "right")] += 1
    return {
        "total": len(schedule),
        "families": dict(sorted(families.items())),
        "splits": dict(sorted(splits.items())),
        "target_fruits": dict(sorted(targets.items())),
        "target_slots": target_slots,
        "target_arms": {
            fruit: {side: target_arms[(fruit, side)] for side in ("left", "right")}
            for fruit in FRUITS
        },
        "layouts": dict(sorted(Counter(spec.layout_index for spec in schedule).items())),
        "preloaded_counts": dict(sorted(Counter(len(spec.preloaded) for spec in schedule).items())),
        "recovery_types": dict(sorted(Counter(
            spec.recovery_type for spec in schedule if spec.recovery_type
        ).items())),
        "paraphrases": sum(spec.is_paraphrase for spec in schedule),
    }


def validate_schedule(schedule: Sequence[AtomicEpisodeSpec], *,
                      include_paraphrases: bool = False) -> None:
    expected_total = 2240 if include_paraphrases else 2000
    if len(schedule) != expected_total:
        raise ValueError(f"expected {expected_total} episodes, got {len(schedule)}")
    if [spec.plan_index for spec in schedule] != list(range(len(schedule))):
        raise ValueError("plan indexes are not consecutive")
    for spec in schedule:
        if spec.target_fruit in spec.preloaded:
            raise ValueError("target fruit is preloaded")
        if set(spec.slot_order) != set(FRUITS):
            raise ValueError("slot_order is not a fruit permutation")
        if spec.task == "place_one" and spec.target_fruit is None:
            raise ValueError("place_one has no target")
    if not include_paraphrases:
        if Counter(spec.scenario_family for spec in schedule) != Counter(FAMILY_TOTALS):
            raise ValueError("family quotas do not match the 2,000-episode design")
        if Counter(spec.split for spec in schedule) != Counter(SPLIT_TOTALS):
            raise ValueError("split quotas do not match the 80/10/10 design")
        if Counter(spec.target_fruit for spec in schedule if spec.target_fruit) != Counter(
            {fruit: 480 for fruit in FRUITS}
        ):
            raise ValueError("target fruit exposure is not balanced")
        for fruit in FRUITS:
            slots = [0, 0, 0, 0]
            for spec in schedule:
                if spec.target_fruit == fruit:
                    slots[spec.slot_order.index(fruit)] += 1
            if slots != [120, 120, 120, 120]:
                raise ValueError(f"{fruit} target slots are not balanced: {slots}")
    groups: dict[str, list[AtomicEpisodeSpec]] = defaultdict(list)
    for spec in schedule:
        if spec.sequence_group_id:
            groups[spec.sequence_group_id].append(spec)
    for group_id, members in groups.items():
        if len({member.split for member in members}) != 1:
            raise ValueError(f"sequence group {group_id} crosses dataset splits")
        if len({member.layout_index for member in members}) != 1 or \
           len({member.slot_order for member in members}) != 1:
            raise ValueError(f"sequence group {group_id} does not share one scene")


def _command(spec: AtomicEpisodeSpec, output_dir: Path, args) -> list[str]:
    command = [
        sys.executable,
        "-m", "rby1_manipulation.tasks.transport_atomic",
        "--task", spec.task,
        "--layout-index", str(spec.layout_index),
        "--slot-order", *spec.slot_order,
        "--seed", str(spec.seed),
        "--headless",
        "--log-dataset", str(output_dir),
        "--log-fps", str(args.log_fps),
        "--speed-scale", str(args.speed_scale),
        "--initial-hold", str(args.initial_hold),
        "--terminal-hold", str(args.terminal_hold),
        "--task-prompt", spec.task_prompt,
        "--canonical-prompt", spec.canonical_prompt,
        "--scenario-family", spec.scenario_family,
        "--plan-index", str(spec.plan_index),
        "--split", spec.split,
        "--sequence-step", str(spec.sequence_step),
        "--sequence-length", str(spec.sequence_length),
        "--basket-jitter", str(args.basket_jitter),
        "--basket-yaw-jitter", str(args.basket_yaw_jitter),
        "--fruit-jitter", str(args.fruit_jitter),
    ]
    if spec.target_fruit:
        command.extend(("--target-fruit", spec.target_fruit))
    if spec.preloaded:
        command.extend(("--preloaded", *spec.preloaded))
    if spec.recovery_type:
        command.extend(("--recovery-type", spec.recovery_type))
    if spec.sequence_group_id:
        command.extend(("--sequence-group-id", spec.sequence_group_id))
    if spec.is_paraphrase:
        command.append("--is-paraphrase")
    if not args.no_random_scene:
        command.append("--random-scene")
    return command


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def _finalize_dataset_splits(output_dir: Path,
                             schedule: Sequence[AtomicEpisodeSpec]) -> None:
    info_path = output_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    ranges: dict[str, str] = {}
    split_metadata: dict[str, dict] = {}
    start = 0
    for split in SPLITS:
        count = sum(spec.split == split for spec in schedule)
        end = start + count
        expected = schedule[start:end]
        if any(spec.split != split for spec in expected):
            raise RuntimeError(f"{split} episodes are not contiguous")
        ranges[split] = f"{start}:{end}"
        split_metadata[split] = {
            "start_episode_index": start,
            "end_episode_index_exclusive": end,
            "episodes": count,
        }
        start = end
    info["splits"] = ranges
    _write_json(info_path, info)
    _write_json(output_dir / "meta" / "atomic_splits.json", {
        "version": 1,
        "splits": split_metadata,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--schedule-seed", type=int, default=20260820)
    parser.add_argument("--include-paraphrases", action="store_true")
    parser.add_argument("--log-fps", type=int, default=15)
    parser.add_argument("--speed-scale", type=float, default=1.25)
    parser.add_argument("--initial-hold", type=float, default=1.0)
    parser.add_argument("--terminal-hold", type=float, default=2.0)
    parser.add_argument("--basket-jitter", type=float, default=0.010)
    parser.add_argument("--basket-yaw-jitter", type=float, default=0.05236)
    parser.add_argument("--fruit-jitter", type=float, default=0.006)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--no-random-scene", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-limit", type=int, default=12)
    args = parser.parse_args()

    schedule = build_schedule(
        seed=args.schedule_seed,
        include_paraphrases=args.include_paraphrases,
    )
    summary = schedule_summary(schedule)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "version": ATOMIC_SCHEMA_VERSION,
        "dataset": "rby1_atomic_basket_14d",
        "schema": "rby1_14",
        "container_semantic": "basket",
        "internal_container_body": "crate",
        "schedule_seed": args.schedule_seed,
        "include_paraphrases": args.include_paraphrases,
        "log_fps": args.log_fps,
        "speed_scale": args.speed_scale,
        "initial_hold_seconds": args.initial_hold,
        "terminal_hold_seconds": args.terminal_hold,
        "basket_jitter_m": args.basket_jitter,
        "basket_yaw_jitter_rad": args.basket_yaw_jitter,
        "fruit_jitter_m": args.fruit_jitter,
        "random_scene": not args.no_random_scene,
        "phase_labels": {str(key): value for key, value in PHASE_NAMES.items()},
        "fruit_grid_fingerprint": fruit_grid_fingerprint(load_fruit_grid_config()),
        "summary": summary,
        "episodes": [asdict(spec) for spec in schedule],
    }
    plan_path = output_dir / "atomic_collection_plan.json"
    if not args.dry_run:
        _check_dataset_schema(output_dir)
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if int(existing.get("version", -1)) != ATOMIC_SCHEMA_VERSION:
            raise RuntimeError(
                f"output directory contains atomic collection plan v"
                f"{existing.get('version')}; use a new output directory for "
                f"corrected v{ATOMIC_SCHEMA_VERSION} data"
            )
        comparison = (
            "schedule_seed", "include_paraphrases", "log_fps", "speed_scale",
            "initial_hold_seconds", "terminal_hold_seconds", "basket_jitter_m",
            "basket_yaw_jitter_rad", "fruit_jitter_m", "random_scene",
            "fruit_grid_fingerprint",
        )
        if any(existing.get(key) != plan.get(key) for key in comparison):
            raise RuntimeError("output directory contains a different atomic collection plan")
    elif not args.dry_run:
        _write_json(plan_path, plan)

    print("=== atomic basket dataset collection ===")
    print(f"  target       : {len(schedule)} successful episodes")
    print(f"  families     : {summary['families']}")
    print(f"  splits       : {summary['splits']}")
    print(f"  fruits       : {summary['target_fruits']}")
    print(f"  target slots : {summary['target_slots']}")
    print(f"  layouts      : min={min(summary['layouts'].values())} "
          f"max={max(summary['layouts'].values())}")

    if args.dry_run:
        for spec in schedule[:args.dry_run_limit]:
            print(f"  [{spec.plan_index:04d}] " + " ".join(_command(spec, output_dir, args)))
        return 0

    stats_path = output_dir / "atomic_collection_stats.json"
    manifest_path = output_dir / "atomic_collection_manifest.jsonl"
    stats = (
        json.loads(stats_path.read_text(encoding="utf-8"))
        if stats_path.exists() else {"version": 1, "episodes": {}}
    )
    completed = sum(value.get("success", False) for value in stats["episodes"].values())
    info_path = output_dir / "meta" / "info.json"
    if info_path.exists():
        saved = int(json.loads(info_path.read_text(encoding="utf-8"))["total_episodes"])
        if saved != completed:
            raise RuntimeError(
                f"resume state mismatch: dataset has {saved}, stats has {completed}"
            )

    started = time.time()
    for sequence_index, spec in enumerate(schedule, start=1):
        key = str(spec.plan_index)
        record = stats["episodes"].setdefault(key, {"attempts": 0, "success": False})
        if record.get("success"):
            continue
        for attempt in range(int(record["attempts"]), args.max_attempts):
            attempt_spec = replace(spec, seed=spec.seed + attempt * 100000)
            command = _command(attempt_spec, output_dir, args)
            episode_started = time.time()
            try:
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=args.timeout
                )
                stdout = result.stdout or ""
                success = result.returncode == 0 and ">>> SUCCESS = True" in stdout
                message = "ok" if success else " | ".join(stdout.splitlines()[-6:])
            except subprocess.TimeoutExpired:
                stdout = ""
                success = False
                message = "TIMEOUT"
            record.update({
                "attempts": attempt + 1,
                "last_seed": attempt_spec.seed,
                "last_message": message[-1000:],
                "success": success,
            })
            if success:
                match = re.search(r"episode (\d+) \(", stdout)
                dataset_index = int(match.group(1)) if match else None
                record["dataset_episode_index"] = dataset_index
                completed += 1
                _append_jsonl(manifest_path, {
                    **asdict(attempt_spec),
                    "dataset_episode_index": dataset_index,
                    "attempt": attempt + 1,
                    "elapsed_seconds": round(time.time() - episode_started, 3),
                })
            _write_json(stats_path, stats)
            status = "OK" if success else f"FAIL {message[:120]}"
            print(f"  [{sequence_index:04d}/{len(schedule):04d}] "
                  f"plan={spec.plan_index:04d} attempt={attempt + 1} "
                  f"completed={completed:04d} {status}")
            if success:
                break

    failed = [key for key, value in stats["episodes"].items() if not value.get("success")]
    elapsed = time.time() - started
    print(f"=== complete: {completed}/{len(schedule)} successes in {elapsed / 3600:.2f} h ===")
    if failed:
        print(f"  exhausted plans: {failed}")
        return 1
    _finalize_dataset_splits(output_dir, schedule)
    print(f"  splits written: {(output_dir / 'meta' / 'atomic_splits.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
