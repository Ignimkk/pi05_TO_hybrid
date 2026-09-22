"""Collect balanced 16-D randomized RBY1 fruit-to-basket demonstrations."""
from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from rby1_manipulation.simulation.randomized_pick_place import (
    canonical_prompt,
    load_randomization_config,
    prompts_for_target,
    randomization_config_fingerprint,
)
from rby1_manipulation.simulation.transport_scene import OBJECT_TYPES


FRUITS = tuple(OBJECT_TYPES)
SPLITS = ("train", "validation", "test")
SPLIT_TARGET_COUNTS = {"train": 400, "validation": 50, "test": 50}
SPLIT_TOTALS = {"train": 1600, "validation": 200, "test": 200}
DATASET_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EpisodeSpec:
    plan_index: int
    split: str
    target_fruit: str
    layout_index: int
    target_slot: int
    slot_order: tuple[str, ...]
    seed: int
    prompt: str
    canonical_prompt: str


def _balanced_counts(total: int, buckets: int, phase: int) -> list[int]:
    counts = [total // buckets] * buckets
    for index in range(total % buckets):
        counts[(phase + index) % buckets] += 1
    return counts


def build_schedule(*, seed: int = 20260918, config=None) -> list[EpisodeSpec]:
    config = config or load_randomization_config()
    rng = np.random.default_rng(seed)
    raw: list[tuple[str, str, int, str]] = []
    for split_index, split in enumerate(SPLITS):
        count = SPLIT_TARGET_COUNTS[split]
        for fruit_index, fruit in enumerate(FRUITS):
            if split == "train":
                slot_counts = [100, 100, 100, 100]
            elif split == "validation":
                slot_counts = _balanced_counts(count, 4, fruit_index)
            else:
                validation = _balanced_counts(SPLIT_TARGET_COUNTS["validation"], 4, fruit_index)
                slot_counts = [25 - value for value in validation]
            slots = [slot for slot, amount in enumerate(slot_counts) for _ in range(amount)]
            prompts = prompts_for_target(config, fruit)
            prompt_counts = _balanced_counts(count, len(prompts), fruit_index + split_index)
            prompt_values = [
                prompt for prompt, amount in zip(prompts, prompt_counts) for _ in range(amount)
            ]
            rng.shuffle(slots)
            rng.shuffle(prompt_values)
            raw.extend((split, fruit, slot, prompt) for slot, prompt in zip(slots, prompt_values))

    ordered: list[tuple[str, str, int, str]] = []
    for split in SPLITS:
        members = [item for item in raw if item[0] == split]
        rng.shuffle(members)
        ordered.extend(members)

    remaining_cycles = {
        fruit: itertools.cycle(itertools.permutations([other for other in FRUITS if other != fruit]))
        for fruit in FRUITS
    }
    schedule: list[EpisodeSpec] = []
    for plan_index, (split, fruit, target_slot, prompt) in enumerate(ordered):
        remaining = next(remaining_cycles[fruit])
        slot_order = list(remaining)
        slot_order.insert(target_slot, fruit)
        schedule.append(EpisodeSpec(
            plan_index=plan_index,
            split=split,
            target_fruit=fruit,
            layout_index=plan_index % 16,
            target_slot=target_slot,
            slot_order=tuple(slot_order),
            seed=10_000 + plan_index,
            prompt=prompt,
            canonical_prompt=canonical_prompt(fruit),
        ))
    validate_schedule(schedule, config=config)
    return schedule


def validate_schedule(schedule: Sequence[EpisodeSpec], *, config=None) -> None:
    config = config or load_randomization_config()
    if len(schedule) != 2000:
        raise ValueError(f"expected 2000 episodes, got {len(schedule)}")
    if Counter(spec.split for spec in schedule) != Counter(SPLIT_TOTALS):
        raise ValueError("split totals are not 1600/200/200")
    if [spec.split for spec in schedule] != [
        *(["train"] * 1600), *(["validation"] * 200), *(["test"] * 200)
    ]:
        raise ValueError("dataset splits are not contiguous")
    if Counter(spec.target_fruit for spec in schedule) != Counter({fruit: 500 for fruit in FRUITS}):
        raise ValueError("target fruits are not balanced")
    if Counter(spec.layout_index for spec in schedule) != Counter({index: 125 for index in range(16)}):
        raise ValueError("layouts are not balanced")
    arm_counts = Counter("left" if spec.target_slot < 2 else "right" for spec in schedule)
    if arm_counts != Counter({"left": 1000, "right": 1000}):
        raise ValueError("target arms are not balanced")
    for fruit in FRUITS:
        members = [spec for spec in schedule if spec.target_fruit == fruit]
        if Counter(spec.target_slot for spec in members) != Counter({index: 125 for index in range(4)}):
            raise ValueError(f"target slots are not balanced for {fruit}")
        expected_prompts = set(prompts_for_target(config, fruit))
        if set(spec.prompt for spec in members) != expected_prompts:
            raise ValueError(f"prompt coverage is incomplete for {fruit}")
        for split, count in SPLIT_TARGET_COUNTS.items():
            split_members = [spec for spec in members if spec.split == split]
            if len(split_members) != count:
                raise ValueError(f"{fruit} has the wrong {split} count")
            prompt_counts = Counter(spec.prompt for spec in split_members)
            if max(prompt_counts.values()) - min(prompt_counts.values()) > 1:
                raise ValueError(f"{fruit} prompts are not balanced within {split}")


def schedule_summary(schedule: Sequence[EpisodeSpec]) -> dict:
    return {
        "total": len(schedule),
        "splits": dict(sorted(Counter(spec.split for spec in schedule).items())),
        "targets": dict(sorted(Counter(spec.target_fruit for spec in schedule).items())),
        "layouts": dict(sorted(Counter(spec.layout_index for spec in schedule).items())),
        "target_slots": {
            fruit: dict(sorted(Counter(
                spec.target_slot for spec in schedule if spec.target_fruit == fruit
            ).items()))
            for fruit in FRUITS
        },
        "prompts": dict(sorted(Counter(spec.prompt for spec in schedule).items())),
    }


def smoke_schedule(schedule: Sequence[EpisodeSpec], per_fruit: int) -> list[EpisodeSpec]:
    limited: list[EpisodeSpec] = []
    for fruit in FRUITS:
        members = [spec for spec in schedule if spec.split == "train" and spec.target_fruit == fruit][:per_fruit]
        limited.extend(members)
    return [replace(spec, plan_index=index, split="train") for index, spec in enumerate(limited)]


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def _load_preflight(path: Path, fingerprint: str) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("passed"):
        raise RuntimeError("preflight report did not pass")
    if int(report.get("requested_samples", 0)) < 100:
        raise RuntimeError("preflight report must cover at least 100 resets")
    if report.get("config_fingerprint") != fingerprint:
        raise RuntimeError("preflight report was produced from a different randomization config")
    return report


def _command(spec: EpisodeSpec, output_dir: Path, args) -> list[str]:
    command = [
        sys.executable,
        "-m", "rby1_manipulation.tasks.transport_atomic",
        "--task", "place_one",
        "--target-fruit", spec.target_fruit,
        "--layout-index", str(spec.layout_index),
        "--slot-order", *spec.slot_order,
        "--seed", str(spec.seed),
        "--headless",
        "--log-dataset", str(output_dir),
        "--schema", "rby1_16",
        "--randomization-config", str(Path(args.randomization_config).resolve()),
        "--task-prompt", spec.prompt,
        "--canonical-prompt", spec.canonical_prompt,
        "--scenario-family", "randomized_pick_place",
        "--plan-index", str(spec.plan_index),
        "--split", spec.split,
        "--log-fps", str(args.log_fps),
        "--speed-scale", str(args.speed_scale),
        "--initial-hold", str(args.initial_hold),
        "--terminal-hold", str(args.terminal_hold),
    ]
    if spec.prompt != spec.canonical_prompt:
        command.append("--is-paraphrase")
    return command


def _extract_json(stdout: str, marker: str):
    match = re.search(rf"^>>> {re.escape(marker)} = (.+)$", stdout, flags=re.MULTILINE)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def _finalize_splits(output_dir: Path) -> None:
    metadata_path = output_dir / "meta" / "randomized_episodes.jsonl"
    rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line]
    splits: dict[str, dict] = {}
    ranges: dict[str, str] = {}
    cursor = 0
    for split in SPLITS:
        count = sum(row["split"] == split for row in rows)
        if count:
            if any(row["split"] != split for row in rows[cursor:cursor + count]):
                raise RuntimeError(f"{split} episodes are not contiguous")
            ranges[split] = f"{cursor}:{cursor + count}"
            splits[split] = {
                "start_episode_index": cursor,
                "end_episode_index_exclusive": cursor + count,
                "episodes": count,
            }
            cursor += count
    info_path = output_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["splits"] = ranges
    _write_json(info_path, info)
    _write_json(output_dir / "meta" / "randomized_splits.json", {
        "version": 1, "splits": splits
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--randomization-config", required=True)
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--schedule-seed", type=int, default=20260918)
    parser.add_argument("--log-fps", type=int, default=15)
    parser.add_argument("--speed-scale", type=float, default=1.25)
    parser.add_argument("--initial-hold", type=float, default=1.0)
    parser.add_argument("--terminal-hold", type=float, default=2.0)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--smoke-per-fruit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-limit", type=int, default=12)
    parser.add_argument(
        "--defer-finalize",
        action="store_true",
        help="Leave split metadata unfinished so a recovery reindex pass can run.",
    )
    args = parser.parse_args()

    config = load_randomization_config(args.randomization_config)
    fingerprint = randomization_config_fingerprint(config)
    preflight = _load_preflight(Path(args.preflight_report), fingerprint)
    full_schedule = build_schedule(seed=args.schedule_seed, config=config)
    schedule = (
        smoke_schedule(full_schedule, args.smoke_per_fruit)
        if args.smoke_per_fruit > 0 else full_schedule
    )
    mode = "smoke" if args.smoke_per_fruit > 0 else "full"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "version": DATASET_SCHEMA_VERSION,
        "dataset": "rby1_randomized_pick_place_16d_v1",
        "mode": mode,
        "schema": "rby1_16",
        "schedule_seed": args.schedule_seed,
        "randomization_config_fingerprint": fingerprint,
        "preflight_report": str(Path(args.preflight_report).resolve()),
        "preflight_seed": preflight["seed"],
        "log_fps": args.log_fps,
        "speed_scale": args.speed_scale,
        "initial_hold_seconds": args.initial_hold,
        "terminal_hold_seconds": args.terminal_hold,
        "summary": schedule_summary(schedule),
        "episodes": [asdict(spec) for spec in schedule],
    }
    plan = json.loads(json.dumps(plan))
    plan_path = output_dir / "randomized_collection_plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        comparison = (
            "version", "dataset", "mode", "schema", "schedule_seed",
            "randomization_config_fingerprint", "log_fps", "speed_scale",
            "initial_hold_seconds", "terminal_hold_seconds", "summary", "episodes",
        )
        if any(existing.get(key) != plan.get(key) for key in comparison):
            raise RuntimeError("output directory contains a different collection plan")
    elif not args.dry_run:
        _write_json(plan_path, plan)

    print("=== randomized 16-D pick-place collection ===")
    print(f"  mode         : {mode}")
    print(f"  target       : {len(schedule)} successful episodes")
    print(f"  summary      : {plan['summary']}")
    print(f"  config hash  : {fingerprint}")
    if args.dry_run:
        for spec in schedule[:args.dry_run_limit]:
            print(f"  [{spec.plan_index:04d}] " + " ".join(_command(spec, output_dir, args)))
        return 0

    stats_path = output_dir / "randomized_collection_stats.json"
    attempts_path = output_dir / "randomized_collection_attempts.jsonl"
    manifest_path = output_dir / "randomized_collection_manifest.jsonl"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {
        "version": 1, "episodes": {}
    }
    completed = sum(value.get("success", False) for value in stats["episodes"].values())
    info_path = output_dir / "meta" / "info.json"
    if info_path.exists():
        saved = int(json.loads(info_path.read_text(encoding="utf-8"))["total_episodes"])
        if saved != completed:
            raise RuntimeError(f"resume state mismatch: dataset has {saved}, stats has {completed}")

    started = time.time()
    for sequence_index, base_spec in enumerate(schedule, start=1):
        key = str(base_spec.plan_index)
        record = stats["episodes"].setdefault(key, {"attempts": 0, "success": False})
        if record.get("success"):
            continue
        for attempt in range(int(record["attempts"]), args.max_attempts):
            spec = replace(base_spec, seed=base_spec.seed + attempt * 100_000)
            command = _command(spec, output_dir, args)
            episode_started = time.time()
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
                stdout = result.stdout or ""
                success = result.returncode == 0 and ">>> SUCCESS = True" in stdout
                failure_match = re.search(r"^>>> FAILURE_REASON = (.+)$", stdout, re.MULTILINE)
                failure_reason = None if success else (
                    failure_match.group(1) if failure_match else f"returncode_{result.returncode}"
                )
            except subprocess.TimeoutExpired as error:
                stdout = error.stdout or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode(errors="replace")
                success = False
                failure_reason = "timeout"
            scene = _extract_json(stdout, "RANDOMIZED_SCENE")
            attempt_row = {
                **asdict(spec),
                "attempt": attempt + 1,
                "success": success,
                "failure_reason": failure_reason,
                "elapsed_seconds": round(time.time() - episode_started, 3),
                "randomized_scene": scene,
            }
            _append_jsonl(attempts_path, attempt_row)
            record.update({
                "attempts": attempt + 1,
                "last_seed": spec.seed,
                "last_failure_reason": failure_reason,
                "success": success,
            })
            if success:
                match = re.search(r"episode (\d+) \(", stdout)
                dataset_index = int(match.group(1)) if match else None
                record["dataset_episode_index"] = dataset_index
                attempt_row["dataset_episode_index"] = dataset_index
                _append_jsonl(manifest_path, attempt_row)
                completed += 1
            _write_json(stats_path, stats)
            status = "OK" if success else f"FAIL {failure_reason}"
            print(
                f"  [{sequence_index:04d}/{len(schedule):04d}] plan={base_spec.plan_index:04d} "
                f"attempt={attempt + 1} completed={completed:04d} {status}"
            )
            if success:
                break

    failed = [key for key, value in stats["episodes"].items() if not value.get("success")]
    print(
        f"=== complete: {completed}/{len(schedule)} successes in "
        f"{(time.time() - started) / 3600:.2f} h ==="
    )
    if failed:
        print(f"  exhausted plans: {failed}")
        return 1
    if args.defer_finalize:
        print("  split finalization deferred for recovery reindex")
        return 0
    _finalize_splits(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
