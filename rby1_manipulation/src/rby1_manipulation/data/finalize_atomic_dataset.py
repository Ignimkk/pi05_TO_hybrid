"""Finalize LeRobot split metadata for a complete or intentionally partial atomic dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


SPLIT_ORDER = ("train", "validation", "test")


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def split_ranges(episodes: Sequence[dict]) -> tuple[dict[str, str], dict[str, dict]]:
    """Return contiguous LeRobot ranges after validating atomic episode order."""
    if not episodes:
        raise ValueError("atomic dataset has no episodes")
    indexes = [int(episode["episode_index"]) for episode in episodes]
    if indexes != list(range(len(episodes))):
        raise ValueError("atomic episode indexes are not consecutive from zero")

    runs: list[tuple[str, int, int]] = []
    start = 0
    current = str(episodes[0]["split"])
    for index, episode in enumerate(episodes[1:], start=1):
        split = str(episode["split"])
        if split != current:
            runs.append((current, start, index))
            current, start = split, index
    runs.append((current, start, len(episodes)))

    names = [name for name, _, _ in runs]
    expected = list(SPLIT_ORDER[:len(names)])
    if names != expected:
        raise ValueError(
            f"atomic splits must be contiguous in {SPLIT_ORDER} order; got {names}"
        )

    ranges = {name: f"{start}:{end}" for name, start, end in runs}
    details = {
        name: {
            "start_episode_index": start,
            "end_episode_index_exclusive": end,
            "episodes": end - start,
        }
        for name, start, end in runs
    }
    return ranges, details


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def finalize_dataset(root: Path, *, dry_run: bool = False) -> dict:
    root = root.resolve()
    info_path = root / "meta" / "info.json"
    atomic_path = root / "meta" / "atomic_episodes.jsonl"
    if not info_path.exists() or not atomic_path.exists():
        raise FileNotFoundError("dataset is missing meta/info.json or meta/atomic_episodes.jsonl")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    episodes = _read_jsonl(atomic_path)
    total = int(info.get("total_episodes", -1))
    if total != len(episodes):
        raise ValueError(
            f"episode count mismatch: info.json={total}, atomic metadata={len(episodes)}"
        )
    ranges, details = split_ranges(episodes)

    plan_path = root / "atomic_collection_plan.json"
    planned = total
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        planned = len(plan.get("episodes", [])) or total
    collected_plans = {
        int(episode["plan_index"])
        for episode in episodes
        if episode.get("plan_index") is not None
    }
    missing_plans = sorted(set(range(planned)) - collected_plans)
    split_document = {
        "version": 1,
        "collected_episodes": total,
        "planned_episodes": planned,
        "intentionally_partial": total != planned,
        "missing_plan_indexes": missing_plans,
        "splits": details,
    }

    if not dry_run:
        info["splits"] = ranges
        _write_json_atomic(info_path, info)
        _write_json_atomic(root / "meta" / "atomic_splits.json", split_document)
    return {"ranges": ranges, **split_document}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = finalize_dataset(Path(args.dataset), dry_run=args.dry_run)
    print("atomic split finalization" + (" (dry run)" if args.dry_run else ""))
    print(f"  collected/planned: {result['collected_episodes']}/{result['planned_episodes']}")
    print(f"  ranges: {result['ranges']}")
    print(f"  missing plans: {result['missing_plan_indexes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
