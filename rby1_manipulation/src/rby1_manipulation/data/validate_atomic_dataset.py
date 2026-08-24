"""Validate structure, timing and semantic sidecars of an atomic basket dataset."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from rby1_manipulation.data.collect_atomic_transport_dataset import (
    FAMILY_TOTALS,
    FRUITS,
    SPLIT_TOTALS,
)
from rby1_manipulation.data.episode import CAMERAS, CHUNK_SIZE
from rby1_manipulation.data.finalize_atomic_dataset import split_ranges
from rby1_manipulation.tasks.transport_atomic import ATOMIC_SCHEMA_VERSION, Phase


REQUIRED_COLUMNS = {
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "task_index",
    "phase_index",
    "prompt_timestamp",
}


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_dataset(root: Path, *, check_video_frames: bool = False,
                     progress_every: int = 0) -> list[str]:
    errors: list[str] = []
    required = (
        root / "meta" / "info.json",
        root / "meta" / "episodes.jsonl",
        root / "meta" / "tasks.jsonl",
        root / "meta" / "atomic_schema.json",
        root / "meta" / "atomic_episodes.jsonl",
    )
    for path in required:
        if not path.exists():
            errors.append(f"missing {path.relative_to(root)}")
    if errors:
        return errors

    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    atomic_schema = json.loads(
        (root / "meta" / "atomic_schema.json").read_text(encoding="utf-8")
    )
    episodes = _jsonl(root / "meta" / "episodes.jsonl")
    tasks = _jsonl(root / "meta" / "tasks.jsonl")
    atomic = _jsonl(root / "meta" / "atomic_episodes.jsonl")
    total = int(info["total_episodes"])
    fps = int(info["fps"])
    if int(atomic_schema.get("version", -1)) != ATOMIC_SCHEMA_VERSION:
        errors.append(
            f"atomic schema is v{atomic_schema.get('version')}, "
            f"expected v{ATOMIC_SCHEMA_VERSION}"
        )
    if not (total == len(episodes) == len(atomic)):
        errors.append(
            f"episode count mismatch: info={total}, episodes={len(episodes)}, atomic={len(atomic)}"
        )
    if info["features"]["observation.state"]["shape"] != [14] or \
       info["features"]["action"]["shape"] != [14]:
        errors.append("state/action schema is not 14-D")
    if any("crate" in task["task"].lower() for task in tasks):
        errors.append("task vocabulary mixes crate into the basket dataset")

    episode_lengths = {int(entry["episode_index"]): int(entry["length"])
                       for entry in episodes}
    metadata_by_episode = {int(entry["episode_index"]): entry for entry in atomic}
    if len(metadata_by_episode) != len(atomic):
        errors.append("duplicate episode_index in atomic_episodes.jsonl")

    parquet_files = list((root / "data").glob("chunk-*/episode_*.parquet"))
    backup_files = list((root / "data").glob("chunk-*/*.bak"))
    video_files = list((root / "videos").glob("chunk-*/observation.images.*/episode_*.mp4"))
    if len(parquet_files) != total:
        errors.append(f"parquet file count mismatch: files={len(parquet_files)}, info={total}")
    if backup_files:
        errors.append(f"backup files inside data chunks: {len(backup_files)}")
    if len(video_files) != total * len(CAMERAS):
        errors.append(
            f"video file count mismatch: files={len(video_files)}, expected={total * len(CAMERAS)}"
        )

    try:
        expected_ranges, expected_split_details = split_ranges(atomic)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"invalid atomic split order: {exc}")
        expected_ranges, expected_split_details = {}, {}
    if expected_ranges and info.get("splits") != expected_ranges:
        errors.append(
            f"LeRobot split ranges mismatch: actual={info.get('splits')}, "
            f"expected={expected_ranges}"
        )
    split_path = root / "meta" / "atomic_splits.json"
    if not split_path.exists():
        errors.append("missing meta/atomic_splits.json")
    else:
        split_document = json.loads(split_path.read_text(encoding="utf-8"))
        if split_document.get("splits") != expected_split_details:
            errors.append("meta/atomic_splits.json does not match atomic episode metadata")

    for episode_index in range(total):
        if progress_every and episode_index and episode_index % progress_every == 0:
            print(f"  validated {episode_index}/{total} episodes", flush=True)
        chunk = episode_index // CHUNK_SIZE
        parquet_path = root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
        if not parquet_path.exists():
            errors.append(f"episode {episode_index}: missing parquet")
            continue
        table = pq.read_table(parquet_path)
        missing_columns = REQUIRED_COLUMNS - set(table.column_names)
        if missing_columns:
            errors.append(f"episode {episode_index}: missing columns {sorted(missing_columns)}")
            continue
        n = table.num_rows
        metadata = metadata_by_episode.get(episode_index)
        if metadata is None:
            errors.append(f"episode {episode_index}: missing atomic metadata")
            continue
        if n != episode_lengths.get(episode_index) or n != int(metadata["episode_length"]):
            errors.append(f"episode {episode_index}: length mismatch")
        state_type = table.schema.field("observation.state").type
        action_type = table.schema.field("action").type
        if getattr(state_type, "list_size", None) != 14 or \
           getattr(action_type, "list_size", None) != 14:
            errors.append(f"episode {episode_index}: state/action list size is not 14")
        timestamps = table["timestamp"].to_numpy()
        expected = np.arange(n, dtype=np.float64) / fps
        if not np.allclose(timestamps, expected, atol=1e-4, rtol=0.0):
            errors.append(f"episode {episode_index}: timestamp grid mismatch")
        if not np.array_equal(table["frame_index"].to_numpy(), np.arange(n)):
            errors.append(f"episode {episode_index}: frame_index mismatch")
        if np.any(table["prompt_timestamp"].to_numpy() != 0.0):
            errors.append(f"episode {episode_index}: prompt timestamp is not zero")
        phases = table["phase_index"].to_numpy()
        if n == 0 or int(phases[0]) != int(Phase.INITIAL_HOLD):
            errors.append(f"episode {episode_index}: no recorded initial_hold")
        if np.count_nonzero(phases == int(Phase.INITIAL_HOLD)) < fps:
            errors.append(f"episode {episode_index}: initial_hold shorter than one second")
        if np.count_nonzero(phases == int(Phase.TERMINAL_HOLD)) < fps:
            errors.append(f"episode {episode_index}: terminal_hold shorter than one second")

        events = metadata.get("events", {})
        ordered = [
            events.get("release_end_frame"),
            events.get("retreat_start_frame"),
            events.get("retreat_end_frame"),
            events.get("return_to_ready_start_frame"),
            events.get("return_to_ready_end_frame"),
            events.get("success_frame"),
            events.get("terminal_hold_end_frame"),
        ]
        if metadata["task_type"] == "place_one" and (
            any(value is None for value in ordered)
            or ordered != sorted(ordered)
        ):
            errors.append(f"episode {episode_index}: release/retreat/success event order invalid")
        if not metadata.get("success") or metadata.get("failure_reason") is not None:
            errors.append(f"episode {episode_index}: unsuccessful record in training dataset")
        validation = metadata.get("validation", {})
        required_semantics = (
            ("target_newly_inside", "fully_released", "non_target_unchanged",
             "preloaded_fruits_remain", "safe_retreat", "returned_to_ready",
             "wrist_camera_view_valid", "terminal_hold_valid")
            if metadata["task_type"] == "place_one"
            else ("basket_grasp_held", "preloaded_fruits_remain", "terminal_hold_valid")
        )
        for key in required_semantics:
            if not validation.get(key, False):
                errors.append(f"episode {episode_index}: semantic check failed: {key}")

        for camera in CAMERAS:
            video = (
                root / "videos" / f"chunk-{chunk:03d}"
                / f"observation.images.{camera}" / f"episode_{episode_index:06d}.mp4"
            )
            if not video.exists() or video.stat().st_size == 0:
                errors.append(f"episode {episode_index}: missing/empty {camera} video")
                continue
            if check_video_frames:
                import imageio.v2 as imageio
                reader = imageio.get_reader(video)
                try:
                    video_frames = reader.count_frames()
                finally:
                    reader.close()
                if video_frames != n:
                    errors.append(
                        f"episode {episode_index}: {camera} has {video_frames} frames, expected {n}"
                    )

    if total == 2000:
        families = Counter(entry["scenario_family"] for entry in atomic)
        splits = Counter(entry["split"] for entry in atomic)
        targets = Counter(entry["target_fruit"] for entry in atomic if entry["target_fruit"])
        if families != Counter(FAMILY_TOTALS):
            errors.append(f"family distribution mismatch: {dict(families)}")
        if splits != Counter(SPLIT_TOTALS):
            errors.append(f"split distribution mismatch: {dict(splits)}")
        if targets != Counter({fruit: 480 for fruit in FRUITS}):
            errors.append(f"target fruit distribution mismatch: {dict(targets)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--check-video-frames", action="store_true")
    args = parser.parse_args()
    root = Path(args.dataset).resolve()
    errors = validate_dataset(
        root,
        check_video_frames=args.check_video_frames,
        progress_every=100 if args.check_video_frames else 0,
    )
    if errors:
        print(f"FAIL: {len(errors)} validation error(s)")
        for error in errors[:100]:
            print(f"  - {error}")
        if len(errors) > 100:
            print(f"  ... {len(errors) - 100} more")
        return 1
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    print(f"OK: {root}")
    print(f"  episodes={info['total_episodes']} frames={info['total_frames']} fps={info['fps']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
