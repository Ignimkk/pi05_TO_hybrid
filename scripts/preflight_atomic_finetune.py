"""Strict, dependency-light preflight for the RBY1 atomic basket fine-tuning dataset."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
EXPECTED_SPLITS = {
    "train": "0:1591",
    "validation": "1591:1790",
    "test": "1790:1989",
}


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--require-stats", action="store_true")
    args = parser.parse_args()

    root = Path(args.dataset).resolve()
    errors: list[str] = []
    info = json.loads((root / "meta" / "info.json").read_text())
    episodes = jsonl(root / "meta" / "episodes.jsonl")
    atomic = jsonl(root / "meta" / "atomic_episodes.jsonl")
    total = int(info["total_episodes"])
    fps = int(info["fps"])
    expected_frames = int(info["total_frames"])

    if (total, expected_frames, fps) != (1989, 622261, 15):
        errors.append(
            f"unexpected totals: episodes={total}, frames={expected_frames}, fps={fps}"
        )
    if len(episodes) != total or len(atomic) != total:
        errors.append("metadata episode counts do not match info.json")
    if info.get("splits") != EXPECTED_SPLITS:
        errors.append(f"unexpected info.json splits: {info.get('splits')}")

    split_counts = Counter(entry["split"] for entry in atomic)
    if split_counts != Counter({"train": 1591, "validation": 199, "test": 199}):
        errors.append(f"unexpected atomic split counts: {dict(split_counts)}")
    transitions = []
    previous = None
    for entry in atomic:
        if entry["split"] != previous:
            transitions.append((int(entry["episode_index"]), entry["split"]))
            previous = entry["split"]
    if transitions != [(0, "train"), (1591, "validation"), (1790, "test")]:
        errors.append(f"non-contiguous split order: {transitions}")
    if any(not entry.get("success", False) for entry in atomic):
        errors.append("atomic metadata contains unsuccessful episodes")

    backups = list((root / "data").glob("chunk-*/*.bak"))
    temporary = list(root.rglob("*.tmp"))
    if backups:
        errors.append(f"{len(backups)} .bak files found inside data chunks")
    if temporary:
        errors.append(f"{len(temporary)} temporary files found in dataset")

    parquet_files = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    if len(parquet_files) != total:
        errors.append(f"parquet count is {len(parquet_files)}, expected {total}")
    rows = 0
    train_rows = 0
    for index, path in enumerate(parquet_files):
        table = pq.read_table(
            path,
            columns=["timestamp", "episode_index", "observation.state", "action"],
        )
        n = table.num_rows
        rows += n
        if index < 1591:
            train_rows += n
        timestamp = table["timestamp"].to_numpy(zero_copy_only=False)
        if not np.allclose(timestamp, np.arange(n) / fps, atol=1e-4, rtol=0.0):
            errors.append(f"timestamp grid mismatch: {path.name}")
        if n and set(table["episode_index"].to_pylist()) != {index}:
            errors.append(f"episode_index mismatch: {path.name}")
        if getattr(table.schema.field("observation.state").type, "list_size", None) != 14:
            errors.append(f"state is not 14-D: {path.name}")
        if getattr(table.schema.field("action").type, "list_size", None) != 14:
            errors.append(f"action is not 14-D: {path.name}")
        if (index + 1) % 250 == 0:
            print(f"  parquet {index + 1}/{total}", flush=True)
    if rows != expected_frames:
        errors.append(f"parquet rows={rows}, expected {expected_frames}")
    if train_rows != 497754:
        errors.append(f"train rows={train_rows}, expected 497754")

    for camera in CAMERAS:
        videos = list(
            (root / "videos").glob(f"chunk-*/observation.images.{camera}/episode_*.mp4")
        )
        if len(videos) != total:
            errors.append(f"{camera} videos={len(videos)}, expected {total}")
    if args.require_stats:
        stats = root / "meta" / "stats.json"
        if not stats.is_file() or stats.stat().st_size == 0:
            errors.append("meta/stats.json is missing or empty")
        else:
            json.loads(stats.read_text())

    if errors:
        print(f"FAIL: {len(errors)} preflight error(s)")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("OK: atomic fine-tuning dataset preflight")
    print(f"  all episodes/frames : {total}/{rows}")
    print(f"  train episodes/frames: 1591/{train_rows}")
    print("  timestamp grid       : exact nominal 15 fps")
    print("  parquet backups      : 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
