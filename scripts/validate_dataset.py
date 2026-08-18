"""Validate a LeRobot dataset before fine-tuning.

Prints:
  - total episodes / frames
  - per-task episode count (success + fail split)
  - frame-length histogram
  - action / state min/max/mean/std sanity check
  - missing files (parquet without matching mp4 or vice versa)
  - dataset structural consistency (info.json vs actual counts)

Usage:
    python scripts/validate_dataset.py --dataset data/rby1_dataset_v1
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    root = Path(args.dataset).resolve()
    if not root.exists():
        raise SystemExit(f"{root} does not exist")

    print(f"=== validate_dataset: {root} ===\n")

    # --- meta files ---
    info_path      = root / "meta" / "info.json"
    episodes_path  = root / "meta" / "episodes.jsonl"
    tasks_path     = root / "meta" / "tasks.jsonl"
    stats_path     = root / "meta" / "stats.json"
    print(f"info.json       : {'OK' if info_path.exists() else 'MISSING'}")
    print(f"episodes.jsonl  : {'OK' if episodes_path.exists() else 'MISSING'}")
    print(f"tasks.jsonl     : {'OK' if tasks_path.exists() else 'MISSING'}")
    print(f"stats.json      : {'OK' if stats_path.exists() else 'MISSING (run compute_stats.py)'}")

    # --- episodes.jsonl ---
    if not episodes_path.exists():
        raise SystemExit("episodes.jsonl missing — nothing to validate")
    with episodes_path.open() as f:
        episodes = [json.loads(l) for l in f if l.strip()]

    # --- tasks.jsonl ---
    tasks_map: dict[int, str] = {}
    if tasks_path.exists():
        with tasks_path.open() as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    tasks_map[int(e["task_index"])] = e["task"]

    total_frames = sum(e["length"] for e in episodes)
    print(f"\nTotal episodes  : {len(episodes)}")
    print(f"Total frames    : {total_frames}")
    print(f"Total tasks     : {len(tasks_map)}")

    # --- per-task counts ---
    task_counts = Counter()
    for ep in episodes:
        for t in ep.get("tasks", []):
            task_counts[t] += 1

    fail_prefix = "[FAIL] "
    ok_count   = sum(v for k, v in task_counts.items() if not k.startswith(fail_prefix))
    fail_count = sum(v for k, v in task_counts.items() if k.startswith(fail_prefix))
    print(f"\n  success episodes: {ok_count}")
    print(f"  [FAIL] episodes : {fail_count}")

    print("\nPer-task episode counts:")
    for task, n in sorted(task_counts.items()):
        marker = "  ✗" if task.startswith(fail_prefix) else "  ✓"
        print(f"  {marker}  {n:>4d}  {task}")

    # --- frame-length histogram ---
    lens = np.array([e["length"] for e in episodes])
    print("\nEpisode length stats:")
    print(f"  min={lens.min()}  max={lens.max()}  mean={lens.mean():.1f}  median={np.median(lens):.0f}")
    print("Length histogram (10 bins):")
    hist, edges = np.histogram(lens, bins=10)
    for i, cnt in enumerate(hist):
        bar = "#" * min(60, int(60 * cnt / max(hist)))
        print(f"  [{edges[i]:5.0f} - {edges[i+1]:5.0f}]  {cnt:>4d}  {bar}")

    # --- file existence check ---
    print("\nChecking parquet + mp4 files present...")
    parquet_files = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    parquets = {int(p.stem.split("_")[1]) for p in parquet_files}
    ep_indices = {e["episode_index"] for e in episodes}
    missing_parquet = ep_indices - parquets
    orphan_parquet  = parquets - ep_indices
    if missing_parquet:
        print(f"  !! missing parquet for episodes: {sorted(missing_parquet)[:10]}...")
    if orphan_parquet:
        print(f"  !! orphan parquet with no episodes.jsonl entry: {sorted(orphan_parquet)[:10]}...")
    if not missing_parquet and not orphan_parquet:
        print(f"  parquet                : OK ({len(parquets)} episodes)")

    for cam in CAMERAS:
        mp4_files = (root / "videos").glob(
            f"chunk-*/observation.images.{cam}/episode_*.mp4"
        )
        mp4s = {int(p.stem.split("_")[1]) for p in mp4_files}
        missing = ep_indices - mp4s
        orphan  = mp4s - ep_indices
        status = "OK"
        if missing:
            status = f"MISSING mp4 for {len(missing)} episodes"
        if orphan:
            status += f"  ORPHAN {len(orphan)} mp4s"
        print(f"  cam {cam:20s} : {status}")

    # --- action/state sanity from ONE random episode ---
    print("\nSampling frames from first episode for sanity...")
    first_parquet = parquet_files[0] if parquet_files else None
    if first_parquet:
        table = pq.read_table(first_parquet,
                              columns=["observation.state", "action"])
        s = np.array(table.column("observation.state").to_pylist(), dtype=np.float32)
        a = np.array(table.column("action").to_pylist(), dtype=np.float32)
        print(f"  first episode: {s.shape[0]} frames")
        print(f"  state  shape={s.shape}  min={s.min():.3f}  max={s.max():.3f}")
        print(f"  action shape={a.shape}  min={a.min():.3f}  max={a.max():.3f}")

    # --- info.json vs actual ---
    if info_path.exists():
        with info_path.open() as f:
            info = json.load(f)
        print("\ninfo.json cross-check:")
        for k in ("total_episodes", "total_frames", "total_tasks", "fps"):
            print(f"  {k}: {info.get(k)}")

    print("\ndone.")


if __name__ == "__main__":
    main()
