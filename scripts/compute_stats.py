"""Compute meta/stats.json for a LeRobot dataset.

pi0.5 / openpi's LeRobotDataConfig relies on per-feature statistics for
input normalization. This script scans every episode_XXXXXX.parquet in
`data/chunk-000/` and produces LeRobot-compliant stats:

  observation.state -> per-dim {min, max, mean, std, q01, q99}   (14,)
  action            -> per-dim {min, max, mean, std, q01, q99}   (14,)
  observation.images.<cam> -> per-channel {min, max, mean, std}  (1, 1, 3)
                              [computed from ImageNet defaults; openpi
                               applies its own vision-encoder normalization
                               later, so exact values here don't matter]

Output: <dataset_root>/meta/stats.json

Usage:
    python scripts/compute_stats.py --dataset data/rby1_dataset_v1
    python scripts/compute_stats.py --dataset data/rby1_dataset_v1 --skip-fail
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_fail_task_indices(dataset_root: Path) -> set[int]:
    """Return the set of task_index values whose task string starts with
    '[FAIL] '. Frames belonging to these tasks will be excluded from the
    stats computation when --skip-fail is set."""
    tasks_jsonl = dataset_root / "meta" / "tasks.jsonl"
    fail_indices: set[int] = set()
    if not tasks_jsonl.exists():
        return fail_indices
    with tasks_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry["task"].startswith("[FAIL] "):
                fail_indices.add(int(entry["task_index"]))
    return fail_indices


def collect_state_action(dataset_root: Path, skip_fail: bool) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (all_states, all_actions, n_frames_used)."""
    parquet_files = sorted((dataset_root / "data" / "chunk-000").glob("episode_*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No parquet files under {dataset_root}/data/chunk-000")

    fail_indices = load_fail_task_indices(dataset_root) if skip_fail else set()
    if fail_indices:
        print(f"Excluding {len(fail_indices)} [FAIL] task_index(es) from stats")

    states_list: list[np.ndarray] = []
    actions_list: list[np.ndarray] = []
    dropped_frames = 0

    for i, pq_file in enumerate(parquet_files):
        table = pq.read_table(
            pq_file,
            columns=["observation.state", "action", "task_index"],
        )
        state_col  = table.column("observation.state").to_pylist()
        action_col = table.column("action").to_pylist()
        task_col   = table.column("task_index").to_pylist()
        for s, a, t in zip(state_col, action_col, task_col):
            if int(t) in fail_indices:
                dropped_frames += 1
                continue
            states_list.append(np.asarray(s, dtype=np.float32))
            actions_list.append(np.asarray(a, dtype=np.float32))
        if (i + 1) % 100 == 0:
            print(f"  scanned {i+1}/{len(parquet_files)} episodes")

    states  = np.stack(states_list,  axis=0) if states_list  else np.zeros((0, 14), dtype=np.float32)
    actions = np.stack(actions_list, axis=0) if actions_list else np.zeros((0, 14), dtype=np.float32)
    if dropped_frames:
        print(f"  dropped {dropped_frames} frames from [FAIL] episodes")
    return states, actions, states.shape[0]


def stats_1d(arr: np.ndarray) -> dict:
    """Per-dim stats for a (N, D) array."""
    return {
        "min":  arr.min(axis=0).tolist(),
        "max":  arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std":  arr.std(axis=0).tolist(),
        "q01":  np.quantile(arr, 0.01, axis=0).tolist(),
        "q99":  np.quantile(arr, 0.99, axis=0).tolist(),
    }


def image_stats() -> dict:
    """Per-channel image stats in LeRobot's expected (1, 1, 3) shape.
    Values are ImageNet defaults; the vision encoder in pi0 re-normalizes
    internally so exact numbers here don't affect training."""
    return {
        "min":  [[[0.0, 0.0, 0.0]]],
        "max":  [[[1.0, 1.0, 1.0]]],
        "mean": [[[IMAGENET_MEAN[0], IMAGENET_MEAN[1], IMAGENET_MEAN[2]]]],
        "std":  [[[IMAGENET_STD[0],  IMAGENET_STD[1],  IMAGENET_STD[2]]]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="LeRobot dataset root")
    ap.add_argument("--skip-fail", action="store_true",
                    help="Exclude frames whose task string starts with '[FAIL] '")
    ap.add_argument("--cameras", nargs="+",
                    default=["cam_high", "cam_left_wrist", "cam_right_wrist"])
    args = ap.parse_args()

    root = Path(args.dataset).resolve()
    if not root.exists():
        raise SystemExit(f"dataset root {root} does not exist")

    print(f"Computing stats for {root}")
    states, actions, n_frames = collect_state_action(root, args.skip_fail)
    print(f"Aggregated {n_frames} frames")
    if n_frames == 0:
        raise SystemExit("No frames collected — nothing to compute stats over.")

    stats: dict = {}
    stats["observation.state"] = stats_1d(states)
    stats["action"]            = stats_1d(actions)
    for cam in args.cameras:
        stats[f"observation.images.{cam}"] = image_stats()

    stats_path = root / "meta" / "stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {stats_path}")

    # Quick sanity print
    print("\nobservation.state per-dim mean:")
    for i, m in enumerate(stats["observation.state"]["mean"]):
        print(f"  [{i:2d}] mean={m:+.4f}  std={stats['observation.state']['std'][i]:+.4f}  "
              f"[min={stats['observation.state']['min'][i]:+.3f}, "
              f"max={stats['observation.state']['max'][i]:+.3f}]")
    print("\naction per-dim mean:")
    for i, m in enumerate(stats["action"]["mean"]):
        print(f"  [{i:2d}] mean={m:+.4f}  std={stats['action']['std'][i]:+.4f}  "
              f"[min={stats['action']['min'][i]:+.3f}, "
              f"max={stats['action']['max'][i]:+.3f}]")


if __name__ == "__main__":
    main()
