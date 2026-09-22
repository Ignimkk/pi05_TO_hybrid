"""Validate a randomized 16-D RBY1 pick-place LeRobot dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from rby1_manipulation.data.collect_randomized_pick_place_dataset import (
    FRUITS,
    SPLIT_TOTALS,
)
from rby1_manipulation.data.episode import CAMERAS


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def validate_dataset(root: Path, *, check_video_frames: bool = False) -> list[str]:
    errors: list[str] = []
    required = (
        root / "randomized_collection_plan.json",
        root / "meta" / "info.json",
        root / "meta" / "episodes.jsonl",
        root / "meta" / "tasks.jsonl",
        root / "meta" / "randomized_schema.json",
        root / "meta" / "randomized_episodes.jsonl",
        root / "meta" / "randomized_splits.json",
    )
    for path in required:
        if not path.exists():
            errors.append(f"missing {path.relative_to(root)}")
    if errors:
        return errors

    plan = json.loads((root / "randomized_collection_plan.json").read_text(encoding="utf-8"))
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    schema = json.loads((root / "meta" / "randomized_schema.json").read_text(encoding="utf-8"))
    episodes = _jsonl(root / "meta" / "episodes.jsonl")
    tasks = _jsonl(root / "meta" / "tasks.jsonl")
    metadata = _jsonl(root / "meta" / "randomized_episodes.jsonl")
    split_metadata = json.loads(
        (root / "meta" / "randomized_splits.json").read_text(encoding="utf-8")
    )
    expected = len(plan["episodes"])
    total = int(info["total_episodes"])
    if not (expected == total == len(episodes) == len(metadata)):
        errors.append(
            f"episode count mismatch: plan={expected}, info={total}, "
            f"episodes={len(episodes)}, metadata={len(metadata)}"
        )
    if schema.get("state_action_schema") != "rby1_16":
        errors.append("randomized schema is not rby1_16")
    if schema.get("randomization_config_fingerprint") != plan.get("randomization_config_fingerprint"):
        errors.append("schema and plan randomization fingerprints differ")
    features = info.get("features", {})
    for key in ("observation.state", "action"):
        if features.get(key, {}).get("shape") != [16]:
            errors.append(f"{key} is not 16-D")
    expected_names = (
        [f"left_arm_{index}" for index in range(7)] + ["left_gripper"]
        + [f"right_arm_{index}" for index in range(7)] + ["right_gripper"]
    )
    if features.get("observation.state", {}).get("names") != expected_names:
        errors.append("16-D state names/order are incorrect")
    if features.get("action", {}).get("names") != expected_names:
        errors.append("16-D action names/order are incorrect")
    for camera in CAMERAS:
        feature = features.get(f"observation.images.{camera}", {})
        if feature.get("shape") != [224, 224, 3]:
            errors.append(f"{camera} is not 224x224 RGB")
        if feature.get("video_info", {}).get("video.fps") != 15:
            errors.append(f"{camera} is not 15 FPS")

    parquet_files = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    video_files = sorted((root / "videos").glob("chunk-*/observation.images.*/episode_*.mp4"))
    if len(parquet_files) != total:
        errors.append(f"parquet count is {len(parquet_files)}, expected {total}")
    if len(video_files) != total * len(CAMERAS):
        errors.append(f"video count is {len(video_files)}, expected {total * len(CAMERAS)}")
    task_indexes = {int(row["task_index"]): row["task"] for row in tasks}
    action_digests: set[str] = set()
    for path in parquet_files:
        table = pq.read_table(path)
        episode_index = int(path.stem.split("_")[-1])
        for key in ("observation.state", "action"):
            if getattr(table.schema.field(key).type, "list_size", None) != 16:
                errors.append(f"{path.name}: {key} is not fixed-size 16-D")
        timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=float)
        expected_timestamps = np.arange(len(table), dtype=float) / int(info["fps"])
        if not np.allclose(timestamps, expected_timestamps, atol=2e-6):
            errors.append(f"{path.name}: timestamps are not on the FPS grid")
        indexes = set(int(value) for value in table["task_index"].to_pylist())
        if len(indexes) != 1 or not indexes <= set(task_indexes):
            errors.append(f"{path.name}: invalid task_index")
        elif episode_index < len(metadata):
            task_index = next(iter(indexes))
            if task_indexes[task_index] != metadata[episode_index].get("prompt"):
                errors.append(f"{path.name}: task_index does not resolve to metadata prompt")
        if len(table) != int(episodes[episode_index]["length"]):
            errors.append(f"{path.name}: frame count differs from episodes.jsonl")
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        action_digests.add(hashlib.sha256(actions.tobytes()).hexdigest())

    required_metadata = {
        "task_id", "target_fruit", "prompt", "requested_target_pose",
        "settled_target_pose", "requested_goal_pose", "settled_goal_pose",
        "initial_robot_state", "initial_arm_joint_qpos", "scene_validity",
        "sampling_attempts", "success",
    }
    for index, row in enumerate(metadata):
        missing = required_metadata - set(row)
        if missing:
            errors.append(f"metadata episode {index}: missing {sorted(missing)}")
            continue
        if row["episode_index"] != index:
            errors.append(f"metadata episode indexes are not contiguous at {index}")
        if row["target_fruit"] not in FRUITS:
            errors.append(f"metadata episode {index}: invalid target")
        if len(row["initial_robot_state"]) != 16 or len(row["initial_arm_joint_qpos"]) != 14:
            errors.append(f"metadata episode {index}: invalid initial robot state")
        if not row["success"] or not row["scene_validity"].get("valid"):
            errors.append(f"metadata episode {index}: unsuccessful or invalid scene was saved")
        if row.get("randomization_config_fingerprint") != plan.get("randomization_config_fingerprint"):
            errors.append(f"metadata episode {index}: config fingerprint mismatch")
        if bool(row.get("is_paraphrase")) != (row["prompt"] != row["canonical_prompt"]):
            errors.append(f"metadata episode {index}: is_paraphrase is inconsistent")

    if total > 1:
        seeds = {int(row["seed"]) for row in metadata}
        scene_poses = {
            json.dumps(
                [row["requested_target_pose"], row["requested_goal_pose"]],
                separators=(",", ":"),
            )
            for row in metadata
        }
        if len(seeds) < 2:
            errors.append("multiple episodes reuse one seed")
        if len(scene_poses) < 2:
            errors.append("different seeds did not produce different requested scene poses")
        if len(action_digests) < 2:
            errors.append("different scenes did not produce different IK action trajectories")

    if plan.get("mode") == "full" and total == 2000:
        expected_ranges = {"train": "0:1600", "validation": "1600:1800", "test": "1800:2000"}
        if info.get("splits") != expected_ranges:
            errors.append(f"full dataset split ranges are incorrect: {info.get('splits')}")
        expected_split_metadata = {
            split: {
                "start_episode_index": int(bounds.split(":")[0]),
                "end_episode_index_exclusive": int(bounds.split(":")[1]),
                "episodes": int(bounds.split(":")[1]) - int(bounds.split(":")[0]),
            }
            for split, bounds in expected_ranges.items()
        }
        if split_metadata.get("splits") != expected_split_metadata:
            errors.append("randomized_splits.json ranges are incorrect")
        if Counter(row["split"] for row in metadata) != Counter(SPLIT_TOTALS):
            errors.append("full dataset split distribution is incorrect")
        if Counter(row["target_fruit"] for row in metadata) != Counter({fruit: 500 for fruit in FRUITS}):
            errors.append("full dataset target distribution is incorrect")
        if Counter(row["used_arm"] for row in metadata) != Counter({"left": 1000, "right": 1000}):
            errors.append("full dataset used-arm distribution is not balanced")
        specs = plan.get("episodes", [])
        if len(specs) != 2000:
            errors.append("full collection plan does not contain 2000 episodes")
        else:
            for index, (row, spec) in enumerate(zip(metadata, specs)):
                if row.get("episode_index") != index or row.get("plan_index") != index:
                    errors.append(f"episode {index}: dataset is not in plan_index order")
                    break
                if any(row.get(key) != spec.get(key) for key in ("split", "target_fruit", "prompt")):
                    errors.append(f"episode {index}: metadata does not match collection plan")
                    break

    if check_video_frames:
        try:
            import imageio.v2 as imageio
        except ImportError:
            import imageio
        lengths = {int(row["episode_index"]): int(row["length"]) for row in episodes}
        def count_video(path: Path) -> tuple[Path, int | None, str | None]:
            episode_index = int(path.stem.split("_")[-1])
            try:
                reader = imageio.get_reader(path)
                try:
                    frames = int(reader.count_frames())
                finally:
                    reader.close()
            except Exception as error:  # pragma: no cover - backend-specific diagnostic
                return path, None, f"cannot decode ({error})"
            if frames != lengths.get(episode_index):
                return path, frames, f"frames={frames}, expected={lengths.get(episode_index)}"
            return path, frames, None

        workers = min(8, max(1, len(video_files)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for completed, (path, _, error) in enumerate(
                executor.map(count_video, video_files), start=1
            ):
                if error is not None:
                    errors.append(f"{path.relative_to(root)}: {error}")
                if completed % 250 == 0 or completed == len(video_files):
                    print(f"video frames checked: {completed}/{len(video_files)}", flush=True)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--check-video-frames", action="store_true")
    args = parser.parse_args()
    root = Path(args.dataset).resolve()
    errors = validate_dataset(root, check_video_frames=args.check_video_frames)
    if errors:
        print("randomized dataset validation FAILED")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(f"randomized dataset validation OK: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
