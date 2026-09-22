"""Rebuild a recovered randomized dataset in contiguous plan/split order."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from rby1_manipulation.data.episode import CAMERAS, CHUNK_SIZE


SPLIT_RANGES = {
    "train": (0, 1600),
    "validation": (1600, 1800),
    "test": (1800, 2000),
}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def rebuild_dataset(source: Path, output: Path) -> dict:
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("source and output must be different directories")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    building = output.with_name(output.name + ".building")
    if building.exists():
        raise FileExistsError(f"stale build directory exists: {building}")

    plan = json.loads((source / "randomized_collection_plan.json").read_text(encoding="utf-8"))
    specs = plan.get("episodes", [])
    if len(specs) != 2000 or [int(row["plan_index"]) for row in specs] != list(range(2000)):
        raise RuntimeError("collection plan is not the expected contiguous 0..1999 schedule")
    metadata = _jsonl(source / "meta" / "randomized_episodes.jsonl")
    episodes = _jsonl(source / "meta" / "episodes.jsonl")
    if len(metadata) != 2000 or len(episodes) != 2000:
        raise RuntimeError(
            f"recovery is incomplete: metadata={len(metadata)}, episodes={len(episodes)}"
        )
    metadata_by_plan = {int(row["plan_index"]): row for row in metadata}
    if set(metadata_by_plan) != set(range(2000)) or len(metadata_by_plan) != len(metadata):
        raise RuntimeError("metadata does not contain exactly one row for every plan_index")
    episodes_by_index = {int(row["episode_index"]): row for row in episodes}
    if len(episodes_by_index) != 2000:
        raise RuntimeError("episodes.jsonl has duplicate or missing episode indexes")

    stats_path = source / "randomized_collection_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if len(stats.get("episodes", {})) != 2000 or not all(
        row.get("success") for row in stats["episodes"].values()
    ):
        raise RuntimeError("collection stats do not report 2000 successful plans")

    building.mkdir(parents=True)
    new_metadata: list[dict] = []
    new_episodes: list[dict] = []
    old_to_new: dict[int, int] = {}
    total_frames = 0
    try:
        for new_index, spec in enumerate(specs):
            plan_index = int(spec["plan_index"])
            row = dict(metadata_by_plan[plan_index])
            old_index = int(row["episode_index"])
            old_to_new[old_index] = new_index
            expected_split = next(
                split for split, (start, end) in SPLIT_RANGES.items()
                if start <= new_index < end
            )
            for key, expected in (
                ("split", expected_split),
                ("target_fruit", spec["target_fruit"]),
                ("prompt", spec["prompt"]),
            ):
                if row.get(key) != expected:
                    raise RuntimeError(
                        f"plan {plan_index}: metadata {key}={row.get(key)!r}, expected {expected!r}"
                    )

            old_chunk = old_index // CHUNK_SIZE
            new_chunk = new_index // CHUNK_SIZE
            source_parquet = (
                source / "data" / f"chunk-{old_chunk:03d}" / f"episode_{old_index:06d}.parquet"
            )
            table = pq.read_table(source_parquet)
            episode_column = table.schema.get_field_index("episode_index")
            table = table.set_column(
                episode_column,
                "episode_index",
                pa.array([new_index] * len(table), type=pa.int64()),
            )
            destination_parquet = (
                building / "data" / f"chunk-{new_chunk:03d}" / f"episode_{new_index:06d}.parquet"
            )
            destination_parquet.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, destination_parquet)

            for camera in CAMERAS:
                source_video = (
                    source / "videos" / f"chunk-{old_chunk:03d}"
                    / f"observation.images.{camera}" / f"episode_{old_index:06d}.mp4"
                )
                destination_video = (
                    building / "videos" / f"chunk-{new_chunk:03d}"
                    / f"observation.images.{camera}" / f"episode_{new_index:06d}.mp4"
                )
                _link_or_copy(source_video, destination_video)

            episode_row = dict(episodes_by_index[old_index])
            episode_row["episode_index"] = new_index
            row["episode_index"] = new_index
            new_episodes.append(episode_row)
            new_metadata.append(row)
            total_frames += int(episode_row["length"])

        (building / "meta").mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "meta" / "tasks.jsonl", building / "meta" / "tasks.jsonl")
        schema = json.loads(
            (source / "meta" / "randomized_schema.json").read_text(encoding="utf-8")
        )
        recovered_plans = sorted(
            int(plan_index) for plan_index, value in stats["episodes"].items()
            if int(value.get("attempts", 0)) > 10
        )
        schema["expert_trajectory_revisions"] = [{
            "reason": "recover banana plans exhausted by fingertip/table contact",
            "plan_indexes": recovered_plans,
            "previous_banana_grasp_dz_m": 0.022,
            "recovery_banana_grasp_dz_m": 0.030,
        }]
        _write_json(building / "meta" / "randomized_schema.json", schema)
        _write_jsonl(building / "meta" / "episodes.jsonl", new_episodes)
        _write_jsonl(building / "meta" / "randomized_episodes.jsonl", new_metadata)

        info = json.loads((source / "meta" / "info.json").read_text(encoding="utf-8"))
        info.update({
            "total_episodes": 2000,
            "total_frames": total_frames,
            "total_chunks": 2,
            "chunks_size": CHUNK_SIZE,
            "splits": {split: f"{start}:{end}" for split, (start, end) in SPLIT_RANGES.items()},
        })
        _write_json(building / "meta" / "info.json", info)
        _write_json(building / "meta" / "randomized_splits.json", {
            "version": 1,
            "splits": {
                split: {
                    "start_episode_index": start,
                    "end_episode_index_exclusive": end,
                    "episodes": end - start,
                }
                for split, (start, end) in SPLIT_RANGES.items()
            },
        })

        _write_json(building / "randomized_collection_plan.json", plan)
        new_stats = json.loads(json.dumps(stats))
        for plan_index, value in new_stats["episodes"].items():
            value["dataset_episode_index"] = int(plan_index)
        _write_json(building / "randomized_collection_stats.json", new_stats)
        shutil.copy2(
            source / "randomized_collection_attempts.jsonl",
            building / "randomized_collection_attempts.jsonl",
        )
        manifest = _jsonl(source / "randomized_collection_manifest.jsonl")
        if len(manifest) != 2000:
            raise RuntimeError(f"success manifest has {len(manifest)} rows, expected 2000")
        for row in manifest:
            row["dataset_episode_index"] = int(row["plan_index"])
        manifest.sort(key=lambda row: int(row["plan_index"]))
        _write_jsonl(building / "randomized_collection_manifest.jsonl", manifest)

        report = {
            "version": 1,
            "source": str(source),
            "output": str(output),
            "episodes": 2000,
            "total_frames": total_frames,
            "hardlinked_videos": 2000 * len(CAMERAS),
            "recovered_plan_indexes": recovered_plans,
            "banana_grasp_dz_m": {"original": 0.022, "recovery": 0.030},
            "old_to_new_episode_index": {str(key): value for key, value in sorted(old_to_new.items())},
        }
        _write_json(building / "reindex_report.json", report)
        building.replace(output)
        return report
    except Exception:
        # Keep the build directory for diagnosis. The source dataset is never modified.
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = rebuild_dataset(Path(args.source), Path(args.output))
    print(json.dumps({key: value for key, value in report.items() if key != "old_to_new_episode_index"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
