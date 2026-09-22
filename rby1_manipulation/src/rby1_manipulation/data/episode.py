"""LeRobot-style episode logger for the dual-arm ALOHA schema.

Per-frame observation collected during scripted rollouts:
    observation.state           : selected writer schema (14-D, 16-D, or 17-D)
    observation.images.cam_high        : zed_left    RGB (H, W, 3) uint8
    observation.images.cam_left_wrist  : wrist_cam_l RGB (H, W, 3) uint8
    observation.images.cam_right_wrist : wrist_cam_r RGB (H, W, 3) uint8
    action                      : same layout as state, but the *commanded*
                                  next joint position (i.e. data.ctrl at the time
                                  the frame was captured)
    timestamp                   : seconds since episode start
    frame_index                 : int
    episode_index               : int
    task                        : language prompt string

Directory layout (LeRobot compliant):
    <root>/
      meta/
        info.json          - fps, cameras, dimensions, feature dtypes
        episodes.jsonl     - one JSON line per episode with length + task
        tasks.jsonl        - dedup'd task -> task_index (a LeRobot convention)
      data/chunk-000/
        episode_XXXXXX.parquet     - all non-image features per frame
      videos/chunk-000/observation.images.<cam>/
        episode_XXXXXX.mp4         - one mp4 per camera per episode

This is intentionally minimal: we do NOT compute stats.json here (LeRobot
recomputes stats when the dataset is uploaded/loaded). Chunk size is fixed at
1000 (LeRobot default) so long collection runs still fit in chunk-000 for
now — if you cross that, add chunk splitting.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
CHUNK_SIZE = 1000  # episodes per chunk directory (LeRobot default)

# Schema registry. "rby1_14" is the original fixed-arm layout and stays the
# default, so every existing caller writes byte-identical datasets. Randomized
# pick-place opts into full-arm ``rby1_16``; mobile scenarios opt into
# ``rby1_17_mobile``, whose first 14 entries retain the legacy quantities and
# append the planar base pose.
_ARM14 = [
    "left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3", "left_arm_4", "left_arm_5",
    "left_gripper",
    "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4", "right_arm_5",
    "right_gripper",
]
_ARM16 = [
    "left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3", "left_arm_4", "left_arm_5",
    "left_arm_6", "left_gripper",
    "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4", "right_arm_5",
    "right_arm_6", "right_gripper",
]
SCHEMAS = {
    "rby1_14": {"dim": 14, "names": _ARM14, "robot_type": "rby1"},
    "rby1_16": {"dim": 16, "names": _ARM16, "robot_type": "rby1"},
    "rby1_17_mobile": {"dim": 17,
                       "names": _ARM14 + ["base_x", "base_y", "base_yaw"],
                       "robot_type": "rby1_mobile"},
}
DEFAULT_SCHEMA = "rby1_14"


@dataclass
class Frame:
    state: np.ndarray                      # (14,), (16,), or (17,) float32, per writer schema
    action: np.ndarray                     # same shape as state
    images: Dict[str, np.ndarray]          # cam name -> HxWx3 uint8
    timestamp: float
    frame_index: int
    # Optional atomic-task annotations. Existing datasets omit the columns;
    # atomic collectors enable them through LeRobotWriter(frame_metadata=True).
    phase_index: int = -1
    prompt_timestamp: float = 0.0


@dataclass
class EpisodeBuffer:
    """One episode's frames + metadata, held in memory until save()."""
    episode_index: int
    task: str
    frames: List[Frame] = field(default_factory=list)

    def append(self, frame: Frame) -> None:
        self.frames.append(frame)

    def __len__(self) -> int:
        return len(self.frames)


class LeRobotWriter:
    """Streaming writer for a LeRobot dataset.

    Usage:
        writer = LeRobotWriter("/path/to/dataset", fps=30, image_wh=(224, 224))
        ep = writer.new_episode(task="pick up the red block")
        # ... during rollout: ep.append(Frame(...)) ...
        writer.save_episode(ep)
        writer.finalize()   # writes meta/info.json etc.
    """

    def __init__(self, root: str | pathlib.Path, *, fps: int,
                 image_wh: tuple[int, int], schema: str = DEFAULT_SCHEMA,
                 frame_metadata: bool = False):
        if schema not in SCHEMAS:
            raise ValueError(f"unknown schema {schema!r}; known: {sorted(SCHEMAS)}")
        spec = SCHEMAS[schema]
        self.schema = schema
        self.state_dim = spec["dim"]
        self.feature_names = spec["names"]
        self.robot_type = spec["robot_type"]
        self.root = pathlib.Path(root)
        self.fps = fps
        self.image_w, self.image_h = image_wh
        self.frame_metadata = frame_metadata
        self._episodes_written: List[Dict[str, Any]] = []
        self._task_to_index: Dict[str, int] = {}

        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        (self.root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        for cam in CAMERAS:
            (self.root / "videos" / "chunk-000" / f"observation.images.{cam}").mkdir(parents=True, exist_ok=True)

        # RESUME: if the dataset root already contains episodes (e.g. from a
        # prior subprocess run of collect_dataset.py), load their metadata so
        # new_episode() picks up the next available episode_index and
        # finalize() rewrites the jsonl files with the accumulated set (not
        # just this subprocess's episodes). Without this, every subprocess
        # starts from episode_index=0 and overwrites episode_000000.*.
        self._load_existing_state()

    def _load_existing_state(self) -> None:
        """Populate _episodes_written and _task_to_index from any existing
        episodes.jsonl / tasks.jsonl in the dataset root."""
        episodes_jsonl = self.root / "meta" / "episodes.jsonl"
        if episodes_jsonl.exists():
            with open(episodes_jsonl) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._episodes_written.append(json.loads(line))
        tasks_jsonl = self.root / "meta" / "tasks.jsonl"
        if tasks_jsonl.exists():
            with open(tasks_jsonl) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        self._task_to_index[entry["task"]] = int(entry["task_index"])

    # ---------- public API ----------

    def new_episode(self, task: str, *, episode_index: Optional[int] = None) -> EpisodeBuffer:
        if episode_index is None:
            episode_index = len(self._episodes_written)
        if task not in self._task_to_index:
            self._task_to_index[task] = len(self._task_to_index)
        return EpisodeBuffer(episode_index=episode_index, task=task)

    def save_episode(self, ep: EpisodeBuffer) -> None:
        if len(ep) == 0:
            raise ValueError("cannot save an empty episode")
        # Auto-register the task string if the caller mutated ep.task after
        # new_episode() (e.g. adding a "[FAIL] " prefix for failed episodes).
        if ep.task not in self._task_to_index:
            self._task_to_index[ep.task] = len(self._task_to_index)
        first = ep.frames[0]
        for name, arr in (("state", first.state), ("action", first.action)):
            if arr.shape != (self.state_dim,):
                raise ValueError(
                    f"{name} has shape {arr.shape}, but schema {self.schema!r} "
                    f"expects ({self.state_dim},)")
        self._write_parquet(ep)
        self._write_videos(ep)
        self._episodes_written.append({
            "episode_index": ep.episode_index,
            "length": len(ep),
            "tasks": [ep.task],
        })

    def finalize(self) -> None:
        self._write_info()
        self._write_episodes_jsonl()
        self._write_tasks_jsonl()

    # ---------- internals ----------

    def _write_parquet(self, ep: EpisodeBuffer) -> None:
        n = len(ep)
        state  = np.stack([f.state  for f in ep.frames]).astype(np.float32)
        action = np.stack([f.action for f in ep.frames]).astype(np.float32)
        timestamp    = np.array([f.timestamp   for f in ep.frames], dtype=np.float32)
        frame_index  = np.array([f.frame_index for f in ep.frames], dtype=np.int64)
        episode_idx  = np.full(n, ep.episode_index, dtype=np.int64)
        task_idx     = np.full(n, self._task_to_index[ep.task], dtype=np.int64)
        # LeRobot expects index (global) and next.done / next.reward, we can
        # populate a simple sequence-local index and dummy reward=0/done at end.
        index        = np.arange(n, dtype=np.int64)
        next_done    = np.zeros(n, dtype=bool); next_done[-1] = True
        next_reward  = np.zeros(n, dtype=np.float32)

        columns = {
            "observation.state": pa.array(state.tolist(),
                                          type=pa.list_(pa.float32(), self.state_dim)),
            "action":            pa.array(action.tolist(),
                                          type=pa.list_(pa.float32(), self.state_dim)),
            "timestamp":         pa.array(timestamp),
            "frame_index":       pa.array(frame_index),
            "episode_index":     pa.array(episode_idx),
            "index":             pa.array(index),
            "task_index":        pa.array(task_idx),
            "next.done":         pa.array(next_done),
            "next.reward":       pa.array(next_reward),
        }
        if self.frame_metadata:
            columns["phase_index"] = pa.array(
                np.asarray([frame.phase_index for frame in ep.frames], dtype=np.int64)
            )
            columns["prompt_timestamp"] = pa.array(
                np.asarray([frame.prompt_timestamp for frame in ep.frames], dtype=np.float32)
            )
        table = pa.table(columns)
        chunk = ep.episode_index // CHUNK_SIZE
        chunk_dir = self.root / "data" / f"chunk-{chunk:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        path = chunk_dir / f"episode_{ep.episode_index:06d}.parquet"
        pq.write_table(table, path)

    def _write_videos(self, ep: EpisodeBuffer) -> None:
        try:
            import imageio.v2 as imageio
        except ImportError:
            import imageio
        for cam in CAMERAS:
            frames = [f.images[cam] for f in ep.frames]
            chunk = ep.episode_index // CHUNK_SIZE
            video_dir = (
                self.root
                / "videos"
                / f"chunk-{chunk:03d}"
                / f"observation.images.{cam}"
            )
            video_dir.mkdir(parents=True, exist_ok=True)
            path = video_dir / f"episode_{ep.episode_index:06d}.mp4"
            imageio.mimsave(str(path), frames, fps=self.fps,
                            codec="libx264", quality=8)

    def _write_info(self) -> None:
        info = {
            "codebase_version": "v2.0",
            "robot_type": self.robot_type,
            "total_episodes": len(self._episodes_written),
            "total_frames": sum(e["length"] for e in self._episodes_written),
            "total_tasks": len(self._task_to_index),
            "total_chunks": max(
                1,
                (len(self._episodes_written) + CHUNK_SIZE - 1) // CHUNK_SIZE,
            ),
            "chunks_size": CHUNK_SIZE,
            "fps": self.fps,
            "splits": {"train": f"0:{len(self._episodes_written)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": ("videos/chunk-{episode_chunk:03d}/{video_key}/"
                           "episode_{episode_index:06d}.mp4"),
            "features": self._features_dict(),
        }
        with open(self.root / "meta" / "info.json", "w") as f:
            json.dump(info, f, indent=2)

    def _features_dict(self) -> Dict[str, Any]:
        images = {
            f"observation.images.{cam}": {
                "dtype": "video",
                "shape": [self.image_h, self.image_w, 3],
                "names": ["height", "width", "channel"],
                "video_info": {"video.fps": self.fps, "video.codec": "h264",
                               "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                               "has_audio": False},
            }
            for cam in CAMERAS
        }
        features = {
            **images,
            "observation.state": {"dtype": "float32", "shape": [self.state_dim],
                                  "names": list(self.feature_names)},
            "action": {"dtype": "float32", "shape": [self.state_dim],
                       "names": list(self.feature_names)},
            "timestamp":     {"dtype": "float32", "shape": [1], "names": ["timestamp"]},
            "frame_index":   {"dtype": "int64",   "shape": [1], "names": ["frame_index"]},
            "episode_index": {"dtype": "int64",   "shape": [1], "names": ["episode_index"]},
            "index":         {"dtype": "int64",   "shape": [1], "names": ["index"]},
            "task_index":    {"dtype": "int64",   "shape": [1], "names": ["task_index"]},
            "next.done":     {"dtype": "bool",    "shape": [1], "names": ["done"]},
            "next.reward":   {"dtype": "float32", "shape": [1], "names": ["reward"]},
        }
        if self.frame_metadata:
            features.update({
                "phase_index": {
                    "dtype": "int64", "shape": [1], "names": ["phase_index"]
                },
                "prompt_timestamp": {
                    "dtype": "float32", "shape": [1], "names": ["prompt_timestamp"]
                },
            })
        return features

    def _write_episodes_jsonl(self) -> None:
        with open(self.root / "meta" / "episodes.jsonl", "w") as f:
            for ep in self._episodes_written:
                f.write(json.dumps(ep) + "\n")

    def _write_tasks_jsonl(self) -> None:
        with open(self.root / "meta" / "tasks.jsonl", "w") as f:
            for task, idx in sorted(self._task_to_index.items(), key=lambda kv: kv[1]):
                f.write(json.dumps({"task_index": idx, "task": task}) + "\n")
