"""Per-step capture shared by both transport scenarios.

Wraps the two things a scenario wants from a rollout - a third-person mp4 and a
LeRobot episode - behind one `on_step` callback, so the motion code stays free of
recording concerns.

Sampling matches the block pipeline: frames are taken at `fps`, not every
physics step (the sim runs at 500 Hz), and the three policy cameras keep their
existing names and 224x224 size so the observation interface is unchanged.
"""
from __future__ import annotations

import pathlib
from typing import Callable, Dict, Optional

import mujoco
import numpy as np

from rby1_manipulation.data.episode import EpisodeBuffer, Frame, LeRobotWriter

POLICY_IMAGE_SIZE = 224
RECORD_SIZE = (640, 480)
# Third-person view, matching pi05_ex_infer's "front" preset.
RECORD_LOOKAT = np.array([0.35, -0.6, 0.85])
# Keep the camera INSIDE the office room: at this azimuth/elevation it sits at
# about x = -2.09, and the room interior spans x in [-2.9, 3.5]. Pushing much
# past 3.5 m puts it in the wall and every recorded video shows only plaster.
RECORD_DISTANCE = 3.0
RECORD_AZIMUTH = 150.0
RECORD_ELEVATION = -20.0


class EpisodeRecorder:
    """Collects dataset frames and/or a third-person video during a rollout."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        state_fn: Callable[[], np.ndarray],
        action_fn: Callable[[], np.ndarray],
        cam_name_map: Dict[str, str],
        task: str,
        record_path: Optional[str] = None,
        dataset_root: Optional[str] = None,
        fps: int = 15,
        schema: str = "rby1_17_mobile",
    ):
        self.model, self.data = model, data
        self.state_fn, self.action_fn = state_fn, action_fn
        self.cam_name_map = cam_name_map
        self.fps = fps
        self.record_path = record_path
        self._steps = 0
        self._every = max(1, int(round(1.0 / (fps * model.opt.timestep))))

        self.writer: Optional[LeRobotWriter] = None
        self.episode: Optional[EpisodeBuffer] = None
        self.policy_renderer: Optional[mujoco.Renderer] = None
        if dataset_root is not None:
            self.writer = LeRobotWriter(dataset_root, fps=fps,
                                        image_wh=(POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE),
                                        schema=schema)
            self.episode = self.writer.new_episode(task=task)
            self.policy_renderer = mujoco.Renderer(model, POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE)

        self.video_renderer: Optional[mujoco.Renderer] = None
        self.video_frames: list = []
        self.video_camera: Optional[mujoco.MjvCamera] = None
        if record_path is not None:
            self.video_renderer = mujoco.Renderer(model, RECORD_SIZE[1], RECORD_SIZE[0])
            cam = mujoco.MjvCamera()
            cam.lookat[:] = RECORD_LOOKAT
            cam.distance = RECORD_DISTANCE
            cam.azimuth = RECORD_AZIMUTH
            cam.elevation = RECORD_ELEVATION
            self.video_camera = cam

    # ---------- rollout hook ----------

    def on_step(self) -> None:
        self._steps += 1
        if self._steps % self._every:
            return
        t = self._steps * self.model.opt.timestep

        if self.episode is not None and self.policy_renderer is not None:
            images = {}
            for logical, cam in self.cam_name_map.items():
                self.policy_renderer.update_scene(self.data, camera=cam)
                images[logical] = self.policy_renderer.render().copy()
            self.episode.append(Frame(
                state=np.asarray(self.state_fn(), dtype=np.float32),
                action=np.asarray(self.action_fn(), dtype=np.float32),
                images=images,
                timestamp=float(t),
                frame_index=len(self.episode),
            ))

        if self.video_renderer is not None:
            self.video_renderer.update_scene(self.data, camera=self.video_camera)
            self.video_frames.append(self.video_renderer.render().copy())

    # ---------- teardown ----------

    def finish(self, *, success: bool, save_failed: bool = False) -> None:
        if self.video_frames and self.record_path:
            try:
                import imageio.v2 as imageio
            except ImportError:
                import imageio
            path = pathlib.Path(self.record_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(str(path), self.video_frames, fps=self.fps,
                            codec="libx264", quality=8)
            print(f"    video -> {path}")

        if self.writer is None or self.episode is None:
            return
        if not success and not save_failed:
            print("    episode not saved (SUCCESS=False; pass --save-failed to keep it)")
            return
        if not success:
            # Same convention as collect_dataset.py so failures are filterable.
            self.episode.task = f"[FAIL] {self.episode.task}"
        self.writer.save_episode(self.episode)
        self.writer.finalize()
        print(f"    episode {self.episode.episode_index} "
              f"({len(self.episode)} frames) -> {self.writer.root}")
