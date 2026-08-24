"""Per-step capture shared by both transport scenarios.

Wraps the two things a scenario wants from a rollout - a third-person mp4 and a
LeRobot episode - behind one `on_step` callback, so the motion code stays free of
recording concerns.

Sampling matches the block pipeline: frames are taken at `fps`, not every
physics step (the sim runs at 500 Hz), and the three policy cameras keep their
existing names and 224x224 size so the observation interface is unchanged.
"""
from __future__ import annotations

import concurrent.futures
import multiprocessing
import os
import pathlib
from typing import Callable, Dict, Optional

import mujoco
import numpy as np
from PIL import Image

from rby1_manipulation.data.episode import EpisodeBuffer, Frame, LeRobotWriter

POLICY_IMAGE_SIZE = 224
# Render with the physical camera's 4:3 field of view, then resize to the
# square tensor consumed by the policy. Rendering MuJoCo directly at 224x224
# changes the horizontal field of view (90 degrees instead of about 106 degrees
# for fovy=90), cropping the fingers and nearby objects from wrist views.
POLICY_SOURCE_HEIGHT = POLICY_IMAGE_SIZE
POLICY_SOURCE_WIDTH = round(POLICY_IMAGE_SIZE * 4 / 3)
# Software OSMesa spends over half of policy-camera render time on scene
# reflections. They are not task observations, so disable only that render flag
# while preserving RGB resolution, camera poses, lighting, shadows, and FPS.
POLICY_RENDER_REFLECTIONS = False
RECORD_SIZE = (640, 480)
# Third-person view, matching pi05_ex_infer's "front" preset.
RECORD_LOOKAT = np.array([0.35, -0.6, 0.85])
# Keep the camera INSIDE the office room: at this azimuth/elevation it sits at
# about x = -2.09, and the room interior spans x in [-2.9, 3.5]. Pushing much
# past 3.5 m puts it in the wall and every recorded video shows only plaster.
RECORD_DISTANCE = 3.0
RECORD_AZIMUTH = 150.0
RECORD_ELEVATION = -20.0


def resize_policy_image(image: np.ndarray) -> np.ndarray:
    """Resize a native 4:3 camera frame to the policy's 224x224 tensor."""
    return np.asarray(
        Image.fromarray(image).resize(
            (POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE),
            resample=Image.Resampling.BILINEAR,
        )
    )


def _render_policy_camera(
    model_xml_path: str,
    logical_name: str,
    camera_name: str,
    qpos_sequence: np.ndarray,
) -> tuple[str, list[np.ndarray]]:
    """Render one policy camera in an isolated worker process."""
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    model = mujoco.MjModel.from_xml_path(model_xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(
        model, height=POLICY_SOURCE_HEIGHT, width=POLICY_SOURCE_WIDTH
    )
    images: list[np.ndarray] = []
    try:
        for qpos in qpos_sequence:
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera_name)
            if not POLICY_RENDER_REFLECTIONS:
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            images.append(resize_policy_image(renderer.render()))
    finally:
        renderer.close()
    return logical_name, images


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
        defer_policy_rendering: bool = False,
        model_xml_path: Optional[str] = None,
        phase_fn: Optional[Callable[[], int]] = None,
        prompt_timestamp: float = 0.0,
    ):
        self.model, self.data = model, data
        self.state_fn, self.action_fn = state_fn, action_fn
        self.cam_name_map = cam_name_map
        self.fps = fps
        self.record_path = record_path
        self.defer_policy_rendering = defer_policy_rendering
        self.model_xml_path = model_xml_path
        self.phase_fn = phase_fn
        self.prompt_timestamp = float(prompt_timestamp)
        self._policy_qpos: list[np.ndarray] = []
        self._steps = 0
        self._every = max(1, int(round(1.0 / (fps * model.opt.timestep))))

        if defer_policy_rendering and dataset_root is not None and model_xml_path is None:
            raise ValueError("model_xml_path is required for deferred policy rendering")

        self.writer: Optional[LeRobotWriter] = None
        self.episode: Optional[EpisodeBuffer] = None
        self.policy_renderer: Optional[mujoco.Renderer] = None
        if dataset_root is not None:
            self.writer = LeRobotWriter(dataset_root, fps=fps,
                                        image_wh=(POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE),
                                        schema=schema,
                                        frame_metadata=phase_fn is not None)
            self.episode = self.writer.new_episode(task=task)
            if not defer_policy_rendering:
                self.policy_renderer = mujoco.Renderer(
                    model,
                    height=POLICY_SOURCE_HEIGHT,
                    width=POLICY_SOURCE_WIDTH,
                )

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
        if self.episode is not None:
            images: Dict[str, np.ndarray] = {}
            if self.defer_policy_rendering:
                self._policy_qpos.append(self.data.qpos.copy())
            elif self.policy_renderer is not None:
                for logical, cam in self.cam_name_map.items():
                    self.policy_renderer.update_scene(self.data, camera=cam)
                    if not POLICY_RENDER_REFLECTIONS:
                        self.policy_renderer.scene.flags[
                            mujoco.mjtRndFlag.mjRND_REFLECTION
                        ] = 0
                    images[logical] = resize_policy_image(
                        self.policy_renderer.render()
                    )
            self.episode.append(Frame(
                state=np.asarray(self.state_fn(), dtype=np.float32),
                action=np.asarray(self.action_fn(), dtype=np.float32),
                images=images,
                # Videos are encoded at nominal CFR. Use their exact PTS grid
                # rather than rounded physics-step time (e.g. 33 * 0.002 =
                # 0.066, which does not satisfy a nominal 15 Hz timeline).
                timestamp=float(len(self.episode) / self.fps),
                frame_index=len(self.episode),
                phase_index=int(self.phase_fn()) if self.phase_fn is not None else -1,
                prompt_timestamp=self.prompt_timestamp,
            ))

        if self.video_renderer is not None:
            self.video_renderer.update_scene(self.data, camera=self.video_camera)
            self.video_frames.append(self.video_renderer.render().copy())

    # ---------- teardown ----------

    def _render_deferred_images(self) -> None:
        if not self.defer_policy_rendering or self.episode is None:
            return
        if len(self._policy_qpos) != len(self.episode):
            raise RuntimeError(
                "deferred render state count does not match recorded frame count"
            )
        if not self._policy_qpos:
            return

        qpos_sequence = np.stack(self._policy_qpos)
        print(
            f"    rendering {len(self.episode)} frames from "
            f"{len(self.cam_name_map)} policy cameras in parallel ..."
        )
        context = multiprocessing.get_context("spawn")
        rendered: Dict[str, list[np.ndarray]] = {}
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(self.cam_name_map), mp_context=context
        ) as executor:
            futures = [
                executor.submit(
                    _render_policy_camera,
                    str(self.model_xml_path),
                    logical,
                    camera,
                    qpos_sequence,
                )
                for logical, camera in self.cam_name_map.items()
            ]
            for future in concurrent.futures.as_completed(futures):
                logical, images = future.result()
                rendered[logical] = images

        for frame_index, frame in enumerate(self.episode.frames):
            frame.images.update({
                logical: rendered[logical][frame_index]
                for logical in self.cam_name_map
            })

    def finish(self, *, success: bool, save_failed: bool = False) -> Optional[int]:
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
            return None
        if not success and not save_failed:
            print("    episode not saved (SUCCESS=False; pass --save-failed to keep it)")
            return None
        if not success:
            # Same convention as collect_dataset.py so failures are filterable.
            self.episode.task = f"[FAIL] {self.episode.task}"
        self._render_deferred_images()
        self.writer.save_episode(self.episode)
        self.writer.finalize()
        print(f"    episode {self.episode.episode_index} "
              f"({len(self.episode)} frames) -> {self.writer.root}")
        return self.episode.episode_index
