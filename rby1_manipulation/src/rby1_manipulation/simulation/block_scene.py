"""Names and pose helpers for the block pick-and-place scene."""

from __future__ import annotations

import mujoco
import numpy as np

from rby1_manipulation.paths import BLOCK_MODEL_XML


MODEL_XML = str(BLOCK_MODEL_XML)
BLOCK_BODIES = {"red": "red_block", "green": "green_block", "blue": "blue_block"}
CONTAINER_BODY = "container"
BLOCK_HALF_SIZE = 0.025


def body_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    """Return a copy of a named body's world position."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xpos[body_id].copy()
