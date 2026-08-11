"""Success checks for block pick-and-place tasks."""

from __future__ import annotations

import mujoco
import numpy as np

from rby1_manipulation.simulation.block_scene import body_pos


def check_success(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    block_name: str,
    container_pos: np.ndarray,
    container_half_xy: float = 0.11,
    table_top_z: float = 0.82,
) -> bool:
    position = body_pos(model, data, block_name)
    inside_xy = (
        abs(position[0] - container_pos[0]) < container_half_xy
        and abs(position[1] - container_pos[1]) < container_half_xy
    )
    return inside_xy and position[2] > table_top_z - 0.01
