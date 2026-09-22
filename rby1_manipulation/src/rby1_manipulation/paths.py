"""Runtime paths shared by the manipulation package.

Configuration is shipped inside this package.  The robot description remains a
separate workspace package and can be selected explicitly with
``RBY1_DESCRIPTION_ROOT`` when this package is installed outside the workspace.
"""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PACKAGE_DIR / "config"
BLOCK_GRID_CONFIG = CONFIG_DIR / "grids" / "block_grid.json"
BLOCK_FALSE_GRID_CONFIG = CONFIG_DIR / "grids" / "block_false_grid.json"
FRUIT_GRID_CONFIG = CONFIG_DIR / "grids" / "fruit_grid.json"
TRANSPORT_LAYOUT_CONFIG = CONFIG_DIR / "transport_layout.json"
PICK_PLACE_OBSTACLE_CONFIG = CONFIG_DIR / "pick_place_obstacles.json"
RANDOMIZED_PICK_PLACE_CONFIG = CONFIG_DIR / "randomized_pick_place.json"


def _description_root() -> Path:
    override = os.environ.get("RBY1_DESCRIPTION_ROOT")
    if override:
        path = Path(override).expanduser().resolve()
        if path.is_dir():
            return path
        raise FileNotFoundError(
            f"RBY1_DESCRIPTION_ROOT does not name a directory: {path}"
        )

    for parent in PACKAGE_DIR.parents:
        candidate = parent / "rby1_description"
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Cannot locate rby1_description. Source the workspace or set "
        "RBY1_DESCRIPTION_ROOT to that package directory."
    )


DESCRIPTION_ROOT = _description_root()
WORKSPACE_SRC_ROOT = DESCRIPTION_ROOT.parent
WORKSPACE_ROOT = WORKSPACE_SRC_ROOT.parent
MUJOCO_MODEL_DIR = DESCRIPTION_ROOT / "models" / "rby1a" / "mujoco"
BLOCK_MODEL_XML = MUJOCO_MODEL_DIR / "model.xml"
PICK_PLACE_OBSTACLE_MODEL_XML = MUJOCO_MODEL_DIR / "model_pick_place_obstacles.xml"
TRANSPORT_MODEL_XML = MUJOCO_MODEL_DIR / "model_transport.xml"
TRANSPORT_PICK_PLACE_OBSTACLE_MODEL_XML = (
    MUJOCO_MODEL_DIR / "model_transport_pick_place_obstacles.xml"
)
TRANSPORT_WHEEL_MODEL_XML = MUJOCO_MODEL_DIR / "model_transport_wheels.xml"
