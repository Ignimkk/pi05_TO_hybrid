"""Preview static pick-and-place obstacles through viewer and policy cameras."""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

if "--headless" in sys.argv and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "osmesa"

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from rby1_manipulation.paths import (
    PICK_PLACE_OBSTACLE_MODEL_XML,
    TRANSPORT_PICK_PLACE_OBSTACLE_MODEL_XML,
)
from rby1_manipulation.simulation.pick_place_obstacles import (
    PickPlaceObstacleManager,
    load_pick_place_obstacle_config,
    profile_names,
)
from rby1_manipulation.simulation.fruit_grid import (
    OBJECT_TYPES as FRUIT_TYPES,
    load_fruit_grid_config,
    reset_fruit_grid_scene,
)
from rby1_manipulation.simulation.transport_scene import load_layout_config


POLICY_CAMERAS = {
    "cam_high": "zed_left",
    "cam_left_wrist": "wrist_cam_l",
    "cam_right_wrist": "wrist_cam_r",
}


def reset_scene(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "teleop")
    mujoco.mj_resetDataKeyframe(model, data, key)
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        data.ctrl[actuator_id] = data.qpos[model.jnt_qposadr[joint_id]]
    mujoco.mj_forward(model, data)


def save_camera_preview(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    output_dir: pathlib.Path,
    profile: str,
) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=224, width=224)
    frames: list[tuple[str, np.ndarray]] = []
    try:
        for logical_name, camera_name in POLICY_CAMERAS.items():
            renderer.update_scene(data, camera=camera_name)
            image = renderer.render().copy()
            frames.append((logical_name, image))
            Image.fromarray(image).save(output_dir / f"{profile}_{logical_name}.png")
    finally:
        renderer.close()

    title_height = 28
    montage = Image.new("RGB", (224 * len(frames), 224 + title_height), "white")
    draw = ImageDraw.Draw(montage)
    for index, (logical_name, image) in enumerate(frames):
        x = index * 224
        montage.paste(Image.fromarray(image), (x, title_height))
        draw.text((x + 6, 7), logical_name, fill="black")
    montage_path = output_dir / f"{profile}_policy_cameras.png"
    montage.save(montage_path)
    return montage_path


def main() -> int:
    config = load_pick_place_obstacle_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=profile_names(config), default="right_bollard")
    parser.add_argument(
        "--scene", choices=("block", "fruit"), default="block",
        help="use block pick-and-place or fruit-to-crate transport scene",
    )
    parser.add_argument("--config", type=pathlib.Path, default=None)
    parser.add_argument(
        "--fruit-layout-index",
        type=int,
        default=None,
        help="use one of the 16 training fruit-grid layouts",
    )
    parser.add_argument(
        "--fruit-slot-order",
        nargs=4,
        choices=FRUIT_TYPES,
        default=None,
        metavar=("FRUIT1", "FRUIT2", "FRUIT3", "FRUIT4"),
    )
    parser.add_argument(
        "--fruit-preloaded",
        nargs="*",
        choices=FRUIT_TYPES,
        default=None,
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="save the three exact 224x224 policy-camera views and a montage",
    )
    args = parser.parse_args()

    if args.fruit_layout_index is not None and args.scene != "fruit":
        parser.error("--fruit-layout-index requires --scene fruit")
    if args.fruit_layout_index is None and (
        args.fruit_slot_order is not None or args.fruit_preloaded is not None
    ):
        parser.error("--fruit-slot-order/--fruit-preloaded require --fruit-layout-index")
    if args.fruit_slot_order is not None and len(set(args.fruit_slot_order)) != 4:
        parser.error("--fruit-slot-order must contain each fruit exactly once")
    if args.fruit_preloaded is not None and len(set(args.fruit_preloaded)) != len(
        args.fruit_preloaded
    ):
        parser.error("--fruit-preloaded must not contain duplicates")

    if args.config is not None:
        config = load_pick_place_obstacle_config(args.config)
        if args.profile not in config["profiles"]:
            parser.error(f"profile {args.profile!r} is not present in {args.config}")
    profile_scene = config["profiles"][args.profile]["scene"]
    if profile_scene not in ("any", args.scene):
        parser.error(
            f"profile {args.profile!r} is for scene={profile_scene!r}; "
            f"use --scene {profile_scene}"
        )

    model_path = (
        TRANSPORT_PICK_PLACE_OBSTACLE_MODEL_XML
        if args.scene == "fruit"
        else PICK_PLACE_OBSTACLE_MODEL_XML
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    fruit_scene = None
    if args.fruit_layout_index is None:
        reset_scene(model, data)
    else:
        try:
            fruit_scene = reset_fruit_grid_scene(
                model,
                data,
                load_layout_config(),
                load_fruit_grid_config(),
                layout_index=args.fruit_layout_index,
                slot_order=args.fruit_slot_order or FRUIT_TYPES,
                preloaded_objects=args.fruit_preloaded or (),
                settle_seconds=1.5,
            )
        except ValueError as exc:
            parser.error(str(exc))
    obstacles = PickPlaceObstacleManager(model, data, config)
    obstacles.activate(args.profile)

    for _ in range(int(round(0.5 / model.opt.timestep))):
        mujoco.mj_step(model, data)
        obstacles.observe_contacts()

    robot_clearance = obstacles.robot_clearance()
    object_clearance = obstacles.object_clearance()
    print(f"scene={args.scene!r} profile={args.profile!r} active={obstacles.active_slots}")
    if fruit_scene is not None:
        print(
            f"fruit layout={fruit_scene.layout_index} slots={fruit_scene.slot_order} "
            f"preloaded={fruit_scene.preloaded_objects}"
        )
    print(f"positions={{{', '.join(f'{k!r}: {v.round(3).tolist()}' for k, v in obstacles.positions().items())}}}")
    print(f"initial robot clearance={robot_clearance:.4f} m")
    print(f"initial movable-object clearance={object_clearance:.4f} m")
    print(f"initial contacts={tuple(sorted(obstacles.contact_pairs))}")

    if args.output_dir is not None:
        artifact_name = args.profile
        if fruit_scene is not None:
            artifact_name += f"_layout_{fruit_scene.layout_index:02d}"
        montage = save_camera_preview(model, data, args.output_dir, artifact_name)
        print(f"saved policy-camera preview: {montage}")

    if not args.headless:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as viewer:
            print("Close the MuJoCo viewer to exit.")
            while viewer.is_running():
                mujoco.mj_step(model, data)
                obstacles.observe_contacts()
                viewer.sync()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
