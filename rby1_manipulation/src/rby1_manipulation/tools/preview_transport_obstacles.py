"""Preview deterministic static/dynamic transport obstacle profiles."""
from __future__ import annotations

import argparse

import mujoco

from rby1_manipulation.simulation.obstacles import (
    ObstacleCollisionMonitor,
    TransportObstacleManager,
    load_obstacle_config,
    profile_names,
)
from rby1_manipulation.simulation.transport_scene import (
    MODEL_XML,
    MODEL_XML_WHEELS,
    load_layout_config,
    reset_transport_scene,
)


def main() -> int:
    config = load_obstacle_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=profile_names(config), default="dynamic_crossing")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--wheel-model", action="store_true")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(MODEL_XML_WHEELS if args.wheel_model else MODEL_XML)
    data = mujoco.MjData(model)
    reset_transport_scene(model, data, load_layout_config(), settle_seconds=0.2)
    obstacles = TransportObstacleManager(model, data, config)
    obstacles.activate(args.profile)
    monitor = ObstacleCollisionMonitor(model, data, obstacles)
    start_time = float(data.time)

    print(f"profile={args.profile!r} duration={args.duration:.1f}s")
    print("start positions=" + repr({k: v.round(3).tolist() for k, v in obstacles.positions().items()}))

    def run(viewer=None) -> None:
        steps = int(round(args.duration / model.opt.timestep))
        for _ in range(steps):
            obstacles.update(float(data.time) - start_time)
            mujoco.mj_step(model, data)
            monitor.observe()
            if viewer is not None:
                viewer.sync()

    if args.headless:
        run()
    else:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as viewer:
            run(viewer)

    print("end positions=" + repr({k: v.round(3).tolist() for k, v in obstacles.positions().items()}))
    print(f"collision={monitor.collided} contact_steps={monitor.contact_steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
