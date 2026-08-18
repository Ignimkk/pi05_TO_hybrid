"""Preflight and optional motion test for the MuJoCo wheel-drive model."""
from __future__ import annotations

import argparse
import math

import mujoco

from rby1_manipulation.control.mobile_base import drive_base_with_wheels, wheel_preflight
from rby1_manipulation.simulation.transport_scene import (
    MODEL_XML_WHEELS,
    base_handles,
    load_layout_config,
    reset_transport_scene,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-test", action="store_true",
                        help="turn -90 deg and drive 0.5 m in simulation")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(MODEL_XML_WHEELS)
    report = wheel_preflight(model)
    print(f"wheel preflight ready={report.ready}")
    print(f"  actuator ids={report.actuator_ids}")
    print(f"  ctrl ranges={report.ctrl_ranges}")
    for issue in report.issues:
        print(f"  FAIL: {issue}")
    if not report.ready:
        return 1
    if not args.motion_test:
        return 0

    data = mujoco.MjData(model)
    reset_transport_scene(model, data, load_layout_config(), settle_seconds=0.2)
    base = base_handles(model, require_actuators=False)
    tests = (
        ("turn", [0.0, 0.0, -math.pi / 2.0], 5.0),
        ("drive", [0.0, -0.5, -math.pi / 2.0], 5.0),
    )
    ok = True
    for label, target, duration in tests:
        result = drive_base_with_wheels(
            model, data, base, target, duration, return_result=True
        )
        print(
            f"  {label}: reached={result.reached} reason={result.reason} "
            f"error={result.error.round(4).tolist()} path={result.path_length:.3f}m"
        )
        ok &= result.reached
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
