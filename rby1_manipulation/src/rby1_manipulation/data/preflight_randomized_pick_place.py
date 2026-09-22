"""Reset-only distribution and validity preflight for randomized pick-place."""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from rby1_manipulation.simulation.fruit_grid import (
    layout_count,
    load_fruit_grid_config,
    table_placements,
)
from rby1_manipulation.simulation.randomized_pick_place import (
    load_default_model,
    load_randomization_config,
    prompts_for_target,
    randomization_config_fingerprint,
    ready_arm_joint_qpos,
    sample_valid_scene,
)
from rby1_manipulation.simulation.transport_scene import (
    OBJECT_TYPES,
    TABLE_X_RANGE,
    TABLE_Y_RANGE,
    load_layout_config,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _span_ratio(values: list[float], requested_width: float) -> float:
    if not values or requested_width <= 0.0:
        return 1.0
    low, high = np.percentile(values, (5.0, 95.0))
    return float((high - low) / requested_width)


def _stats(values: list[float]) -> dict:
    if not values:
        return {}
    array = np.asarray(values, dtype=float)
    return {
        "min": float(array.min()),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def _scatter(path: Path, rows: list[dict]) -> None:
    width, height, margin = 900, 600, 55
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((margin, margin, width - margin, height - margin), outline="#333333", width=2)

    def pixel(x: float, y: float) -> tuple[int, int]:
        px = margin + int((x - TABLE_X_RANGE[0]) / (TABLE_X_RANGE[1] - TABLE_X_RANGE[0]) * (width - 2 * margin))
        py = height - margin - int((y - TABLE_Y_RANGE[0]) / (TABLE_Y_RANGE[1] - TABLE_Y_RANGE[0]) * (height - 2 * margin))
        return px, py

    for row in rows:
        if not row.get("valid"):
            continue
        tx, ty = row["requested_target_pose"][:2]
        gx, gy = row["requested_goal_pose"][:2]
        px, py = pixel(tx, ty)
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill="#ef4444")
        px, py = pixel(gx, gy)
        draw.rectangle((px - 3, py - 3, px + 3, py + 3), fill="#2563eb")
    draw.text((margin, 18), "Randomized target (red) and goal (blue) XY", fill="#111111")
    draw.text((margin, height - 35), f"x={TABLE_X_RANGE}, y={TABLE_Y_RANGE}", fill="#444444")
    image.save(path)


def run_preflight(*, config_path: str, samples: int, seed: int, output_dir: Path) -> dict:
    if samples <= 0:
        raise ValueError("samples must be positive")
    config = load_randomization_config(config_path)
    fingerprint = randomization_config_fingerprint(config)
    layout = load_layout_config()
    grid = load_fruit_grid_config()
    model, data = load_default_model()
    rng = np.random.default_rng(seed)
    permutations = tuple(itertools.permutations(OBJECT_TYPES))
    nominal_joints = ready_arm_joint_qpos(model, config)

    rows: list[dict] = []
    failures: Counter[str] = Counter()
    target_offsets = {"x": [], "y": []}
    goal_offsets = {"x": [], "y": []}
    yaw_values: list[float] = []
    joint_noise_values: list[float] = []
    attempts: list[int] = []
    prompt_counts: Counter[str] = Counter()
    ik_maxima: list[float] = []

    for index in range(samples):
        target = OBJECT_TYPES[index % len(OBJECT_TYPES)]
        layout_index = index % layout_count(grid)
        slot_order = permutations[index % len(permutations)]
        prompts = prompts_for_target(config, target)
        prompt = prompts[int(rng.integers(0, len(prompts)))]
        try:
            scene = sample_valid_scene(
                model, data, layout, grid, config,
                target=target,
                layout_index=layout_index,
                slot_order=slot_order,
                rng=rng,
            )
        except RuntimeError as error:
            rows.append({
                "sample_index": index,
                "target_fruit": target,
                "layout_index": layout_index,
                "slot_order": list(slot_order),
                "selected_language_instruction": prompt,
                "valid": False,
                "error": str(error),
            })
            failures["sampling_exhausted"] += 1
            continue

        row = {
            "sample_index": index,
            "target_fruit": target,
            "layout_index": layout_index,
            "slot_order": list(slot_order),
            "selected_language_instruction": prompt,
            "valid": True,
            **scene.to_dict(),
        }
        rows.append(row)
        nominal = table_placements(grid, layout, layout_index=layout_index, slot_order=slot_order)[target]
        target_offsets["x"].append(float(scene.requested_target_pose[0] - nominal[0]))
        target_offsets["y"].append(float(scene.requested_target_pose[1] - nominal[1]))
        goal_offsets["x"].append(float(scene.requested_goal_pose[0] - layout["crate"]["xy"][0]))
        goal_offsets["y"].append(float(scene.requested_goal_pose[1] - layout["crate"]["xy"][1]))
        yaw_values.append(float(scene.target_yaw_rad))
        joint_noise_values.extend(
            value - nominal_joints[name]
            for name, value in scene.initial_arm_joint_qpos.items()
        )
        attempts.append(scene.sampling_attempts)
        prompt_counts[prompt] += 1
        ik_maxima.append(float(scene.validity["max_ik_position_error_m"]))
        failures.update(scene.rejected_reasons)
        print(
            f"[{index + 1:03d}/{samples:03d}] {target:6s} "
            f"target=({scene.requested_target_pose[0]:+.3f},{scene.requested_target_pose[1]:+.3f}) "
            f"goal=({scene.requested_goal_pose[0]:+.3f},{scene.requested_goal_pose[1]:+.3f}) "
            f"yaw={np.rad2deg(scene.target_yaw_rad):+5.1f}deg attempts={scene.sampling_attempts} valid=True"
        )

    valid_count = sum(bool(row.get("valid")) for row in rows)
    total_candidates = sum(attempts) if valid_count else 0
    acceptance_rate = valid_count / total_candidates if total_candidates else 0.0
    span_ratios = {
        "target_x": _span_ratio(target_offsets["x"], 2 * config.target_position.values[0])
        if config.target_position.enabled else 1.0,
        "target_y": _span_ratio(target_offsets["y"], 2 * config.target_position.values[1])
        if config.target_position.enabled else 1.0,
        "goal_x": _span_ratio(goal_offsets["x"], 2 * config.goal_position.values[0])
        if config.goal_position.enabled else 1.0,
        "goal_y": _span_ratio(goal_offsets["y"], 2 * config.goal_position.values[1])
        if config.goal_position.enabled else 1.0,
        "target_yaw": _span_ratio(
            yaw_values, config.target_orientation.values[1] - config.target_orientation.values[0]
        ) if config.target_orientation.enabled else 1.0,
        "robot_joint_noise": _span_ratio(
            joint_noise_values,
            config.robot_initial_configuration.values[1]
            - config.robot_initial_configuration.values[0],
        ) if config.robot_initial_configuration.enabled else 1.0,
    }
    p95_attempts = float(np.percentile(attempts, 95)) if attempts else float("inf")
    checks = {
        "all_resets_valid": valid_count == samples,
        "candidate_acceptance_at_least_20pct": acceptance_rate >= 0.20,
        "sampling_attempts_p95_at_most_20": p95_attempts <= 20.0,
        "ik_within_tolerance": bool(ik_maxima) and max(ik_maxima) <= config.max_ik_position_error_m,
        "range_coverage_at_least_60pct": all(value >= 0.60 for value in span_ratios.values()),
        "all_language_templates_observed": (
            not config.language_instruction.enabled
            or all(
                any(prompt == template.format(target=target) for prompt in prompt_counts)
                for target in OBJECT_TYPES
                for template in config.language_instruction.templates
            )
        ),
    }
    summary = {
        "version": 1,
        "passed": all(checks.values()),
        "config_path": str(Path(config_path).resolve()),
        "config_fingerprint": fingerprint,
        "seed": seed,
        "requested_samples": samples,
        "valid_samples": valid_count,
        "total_candidate_draws": total_candidates,
        "candidate_acceptance_rate": acceptance_rate,
        "sampling_attempts": _stats([float(value) for value in attempts]),
        "rejection_reasons": dict(sorted(failures.items())),
        "target_offset_x_m": _stats(target_offsets["x"]),
        "target_offset_y_m": _stats(target_offsets["y"]),
        "goal_offset_x_m": _stats(goal_offsets["x"]),
        "goal_offset_y_m": _stats(goal_offsets["y"]),
        "target_yaw_rad": _stats(yaw_values),
        "robot_joint_noise_rad": _stats(joint_noise_values),
        "max_ik_position_error_m": _stats(ik_maxima),
        "prompt_counts": dict(sorted(prompt_counts.items())),
        "span_ratios": span_ratios,
        "checks": checks,
        "config": asdict(config),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "randomization_samples.jsonl", rows)
    _write_json(output_dir / "randomization_summary.json", summary)
    _scatter(output_dir / "randomization_xy.png", rows)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = run_preflight(
        config_path=args.config,
        samples=args.samples,
        seed=args.seed,
        output_dir=Path(args.output_dir).resolve(),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
