"""Batch data collection for pi0.5 LoRA fine-tuning.

Data collection matrix (12 configs, 75 successful episodes each => 900 total):

    For each color in {red, green, blue} and for each of these 4 cases:

    | Config       | block spawn side | task instruction ("...hand")   | scenario         |
    |--------------|-------------------|--------------------------------|------------------|
    | LEFT_SINGLE  | LEFT envelope    | left                          | scenario 1 LEFT  |
    | LEFT_HANDOFF | LEFT envelope    | right (needs L->R handoff)    | scenario 2       |
    | RIGHT_HANDOFF| RIGHT envelope   | left  (needs R->L handoff)    | scenario 3       |
    | RIGHT_SINGLE | RIGHT envelope   | right                         | scenario 1 RIGHT |

Each subprocess writes to the SAME LeRobot dataset root, appending episodes.
The dataset ends up as a single mixed-task LeRobot dataset in ALOHA schema
(ready for pi0.5 baseline consumption).

Usage:
    python collect_dataset.py --output-dir /path/to/rby1_dataset
    python collect_dataset.py --output-dir ... --per-config 75 --base-seed 1000
    python collect_dataset.py --output-dir ... --only left_single --only right_single
    python collect_dataset.py --output-dir ... --dry-run     # print planned commands
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

COLORS = ["red", "green", "blue"]


@dataclass
class ConfigSpec:
    label: str                      # e.g. "left_single_red"
    scenario_module: str            # importable module passed to ``python -m``
    extra_args: List[str]           # per-episode CLI args (excluding --seed, --headless, --log-*)
    task_prompt: str                # LeRobot task string (overrides scenario default)


def build_configs() -> List[ConfigSpec]:
    """Return the full 12-config matrix."""
    configs: List[ConfigSpec] = []
    for color in COLORS:
        # A) LEFT single-arm  (block on LEFT side, LEFT places)
        configs.append(ConfigSpec(
            label=f"left_single_{color}",
            scenario_module="rby1_manipulation.tasks.block_pick",
            extra_args=["--arm", "left", "--block", color,
                        "--spawn-side", "left", "--random"],
            task_prompt=f"put the {color} block in the brown box with your left hand",
        ))
        # B) LEFT -> RIGHT handoff  (block on LEFT side, RIGHT places)
        configs.append(ConfigSpec(
            label=f"left_handoff_{color}",
            scenario_module="rby1_manipulation.tasks.handoff_left_to_right",
            extra_args=["--block", color, "--random"],
            task_prompt=f"put the {color} block in the brown box with your right hand",
        ))
        # C) RIGHT -> LEFT handoff  (block on RIGHT side, LEFT places)
        configs.append(ConfigSpec(
            label=f"right_handoff_{color}",
            scenario_module="rby1_manipulation.tasks.handoff_right_to_left",
            extra_args=["--block", color, "--random"],
            task_prompt=f"put the {color} block in the brown box with your left hand",
        ))
        # D) RIGHT single-arm  (block on RIGHT side, RIGHT places)
        configs.append(ConfigSpec(
            label=f"right_single_{color}",
            scenario_module="rby1_manipulation.tasks.block_pick",
            extra_args=["--arm", "right", "--block", color,
                        "--spawn-side", "right", "--random"],
            task_prompt=f"put the {color} block in the brown box with your right hand",
        ))
    return configs


def run_one_episode(cfg: ConfigSpec, seed: int, output_dir: Path,
                    log_fps: int, timeout_secs: int,
                    dry_run: bool, save_failed: bool = False) -> tuple[bool, str]:
    """Run one episode as a subprocess. Return (success, short_msg)."""
    cmd = [
        sys.executable,
        "-m",
        cfg.scenario_module,
        *cfg.extra_args,
        "--seed", str(seed),
        "--headless",
        "--log-dataset", str(output_dir),
        "--log-fps", str(log_fps),
        "--task-prompt", cfg.task_prompt,
    ]
    if save_failed:
        cmd.append("--save-failed")
    if dry_run:
        return True, " ".join(cmd)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout_secs)
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    # The scenarios print ">>> SUCCESS = True" on success, and only write to
    # the dataset if success. We rely on the stdout marker as the source of truth.
    stdout = result.stdout or ""
    success = ">>> SUCCESS = True" in stdout
    if success:
        return True, "ok"
    # Extract the last few lines for a compact error hint.
    tail = "\n".join(stdout.splitlines()[-3:]) if stdout else (result.stderr or "")[-200:]
    return False, tail.replace("\n", " | ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True,
                    help="LeRobot dataset root directory (created if missing)")
    ap.add_argument("--per-config", type=int, default=100,
                    help="Target number of SUCCESSFUL episodes per config")
    ap.add_argument("--max-attempts-mult", type=float, default=1.5,
                    help="Max attempts per config = per-config * this. Failures "
                         "beyond this budget are logged and the config moves on.")
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--log-fps", type=int, default=15)
    ap.add_argument("--timeout", type=int, default=180,
                    help="Per-episode subprocess timeout in seconds")
    ap.add_argument("--only", action="append", default=None,
                    help="Only run configs whose label CONTAINS any of these "
                         "substrings. Can repeat: --only left_single --only red")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the commands that would run and exit")
    ap.add_argument("--save-failed", action="store_true",
                    help="Forward --save-failed to each scenario so failure "
                         "episodes are also written to the LeRobot dataset "
                         "(task prompt prefixed with '[FAIL] '). Use for "
                         "smoke tests / debugging; keep OFF for real collection.")
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = build_configs()
    if args.only:
        configs = [c for c in configs if any(o in c.label for o in args.only)]

    stats_path = output_dir / "collect_dataset_stats.json"
    if stats_path.exists():
        with stats_path.open() as f:
            stats = json.load(f)
    else:
        stats = {}

    max_attempts_per_cfg = int(args.per_config * args.max_attempts_mult)
    total_targets = len(configs) * args.per_config
    print(f"=== collect_dataset ===")
    print(f"  output_dir       : {output_dir}")
    print(f"  configs          : {len(configs)}")
    print(f"  per_config       : {args.per_config} (max attempts {max_attempts_per_cfg})")
    print(f"  target successes : {total_targets}")
    print(f"  base_seed        : {args.base_seed}")
    print()

    grand_start = time.time()

    for cfg_idx, cfg in enumerate(configs):
        cfg_stats = stats.setdefault(cfg.label, {"success": 0, "fail": 0,
                                                 "next_seed_offset": 0})
        already_ok = cfg_stats["success"]
        remaining = max(0, args.per_config - already_ok)
        if remaining == 0:
            print(f"[{cfg_idx+1}/{len(configs)}] {cfg.label:22s}  already has "
                  f"{already_ok} successes, skipping.")
            continue

        print(f"[{cfg_idx+1}/{len(configs)}] {cfg.label:22s}  target "
              f"{args.per_config} (have {already_ok})")

        attempts = 0
        while cfg_stats["success"] < args.per_config and attempts < max_attempts_per_cfg:
            seed = args.base_seed + cfg_idx * 10000 + cfg_stats["next_seed_offset"]
            attempts += 1
            ep_start = time.time()
            success, msg = run_one_episode(cfg, seed, output_dir, args.log_fps,
                                            args.timeout, args.dry_run,
                                            save_failed=args.save_failed)
            ep_secs = time.time() - ep_start

            if args.dry_run:
                # Preview mode: print the command and move on. Don't touch
                # stats, don't advance seed offset, don't persist.
                print(f"    [DRY-RUN seed={seed}] {msg}")
                if attempts >= 1:  # one command per config is enough to inspect
                    break
                continue

            cfg_stats["next_seed_offset"] += 1
            if success:
                cfg_stats["success"] += 1
                status = "OK"
            else:
                cfg_stats["fail"] += 1
                status = f"FAIL: {msg[:80]}"
            print(f"    seed={seed:>7d}  attempt {attempts:>3d}  "
                  f"{cfg_stats['success']:>3d}/{args.per_config}  "
                  f"({ep_secs:4.1f}s)  {status}")

            # Persist stats after every episode for safe resume.
            with stats_path.open("w") as f:
                json.dump(stats, f, indent=2)

        if not args.dry_run and cfg_stats["success"] < args.per_config:
            print(f"    !! config exhausted attempt budget "
                  f"({cfg_stats['success']}/{args.per_config})")

    grand_secs = time.time() - grand_start
    print()
    print(f"=== summary  (elapsed {grand_secs/60:.1f} min) ===")
    total_ok, total_attempts = 0, 0
    for label, s in stats.items():
        n = s["success"] + s["fail"]
        rate = 100.0 * s["success"] / n if n else 0.0
        print(f"  {label:22s}  {s['success']:>3d}/{s['success']+s['fail']:<3d}  "
              f"({rate:5.1f}%)")
        total_ok += s["success"]
        total_attempts += n
    if total_attempts:
        print(f"  TOTAL              {total_ok:>3d}/{total_attempts:<3d}  "
              f"({100.0*total_ok/total_attempts:5.1f}%)")


if __name__ == "__main__":
    main()
