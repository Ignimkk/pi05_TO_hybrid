"""Realign parquet timestamps to match video PTS.

Root cause: parquet 'timestamp' can store raw simulation sampling time,
whose step does not exactly equal 1 / nominal_fps when the number of physics
steps per sample is rounded. Video is encoded at nominal CFR starting at t=0.
This makes LeRobot's timestamp validation and timestamp -> PTS lookup fail.

Physical alignment (which is what matters for training) is already
correct: parquet row i, video frame i, and state[i]/action[i] all
represent the same simulated instant. Only the timestamp *labels* are
mis-recorded.

Fix: overwrite parquet timestamp column with i / nominal_fps (i.e. what
the video PTS would report). Videos are left untouched. frame_index is
unchanged.

Idempotent: safe to run multiple times. Backups optional.
By default backups are written next to the dataset root, never inside its
``data/`` tree, so dataset loaders cannot mistake them for training shards.

Usage:
    python patch_dataset_timestamps.py --dataset /root/work/pi05_TO_hybrid/data/rby1_dataset_v1
    python patch_dataset_timestamps.py --dataset ... --no-backup
    python patch_dataset_timestamps.py --dataset ... --dry-run   # preview first
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def patch_one(path: Path, nominal_fps: float, backup_path: Path | None,
              dry_run: bool) -> tuple[int, float, float, float, float]:
    """Rewrite `timestamp` in one parquet. Return (n_rows, old_t0, old_t_last,
    new_t0, new_t_last) for diagnostics."""
    table = pq.read_table(path)
    n = table.num_rows
    old_ts = np.asarray(table.column("timestamp"))
    new_ts = (np.arange(n, dtype=np.float64) / nominal_fps).astype(np.float32)

    if dry_run:
        return n, float(old_ts[0]), float(old_ts[-1]), float(new_ts[0]), float(new_ts[-1])

    if backup_path is not None and not backup_path.exists():
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_path)

    cols = {name: table.column(name) for name in table.column_names}
    cols["timestamp"] = pa.array(new_ts, type=pa.float32())
    new_table = pa.table(cols)
    pq.write_table(new_table, path)
    return n, float(old_ts[0]), float(old_ts[-1]), float(new_ts[0]), float(new_ts[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="LeRobot dataset root")
    ap.add_argument("--fps", type=float, default=15.0,
                    help="Nominal fps (must match info.json + video encoder). Default 15.")
    ap.add_argument("--no-backup", action="store_true",
                    help="Skip the external timestamp-backup directory.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print planned changes for first / mid / last episodes and exit.")
    args = ap.parse_args()

    root = Path(args.dataset).resolve()
    data_dir = root / "data"
    parquets = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    if not parquets:
        raise SystemExit(f"No parquet files under {data_dir}/chunk-*")

    print(f"Patching {len(parquets)} parquet files under {data_dir}/chunk-*")
    print(f"  nominal fps      : {args.fps}")
    backup_root = None if args.no_backup else root.parent / f"{root.name}_timestamp_backup"
    print(f"  backup directory : {backup_root if backup_root else 'skipped'}")
    print(f"  dry run          : {args.dry_run}")
    print()

    # Always show preview for a few episodes first.
    preview_indices = [0, len(parquets) // 2, len(parquets) - 1]
    print("Preview (first / mid / last):")
    for idx in preview_indices:
        n, o0, oL, n0, nL = patch_one(parquets[idx], args.fps,
                                      backup_path=None, dry_run=True)
        print(f"  ep {parquets[idx].name}: n={n:4d}  "
              f"old ts=[{o0:+.4f}, {oL:+.4f}]  -> new ts=[{n0:.4f}, {nL:.4f}]")

    if args.dry_run:
        print("\ndry-run only, nothing modified.")
        return

    # Confirmation prompt (skip if -y or piped)
    print()
    resp = input("Proceed with in-place patch? (y/N) ").strip().lower()
    if resp != "y":
        print("aborted.")
        return

    total_rows = 0
    for i, p in enumerate(parquets):
        backup_path = None if backup_root is None else backup_root / p.relative_to(data_dir)
        n, *_ = patch_one(p, args.fps,
                          backup_path=backup_path,
                          dry_run=False)
        total_rows += n
        if (i + 1) % 100 == 0:
            print(f"  patched {i+1}/{len(parquets)}  (cumulative rows: {total_rows})")
    print(f"\ndone. patched {len(parquets)} files, {total_rows} rows total.")

    # Verify one file post-write
    verify_table = pq.read_table(parquets[0])
    v_ts = np.asarray(verify_table.column("timestamp"))
    print(f"\nverify {parquets[0].name}: ts[0]={v_ts[0]:.4f}  "
          f"ts[1]-ts[0]={v_ts[1]-v_ts[0]:.6f}  "
          f"ts[-1]={v_ts[-1]:.4f}  (expected step {1.0/args.fps:.6f})")


if __name__ == "__main__":
    main()
