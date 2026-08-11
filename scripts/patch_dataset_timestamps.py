"""Realign parquet timestamps to match video PTS.

Root cause (see conversation log): parquet 'timestamp' column stores raw
sim time (data.time), which starts at ~1.5s (settle_scene offset) and
steps at 0.068s (14.71Hz true rate, not the nominal 15Hz). Video is
encoded at nominal 15fps CFR starting at t=0. This makes LeRobot's
timestamp -> PTS video lookup return wrong frames.

Physical alignment (which is what matters for training) is already
correct: parquet row i, video frame i, and state[i]/action[i] all
represent the same simulated instant. Only the timestamp *labels* are
mis-recorded.

Fix: overwrite parquet timestamp column with i / nominal_fps (i.e. what
the video PTS would report). Videos are left untouched. frame_index is
unchanged.

Idempotent: safe to run multiple times. Backups optional.

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


def patch_one(path: Path, nominal_fps: float, make_backup: bool,
              dry_run: bool) -> tuple[int, float, float, float, float]:
    """Rewrite `timestamp` in one parquet. Return (n_rows, old_t0, old_t_last,
    new_t0, new_t_last) for diagnostics."""
    table = pq.read_table(path)
    n = table.num_rows
    old_ts = np.asarray(table.column("timestamp"))
    new_ts = (np.arange(n, dtype=np.float64) / nominal_fps).astype(np.float32)

    if dry_run:
        return n, float(old_ts[0]), float(old_ts[-1]), float(new_ts[0]), float(new_ts[-1])

    if make_backup:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(path, bak)

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
                    help="Skip .bak files (saves ~2GB disk).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print planned changes for first / mid / last episodes and exit.")
    args = ap.parse_args()

    root = Path(args.dataset).resolve()
    chunk_dir = root / "data" / "chunk-000"
    parquets = sorted(chunk_dir.glob("episode_*.parquet"))
    if not parquets:
        raise SystemExit(f"No parquet files under {chunk_dir}")

    print(f"Patching {len(parquets)} parquet files under {chunk_dir}")
    print(f"  nominal fps      : {args.fps}")
    print(f"  backup .bak      : {'skipped' if args.no_backup else 'yes'}")
    print(f"  dry run          : {args.dry_run}")
    print()

    # Always show preview for a few episodes first.
    preview_indices = [0, len(parquets) // 2, len(parquets) - 1]
    print("Preview (first / mid / last):")
    for idx in preview_indices:
        n, o0, oL, n0, nL = patch_one(parquets[idx], args.fps,
                                      make_backup=False, dry_run=True)
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
        n, *_ = patch_one(p, args.fps,
                          make_backup=not args.no_backup,
                          dry_run=False)
        total_rows += n
        if (i + 1) % 100 == 0:
            print(f"  patched {i+1}/{len(parquets)}  (cumulative rows: {total_rows})")
    print(f"\ndone. patched {len(parquets)} files, {total_rows} rows total.")

    # Verify one file post-write
    verify_table = pq.read_table(parquets[0])
    v_ts = np.asarray(verify_table.column("timestamp"))
    print(f"\nverify ep_000000: ts[0]={v_ts[0]:.4f}  ts[1]-ts[0]={v_ts[1]-v_ts[0]:.6f}  "
          f"ts[-1]={v_ts[-1]:.4f}  (expected step {1.0/args.fps:.6f})")


if __name__ == "__main__":
    main()
