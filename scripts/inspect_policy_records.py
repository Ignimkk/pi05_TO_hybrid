"""Inspect policy_records/*.npy dumped by openpi's PolicyRecorder.

Each step_N.npy holds one flattened dict of the obs sent to the policy and the
action chunk it returned:

    inputs/state                    (14,)         [L 6 joint, L grip, R 6 joint, R grip]
    inputs/images/cam_high          (3, 224, 224) CHW uint8
    inputs/images/cam_left_wrist    (3, 224, 224)
    inputs/images/cam_right_wrist   (3, 224, 224)
    inputs/prompt                   str
    outputs/actions                 (50, 14)      absolute joint targets (rad)
    outputs/policy_timing/infer_ms  float

Usage:
    python scripts/inspect_policy_records.py data/policy_records            # summary
    python scripts/inspect_policy_records.py data/policy_records --step 12  # one step's chunk
    python scripts/inspect_policy_records.py data/policy_records --images 0 5 10
    python scripts/inspect_policy_records.py data/policy_records --plot /tmp/actions.png
"""
import argparse
import pathlib

import numpy as np

# Must match pi05_infer.py: only the first OPEN_LOOP_HORIZON steps of each
# 50-step chunk are ever executed, so that's the slice worth analyzing.
OPEN_LOOP_HORIZON = 8
CTRL_HZ = 15
ARM_DIMS = list(range(6)) + list(range(7, 13))  # joint dims only, grippers excluded
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def load_records(record_dir):
    record_dir = pathlib.Path(record_dir)
    n = len(list(record_dir.glob("step_*.npy")))
    if n == 0:
        raise SystemExit(f"no step_*.npy under {record_dir}")
    # Load by index, not glob order -- glob sorts step_10 before step_2.
    return [np.load(record_dir / f"step_{i}.npy", allow_pickle=True).item() for i in range(n)]


def summarize(records):
    actions = [r["outputs/actions"] for r in records]
    states = [r["inputs/state"] for r in records]
    timings = np.array([r["outputs/policy_timing/infer_ms"] for r in records])

    print(f"records            : {len(records)}")
    print(f"chunk shape        : {actions[0].shape}  (executed: first {OPEN_LOOP_HORIZON})")
    print(f"prompt             : {records[0]['inputs/prompt']!r}")
    prompts = {r["inputs/prompt"] for r in records}
    if len(prompts) > 1:
        print(f"  !! {len(prompts)} distinct prompts in this run: {prompts}")
    print(f"infer_ms           : first={timings[0]:.0f}  median={np.median(timings[1:]):.0f}  "
          f"max={timings[1:].max():.0f}   (first call includes JIT compile)")

    exe = [a[:OPEN_LOOP_HORIZON] for a in actions]

    within = np.array([np.abs(np.diff(a, axis=0))[:, ARM_DIMS].max() for a in exe])
    print(f"\nchunk 내 스텝당 최대 관절변화 : median={np.median(within):.4f} rad  max={within.max():.4f} rad")
    print(f"  -> {CTRL_HZ}Hz 환산 각속도    : median={np.median(within)*CTRL_HZ:.2f} rad/s  "
          f"max={within.max()*CTRL_HZ:.2f} rad/s")

    # How far the first commanded target sits from the measured state. Large values
    # mean the policy is asking for a step the position actuator will chase hard.
    jump = np.array([np.abs(a[0][ARM_DIMS] - s[ARM_DIMS]).max() for a, s in zip(exe, states)])
    print(f"state -> chunk[0] 점프        : median={np.median(jump):.4f} rad  max={jump.max():.4f} rad")

    # Discontinuity where one chunk's executed tail meets the next chunk's head.
    bound = np.array([np.abs(exe[i][0][ARM_DIMS] - exe[i - 1][-1][ARM_DIMS]).max()
                      for i in range(1, len(exe))])
    print(f"chunk 경계 점프               : median={np.median(bound):.4f} rad  max={bound.max():.4f} rad")

    outliers = [(i + 1, v) for i, v in enumerate(bound) if v > 0.2]
    if outliers:
        print("  경계 점프 > 0.2 rad 인 step:")
        for i, v in sorted(outliers, key=lambda x: -x[1])[:10]:
            print(f"    step_{i}: {v:.3f} rad")

    grip = np.array([[a[6], a[13]] for a in actions])
    print(f"\ngripper 명령 범위 (L, R)      : "
          f"L[{grip[:, 0].min():.2f}, {grip[:, 0].max():.2f}]  "
          f"R[{grip[:, 1].min():.2f}, {grip[:, 1].max():.2f}]   (1=open, 0=closed)")


def show_step(records, step):
    r = records[step]
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print(f"=== step_{step}  prompt={r['inputs/prompt']!r} ===")
    s = r["inputs/state"]
    print(f"state  L_joint={s[:6]} L_grip={s[6]:.2f}")
    print(f"       R_joint={s[7:13]} R_grip={s[13]:.2f}")
    a = r["outputs/actions"]
    print(f"\nactions[:{OPEN_LOOP_HORIZON}] (실행되는 구간) — [L 6 joint | L grip | R 6 joint | R grip]")
    for i, row in enumerate(a[:OPEN_LOOP_HORIZON]):
        print(f"  [{i}] {row}")
    print(f"\n(나머지 {len(a) - OPEN_LOOP_HORIZON} 스텝은 예측만 되고 버려짐)")


def show_images(records, steps, out_dir):
    from PIL import Image

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for step in steps:
        imgs = []
        for cam in CAMERAS:
            img = records[step][f"inputs/images/{cam}"]
            imgs.append(np.asarray(img).transpose(1, 2, 0))  # CHW -> HWC
        path = out_dir / f"step_{step}_inputs.png"
        Image.fromarray(np.hstack(imgs)).save(path)
        print(f"wrote {path}   (cam_high | cam_left_wrist | cam_right_wrist)")


def plot(records, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Stitch together only the executed slice of each chunk -- this is the
    # trajectory the robot actually followed.
    traj = np.concatenate([r["outputs/actions"][:OPEN_LOOP_HORIZON] for r in records], axis=0)
    t = np.arange(len(traj)) / CTRL_HZ

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, (name, dims, gdim) in zip(axes, [("left", range(6), 6), ("right", range(7, 13), 13)]):
        for j, dim in enumerate(dims):
            ax.plot(t, traj[:, dim], label=f"joint_{j}", lw=1)
        ax.plot(t, traj[:, gdim], "k--", label="gripper", lw=1.2)
        # Mark every chunk boundary -- visible steps here are the discontinuities.
        for b in range(OPEN_LOOP_HORIZON, len(traj), OPEN_LOOP_HORIZON):
            ax.axvline(b / CTRL_HZ, color="0.85", lw=0.5, zorder=0)
        ax.set_ylabel(f"{name} arm (rad)")
        ax.legend(ncol=8, fontsize=7)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("commanded absolute joint targets (executed chunk slices)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record_dir", nargs="?", default="data/policy_records")
    ap.add_argument("--step", type=int, help="print one step's full action chunk")
    ap.add_argument("--images", type=int, nargs="+", metavar="STEP",
                    help="dump the 3 policy-input camera frames for these steps as PNG")
    ap.add_argument("--image-dir", default="/tmp/policy_record_images")
    ap.add_argument("--plot", metavar="PNG", help="plot the stitched commanded trajectory")
    args = ap.parse_args()

    records = load_records(args.record_dir)

    if args.step is not None:
        show_step(records, args.step)
    elif args.images:
        show_images(records, args.images, args.image_dir)
    elif args.plot:
        plot(records, args.plot)
    else:
        summarize(records)


if __name__ == "__main__":
    main()
