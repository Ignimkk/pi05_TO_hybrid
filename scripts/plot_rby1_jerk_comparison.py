"""Visualize SEAM paper-style discrete jerk for two RBY1 trajectory files.

For absolute action targets ``a[t]``, the project metric is:

    j[t] = ||a[t+1] - 2*a[t] + a[t-1]||_2

Only the 12 arm-joint dimensions are used; grippers are excluded. This is the
SEAM paper's discrete "jerk" definition. Since ``a`` is a position target, it is
mathematically a second finite difference, not SI jerk in rad/s^3.

Example:
    python scripts/plot_rby1_jerk_comparison.py \
        --baseline /tmp/rby1_base.npz \
        --seam /tmp/rby1_seam.npz \
        --output docs/assets/rby1_discrete_jerk_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ARM_DIMS = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)
WIDTH, HEIGHT = 1920, 1080
BACKGROUND = "#F5F7FA"
PANEL = "#FFFFFF"
TEXT = "#172033"
MUTED = "#5E6A7D"
GRID = "#DDE3EC"
BOUNDARY = "#C5CEDA"
BASELINE = "#367BF5"
SEAM = "#F27D42"
GOOD = "#26705B"
FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size=size)


def scalar(data: np.lib.npyio.NpzFile, key: str, default):
    return np.asarray(data[key]).item() if key in data.files else default


def load_run(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {
            "action": np.asarray(data["executed_actions"], dtype=np.float64)[:, ARM_DIMS],
            "qpos": np.asarray(data["measured_qpos"], dtype=np.float64)[:, ARM_DIMS],
            "prompt": str(scalar(data, "prompt", "")),
            "fps": float(scalar(data, "control_hz", 15.0)),
            "k": int(scalar(data, "execution_length", 8)),
        }


def discrete_jerk(values: np.ndarray) -> np.ndarray:
    second_difference = values[2:] - 2.0 * values[1:-1] + values[:-2]
    return np.linalg.norm(second_difference, axis=-1)


def boundary_centers(num_steps: int, k: int) -> np.ndarray:
    centers = np.arange(1, num_steps - 1)
    return centers[centers % k == 0]


def metrics(values: np.ndarray, k: int) -> dict:
    jerk = discrete_jerk(values)
    centers = np.arange(1, len(values) - 1)
    boundaries = boundary_centers(len(values), k)
    boundary_mask = np.isin(centers, boundaries)
    boundary_jerk = jerk[boundary_mask]
    interior_jerk = jerk[~boundary_mask]
    boundary_delta = np.linalg.norm(values[boundaries] - values[boundaries - 1], axis=-1)
    return {
        "BJ": float(np.mean(boundary_jerk)),
        "IJ": float(np.mean(interior_jerk)),
        "CD": float(np.mean(boundary_delta)),
        "AVb": float(np.var(boundary_jerk)),
    }


def pct_change(base: float, seam: float) -> float:
    return (seam - base) / abs(base) * 100.0


def point(
    center: int,
    value: float,
    plot: tuple[int, int, int, int],
    num_steps: int,
    vmax: float,
) -> tuple[int, int]:
    x0, y0, x1, y1 = plot
    x = x0 + center * (x1 - x0) / max(1, num_steps - 1)
    y = y1 - value * (y1 - y0) / max(1e-12, vmax)
    return int(x), int(y)


def draw_plot(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    base_values: np.ndarray,
    seam_values: np.ndarray,
    *,
    title: str,
    fps: float,
    k: int,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=20, fill=PANEL, outline="#E3E8F0", width=2)
    draw.text((x0 + 28, y0 + 18), title, font=font(27, bold=True), fill=TEXT)
    draw.text(
        (x0 + 28, y0 + 59),
        "j[t] = L2-norm(x[t+1] - 2x[t] + x[t-1])  (12 arm joints)",
        font=font(17),
        fill=MUTED,
    )

    plot = (x0 + 78, y0 + 105, x1 - 28, y1 - 54)
    base_jerk = discrete_jerk(base_values)
    seam_jerk = discrete_jerk(seam_values)
    centers = np.arange(1, len(base_values) - 1)
    boundaries = boundary_centers(len(base_values), k)
    vmax = float(max(np.max(base_jerk), np.max(seam_jerk))) * 1.08

    px0, py0, px1, py1 = plot
    for i in range(5):
        gy = int(py0 + i * (py1 - py0) / 4)
        value = vmax * (1 - i / 4)
        draw.line((px0, gy, px1, gy), fill=GRID, width=1)
        draw.text((px0 - 62, gy - 10), f"{value:.3f}", font=font(13), fill=MUTED)
    duration = (len(base_values) - 1) / fps
    for i in range(5):
        gx = int(px0 + i * (px1 - px0) / 4)
        draw.line((gx, py0, gx, py1), fill=GRID, width=1)
        draw.text((gx - 17, py1 + 8), f"{duration * i / 4:.1f}s", font=font(13), fill=MUTED)

    for center in boundaries:
        gx = point(center, 0.0, plot, len(base_values), vmax)[0]
        draw.line((gx, py0, gx, py1), fill=BOUNDARY, width=1)

    base_points = [
        point(int(center), float(value), plot, len(base_values), vmax)
        for center, value in zip(centers, base_jerk, strict=True)
    ]
    seam_points = [
        point(int(center), float(value), plot, len(base_values), vmax)
        for center, value in zip(centers, seam_jerk, strict=True)
    ]
    draw.line(base_points, fill=BASELINE, width=3)
    draw.line(seam_points, fill=SEAM, width=3)

    for center in boundaries:
        jerk_index = center - 1
        for values, color in ((base_jerk, BASELINE), (seam_jerk, SEAM)):
            px, py = point(center, float(values[jerk_index]), plot, len(base_values), vmax)
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color, outline="white", width=1)
    draw.rectangle(plot, outline="#AEB8C7", width=2)


def draw_metric_table(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    base: dict,
    seam: dict,
    *,
    title: str,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=18, fill=PANEL, outline="#E3E8F0", width=2)
    draw.text((x0 + 22, y0 + 14), title, font=font(22, bold=True), fill=TEXT)
    headers = ("Metric", "Baseline", "SEAM", "Change")
    xs = (x0 + 24, x0 + 175, x0 + 320, x0 + 450)
    for x, header in zip(xs, headers, strict=True):
        draw.text((x, y0 + 54), header, font=font(15, bold=True), fill=MUTED)
    for row, key in enumerate(("BJ", "IJ", "CD", "AVb")):
        y = y0 + 84 + row * 34
        change = pct_change(base[key], seam[key])
        values = (key, f"{base[key]:.6f}", f"{seam[key]:.6f}", f"{change:+.1f}%")
        for col, (x, value) in enumerate(zip(xs, values, strict=True)):
            color = GOOD if col == 3 and change < 0 else TEXT
            draw.text((x, y), value, font=font(15, bold=col in (0, 3)), fill=color)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--seam", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_run(args.baseline)
    seam = load_run(args.seam)
    if baseline["action"].shape != seam["action"].shape:
        raise ValueError(
            f"shape mismatch: baseline={baseline['action'].shape}, SEAM={seam['action'].shape}"
        )
    fps = baseline["fps"]
    k = baseline["k"]

    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((60, 34), "RBY1 Discrete Jerk: Baseline vs SEAM", font=font(38, bold=True), fill=TEXT)
    prompt = baseline["prompt"]
    draw.text((60, 86), f'Prompt: "{prompt}"', font=font(20), fill=MUTED)
    draw.line((1420, 55, 1465, 55), fill=BASELINE, width=6)
    draw.text((1477, 42), "Baseline", font=font(19, bold=True), fill=TEXT)
    draw.line((1600, 55, 1645, 55), fill=SEAM, width=6)
    draw.text((1657, 42), "SEAM", font=font(19, bold=True), fill=TEXT)
    draw.text((1420, 89), "dots/vertical lines = chunk boundaries (K=8)", font=font(15), fill=MUTED)

    draw_plot(
        draw,
        (55, 135, 1270, 545),
        baseline["action"],
        seam["action"],
        title="Executed action target — command smoothness",
        fps=fps,
        k=k,
    )
    draw_plot(
        draw,
        (55, 570, 1270, 980),
        baseline["qpos"],
        seam["qpos"],
        title="Measured MuJoCo qpos — physical response",
        fps=fps,
        k=k,
    )

    action_base = metrics(baseline["action"], k)
    action_seam = metrics(seam["action"], k)
    qpos_base = metrics(baseline["qpos"], k)
    qpos_seam = metrics(seam["qpos"], k)
    draw_metric_table(
        draw,
        (1300, 135, 1865, 360),
        action_base,
        action_seam,
        title="Executed-action metrics",
    )
    draw_metric_table(
        draw,
        (1300, 385, 1865, 610),
        qpos_base,
        qpos_seam,
        title="Measured-qpos metrics",
    )

    draw.rounded_rectangle((1300, 640, 1865, 980), radius=18, fill="#EEF4F1")
    draw.text((1325, 665), "How to read", font=font(24, bold=True), fill=GOOD)
    notes = [
        "• Lower values indicate smoother motion.",
        "• Peaks show abrupt changes in action slope.",
        "• Boundary dots isolate chunk-transition jerk.",
        "• BJ: mean boundary jerk",
        "• IJ: mean interior jerk",
        "• CD: boundary command discontinuity",
        "• AVb: variance of boundary jerk",
    ]
    for i, note in enumerate(notes):
        draw.text((1325, 714 + i * 34), note, font=font(17), fill=TEXT)

    draw.text(
        (60, 1021),
        "Note: this is the project/paper discrete metric (2nd difference), not SI physical jerk (3rd derivative, rad/s³).",
        font=font(17, bold=True),
        fill=MUTED,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, quality=95)
    print(f"wrote {args.output}")
    print("executed action:", action_base, action_seam)
    print("measured qpos:", qpos_base, qpos_seam)


if __name__ == "__main__":
    main()
