"""Plot Baseline and SEAM executed RBY1 actions from trajectory NPZ files.

The trajectory files are produced by ``pi05_infer.py --trajectory-out`` and
contain ``executed_actions`` with shape ``(N, 14)``:

    [left J0..J5, left gripper, right J0..J5, right gripper]

This script uses Pillow instead of Matplotlib so it also works in the lightweight
workspace environment.

Example:
    python scripts/plot_rby1_action_comparison.py \
        --baseline /tmp/rby1_base.npz \
        --seam /tmp/rby1_seam.npz \
        --output docs/assets/rby1_action_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


WIDTH = 1920
HEIGHT = 1500
BACKGROUND = "#F5F7FA"
PANEL = "#FFFFFF"
TEXT = "#172033"
MUTED = "#5E6A7D"
GRID = "#DDE3EC"
BASELINE = "#367BF5"
SEAM = "#F27D42"
CHUNK_LINE = "#C4CCD8"
FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size=size)


def scalar(data: np.lib.npyio.NpzFile, key: str, default):
    if key not in data.files:
        return default
    return np.asarray(data[key]).item()


def load_run(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        action_key = "executed_actions" if "executed_actions" in data.files else "actions"
        if action_key not in data.files:
            raise KeyError(
                f"{path} has keys {data.files}, but neither 'executed_actions' nor 'actions'"
            )
        actions = np.asarray(data[action_key], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] < 14:
            raise ValueError(f"{path}: expected action shape (N, >=14), got {actions.shape}")
        return {
            "actions": actions[:, :14],
            "condition": str(scalar(data, "condition", path.stem)),
            "prompt": str(scalar(data, "prompt", "")),
            "execution_length": int(scalar(data, "execution_length", 0)),
        }


def map_point(
    sample: int,
    value: float,
    plot: tuple[int, int, int, int],
    n: int,
    vmin: float,
    vmax: float,
) -> tuple[int, int]:
    x0, y0, x1, y1 = plot
    x = x0 + sample * (x1 - x0) / max(1, n - 1)
    y = y1 - (value - vmin) * (y1 - y0) / max(1e-9, vmax - vmin)
    return int(x), int(y)


def downsample_points(values: np.ndarray, max_points: int = 1800) -> tuple[np.ndarray, np.ndarray]:
    if len(values) <= max_points:
        return np.arange(len(values)), values
    indices = np.linspace(0, len(values) - 1, max_points).astype(np.int64)
    return indices, values[indices]


def draw_series(
    draw: ImageDraw.ImageDraw,
    plot: tuple[int, int, int, int],
    baseline: np.ndarray,
    seam: np.ndarray,
    *,
    fps: float,
    execution_length: int,
    show_x_labels: bool,
    is_gripper: bool,
) -> None:
    n = len(baseline)
    combined = np.concatenate([baseline, seam])
    if is_gripper:
        vmin = min(-0.05, float(np.nanmin(combined)))
        vmax = max(1.05, float(np.nanmax(combined)))
    else:
        vmin = float(np.nanmin(combined))
        vmax = float(np.nanmax(combined))
        padding = max(0.04, (vmax - vmin) * 0.12)
        vmin -= padding
        vmax += padding

    x0, y0, x1, y1 = plot
    for k in range(3):
        gy = int(y0 + k * (y1 - y0) / 2)
        draw.line((x0, gy, x1, gy), fill=GRID, width=1)
        value = vmax - k * (vmax - vmin) / 2
        draw.text((x0 - 58, gy - 10), f"{value:+.2f}", font=font(13), fill=MUTED)

    duration = (n - 1) / fps
    for k in range(5):
        gx = int(x0 + k * (x1 - x0) / 4)
        draw.line((gx, y0, gx, y1), fill=GRID, width=1)
        if show_x_labels:
            draw.text((gx - 17, y1 + 6), f"{duration * k / 4:.1f}s", font=font(13), fill=MUTED)

    if execution_length > 0:
        for step in range(execution_length, n, execution_length):
            gx = int(x0 + step * (x1 - x0) / max(1, n - 1))
            draw.line((gx, y0, gx, y1), fill=CHUNK_LINE, width=1)

    base_idx, base_values = downsample_points(baseline)
    seam_idx, seam_values = downsample_points(seam)
    base_points = [
        map_point(int(i), float(v), plot, n, vmin, vmax)
        for i, v in zip(base_idx, base_values, strict=True)
    ]
    seam_points = [
        map_point(int(i), float(v), plot, n, vmin, vmax)
        for i, v in zip(seam_idx, seam_values, strict=True)
    ]
    draw.line(base_points, fill=BASELINE, width=3)
    draw.line(seam_points, fill=SEAM, width=3)
    draw.rectangle(plot, outline="#AEB8C7", width=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--seam", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()

    baseline = load_run(args.baseline)
    seam = load_run(args.seam)
    base_actions = baseline["actions"]
    seam_actions = seam["actions"]

    if base_actions.shape != seam_actions.shape:
        raise ValueError(
            f"action shape mismatch: baseline={base_actions.shape}, SEAM={seam_actions.shape}"
        )
    if baseline["prompt"] and seam["prompt"] and baseline["prompt"] != seam["prompt"]:
        raise ValueError(
            f"prompt mismatch: baseline={baseline['prompt']!r}, SEAM={seam['prompt']!r}"
        )

    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((65, 38), "RBY1 Executed Action Comparison", font=font(40, bold=True), fill=TEXT)
    prompt = baseline["prompt"] or seam["prompt"]
    if prompt:
        draw.text((65, 92), f'Prompt: "{prompt}"', font=font(22), fill=MUTED)
    n = len(base_actions)
    summary = f"{n} steps · {n / args.fps:.1f} s · {args.fps:g} Hz"
    summary_width = draw.textbbox((0, 0), summary, font=font(22, bold=True))[2]
    draw.text((WIDTH - 65 - summary_width, 92), summary, font=font(22, bold=True), fill="#26705B")

    draw.line((1420, 56, 1465, 56), fill=BASELINE, width=6)
    draw.text((1478, 42), "Baseline", font=font(20, bold=True), fill=TEXT)
    draw.line((1600, 56, 1645, 56), fill=SEAM, width=6)
    draw.text((1658, 42), "SEAM", font=font(20, bold=True), fill=TEXT)

    panel = (50, 140, WIDTH - 50, HEIGHT - 45)
    draw.rounded_rectangle(panel, radius=24, fill=PANEL, outline="#E3E8F0", width=2)

    left_panel = (115, 210, 930, HEIGHT - 105)
    right_panel = (1020, 210, WIDTH - 75, HEIGHT - 105)
    draw.text((left_panel[0], 165), "Left arm", font=font(30, bold=True), fill=TEXT)
    draw.text((right_panel[0], 165), "Right arm", font=font(30, bold=True), fill=TEXT)

    row_gap = 22
    row_height = (left_panel[3] - left_panel[1] - row_gap * 6) // 7
    execution_length = baseline["execution_length"] or seam["execution_length"]

    for row in range(7):
        y0 = left_panel[1] + row * (row_height + row_gap)
        y1 = y0 + row_height
        label = f"J{row}" if row < 6 else "Gripper"

        left_dim = row
        right_dim = row + 7
        show_x = row == 6
        draw_series(
            draw,
            (left_panel[0], y0, left_panel[2], y1),
            base_actions[:, left_dim],
            seam_actions[:, left_dim],
            fps=args.fps,
            execution_length=execution_length,
            show_x_labels=show_x,
            is_gripper=row == 6,
        )
        draw_series(
            draw,
            (right_panel[0], y0, right_panel[2], y1),
            base_actions[:, right_dim],
            seam_actions[:, right_dim],
            fps=args.fps,
            execution_length=execution_length,
            show_x_labels=show_x,
            is_gripper=row == 6,
        )
        label_width = draw.textbbox((0, 0), label, font=font(15, bold=True))[2]
        for plot_x in (left_panel[0], right_panel[0]):
            draw.rounded_rectangle(
                (plot_x + 6, y0 + 5, plot_x + label_width + 20, y0 + 31),
                radius=5,
                fill="#FFFFFF",
            )
            draw.text((plot_x + 12, y0 + 5), label, font=font(15, bold=True), fill=TEXT)

    draw.text(
        (65, HEIGHT - 31),
        "Solid blue: Baseline · Solid orange: SEAM · thin vertical lines: action-chunk boundaries",
        font=font(16),
        fill=MUTED,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, quality=95)
    print(f"baseline shape: {base_actions.shape}")
    print(f"SEAM shape:     {seam_actions.shape}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
