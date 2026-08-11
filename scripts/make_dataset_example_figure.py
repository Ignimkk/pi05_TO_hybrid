"""Create presentation figures for one RBY1 LeRobot episode.

The script intentionally uses only packages already required by this workspace:
NumPy, PyArrow, Pillow, and ImageIO. It produces:

1. ``episode_XXXXXX_action_schema.png``
2. ``episode_XXXXXX_action_plot.png``
3. ``episode_XXXXXX_dataset_slide.png``
4. ``episode_XXXXXX_dataset_slide_simple.png``

Example:
    python scripts/make_dataset_example_figure.py \
        --dataset data/rby1_dataset_v1 \
        --episode 100 \
        --output-dir docs/assets
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


CANVAS_BG = "#F5F7FA"
PANEL_BG = "#FFFFFF"
TEXT = "#172033"
MUTED = "#5E6A7D"
GRID = "#DDE3EC"
LEFT = "#367BF5"
LEFT_GRIP = "#65B8FF"
RIGHT = "#F27D42"
RIGHT_GRIP = "#F6C453"
JOINT_COLORS = ("#367BF5", "#25A18E", "#7B61FF", "#E15554", "#3BAFDA", "#7A8B99")
FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size=size)


def rounded_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], radius: int = 22) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=PANEL_BG, outline="#E3E8F0", width=2)


def load_task(dataset: Path, task_index: int) -> str:
    with (dataset / "meta" / "tasks.jsonl").open() as f:
        tasks = {int(x["task_index"]): x["task"] for x in map(json.loads, f)}
    return tasks[task_index]


def read_video_frame(path: Path, frame_index: int) -> Image.Image:
    reader = imageio.get_reader(path)
    try:
        frame = reader.get_data(frame_index)
    finally:
        reader.close()
    return Image.fromarray(frame).convert("RGB")


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.copy()
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "black")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def draw_action_schema(
    image: Image.Image,
    box: tuple[int, int, int, int],
    action: np.ndarray,
    *,
    title_size: int = 31,
) -> None:
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = box
    rounded_panel(draw, box)
    draw.text((x0 + 28, y0 + 20), "14차원 state/action 구조", font=font(title_size, bold=True), fill=TEXT)
    draw.text(
        (x0 + 28, y0 + 64),
        "Parquet의 프레임 한 행: float32[14]",
        font=font(21),
        fill=MUTED,
    )

    labels = [f"LJ{i}" for i in range(6)] + ["LG"] + [f"RJ{i}" for i in range(6)] + ["RG"]
    colors = [LEFT] * 6 + [LEFT_GRIP] + [RIGHT] * 6 + [RIGHT_GRIP]
    gap = 5
    cells_x0 = x0 + 28
    cells_x1 = x1 - 28
    cell_w = (cells_x1 - cells_x0 - gap * 13) / 14
    cy0, cy1 = y0 + 108, y0 + 174

    for i, (label, color) in enumerate(zip(labels, colors, strict=True)):
        cx0 = int(cells_x0 + i * (cell_w + gap))
        cx1 = int(cx0 + cell_w)
        draw.rounded_rectangle((cx0, cy0, cx1, cy1), radius=8, fill=color)
        bbox = draw.textbbox((0, 0), label, font=font(16, bold=True))
        draw.text(
            ((cx0 + cx1 - (bbox[2] - bbox[0])) / 2, cy0 + 9),
            label,
            font=font(16, bold=True),
            fill="white" if i not in (13,) else TEXT,
        )
        value = f"{action[i]:.2f}"
        vb = draw.textbbox((0, 0), value, font=font(13))
        draw.text(((cx0 + cx1 - (vb[2] - vb[0])) / 2, cy0 + 38), value, font=font(13), fill="white")

    legend_y = y0 + 196
    legends = [
        (LEFT, "LJ0–LJ5: 왼팔 관절 목표(rad)"),
        (LEFT_GRIP, "LG: 왼쪽 gripper"),
        (RIGHT, "RJ0–RJ5: 오른팔 관절 목표(rad)"),
        (RIGHT_GRIP, "RG: 오른쪽 gripper"),
    ]
    col_w = (x1 - x0 - 56) // 2
    for i, (color, label) in enumerate(legends):
        lx = x0 + 28 + (i % 2) * col_w
        ly = legend_y + (i // 2) * 34
        draw.rounded_rectangle((lx, ly + 5, lx + 19, ly + 24), radius=4, fill=color)
        draw.text((lx + 29, ly), label, font=font(17), fill=TEXT)

    draw.rounded_rectangle((x0 + 28, y1 - 67, x1 - 28, y1 - 22), radius=10, fill="#EEF3FA")
    draw.text(
        (x0 + 45, y1 - 59),
        "학습: (14,) → 50-step (50×14) → joint delta → zero-pad (50×32)",
        font=font(18, bold=True),
        fill=TEXT,
    )


def map_point(
    sample_idx: int,
    value: float,
    plot: tuple[int, int, int, int],
    n: int,
    vmin: float,
    vmax: float,
) -> tuple[int, int]:
    x0, y0, x1, y1 = plot
    x = x0 + sample_idx * (x1 - x0) / max(1, n - 1)
    y = y1 - (value - vmin) * (y1 - y0) / max(1e-8, vmax - vmin)
    return int(x), int(y)


def draw_one_plot(
    draw: ImageDraw.ImageDraw,
    plot: tuple[int, int, int, int],
    values: np.ndarray,
    *,
    title: str,
    fps: float,
    gripper_column: int,
) -> None:
    x0, y0, x1, y1 = plot
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    pad = max(0.1, (vmax - vmin) * 0.08)
    vmin -= pad
    vmax += pad

    draw.text((x0, y0 - 39), title, font=font(22, bold=True), fill=TEXT)
    for k in range(5):
        gy = int(y0 + k * (y1 - y0) / 4)
        draw.line((x0, gy, x1, gy), fill=GRID, width=1)
        val = vmax - k * (vmax - vmin) / 4
        draw.text((x0 - 64, gy - 11), f"{val:+.1f}", font=font(14), fill=MUTED)

    duration = (len(values) - 1) / fps
    for k in range(5):
        gx = int(x0 + k * (x1 - x0) / 4)
        draw.line((gx, y0, gx, y1), fill=GRID, width=1)
        draw.text((gx - 15, y1 + 7), f"{duration * k / 4:.0f}s", font=font(14), fill=MUTED)

    for col in range(values.shape[1]):
        color = RIGHT_GRIP if col == gripper_column else JOINT_COLORS[col if col < 6 else col - 1]
        points = [
            map_point(i, float(values[i, col]), plot, len(values), vmin, vmax)
            for i in range(len(values))
        ]
        draw.line(points, fill=color, width=3 if col == gripper_column else 2)

    draw.rectangle(plot, outline="#AEB8C7", width=2)


def draw_action_plot(
    image: Image.Image,
    box: tuple[int, int, int, int],
    actions: np.ndarray,
    *,
    fps: float,
    title_size: int = 31,
) -> None:
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = box
    rounded_panel(draw, box)
    draw.text((x0 + 28, y0 + 18), "에피소드 전체 action 시계열", font=font(title_size, bold=True), fill=TEXT)
    draw.text(
        (x0 + 28, y0 + 62),
        "관절: absolute target (rad) · gripper: 0=closed, 1=open",
        font=font(18),
        fill=MUTED,
    )

    plot_left = x0 + 92
    plot_right = x1 - 28
    top_plot = (plot_left, y0 + 130, plot_right, y0 + 270)
    bottom_plot = (plot_left, y0 + 340, plot_right, y1 - 52)
    draw_one_plot(draw, top_plot, actions[:, :7], title="Left arm: J0–J5 + gripper", fps=fps, gripper_column=6)
    draw_one_plot(
        draw,
        bottom_plot,
        actions[:, 7:14],
        title="Right arm: J0–J5 + gripper",
        fps=fps,
        gripper_column=6,
    )

    legend_x = x1 - 435
    for i, color in enumerate(JOINT_COLORS):
        lx = legend_x + (i % 3) * 100
        ly = y0 + 22 + (i // 3) * 28
        draw.line((lx, ly + 10, lx + 27, ly + 10), fill=color, width=4)
        draw.text((lx + 34, ly), f"J{i}", font=font(15), fill=TEXT)
    draw.line((x1 - 120, y0 + 32, x1 - 93, y0 + 32), fill=RIGHT_GRIP, width=5)
    draw.text((x1 - 87, y0 + 22), "Grip", font=font(15), fill=TEXT)


def save_schema(path: Path, action: np.ndarray) -> None:
    image = Image.new("RGB", (1500, 410), CANVAS_BG)
    draw_action_schema(image, (20, 20, 1480, 390), action)
    image.save(path, quality=95)


def save_plot(path: Path, actions: np.ndarray, fps: float) -> None:
    image = Image.new("RGB", (1500, 850), CANVAS_BG)
    draw_action_plot(image, (20, 20, 1480, 830), actions, fps=fps)
    image.save(path, quality=95)


def save_slide(
    path: Path,
    dataset: Path,
    episode: int,
    actions: np.ndarray,
    prompt: str,
    fps: float,
    snapshot_frame: int,
) -> None:
    image = Image.new("RGB", (1920, 1080), CANVAS_BG)
    draw = ImageDraw.Draw(image)
    draw.text((60, 34), f"RBY1 LeRobot Dataset — Episode {episode:06d}", font=font(39, bold=True), fill=TEXT)
    draw.text((60, 87), f'Instruction: "{prompt}"', font=font(24), fill=MUTED)
    duration = len(actions) / fps
    summary = f"{len(actions)} frames · {duration:.1f} s · 15 FPS · successful handoff"
    sw = draw.textbbox((0, 0), summary, font=font(21, bold=True))[2]
    draw.text((1860 - sw, 88), summary, font=font(21, bold=True), fill="#26705B")

    camera_defs = [
        ("cam_high", "External camera"),
        ("cam_left_wrist", "Left wrist camera"),
        ("cam_right_wrist", "Right wrist camera"),
    ]
    camera_y0, camera_y1 = 145, 485
    cam_gap = 24
    cam_w = (1800 - cam_gap * 2) // 3
    for i, (camera, label) in enumerate(camera_defs):
        cx0 = 60 + i * (cam_w + cam_gap)
        cx1 = cx0 + cam_w
        rounded_panel(draw, (cx0, camera_y0, cx1, camera_y1))
        video = (
            dataset
            / "videos"
            / "chunk-000"
            / f"observation.images.{camera}"
            / f"episode_{episode:06d}.mp4"
        )
        frame = fit_image(read_video_frame(video, snapshot_frame), (cam_w - 24, 270))
        image.paste(frame, (cx0 + 12, camera_y0 + 52))
        draw.text((cx0 + 20, camera_y0 + 14), label, font=font(22, bold=True), fill=TEXT)
        time_label = f"frame {snapshot_frame} / t={snapshot_frame / fps:.1f}s"
        draw.rounded_rectangle(
            (cx0 + 13, camera_y1 - 44, cx0 + 224, camera_y1 - 15),
            radius=6,
            fill="#101827",
        )
        draw.text((cx0 + 20, camera_y1 - 42), time_label, font=font(17), fill="white")

    draw_action_schema(image, (60, 515, 765, 1045), actions[snapshot_frame], title_size=26)
    draw_action_plot(image, (790, 515, 1860, 1045), actions, fps=fps, title_size=26)
    image.save(path, quality=95)


def draw_format_row(
    draw: ImageDraw.ImageDraw,
    *,
    y: int,
    label: str,
    description: str,
    x0: int = 105,
    x1: int = 1815,
) -> None:
    draw.text((x0, y), label, font=font(29, bold=True), fill=TEXT)
    draw.text((x0 + 155, y + 4), description, font=font(20), fill=MUTED)
    draw.text((x1 - 165, y + 4), "float32[14]", font=font(20, bold=True), fill=TEXT)

    labels = [f"LJ{i}" for i in range(6)] + ["LG"] + [f"RJ{i}" for i in range(6)] + ["RG"]
    colors = [LEFT] * 6 + [LEFT_GRIP] + [RIGHT] * 6 + [RIGHT_GRIP]
    cells_y0, cells_y1 = y + 52, y + 118
    gap = 8
    cell_w = (x1 - x0 - gap * 13) / 14
    for i, (cell_label, color) in enumerate(zip(labels, colors, strict=True)):
        cx0 = int(x0 + i * (cell_w + gap))
        cx1 = int(cx0 + cell_w)
        draw.rounded_rectangle((cx0, cells_y0, cx1, cells_y1), radius=9, fill=color)
        bbox = draw.textbbox((0, 0), cell_label, font=font(18, bold=True))
        draw.text(
            ((cx0 + cx1 - (bbox[2] - bbox[0])) / 2, cells_y0 + 18),
            cell_label,
            font=font(18, bold=True),
            fill=TEXT if i == 13 else "white",
        )


def save_simple_slide(
    path: Path,
    dataset: Path,
    episode: int,
    actions: np.ndarray,
    prompt: str,
    fps: float,
    snapshot_frame: int,
) -> None:
    image = Image.new("RGB", (1920, 1080), CANVAS_BG)
    draw = ImageDraw.Draw(image)
    draw.text((60, 34), f"RBY1 학습 데이터 예시 — Episode {episode:06d}", font=font(39, bold=True), fill=TEXT)
    draw.text((60, 87), f'Instruction: "{prompt}"', font=font(24), fill=MUTED)
    duration = len(actions) / fps
    summary = f"{len(actions)} frames · {duration:.1f} s · 15 FPS · successful handoff"
    sw = draw.textbbox((0, 0), summary, font=font(21, bold=True))[2]
    draw.text((1860 - sw, 88), summary, font=font(21, bold=True), fill="#26705B")

    camera_defs = [
        ("cam_high", "External camera"),
        ("cam_left_wrist", "Left wrist camera"),
        ("cam_right_wrist", "Right wrist camera"),
    ]
    camera_y0, camera_y1 = 145, 485
    cam_gap = 24
    cam_w = (1800 - cam_gap * 2) // 3
    for i, (camera, label) in enumerate(camera_defs):
        cx0 = 60 + i * (cam_w + cam_gap)
        cx1 = cx0 + cam_w
        rounded_panel(draw, (cx0, camera_y0, cx1, camera_y1))
        video = (
            dataset
            / "videos"
            / "chunk-000"
            / f"observation.images.{camera}"
            / f"episode_{episode:06d}.mp4"
        )
        frame = fit_image(read_video_frame(video, snapshot_frame), (cam_w - 24, 270))
        image.paste(frame, (cx0 + 12, camera_y0 + 52))
        draw.text((cx0 + 20, camera_y0 + 14), label, font=font(22, bold=True), fill=TEXT)
        time_label = f"frame {snapshot_frame} / t={snapshot_frame / fps:.1f}s"
        draw.rounded_rectangle(
            (cx0 + 13, camera_y1 - 44, cx0 + 224, camera_y1 - 15),
            radius=6,
            fill="#101827",
        )
        draw.text((cx0 + 20, camera_y1 - 42), time_label, font=font(17), fill="white")

    rounded_panel(draw, (60, 515, 1860, 1045))
    draw.text((90, 538), "한 프레임의 state/action 포맷", font=font(31, bold=True), fill=TEXT)
    draw.text(
        (90, 582),
        "양팔 각각 6개 관절과 1개 gripper를 동일한 순서로 저장",
        font=font(21),
        fill=MUTED,
    )
    draw_format_row(
        draw,
        y=632,
        label="state[t]",
        description="측정된 현재 관절 위치(rad) 및 gripper 상태",
    )
    draw_format_row(
        draw,
        y=802,
        label="action[t]",
        description="actuator에 전달된 절대 관절 목표(rad) 및 gripper 명령",
    )

    legend_y = 957
    legends = [
        (LEFT, "왼팔 관절 6"),
        (LEFT_GRIP, "왼쪽 gripper"),
        (RIGHT, "오른팔 관절 6"),
        (RIGHT_GRIP, "오른쪽 gripper"),
    ]
    lx = 105
    for color, text_label in legends:
        draw.rounded_rectangle((lx, legend_y + 4, lx + 22, legend_y + 26), radius=5, fill=color)
        draw.text((lx + 31, legend_y), text_label, font=font(18), fill=TEXT)
        lx += 245
    draw.text(
        (1210, legend_y),
        "학습 target: 미래 50-step action sequence",
        font=font(20, bold=True),
        fill="#26705B",
    )
    image.save(path, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/rby1_dataset_v1"))
    parser.add_argument("--episode", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    parser.add_argument(
        "--snapshot-frame",
        type=int,
        default=None,
        help="Camera frame used on the slide. Default: middle frame.",
    )
    args = parser.parse_args()

    parquet = args.dataset / "data" / "chunk-000" / f"episode_{args.episode:06d}.parquet"
    table = pq.read_table(parquet, columns=["action", "task_index"])
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    task_index = int(table["task_index"][0].as_py())
    prompt = load_task(args.dataset, task_index)
    with (args.dataset / "meta" / "info.json").open() as f:
        fps = float(json.load(f)["fps"])

    snapshot_frame = args.snapshot_frame if args.snapshot_frame is not None else len(actions) // 2
    if not 0 <= snapshot_frame < len(actions):
        raise ValueError(f"snapshot frame must be in [0, {len(actions) - 1}]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"episode_{args.episode:06d}"
    schema_path = args.output_dir / f"{stem}_action_schema.png"
    plot_path = args.output_dir / f"{stem}_action_plot.png"
    slide_path = args.output_dir / f"{stem}_dataset_slide.png"
    simple_slide_path = args.output_dir / f"{stem}_dataset_slide_simple.png"

    save_schema(schema_path, actions[snapshot_frame])
    save_plot(plot_path, actions, fps)
    save_slide(slide_path, args.dataset, args.episode, actions, prompt, fps, snapshot_frame)
    save_simple_slide(simple_slide_path, args.dataset, args.episode, actions, prompt, fps, snapshot_frame)

    print(f"wrote {schema_path}")
    print(f"wrote {plot_path}")
    print(f"wrote {slide_path}")
    print(f"wrote {simple_slide_path}")


if __name__ == "__main__":
    main()
