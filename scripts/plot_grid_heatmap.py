"""Plot per-position success (or motion-metric) heatmaps from an RBY1 grid evaluation.

Consumes ``results.jsonl`` written by ``rby1_bringup/pi05_ex_infer.py --grid-experiment`` and
projects the trial table onto the 2-D table plane: one small-multiple panel per
(block color x condition x side), each a grid of spawn positions shaded by the chosen metric.

    src/openpi/.venv/bin/python scripts/plot_grid_heatmap.py \
        --results data/rby1_grid_eval/results.jsonl --out docs/assets/grid_success.png

Success rate is a magnitude, so it uses a single-hue sequential ramp (light = low, dark = high).
Every cell is also annotated with its value and trial count, so the reading never depends on
color alone, and cells with no trials are left blank rather than being drawn as 0%.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np

# Sequential blue ramp, light -> dark (validated data-viz palette).
SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
NO_DATA = "#f0efec"

# Metrics that live under record["motion_metrics"][space]; success is top-level.
MOTION_METRICS = ("BJ", "IJ", "CD", "AVb")


def load_records(path, *, condition=None, grid_fingerprint=None):
    """Read results.jsonl, keeping only completed trials of one grid revision."""
    records = []
    with pathlib.Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if record.get("status") in ("setup_error", "interrupted"):
                continue
            if condition is not None and record.get("condition") != condition:
                continue
            if grid_fingerprint is not None and record.get("grid_fingerprint") != grid_fingerprint:
                continue
            records.append(record)
    if not records:
        raise SystemExit(f"no completed trials in {path} for the requested filters")

    fingerprints = {r.get("grid_fingerprint") for r in records}
    if len(fingerprints) > 1:
        raise SystemExit(
            "results.jsonl mixes grid revisions "
            f"({sorted(map(str, fingerprints))}); pass --grid-fingerprint to pick one"
        )
    return records


def metric_value(record, metric, space):
    """Return the scalar this record contributes, or None when it is unavailable."""
    if metric == "success":
        return 1.0 if record.get("success") else 0.0
    value = (record.get("motion_metrics") or {}).get(space, {}).get(metric)
    return None if value is None else float(value)


def spawn_xy(record):
    """Real table coordinates (x forward, y lateral-signed) of this trial's block spawn."""
    x, y = record["requested_xyz"][0], record["requested_xyz"][1]
    return round(float(x), 6), round(float(y), 6)


ALL_COLORS = "all colors"


def build_panels(records, metric, space, *, merge_colors=False):
    """Aggregate records into {(color, condition): {(x, y): (mean, n)}} keyed by real coordinates.

    Left- and right-hand trials share one panel: their spawn positions are disjoint points on the
    same table (left y > 0, right y < 0) and the hand is determined by the side, so a single
    top-down panel is a faithful projection of the physical workspace. With ``merge_colors`` the
    three block colors collapse into one panel too, trading the per-color breakdown for a denser
    per-cell sample.
    """
    buckets = collections.defaultdict(lambda: collections.defaultdict(list))
    for record in records:
        value = metric_value(record, metric, space)
        if value is None:
            continue
        color = ALL_COLORS if merge_colors else record["color"]
        buckets[(color, record["condition"])][spawn_xy(record)].append(value)
    return {
        key: {cell: (float(np.mean(values)), len(values)) for cell, values in cells.items()}
        for key, cells in buckets.items()
    }


def table_axes(records):
    """Distinct table coordinates: forward x (bottom->top) and lateral y (left->right on screen).

    Viewed from above with the robot at the bottom, so x increases upward and +y (the robot's
    left) is drawn on the left. Taken over every record so all panels share one geometry.
    """
    xs = sorted({spawn_xy(r)[0] for r in records})
    ys = sorted({spawn_xy(r)[1] for r in records}, reverse=True)
    return xs, ys


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=pathlib.Path, default=pathlib.Path("data/rby1_grid_eval/results.jsonl"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs/assets/grid_success_heatmap.png"))
    parser.add_argument("--metric", default="success", choices=("success",) + MOTION_METRICS)
    parser.add_argument("--space", default="action", choices=("action", "qpos"),
                        help="motion metrics only: commanded actions or measured physical response")
    parser.add_argument("--condition", default=None, help="restrict to 'baseline' or 'seam'")
    parser.add_argument("--merge-colors", action="store_true",
                        help="pool the three block colors into a single table panel per condition")
    parser.add_argument("--grid-fingerprint", default=None)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    records = load_records(
        args.results, condition=args.condition, grid_fingerprint=args.grid_fingerprint
    )
    panels = build_panels(records, args.metric, args.space, merge_colors=args.merge_colors)
    if not panels:
        raise SystemExit(
            f"no trials carry metric {args.metric!r}"
            + (f" in {args.space!r} space" if args.metric != "success" else "")
        )

    colors = sorted({key[0] for key in panels})
    conditions = sorted({key[1] for key in panels})

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    cmap.set_bad(NO_DATA)
    if args.metric == "success":
        norm = Normalize(vmin=0.0, vmax=1.0)
        label = "Success rate"
    else:
        all_values = [mean for cells in panels.values() for mean, _ in cells.values()]
        norm = Normalize(vmin=0.0, vmax=max(all_values) or 1.0)
        label = f"{args.metric} ({args.space}) — lower is better"

    table_x, table_y = table_axes(records)
    n_rows, n_cols = len(colors), len(conditions)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.05 * len(table_y) * n_cols + 1.8, 0.95 * len(table_x) * n_rows + 1.3),
        squeeze=False,
        layout="constrained",
    )
    fig.patch.set_facecolor(SURFACE)

    for row_index, color in enumerate(colors):
        for col_index, condition in enumerate(conditions):
            ax = axes[row_index][col_index]
            ax.set_facecolor(SURFACE)
            cells = panels.get((color, condition), {})
            # Rows are forward distance x (bottom = nearest the robot), columns are lateral y.
            grid = np.full((len(table_x), len(table_y)), np.nan)
            counts = np.zeros_like(grid)
            for (x, y), (mean, count) in cells.items():
                if x in table_x and y in table_y:
                    grid[table_x.index(x), table_y.index(y)] = mean
                    counts[table_x.index(x), table_y.index(y)] = count

            ax.imshow(
                np.ma.masked_invalid(grid),
                cmap=cmap, norm=norm, origin="lower", aspect="equal",
            )
            # 2px surface gap between cells keeps adjacent fills from bleeding together.
            ax.set_xticks(np.arange(-0.5, len(table_y), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(table_x), 1), minor=True)
            ax.grid(which="minor", color=SURFACE, linewidth=2)
            ax.tick_params(which="minor", length=0)
            # Centreline between the robot's left (+y) and right (-y) halves.
            for split in range(1, len(table_y)):
                if table_y[split - 1] > 0 >= table_y[split]:
                    ax.axvline(split - 0.5, color=TEXT_MUTED, linewidth=1.2, linestyle=(0, (4, 3)))

            for row in range(grid.shape[0]):
                for col in range(grid.shape[1]):
                    value = grid[row, col]
                    if np.isnan(value):
                        ax.text(col, row, "–", ha="center", va="center",
                                fontsize=8, color=TEXT_MUTED)
                        continue
                    text = f"{value:.0%}" if args.metric == "success" else f"{value:.3f}"
                    # Flip the label to white once the fill is dark enough to swallow dark ink.
                    ink = "#ffffff" if norm(value) > 0.55 else TEXT_PRIMARY
                    ax.text(col, row + 0.08, text, ha="center", va="center",
                            fontsize=9, color=ink, fontweight="bold")
                    ax.text(col, row - 0.26, f"n={int(counts[row, col])}",
                            ha="center", va="center", fontsize=6.5, color=ink, alpha=0.75)

            ax.set_xticks(range(len(table_y)))
            ax.set_xticklabels([f"{y:+.3f}" for y in table_y], fontsize=7, color=TEXT_MUTED)
            ax.set_yticks(range(len(table_x)))
            ax.set_yticklabels([f"{x:.3f}" for x in table_x], fontsize=7, color=TEXT_MUTED)
            for spine in ax.spines.values():
                spine.set_color(GRIDLINE)

            if row_index == 0:
                ax.set_title(condition, fontsize=10,
                             color=TEXT_PRIMARY, fontweight="bold", pad=8)
            if col_index == 0:
                heading = color if color == ALL_COLORS else f"{color} block"
                ax.set_ylabel(f"{heading}\nx — forward (m)", fontsize=9, color=TEXT_SECONDARY)
            if row_index == n_rows - 1:
                ax.set_xlabel("y — lateral (m)      [+ = robot's left]",
                              fontsize=8, color=TEXT_MUTED)

    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    colorbar = fig.colorbar(mappable, ax=axes, fraction=0.025, pad=0.015, aspect=30)
    colorbar.set_label(label, fontsize=9, color=TEXT_SECONDARY)
    colorbar.ax.tick_params(labelsize=7, colors=TEXT_MUTED)
    colorbar.outline.set_edgecolor(GRIDLINE)

    total = sum(count for cells in panels.values() for _, count in cells.values())
    fig.suptitle(
        f"RBY1 grid evaluation — {label.split(' —')[0]}   ({total} trials, "
        f"grid {records[0].get('grid_fingerprint')})\n"
        "table seen from above · robot base at the bottom",
        fontsize=12, color=TEXT_PRIMARY, fontweight="bold", x=0.01, ha="left",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor=SURFACE)
    print(f"wrote {args.out}  ({n_rows}x{n_cols} panels, {total} trials)")


if __name__ == "__main__":
    main()
