from __future__ import annotations

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "results" / "figures"
MPL_CONFIG_DIR = PROJECT_ROOT / "data" / "results" / ".mpl-cache"

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


OUTPUT_PNG = OUTPUT_DIR / "missingness_mechanisms_schematic.png"
OUTPUT_PDF = OUTPUT_DIR / "missingness_mechanisms_schematic.pdf"

COLORS = {
    "observed": "#e8eaed",
    "observed_text": "#30343b",
    "masked": "#ef8354",
    "masked_text": "#fffaf5",
    "target_fill": "#dceeff",
    "target_edge": "#3d7ea6",
    "driver_fill": "#d9f3d4",
    "driver_edge": "#4f9d69",
    "excluded_fill": "#70757d",
    "excluded_text": "#f4f5f7",
    "header_fill": "#f6f7f9",
    "header_text": "#23262b",
    "panel_title": "#1f2933",
    "panel_note": "#454b54",
    "arrow": "#2f6b8a",
    "self_arrow": "#b8553c",
    "panel_border": "#d3d8de",
    "background": "#fcfcfd",
}


def _draw_cell(ax, x, y, w, h, text, *, facecolor, edgecolor="#ffffff", textcolor="#1f2933",
               lw=1.0, fontsize=9, fontweight="normal", hatch=None):
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=lw,
            hatch=hatch,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=textcolor,
        fontweight=fontweight,
    )


def _draw_table(
    ax,
    x0,
    y0,
    columns,
    rows,
    *,
    cell_w=1.15,
    cell_h=0.42,
    header_fontsize=9,
    body_fontsize=9,
    target_col=None,
    highlight_cols=None,
    outline_only_cols=None,
    driver_col=None,
    excluded_cols=None,
    masked_cells=None,
    marker_cells=None,
    marker_text="!",
    marker_color=None,
):
    excluded_cols = set(excluded_cols or [])
    highlight_cols = set(highlight_cols or [])
    outline_only_cols = set(outline_only_cols or [])
    if target_col is not None:
        highlight_cols.add(target_col)
    masked_cells = set(masked_cells or [])
    marker_cells = set(marker_cells or [])
    n_rows = len(rows)

    for col_idx, col_name in enumerate(columns):
        x = x0 + col_idx * cell_w
        if col_name in excluded_cols:
            header_fill = COLORS["excluded_fill"]
            header_text = COLORS["excluded_text"]
            edge = COLORS["excluded_fill"]
        elif col_name in highlight_cols:
            header_fill = COLORS["header_fill"]
            header_text = COLORS["header_text"]
            edge = COLORS["target_edge"]
        elif col_name in outline_only_cols:
            header_fill = COLORS["header_fill"]
            header_text = COLORS["header_text"]
            edge = COLORS["target_edge"]
        elif col_name == driver_col:
            header_fill = COLORS["driver_fill"]
            header_text = COLORS["header_text"]
            edge = COLORS["driver_edge"]
        else:
            header_fill = COLORS["header_fill"]
            header_text = COLORS["header_text"]
            edge = "#d7dbe0"

        _draw_cell(
            ax,
            x,
            y0 + n_rows * cell_h,
            cell_w,
            cell_h,
            col_name,
            facecolor=header_fill,
            edgecolor=edge,
            textcolor=header_text,
            lw=1.5 if col_name in {target_col, driver_col} or col_name in excluded_cols else 1.0,
            fontsize=header_fontsize,
            fontweight="semibold",
        )

    for row_idx, row in enumerate(rows):
        for col_idx, col_name in enumerate(columns):
            x = x0 + col_idx * cell_w
            y = y0 + (n_rows - 1 - row_idx) * cell_h
            text = str(row[col_idx])

            if col_name in excluded_cols:
                fill = COLORS["excluded_fill"]
                textcolor = COLORS["excluded_text"]
                edge = COLORS["excluded_fill"]
                hatch = "////"
            elif (row_idx, col_idx) in masked_cells:
                fill = COLORS["masked"]
                textcolor = COLORS["masked_text"]
                edge = "#ffffff"
                hatch = None
                text = "NA"
            elif col_name in highlight_cols:
                fill = COLORS["observed"]
                textcolor = COLORS["observed_text"]
                edge = "#ffffff"
                hatch = None
            elif col_name in outline_only_cols:
                fill = COLORS["observed"]
                textcolor = COLORS["observed_text"]
                edge = "#ffffff"
                hatch = None
            elif col_name == driver_col:
                fill = COLORS["driver_fill"]
                textcolor = COLORS["observed_text"]
                edge = "#ffffff"
                hatch = None
            else:
                fill = COLORS["observed"]
                textcolor = COLORS["observed_text"]
                edge = "#ffffff"
                hatch = None

            _draw_cell(
                ax,
                x,
                y,
                cell_w,
                cell_h,
                text,
                facecolor=fill,
                edgecolor=edge,
                textcolor=textcolor,
                lw=1.0,
                fontsize=body_fontsize,
                fontweight="semibold" if (row_idx, col_idx) in masked_cells else "normal",
                hatch=hatch,
            )

            if (row_idx, col_idx) in marker_cells:
                ax.text(
                    x + cell_w * 0.80,
                    y + cell_h * 0.50,
                    marker_text,
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=marker_color or COLORS["self_arrow"],
                    fontweight="bold",
                )

    for col_name in highlight_cols:
        if col_name not in columns:
            continue
        col_idx = columns.index(col_name)
        ax.add_patch(
            Rectangle(
                (x0 + col_idx * cell_w, y0),
                cell_w,
                (n_rows + 1) * cell_h,
                fill=False,
                edgecolor=COLORS["target_edge"],
                linewidth=2.0,
            )
        )

    for col_name in outline_only_cols:
        if col_name not in columns:
            continue
        col_idx = columns.index(col_name)
        ax.add_patch(
            Rectangle(
                (x0 + col_idx * cell_w, y0),
                cell_w,
                (n_rows + 1) * cell_h,
                fill=False,
                edgecolor=COLORS["target_edge"],
                linewidth=2.0,
            )
        )

    if driver_col in columns:
        col_idx = columns.index(driver_col)
        ax.add_patch(
            Rectangle(
                (x0 + col_idx * cell_w, y0),
                cell_w,
                (n_rows + 1) * cell_h,
                fill=False,
                edgecolor=COLORS["driver_edge"],
                linewidth=2.0,
            )
        )

    return {
        "x0": x0,
        "y0": y0,
        "cell_w": cell_w,
        "cell_h": cell_h,
        "n_rows": n_rows,
        "n_cols": len(columns),
        "columns": columns,
    }


def _col_center(table_meta, col_name):
    idx = table_meta["columns"].index(col_name)
    return table_meta["x0"] + (idx + 0.5) * table_meta["cell_w"]


def _table_mid_y(table_meta):
    return table_meta["y0"] + table_meta["n_rows"] * table_meta["cell_h"] / 2


def _draw_panel_border(ax, bounds):
    x, y, w, h = bounds
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            fill=False,
            edgecolor=COLORS["panel_border"],
            linewidth=1.2,
        )
    )


def _panel_title(ax, bounds, title):
    x, y, w, h = bounds
    ax.text(
        x + 0.18,
        y + h - 0.16,
        title,
        ha="left",
        va="top",
        fontsize=13,
        color=COLORS["panel_title"],
        fontweight="bold",
    )


def _panel_note(ax, bounds, text):
    x, y, w, h = bounds
    ax.text(
        x + 0.18,
        y + 0.12,
        text,
        ha="left",
        va="bottom",
        fontsize=9.4,
        color=COLORS["panel_note"],
        wrap=True,
    )


def _draw_arrow(ax, start, end, color, rad=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=2.2,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def _plot():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    y_shift = 0.55

    fig, ax = plt.subplots(figsize=(15.5, 10.2))
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 10.5)
    ax.axis("off")

    fig.suptitle("Missingness Mechanisms in the Benchmark", fontsize=24, y=0.935)

    panels = {
        "mar": (0.7, 5.3 + y_shift, 6.6, 3.85),
        "mnar": (8.2, 5.3 + y_shift, 6.6, 3.85),
        "mcar_target": (0.7, 0.95 + y_shift, 6.6, 3.85),
        "mcar_global": (8.2, 0.95 + y_shift, 6.6, 3.85),
    }

    for bounds in panels.values():
        _draw_panel_border(ax, bounds)

    columns = ["tenure", "Monthly\nCharges", "Total\nCharges"]
    rows = [
        [12, 55.20, 662.40],
        [24, 79.90, 1917.60],
        [37, 70.35, 2602.95],
        [48, 92.10, 4420.80],
        [61, 74.85, 4565.85],
    ]

    _panel_title(ax, panels["mar"], "MAR_TARGET")
    mar_table = _draw_table(
        ax,
        1.62,
        5.98 + y_shift,
        columns,
        rows,
        target_col="Total\nCharges",
        driver_col="tenure",
        masked_cells={(1, 2), (3, 2)},
        cell_w=1.46,
        cell_h=0.42,
        header_fontsize=8.8,
        body_fontsize=8.8,
    )
    _draw_arrow(
        ax,
        (_col_center(mar_table, "tenure"), mar_table["y0"] - 0.11),
        (_col_center(mar_table, "Total\nCharges"), mar_table["y0"] - 0.11),
        COLORS["arrow"],
    )
    _panel_note(
        ax,
        panels["mar"],
        "Missingness im zu imputierenden Merkmal\nhängt von einer beobachteten Variable ab.",
    )

    _panel_title(ax, panels["mnar"], "MNAR_TARGET")
    mnar_table = _draw_table(
        ax,
        9.12,
        5.98 + y_shift,
        columns,
        rows,
        target_col="Total\nCharges",
        masked_cells={(0, 2), (2, 2), (4, 2)},
        marker_cells={(0, 2), (2, 2), (4, 2)},
        cell_w=1.46,
        cell_h=0.42,
        header_fontsize=8.8,
        body_fontsize=8.8,
    )
    _panel_note(
        ax,
        panels["mnar"],
        "Missingness hängt vom Wert des\nzu imputierenden Merkmals selbst ab.",
    )

    _panel_title(ax, panels["mcar_target"], "MCAR_TARGET")
    _draw_table(
        ax,
        1.62,
        1.58 + y_shift,
        columns,
        rows,
        target_col="Total\nCharges",
        masked_cells={(0, 2), (3, 2)},
        cell_w=1.46,
        cell_h=0.42,
        header_fontsize=8.8,
        body_fontsize=8.8,
    )
    _panel_note(
        ax,
        panels["mcar_target"],
        "Zufällige Missingness nur im\nzu imputierenden Merkmal.",
    )

    _panel_title(ax, panels["mcar_global"], "MCAR_GLOBAL")
    global_columns = ["ID", "tenure", "Monthly\nCharges", "Total\nCharges", "Contract"]
    global_rows = [
        ["A-101", 12, 55.20, 662.40, "Month"],
        ["A-204", 24, 79.90, 1917.60, "2-Year"],
        ["A-317", 37, 70.35, 2602.95, "Month"],
        ["A-488", 48, 92.10, 4420.80, "1-Year"],
        ["A-562", 61, 74.85, 4565.85, "2-Year"],
    ]
    _draw_table(
        ax,
        8.68,
        1.58 + y_shift,
        global_columns,
        global_rows,
        outline_only_cols={"tenure", "Monthly\nCharges", "Total\nCharges", "Contract"},
        excluded_cols={"ID"},
        masked_cells={(0, 1), (1, 3), (2, 4), (3, 2), (4, 3)},
        cell_w=1.16,
        cell_h=0.42,
        header_fontsize=8.0,
        body_fontsize=8.6,
    )
    _panel_note(
        ax,
        panels["mcar_global"],
        "Zufällige Missingness über mehrere\nimputierbare Merkmale.",
    )

    legend_y = 0.22
    legend_items = [
        ("Beobachtete Werte", COLORS["observed"], COLORS["observed_text"]),
        ("Künstlich maskierte Werte", COLORS["masked"], COLORS["masked_text"]),
        ("Treiber-Variable", COLORS["driver_fill"], COLORS["observed_text"]),
        ("Ausgeschlossen", COLORS["excluded_fill"], COLORS["excluded_text"]),
    ]
    x = 1.0
    for label, fill, textcolor in legend_items:
        _draw_cell(ax, x, legend_y, 0.34, 0.18, "", facecolor=fill, textcolor=textcolor, edgecolor="#ffffff")
        ax.text(x + 0.46, legend_y + 0.09, label, va="center", ha="left", fontsize=9.2, color=COLORS["panel_note"])
        x += 2.72

    outline_x = x - 0.08
    ax.add_patch(
        Rectangle(
            (outline_x, legend_y),
            0.34,
            0.18,
            facecolor=COLORS["observed"],
            edgecolor=COLORS["target_edge"],
            linewidth=1.8,
        )
    )
    ax.text(
        outline_x + 0.46,
        legend_y + 0.09,
        "Imputierbare Merkmale (Umrandung)",
        va="center",
        ha="left",
        fontsize=9.2,
        color=COLORS["panel_note"],
    )

    marker_x = outline_x + 3.85
    _draw_cell(
        ax,
        marker_x,
        legend_y,
        0.34,
        0.18,
        "!",
        facecolor=COLORS["masked"],
        edgecolor="#ffffff",
        textcolor=COLORS["self_arrow"],
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        marker_x + 0.46,
        legend_y + 0.09,
        "Missingness abhängig vom eigenen Wert",
        va="center",
        ha="left",
        fontsize=9.2,
        color=COLORS["panel_note"],
    )

    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main():
    _plot()
    print("=" * 80)
    print("MISSINGNESS MECHANISMS SCHEMATIC WRITTEN")
    print("=" * 80)
    print(f"Output PNG: {OUTPUT_PNG.resolve()}")
    print(f"Output PDF: {OUTPUT_PDF.resolve()}")


if __name__ == "__main__":
    main()
