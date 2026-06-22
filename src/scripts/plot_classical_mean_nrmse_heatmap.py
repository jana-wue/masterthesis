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
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
    }
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import DATA_RESULTS


INPUT_PATH = DATA_RESULTS / "benchmark_all_completed_results.csv"
OUTPUT_PNG = OUTPUT_DIR / "classical_mean_nrmse_heatmap.png"
OUTPUT_PDF = OUTPUT_DIR / "classical_mean_nrmse_heatmap.pdf"
OUTPUT_CSV = OUTPUT_DIR / "classical_mean_nrmse_heatmap_table.csv"

METHOD_ORDER = ["meanmode", "medianmode", "mice", "missforest", "dae"]
METHOD_LABELS = {
    "meanmode": "Mean/Mode",
    "medianmode": "Median/Mode",
    "mice": "MICE",
    "missforest": "MissForest",
    "dae": "DAE",
}
DATASET_ORDER = ["German Credit Card", "German Statlog", "Telco"]
SCENARIO_ORDER = ["MAR_TARGET", "MCAR_TARGET", "MNAR_TARGET", "MCAR_GLOBAL"]
RATE_ORDER = [10, 15]
SCENARIO_LABELS = {
    "MAR_TARGET": "MAR-T",
    "MCAR_TARGET": "MCAR-T",
    "MNAR_TARGET": "MNAR-T",
    "MCAR_GLOBAL": "MCAR-G",
}
DATASET_SHORT = {
    "German Credit Card": "Credit",
    "German Statlog": "Statlog",
    "Telco": "Telco",
}


def _load_classical_results() -> pd.DataFrame:
    """Load classical results."""
    df = pd.read_csv(INPUT_PATH)
    work = df[(df["method_family"] == "classical") & (df["status"] == "success")].copy()
    if work.empty:
        raise ValueError("No successful classical benchmark rows found.")
    return work


def _build_task_summary(classical_df: pd.DataFrame) -> pd.DataFrame:
    """Build task summary."""
    group_cols = ["dataset_name", "scenario_family", "rate_pct", "method_key"]
    summary = (
        classical_df.groupby(group_cols, as_index=False)["mean_nrmse"]
        .mean()
        .rename(columns={"mean_nrmse": "mean_nrmse_avg"})
    )
    summary["method_label"] = summary["method_key"].map(METHOD_LABELS)
    summary["task_label"] = summary.apply(
        lambda row: (
            f"{DATASET_SHORT[row['dataset_name']]}\n"
            f"{SCENARIO_LABELS[row['scenario_family']]}\n"
            f"{int(row['rate_pct'])}%"
        ),
        axis=1,
    )
    return summary


def _task_sort_key(task: tuple[str, str, int]) -> tuple[int, int, int]:
    """Handle task sort key."""
    dataset_name, scenario_family, rate_pct = task
    return (
        DATASET_ORDER.index(dataset_name),
        SCENARIO_ORDER.index(scenario_family),
        RATE_ORDER.index(int(rate_pct)),
    )


def _build_metric_matrix(summary: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, str, int]]]:
    """Build metric matrix."""
    tasks = sorted(
        {
            (row["dataset_name"], row["scenario_family"], int(row["rate_pct"]))
            for _, row in summary.iterrows()
        },
        key=_task_sort_key,
    )

    matrix = (
        summary.pivot_table(
            index="method_key",
            columns=["dataset_name", "scenario_family", "rate_pct"],
            values="mean_nrmse_avg",
            aggfunc="first",
        )
        .reindex(index=METHOD_ORDER)
        .reindex(columns=pd.MultiIndex.from_tuples(tasks))
    )

    matrix.index = [METHOD_LABELS[key] for key in matrix.index]
    return matrix, tasks


def _build_display_labels(tasks: list[tuple[str, str, int]]) -> list[str]:
    """Build display labels."""
    return [
        f"{SCENARIO_LABELS[scenario_family]}\n{rate_pct}%"
        for _, scenario_family, rate_pct in tasks
    ]


def _draw_dataset_group_labels(ax: Axes, tasks: list[tuple[str, str, int]]) -> None:
    """Draw dataset group labels."""
    dataset_positions: dict[str, list[int]] = {}
    for idx, (dataset_name, _, _) in enumerate(tasks):
        dataset_positions.setdefault(dataset_name, []).append(idx)

    for dataset_name in DATASET_ORDER:
        positions = dataset_positions.get(dataset_name, [])
        if not positions:
            continue
        center = (positions[0] + positions[-1] + 1) / 2.0
        ax.text(
            center,
            1.12,
            dataset_name,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="semibold",
            color="#2f2f2f",
            clip_on=False,
            transform=ax.get_xaxis_transform(),
        )

    for split_after in [8, 16]:
        ax.axvline(split_after, color="#7a7a7a", linestyle=(0, (6, 6)), linewidth=1.2, alpha=0.9)


def _annotation_color(value: float, vmin: float, vmax: float) -> str:
    """Handle annotation color."""
    midpoint = vmin + 0.55 * (vmax - vmin)
    return "#fffdf7" if value >= midpoint else "#1f2933"


def _plot_heatmap(metric_matrix: pd.DataFrame, tasks: list[tuple[str, str, int]]) -> None:
    """Plot heatmap."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    fig_width = max(12, len(tasks) * 0.62)
    fig_height = 4.9
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = sns.blend_palette(
        ["#0f4c5c", "#4d908e", "#f1f5d8", "#f2cc8f", "#d1495b"],
        as_cmap=True,
    )
    vmin = float(np.nanmin(metric_matrix.to_numpy()))
    vmax = float(np.nanmax(metric_matrix.to_numpy()))

    sns.heatmap(
        metric_matrix,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=False,
        cbar=True,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Mean NRMSE (lower is better)", "shrink": 0.9, "pad": 0.02},
    )

    for row_idx in range(metric_matrix.shape[0]):
        for col_idx in range(metric_matrix.shape[1]):
            value = metric_matrix.iloc[row_idx, col_idx]
            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=9.5,
                fontweight="semibold",
                color=_annotation_color(float(value), vmin, vmax),
            )

    fig.suptitle("Classical Methods: Mean NRMSE on Each Benchmark Task", fontsize=20, y=0.965)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(_build_display_labels(tasks), rotation=55, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
    _draw_dataset_group_labels(ax, tasks)

    plt.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the script entry point."""
    classical_df = _load_classical_results()
    summary = _build_task_summary(classical_df)
    metric_matrix, tasks = _build_metric_matrix(summary)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    metric_matrix.to_csv(OUTPUT_CSV)
    _plot_heatmap(metric_matrix=metric_matrix, tasks=tasks)

    print("=" * 80)
    print("CLASSICAL MEAN NRMSE HEATMAP WRITTEN")
    print("=" * 80)
    print(f"Input CSV : {INPUT_PATH.resolve()}")
    print(f"Output PNG: {OUTPUT_PNG.resolve()}")
    print(f"Output PDF: {OUTPUT_PDF.resolve()}")
    print(f"Output CSV: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
