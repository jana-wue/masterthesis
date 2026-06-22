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
import pandas as pd
import seaborn as sns

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import DATA_RESULTS


INPUT_PATH = DATA_RESULTS / "benchmark_all_completed_results.csv"
OUTPUT_PNG = OUTPUT_DIR / "classical_runtime_vs_performance_scatter.png"
OUTPUT_PDF = OUTPUT_DIR / "classical_runtime_vs_performance_scatter.pdf"
OUTPUT_CSV = OUTPUT_DIR / "classical_runtime_vs_performance_scatter_table.csv"

METHOD_ORDER = ["meanmode", "medianmode", "mice", "missforest", "dae"]
METHOD_LABELS = {
    "meanmode": "Mean/Mode",
    "medianmode": "Median/Mode",
    "mice": "MICE",
    "missforest": "MissForest",
    "dae": "DAE",
}
METHOD_COLORS = {
    "meanmode": "#4c78a8",
    "medianmode": "#f58518",
    "mice": "#54a24b",
    "missforest": "#e45756",
    "dae": "#72b7b2",
}
LABEL_OFFSETS = {
    "meanmode": (10, -4),
    "medianmode": (10, 8),
    "mice": (10, -6),
    "missforest": (10, 2),
    "dae": (10, 8),
}


def _load_classical_results() -> pd.DataFrame:
    """Load classical results."""
    df = pd.read_csv(INPUT_PATH)
    work = df[(df["method_family"] == "classical") & (df["status"] == "success")].copy()
    if work.empty:
        raise ValueError("No successful classical benchmark rows found.")
    if work["runtime_seconds"].isna().any():
        raise ValueError("Classical benchmark rows contain missing runtime values.")
    return work


def _build_summary(classical_df: pd.DataFrame) -> pd.DataFrame:
    """Build summary."""
    summary = (
        classical_df.groupby("method_key", as_index=False)
        .agg(
            mean_runtime_seconds=("runtime_seconds", "mean"),
            mean_nrmse=("mean_nrmse", "mean"),
            runtime_std_seconds=("runtime_seconds", "std"),
            nrmse_std=("mean_nrmse", "std"),
            n_runs=("runtime_seconds", "size"),
        )
        .reindex(columns=[
            "method_key",
            "mean_runtime_seconds",
            "mean_nrmse",
            "runtime_std_seconds",
            "nrmse_std",
            "n_runs",
        ])
    )
    summary["method_label"] = summary["method_key"].map(METHOD_LABELS)
    summary["method_order"] = summary["method_key"].map({key: idx for idx, key in enumerate(METHOD_ORDER)})
    summary = summary.sort_values("method_order").drop(columns="method_order").reset_index(drop=True)
    return summary


def _plot_scatter(summary: pd.DataFrame) -> None:
    """Plot scatter."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    for _, row in summary.iterrows():
        ax.scatter(
            row["mean_runtime_seconds"],
            row["mean_nrmse"],
            s=180,
            color=METHOD_COLORS[row["method_key"]],
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )

        dx, dy = LABEL_OFFSETS[row["method_key"]]
        ax.annotate(
            row["method_label"],
            (row["mean_runtime_seconds"], row["mean_nrmse"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="semibold",
            color="#27313a",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Mean Runtime (seconds, log scale)", fontsize=11)
    ax.set_ylabel("Mean NRMSE", fontsize=11)
    ax.set_title("Classical Methods: Runtime vs. Performance", fontsize=18, pad=14)

    ax.grid(True, which="major", color="#d9dde3", linewidth=0.8)
    ax.grid(True, which="minor", axis="x", color="#eceff3", linewidth=0.5)
    ax.set_axisbelow(True)

    sns.despine(ax=ax)
    plt.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the script entry point."""
    classical_df = _load_classical_results()
    summary = _build_summary(classical_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_CSV, index=False)
    _plot_scatter(summary)

    print("=" * 80)
    print("CLASSICAL RUNTIME VS PERFORMANCE SCATTER WRITTEN")
    print("=" * 80)
    print(f"Input CSV : {INPUT_PATH.resolve()}")
    print(f"Output PNG: {OUTPUT_PNG.resolve()}")
    print(f"Output PDF: {OUTPUT_PDF.resolve()}")
    print(f"Output CSV: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
