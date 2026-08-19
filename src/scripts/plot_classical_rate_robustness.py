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
OUTPUT_PNG = OUTPUT_DIR / "classical_rate_robustness.png"
OUTPUT_PDF = OUTPUT_DIR / "classical_rate_robustness.pdf"
OUTPUT_CSV = OUTPUT_DIR / "classical_rate_robustness_table.csv"

METHOD_ORDER = ["meanmode", "medianmode", "mice", "missforest", "dae"]
METHOD_LABELS = {
    "meanmode": "Mean/Mode",
    "medianmode": "Median/Mode",
    "mice": "MICE",
    "missforest": "MissForest",
    "dae": "DAE",
}
METHOD_COLORS = {
    "meanmode": "#9bbfe0",
    "medianmode": "#6f9fd8",
    "mice": "#4c78a8",
    "missforest": "#2f5c8f",
    "dae": "#6bb6c9",
}
SCENARIO_ORDER = ["MAR_TARGET", "MCAR_TARGET", "MNAR_TARGET", "MCAR_GLOBAL"]
SCENARIO_LABELS = {
    "MAR_TARGET": "MAR target-only",
    "MCAR_TARGET": "MCAR target-only",
    "MNAR_TARGET": "MNAR target-only",
    "MCAR_GLOBAL": "MCAR global",
}
RATE_ORDER = [10, 15]


def _load_rows() -> pd.DataFrame:
    """Load rows."""
    df = pd.read_csv(INPUT_PATH)
    work = df[
        (df["method_family"] == "classical")
        & (df["status"] == "success")
        & (df["scenario_family"].isin(SCENARIO_ORDER))
        & (df["rate_pct"].isin(RATE_ORDER))
    ].copy()
    if work.empty:
        raise ValueError("No successful classical rows found for robustness plot.")
    return work


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build summary."""
    summary = (
        df.groupby(["scenario_family", "rate_pct", "method_key"], as_index=False)["mean_nrmse"]
        .mean()
        .rename(columns={"mean_nrmse": "mean_nrmse_avg"})
    )
    summary["scenario_family"] = pd.Categorical(summary["scenario_family"], categories=SCENARIO_ORDER, ordered=True)
    summary["method_key"] = pd.Categorical(summary["method_key"], categories=METHOD_ORDER, ordered=True)
    summary = summary.sort_values(["scenario_family", "method_key", "rate_pct"]).reset_index(drop=True)
    summary["scenario_label"] = summary["scenario_family"].map(SCENARIO_LABELS)
    summary["method_label"] = summary["method_key"].map(METHOD_LABELS)
    return summary


def _plot(summary: pd.DataFrame) -> None:
    """Plot this helper."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(
        style="whitegrid",
        rc={
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
        },
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.4), sharey=True)
    axes_flat = axes.flatten()

    for ax, scenario_family in zip(axes_flat, SCENARIO_ORDER):
        subset = summary[summary["scenario_family"] == scenario_family]
        for method_key in METHOD_ORDER:
            method_subset = subset[subset["method_key"] == method_key]
            ax.plot(
                method_subset["rate_pct"].astype(int).astype(str),
                method_subset["mean_nrmse_avg"],
                marker="o",
                markersize=6.8,
                linewidth=2.0,
                color=METHOD_COLORS[method_key],
                label=METHOD_LABELS[method_key],
            )

        ax.set_title(SCENARIO_LABELS[scenario_family], fontsize=11, pad=10)
        ax.set_xlabel("")
        ax.grid(True, axis="y", color="#d9dde3", linewidth=0.8)
        ax.grid(False, axis="x")
        ax.set_axisbelow(True)

    axes[0, 0].set_ylabel("Mean NRMSE", fontsize=11)
    axes[1, 0].set_ylabel("Mean NRMSE", fontsize=11)
    axes[0, 1].set_ylabel("")
    axes[1, 1].set_ylabel("")

    for ax in axes[1, :]:
        ax.set_xlabel("Missingness Rate (%)", fontsize=10.5)

    legend_handles, legend_labels = axes_flat[0].get_legend_handles_labels()
    fig.suptitle("Classical Methods: Robustness Across Missingness Rates", fontsize=18, y=0.98)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        frameon=False,
        ncol=5,
        fontsize=9.2,
        columnspacing=1.0,
        handletextpad=0.45,
    )
    fig.text(0.99, 0.02, "Values averaged across all datasets and seeds", ha="right", va="bottom", fontsize=9, color="#5d6773")

    sns.despine(fig=fig)
    plt.tight_layout(rect=(0, 0.04, 1, 0.9))
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the script entry point."""
    rows = _load_rows()
    summary = _build_summary(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_CSV, index=False)
    _plot(summary)

    print("=" * 80)
    print("CLASSICAL RATE ROBUSTNESS PLOT WRITTEN")
    print("=" * 80)
    print(f"Input CSV : {INPUT_PATH.resolve()}")
    print(f"Output PNG: {OUTPUT_PNG.resolve()}")
    print(f"Output PDF: {OUTPUT_PDF.resolve()}")
    print(f"Output CSV: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
