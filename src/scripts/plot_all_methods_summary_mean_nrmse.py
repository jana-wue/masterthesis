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
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    }
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import DATA_RESULTS


INPUT_PATH = DATA_RESULTS / "benchmark_all_completed_results.csv"
OUTPUT_PNG = OUTPUT_DIR / "all_methods_summary_mean_nrmse.png"
OUTPUT_PDF = OUTPUT_DIR / "all_methods_summary_mean_nrmse.pdf"
OUTPUT_CSV = OUTPUT_DIR / "all_methods_summary_mean_nrmse_table.csv"

SCENARIO_ORDER = ["MAR_TARGET", "MCAR_TARGET", "MNAR_TARGET"]
ROW_ORDER = [
    "meanmode",
    "medianmode",
    "mice",
    "missforest",
    "dae",
    "llm_prompt::llama",
    "llm_finetuned::llama",
    "llm_prompt::mistral",
    "llm_finetuned::mistral",
    "llm_prompt::qwen",
    "llm_finetuned::qwen",
]
ROW_LABELS = {
    "meanmode": "Mean/Mode",
    "medianmode": "Median/Mode",
    "mice": "MICE",
    "missforest": "MissForest",
    "dae": "DAE",
    "llm_prompt::llama": "Llama 3.1 Prompt",
    "llm_finetuned::llama": "Llama 3.1 Finetuned",
    "llm_prompt::mistral": "Mistral Prompt",
    "llm_finetuned::mistral": "Mistral Finetuned",
    "llm_prompt::qwen": "Qwen 2.5 Prompt",
    "llm_finetuned::qwen": "Qwen 2.5 Finetuned",
}
FAMILY_COLORS = {
    "classical": "#4c78a8",
    "llm_prompt": "#f28e2b",
    "llm_finetuned": "#6a994e",
}
METHOD_COLORS = {
    "meanmode": FAMILY_COLORS["classical"],
    "medianmode": FAMILY_COLORS["classical"],
    "mice": FAMILY_COLORS["classical"],
    "missforest": FAMILY_COLORS["classical"],
    "dae": FAMILY_COLORS["classical"],
    "llm_prompt::llama": FAMILY_COLORS["llm_prompt"],
    "llm_prompt::mistral": FAMILY_COLORS["llm_prompt"],
    "llm_prompt::qwen": FAMILY_COLORS["llm_prompt"],
    "llm_finetuned::llama": FAMILY_COLORS["llm_finetuned"],
    "llm_finetuned::mistral": FAMILY_COLORS["llm_finetuned"],
    "llm_finetuned::qwen": FAMILY_COLORS["llm_finetuned"],
}


def _load_shared_comparison_rows() -> pd.DataFrame:
    """Load shared comparison rows."""
    df = pd.read_csv(INPUT_PATH)
    work = df[
        (df["status"] == "success")
        & (df["seed"] == 42)
        & (df["rate_pct"] == 10)
        & (df["scenario_family"].isin(SCENARIO_ORDER))
        & (df["method_family"].isin(["classical", "llm_prompt", "llm_finetuned"]))
    ].copy()
    if work.empty:
        raise ValueError("No successful shared comparison rows found.")
    return work


def _build_summary(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Build summary."""
    work = comparison_df.copy()

    classical_mask = work["method_family"] == "classical"
    work.loc[classical_mask, "row_key"] = work.loc[classical_mask, "method_key"].astype(str)

    llm_mask = work["method_family"].isin(["llm_prompt", "llm_finetuned"])
    work.loc[llm_mask, "row_key"] = (
        work.loc[llm_mask, "method_family"].astype(str)
        + "::"
        + work.loc[llm_mask, "model_key"].astype(str)
    )

    summary = (
        work.groupby("row_key", as_index=False)
        .agg(
            mean_nrmse=("mean_nrmse", "mean"),
            nrmse_std=("mean_nrmse", "std"),
            n_tasks=("mean_nrmse", "size"),
        )
    )
    summary["method_label"] = summary["row_key"].map(ROW_LABELS)
    summary["plot_order"] = summary["row_key"].map({key: idx for idx, key in enumerate(ROW_ORDER)})
    summary["color"] = summary["row_key"].map(METHOD_COLORS)
    summary = summary.sort_values("mean_nrmse", ascending=True).reset_index(drop=True)
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
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        },
    )
    fig, ax = plt.subplots(figsize=(9.2, 6.6))

    y_positions = list(range(len(summary)))
    ax.hlines(
        y=y_positions,
        xmin=0,
        xmax=summary["mean_nrmse"],
        color="#d8dde6",
        linewidth=1.5,
        zorder=1,
    )
    ax.scatter(
        summary["mean_nrmse"],
        y_positions,
        s=170,
        c=summary["color"],
        edgecolor="white",
        linewidth=1.1,
        zorder=3,
    )

    for y, (_, row) in enumerate(summary.iterrows()):
        ax.text(
            row["mean_nrmse"] + 0.018,
            y,
            f"{row['mean_nrmse']:.3f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="semibold",
            color="black",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(summary["method_label"], fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlabel("Mean NRMSE on Shared Benchmark Tasks", fontsize=11)
    ax.set_ylabel("")
    ax.grid(True, axis="x", color="#d9dde3", linewidth=0.8)
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=FAMILY_COLORS["classical"], markeredgecolor="white",
               markeredgewidth=1.0, markersize=9, label="Classical"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=FAMILY_COLORS["llm_prompt"], markeredgecolor="white",
               markeredgewidth=1.0, markersize=9, label="Prompting"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=FAMILY_COLORS["llm_finetuned"], markeredgecolor="white",
               markeredgewidth=1.0, markersize=9, label="Finetuning"),
    ]
    fig.suptitle("All Methods: Summary Performance Overview", fontsize=18, y=0.965)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        frameon=False,
        fontsize=9.5,
        ncol=3,
        bbox_to_anchor=(0.5, 0.94),
        columnspacing=1.2,
        handletextpad=0.4,
    )

    note = "Shared comparison space: Seed 42, 10%, MAR/MCAR/MNAR target-only"
    fig.text(0.99, 0.02, note, ha="right", va="bottom", fontsize=9, color="black")

    sns.despine(ax=ax, left=True)
    plt.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the script entry point."""
    comparison_df = _load_shared_comparison_rows()
    summary = _build_summary(comparison_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_CSV, index=False)
    _plot(summary)

    print("=" * 80)
    print("ALL METHODS SUMMARY MEAN NRMSE PLOT WRITTEN")
    print("=" * 80)
    print(f"Input CSV : {INPUT_PATH.resolve()}")
    print(f"Output PNG: {OUTPUT_PNG.resolve()}")
    print(f"Output PDF: {OUTPUT_PDF.resolve()}")
    print(f"Output CSV: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
