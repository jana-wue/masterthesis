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
OUTPUT_PNG = OUTPUT_DIR / "llm_finetuning_scenario_robustness.png"
OUTPUT_PDF = OUTPUT_DIR / "llm_finetuning_scenario_robustness.pdf"
OUTPUT_CSV = OUTPUT_DIR / "llm_finetuning_scenario_robustness_table.csv"

DATASET_ORDER = ["German Credit Card", "German Statlog", "Telco"]
SCENARIO_ORDER = ["MAR_TARGET", "MCAR_TARGET", "MNAR_TARGET"]
SCENARIO_LABELS = {
    "MAR_TARGET": "MAR",
    "MCAR_TARGET": "MCAR",
    "MNAR_TARGET": "MNAR",
}
MODEL_ORDER = ["llama", "mistral", "qwen"]
MODEL_LABELS = {
    "llama": "Llama 3.1",
    "mistral": "Mistral",
    "qwen": "Qwen 2.5",
}
MODEL_COLORS = {
    "llama": "#8ab446",
    "mistral": "#2ca02c",
    "qwen": "#386641",
}


def _load_rows() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH)
    work = df[
        (df["method_family"] == "llm_finetuned")
        & (df["status"] == "success")
        & (df["rate_pct"] == 10)
        & (df["scenario_family"].isin(SCENARIO_ORDER))
        & (df["dataset_name"].isin(DATASET_ORDER))
    ].copy()
    if work.empty:
        raise ValueError("No successful LLM finetuning rows found for robustness plot.")
    return work


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["dataset_name", "scenario_family", "model_key"], as_index=False)["mean_nrmse"]
        .mean()
        .rename(columns={"mean_nrmse": "mean_nrmse_avg"})
    )
    summary["dataset_name"] = pd.Categorical(summary["dataset_name"], categories=DATASET_ORDER, ordered=True)
    summary["scenario_family"] = pd.Categorical(summary["scenario_family"], categories=SCENARIO_ORDER, ordered=True)
    summary["model_key"] = pd.Categorical(summary["model_key"], categories=MODEL_ORDER, ordered=True)
    summary = summary.sort_values(["dataset_name", "model_key", "scenario_family"]).reset_index(drop=True)
    summary["scenario_label"] = summary["scenario_family"].map(SCENARIO_LABELS)
    summary["model_label"] = summary["model_key"].map(MODEL_LABELS)
    return summary


def _plot(summary: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2), sharey=True)

    for ax, dataset_name in zip(axes, DATASET_ORDER):
        subset = summary[summary["dataset_name"] == dataset_name]
        for model_key in MODEL_ORDER:
            model_subset = subset[subset["model_key"] == model_key]
            ax.plot(
                model_subset["scenario_label"],
                model_subset["mean_nrmse_avg"],
                marker="o",
                markersize=7.5,
                linewidth=2.2,
                color=MODEL_COLORS[model_key],
                label=MODEL_LABELS[model_key],
            )

        ax.set_title(dataset_name, fontsize=11, pad=10)
        ax.set_xlabel("")
        ax.grid(True, axis="y", color="#d9dde3", linewidth=0.8)
        ax.grid(False, axis="x")
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Mean NRMSE", fontsize=11)
    for ax in axes[1:]:
        ax.set_ylabel("")

    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.suptitle("LLM Finetuning Robustness Across Missingness Scenarios", fontsize=18, y=0.98)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        frameon=False,
        ncol=3,
        fontsize=9.5,
        columnspacing=1.4,
        handletextpad=0.5,
    )
    fig.text(0.99, 0.02, "Rate fixed at 10%", ha="right", va="bottom", fontsize=9, color="#5d6773")

    sns.despine(fig=fig)
    plt.tight_layout(rect=(0, 0.04, 1, 0.88))
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = _load_rows()
    summary = _build_summary(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_CSV, index=False)
    _plot(summary)

    print("=" * 80)
    print("LLM FINETUNING SCENARIO ROBUSTNESS PLOT WRITTEN")
    print("=" * 80)
    print(f"Input CSV : {INPUT_PATH.resolve()}")
    print(f"Output PNG: {OUTPUT_PNG.resolve()}")
    print(f"Output PDF: {OUTPUT_PDF.resolve()}")
    print(f"Output CSV: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
