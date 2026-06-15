from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scripts.build_unified_benchmark_results import OUTPUT_COLUMNS, main as build_unified_main


RESULTS_DIR = PROJECT_ROOT / "data" / "results"
RAW_PROMPT_PATH = RESULTS_DIR / "benchmark_llm_prompt_runs.csv"
FINAL_PROMPT_PATH = RESULTS_DIR / "benchmark_llm_prompt_runs_final.csv"

DEDUP_KEYS = [
    "dataset_name",
    "target_column",
    "missingness_type",
    "method_key",
    "model_name",
]


def _load_raw_prompt_runs() -> pd.DataFrame:
    if not RAW_PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt run file not found: {RAW_PROMPT_PATH}")

    df = pd.read_csv(RAW_PROMPT_PATH)
    if df.empty:
        raise ValueError(f"Prompt run file is empty: {RAW_PROMPT_PATH}")

    required = set(DEDUP_KEYS + ["run_id", "run_timestamp_utc", "status"])
    missing = required - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Prompt run file is missing required columns: {missing_list}")

    return df


def _clean_prompt_runs(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["run_timestamp_utc"] = pd.to_datetime(work["run_timestamp_utc"], errors="coerce", utc=True)

    # Keep only successful prompt baseline rows and retain the latest run per setting.
    work = work[work["status"].astype(str) == "success"].copy()
    if work.empty:
        raise ValueError("No successful prompt rows found in benchmark_llm_prompt_runs.csv.")

    if "few_shot_k" in work.columns:
        work["few_shot_k"] = pd.to_numeric(work["few_shot_k"], errors="coerce")
        dedup_keys = DEDUP_KEYS + ["few_shot_k"]
    else:
        dedup_keys = list(DEDUP_KEYS)

    work = work.sort_values(by=["run_timestamp_utc", "run_id"], na_position="last")
    work = work.drop_duplicates(subset=dedup_keys, keep="last").copy()
    work = work.sort_values(
        by=[
            "dataset_key",
            "missingness_type",
            "model_key",
            "method_key",
            "run_timestamp_utc",
        ],
        na_position="last",
    ).reset_index(drop=True)

    for col in OUTPUT_COLUMNS:
        if col not in work.columns:
            work[col] = pd.NA

    # Keep extra provenance columns like few_shot_k in the final prompt file;
    # the unified benchmark builder will only read OUTPUT_COLUMNS from it.
    base_columns = OUTPUT_COLUMNS + [col for col in work.columns if col not in OUTPUT_COLUMNS]
    return work[base_columns]


def main() -> None:
    raw = _load_raw_prompt_runs()
    cleaned = _clean_prompt_runs(raw)

    FINAL_PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(FINAL_PROMPT_PATH, index=False)

    print("=" * 80)
    print("LLM PROMPT RESULTS FINALIZED")
    print("=" * 80)
    print(f"Raw prompt rows: {len(raw)}")
    print(f"Successful latest prompt rows: {len(cleaned)}")
    print(f"Output CSV: {FINAL_PROMPT_PATH.resolve()}")

    build_unified_main()


if __name__ == "__main__":
    main()
