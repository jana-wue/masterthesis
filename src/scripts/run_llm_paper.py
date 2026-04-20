from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
import gc

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.imputation.llm_paper import (
    build_paper_prompt,
    build_paper_retry_prompt,
    clean_and_parse_llm_data,
    normalize_imputed_matrix,
)
from src.paths import DATA_RAW, DATA_PROCESSED, DATA_RESULTS


MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DATASET_NAME_PROMPT = "Telco-Customer-Churn"
TARGET_COLUMN = "TotalCharges"
RESULTS_RMSE_PATH = Path("data/results/imputation_results.csv")
RESULTS_NRMSE_PATH = Path("data/results/imputation_results_nrsme.csv")


def model_name_to_file_token(model_name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")
    return token or "unknown_model"


def _append_results_row(csv_path: Path, required_columns: list[str], row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=required_columns)

    for col in required_columns:
        if col not in df.columns:
            df[col] = pd.NA

    row_normalized = {col: row.get(col, pd.NA) for col in required_columns}
    df = pd.concat([df, pd.DataFrame([row_normalized])], ignore_index=True)
    df.to_csv(csv_path, index=False)


def append_to_global_results(dataset, missingness_type, missing_rate, imputation_method,
    mean_rmse, mean_nrmse) -> None:
    _append_results_row(
        csv_path=RESULTS_RMSE_PATH,
        required_columns=[
            "dataset",
            "missingness_type",
            "missing_rate",
            "imputation_method",
            "mean_rmse",
        ],
        row={
            "dataset": dataset,
            "missingness_type": missingness_type,
            "missing_rate": missing_rate,
            "imputation_method": imputation_method,
            "mean_rmse": mean_rmse,
        },
    )

    _append_results_row(
        csv_path=RESULTS_NRMSE_PATH,
        required_columns=[
            "dataset",
            "missingness_type",
            "missing_rate",
            "imputation_method",
            "mean_rmse",
            "mean_nrmse",
        ],
        row={
            "dataset": dataset,
            "missingness_type": missingness_type,
            "missing_rate": missing_rate,
            "imputation_method": imputation_method,
            "mean_rmse": mean_rmse,
            "mean_nrmse": mean_nrmse,
        },
    )


def _load_telco_mar_data():
    df_full = pd.read_csv(DATA_RAW / "Telco-Customer-Churn_cleaned.csv")
    df_missing = pd.read_csv(
        DATA_PROCESSED / "MAR" / "telco_customer_churn_mar_totalcharges_tenure_10pct.csv"
    )
    return df_full, df_missing


def _build_prompt_matrix(df_missing, n_rows):
    work = df_missing.drop(columns=["customerID"], errors="ignore").copy()
    if n_rows is not None:
        work = work.head(n_rows).copy()

    original_indices = work.index.tolist()
    prompt_matrix = work.reset_index(drop=True)

    missing_positions = [
        pos for pos, is_missing in enumerate(prompt_matrix[TARGET_COLUMN].isna()) if is_missing
    ]
    if not missing_positions:
        raise ValueError(
            "Selected prompt matrix has no missing TotalCharges rows. "
            "Increase n_rows or set n_rows=None."
        )

    return prompt_matrix, original_indices, missing_positions


def _split_into_folds(positions, n_folds):
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1.")
    if not positions:
        return []

    k = min(n_folds, len(positions))
    base_size = len(positions) // k
    remainder = len(positions) % k

    folds = []
    start = 0
    for fold_id in range(k):
        fold_size = base_size + (1 if fold_id < remainder else 0)
        end = start + fold_size
        folds.append(positions[start:end])
        start = end
    return folds


def _load_or_get_model(tokenizer=None, model=None):
    owns_model = tokenizer is None or model is None
    if owns_model:
        print("=" * 80)
        print("LOADING MODEL")
        print("=" * 80)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()
    return tokenizer, model


def apply_user_chat_template(prompt_text, tokenizer):
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt_text


def generate_matrix_answer(prompt_text, tokenizer,model, max_new_tokens):
    model_input = apply_user_chat_template(prompt_text, tokenizer)
    inputs = tokenizer(model_input, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def fallback_totalcharges(row, totalcharges_median):
    tenure = row.get("tenure")
    monthly = row.get("MonthlyCharges")

    if pd.notna(tenure) and pd.notna(monthly):
        try:
            return float(tenure) * float(monthly), "fallback_tenure_x_monthlycharges"
        except Exception:
            pass

    return float(totalcharges_median), "fallback_median"


def _append_prediction_rows(output_csv, rows_df):
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if output_csv.exists():
        existing_df = pd.read_csv(output_csv)
        combined = pd.concat([existing_df, rows_df], ignore_index=True, sort=False)
    else:
        combined = rows_df.copy()

    combined.to_csv(output_csv, index=False)


def _debug_dump_prompt(prompt_text, debug_prompts, debug_label):
    if not debug_prompts:
        return

    print("\n" + "=" * 80)
    print(f"DEBUG PROMPT [{debug_label}]")
    print("=" * 80)
    print(prompt_text)
    print("=" * 80 + "\n")

    debug_dir = DATA_RESULTS / "prompt_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / f"{debug_label}.txt"
    debug_path.write_text(prompt_text, encoding="utf-8")
    print(f"Saved debug prompt to: {debug_path}")


def _normalize_or_recover_rows(parsed_df, prompt_matrix):
    expected_columns = list(prompt_matrix.columns)
    expected_rows = len(prompt_matrix)

    try:
        normalized = normalize_imputed_matrix(
            df_imputed=parsed_df,
            expected_columns=expected_columns,
            expected_rows=expected_rows,
        )
        return normalized, None
    except Exception as exc:
        error_text = str(exc)
        if "fewer rows than expected" not in error_text:
            raise

        # Partial-row recovery for truncated model outputs:
        # keep parsed rows and pad the tail with original prompt rows.
        work = parsed_df.copy()
        if len(work.columns) == len(expected_columns) + 1:
            first_col = str(work.columns[0]).strip().lower()
            if first_col.startswith("unnamed") or work.columns[0] not in expected_columns:
                work = work.iloc[:, 1:]

        if set(expected_columns).issubset(set(work.columns)):
            work = work.loc[:, expected_columns]
        elif len(work.columns) == len(expected_columns):
            work.columns = expected_columns
        else:
            raise

        if len(work) > expected_rows:
            work = work.tail(expected_rows)

        if len(work) < expected_rows:
            missing_rows = expected_rows - len(work)
            filler = prompt_matrix.iloc[len(work): len(work) + missing_rows].copy()
            work = pd.concat([work.reset_index(drop=True), filler.reset_index(drop=True)], ignore_index=True)

        recovered_note = f"partial_rows_recovered({len(parsed_df)}->{expected_rows})"
        return work.reset_index(drop=True), recovered_note


def _build_compact_fold_matrix(fold_matrix, fold_positions, max_rows):
    """
    Build a smaller per-fold prompt matrix while keeping all evaluated rows.
    """
    total_rows = len(fold_matrix)
    if total_rows == 0:
        raise ValueError("fold_matrix is empty.")

    required_positions = sorted(set(int(pos) for pos in fold_positions))
    if max_rows is None:
        budget = total_rows
    else:
        budget = max(int(max_rows), len(required_positions))
        budget = min(budget, total_rows)

    if budget >= total_rows:
        selected_positions = list(range(total_rows))
    else:
        observed_positions = [
            pos
            for pos in range(total_rows)
            if pos not in required_positions and pd.notna(fold_matrix.loc[pos, TARGET_COLUMN])
        ]
        observed_positions.sort(key=lambda pos: min(abs(pos - req) for req in required_positions))
        selected_positions = sorted(required_positions + observed_positions[: budget - len(required_positions)])

    compact_matrix = fold_matrix.iloc[selected_positions].reset_index(drop=True)
    pos_map = {orig_pos: new_pos for new_pos, orig_pos in enumerate(selected_positions)}
    compact_fold_positions = [pos_map[pos] for pos in required_positions if pos in pos_map]
    return compact_matrix, pos_map, compact_fold_positions


def _impute_full_matrix(
    prompt_matrix,
    tokenizer,
    model,
    max_new_tokens,
    debug_prompts=False,
    debug_label="run",
):
    def _generate_or_raise(prompt_text: str) -> str:
        try:
            return generate_matrix_answer(
                prompt_text=prompt_text,
                tokenizer=tokenizer,
                model=model,
                max_new_tokens=max_new_tokens,
            )
        except torch.OutOfMemoryError as exc:
            raise RuntimeError(
                "CUDA OOM during generation. Try lower n_rows and/or max_new_tokens. "
                "Recommended starting point on 16GB GPU: n_rows=120, max_new_tokens=4096."
            ) from exc

    primary_prompt = build_paper_prompt(
        dataset_name=DATASET_NAME_PROMPT,
        missing_data=prompt_matrix,
    )
    _debug_dump_prompt(
        prompt_text=primary_prompt,
        debug_prompts=debug_prompts,
        debug_label=f"{debug_label}_primary",
    )
    raw_primary = _generate_or_raise(primary_prompt)

    first_error_text = None
    try:
        parsed_df = clean_and_parse_llm_data(
            response_text=raw_primary,
            expected_shape=prompt_matrix.shape,
        )
        imputed_matrix, recovery_note = _normalize_or_recover_rows(
            parsed_df=parsed_df,
            prompt_matrix=prompt_matrix,
        )
        return imputed_matrix, raw_primary, recovery_note
    except Exception as first_exc:
        first_error_text = str(first_exc)
        print(f"WARNING: first parse failed, retrying with stricter prompt. ({first_error_text})")

    retry_prompt = build_paper_retry_prompt(
        dataset_name=DATASET_NAME_PROMPT,
        missing_data=prompt_matrix,
    )
    _debug_dump_prompt(
        prompt_text=retry_prompt,
        debug_prompts=debug_prompts,
        debug_label=f"{debug_label}_retry",
    )
    raw_retry = _generate_or_raise(retry_prompt)

    try:
        parsed_df = clean_and_parse_llm_data(
            response_text=raw_retry,
            expected_shape=prompt_matrix.shape,
        )
        imputed_matrix, recovery_note = _normalize_or_recover_rows(
            parsed_df=parsed_df,
            prompt_matrix=prompt_matrix,
        )
        if recovery_note:
            parse_note = f"first_parse_failed_but_retry_succeeded|{recovery_note}"
        else:
            parse_note = "first_parse_failed_but_retry_succeeded"
        return imputed_matrix, raw_retry, parse_note
    except Exception as retry_exc:
        retry_error_text = str(retry_exc)
        parse_error = (
            f"first_parse_failed: {first_error_text} | retry_failed: {retry_error_text}"
        )
        print(f"WARNING: retry parse failed, using fallback-only for missing rows. ({parse_error})")
        return prompt_matrix.copy(), raw_retry, parse_error


def _summarize_notes(notes, max_items=5):
    if not notes:
        return None
    unique = []
    for note in notes:
        if note and note not in unique:
            unique.append(note)
    if not unique:
        return None
    shown = unique[:max_items]
    if len(unique) > max_items:
        shown.append(f"... (+{len(unique) - max_items} more)")
    return " | ".join(shown)


def _impute_matrix_in_batches(
    matrix,
    tokenizer,
    model,
    max_new_tokens,
    batch_row=40,
    batch_col=10,
    debug_prompts=False,
    debug_label="run",
):
    """
    Paper-style batching over rows and columns.
    Keeps current output format/parser logic from this project.
    """
    output = matrix.copy()
    n_rows, n_cols = matrix.shape

    raw_blocks = []
    notes = []
    batch_idx = 0

    for row_start in range(0, n_rows, batch_row):
        row_end = min(row_start + batch_row, n_rows)
        actual_start = row_start
        if (row_end - row_start) < batch_row and n_rows >= batch_row:
            actual_start = row_end - batch_row

        for col_start in range(0, n_cols, batch_col):
            col_end = min(col_start + batch_col, n_cols)
            batch_to_prompt = matrix.iloc[actual_start:row_end, col_start:col_end].copy()
            batch_label = (
                f"{debug_label}_batch{batch_idx}_r{actual_start}-{row_end}_c{col_start}-{col_end}"
            )

            imputed_batch, raw_output, parse_error = _impute_full_matrix(
                prompt_matrix=batch_to_prompt,
                tokenizer=tokenizer,
                model=model,
                max_new_tokens=max_new_tokens,
                debug_prompts=debug_prompts,
                debug_label=batch_label,
            )

            raw_blocks.append(
                "\n".join(
                    [
                        f"### {batch_label}",
                        raw_output if raw_output is not None else "",
                    ]
                )
            )
            if parse_error:
                notes.append(f"{batch_label}:{parse_error}")

            rows_needed = row_end - row_start
            clean_imputed = imputed_batch.iloc[-rows_needed:, :].copy()
            expected_cols = list(batch_to_prompt.columns)

            if set(expected_cols).issubset(set(clean_imputed.columns)):
                clean_imputed = clean_imputed.loc[:, expected_cols]
            elif len(clean_imputed.columns) == len(expected_cols):
                clean_imputed.columns = expected_cols
            else:
                notes.append(
                    f"{batch_label}:col_mismatch(expected={len(expected_cols)},got={len(clean_imputed.columns)})"
                )
                batch_idx += 1
                continue

            if clean_imputed.empty or clean_imputed.shape[0] != rows_needed:
                notes.append(
                    f"{batch_label}:row_mismatch(expected={rows_needed},got={clean_imputed.shape[0]})"
                )
                batch_idx += 1
                continue

            target_index = output.index[row_start:row_end]
            assign_failed = False
            for col in expected_cols:
                values = clean_imputed[col]
                if pd.api.types.is_numeric_dtype(output[col]):
                    coerced = pd.to_numeric(values, errors="coerce")
                    if coerced.isna().sum() > values.isna().sum():
                        notes.append(
                            f"{batch_label}:numeric_coerce({col},"
                            f"added_nan={int(coerced.isna().sum() - values.isna().sum())})"
                        )
                    values = coerced
                try:
                    output.loc[target_index, col] = values.values
                except Exception as exc:
                    notes.append(f"{batch_label}:assign_fail({col},{exc})")
                    assign_failed = True
                    break

            if assign_failed:
                batch_idx += 1
                continue

            batch_idx += 1

    raw_output_text = "\n\n".join(raw_blocks)
    parse_note = _summarize_notes(notes)
    return output, raw_output_text, parse_note


def run_telco_paper_prompt(
    n_rows: int | None = 40,
    max_new_tokens: int = 4096,
    batch_row: int = 40,
    batch_col: int = 10,
    tokenizer=None,
    model=None,
    debug_prompts: bool = False,
):
    df_full, df_missing = _load_telco_mar_data()

    prompt_matrix, original_indices, missing_positions = _build_prompt_matrix(
        df_missing=df_missing,
        n_rows=n_rows,
    )
    tokenizer, model = _load_or_get_model(tokenizer=tokenizer, model=model)

    print("\n" + "=" * 80)
    print("RUNNING PAPER PROMPT (NO BATCH LOOP)")
    print("=" * 80)
    print(f"Prompt rows: {len(prompt_matrix)}")
    print(f"Missing rows to evaluate: {len(missing_positions)}")

    imputed_matrix, raw_output, parse_error = _impute_matrix_in_batches(
        matrix=prompt_matrix,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=max_new_tokens,
        batch_row=batch_row,
        batch_col=batch_col,
        debug_prompts=debug_prompts,
        debug_label="single_run",
    )

    observed_target = pd.to_numeric(df_full[TARGET_COLUMN], errors="coerce").dropna()
    totalcharges_median = float(observed_target.median())

    results = []
    for pos in missing_positions:
        original_idx = original_indices[pos]
        row_for_fallback = prompt_matrix.iloc[pos]

        pred_value = pd.to_numeric(imputed_matrix.loc[pos, TARGET_COLUMN], errors="coerce")
        source = "paper_prompt"

        if pd.isna(pred_value):
            pred_value, source = fallback_totalcharges(
                row=row_for_fallback,
                totalcharges_median=totalcharges_median,
            )

        ground_truth = pd.to_numeric(df_full.loc[original_idx, TARGET_COLUMN], errors="coerce")

        results.append(
            {
                "row_index": int(original_idx),
                "prompt_row_position": int(pos),
                "prediction": float(pred_value),
                "ground_truth": float(ground_truth) if pd.notna(ground_truth) else pd.NA,
                "source": source,
                "parse_error": parse_error,
                "fold_id": 1,
                "n_folds_requested": 1,
                "n_folds_effective": 1,
            }
        )

    results_df = pd.DataFrame(results)

    run_timestamp = datetime.now(timezone.utc).isoformat()
    model_token = model_name_to_file_token(MODEL_NAME)
    output_csv = DATA_RESULTS / f"telco_{model_token}_paper_results.csv"
    raw_output_txt = DATA_RESULTS / f"telco_{model_token}_paper_raw_output.txt"

    results_df["model_name"] = MODEL_NAME
    results_df["prompt_rows"] = len(prompt_matrix)
    results_df["n_rows_requested"] = n_rows if n_rows is not None else len(prompt_matrix)
    results_df["max_new_tokens"] = max_new_tokens
    results_df["batch_row"] = batch_row
    results_df["batch_col"] = batch_col
    results_df["run_timestamp_utc"] = run_timestamp
    results_df["raw_output_preview"] = raw_output[:600]

    raw_output_txt.parent.mkdir(parents=True, exist_ok=True)
    raw_output_txt.write_text(raw_output, encoding="utf-8")
    _append_prediction_rows(output_csv=output_csv, rows_df=results_df)

    print("\n" + "=" * 80)
    print("RESULT TABLE")
    print("=" * 80)
    print(results_df.head())
    print(f"\nSaved predictions to: {output_csv}")
    print(f"Saved full model output to: {raw_output_txt}")

    return results_df


def run_telco_paper_prompt_folds(
    n_rows: int | None = 40,
    max_new_tokens: int = 4096,
    n_folds: int = 5,
    fold_prompt_rows: int | None = 25,
    batch_row: int = 40,
    batch_col: int = 10,
    debug_prompts: bool = False,
    tokenizer=None,
    model=None,
):
    df_full, df_missing = _load_telco_mar_data()
    prompt_matrix, original_indices, missing_positions = _build_prompt_matrix(
        df_missing=df_missing,
        n_rows=n_rows,
    )
    fold_splits = _split_into_folds(missing_positions, n_folds=n_folds)
    effective_folds = len(fold_splits)

    tokenizer, model = _load_or_get_model(tokenizer=tokenizer, model=model)

    print("\n" + "=" * 80)
    print("RUNNING PAPER PROMPT WITH FOLDS")
    print("=" * 80)
    print(f"Prompt rows: {len(prompt_matrix)}")
    print(f"Missing rows to evaluate: {len(missing_positions)}")
    print(f"Requested folds: {n_folds} | Effective folds: {effective_folds}")

    observed_target = pd.to_numeric(df_full[TARGET_COLUMN], errors="coerce").dropna()
    totalcharges_median = float(observed_target.median())

    run_timestamp = datetime.now(timezone.utc).isoformat()
    model_token = model_name_to_file_token(MODEL_NAME)
    output_csv = DATA_RESULTS / f"telco_{model_token}_paper_results.csv"

    all_fold_results = []
    for fold_idx, fold_positions in enumerate(fold_splits, start=1):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        fold_set = set(fold_positions)
        fold_matrix = prompt_matrix.copy()

        # Keep missing values only for this fold. Non-fold missing rows are filled with known full-data values.
        for pos in missing_positions:
            if pos in fold_set:
                continue
            original_idx = original_indices[pos]
            known_value = pd.to_numeric(df_full.loc[original_idx, TARGET_COLUMN], errors="coerce")
            if pd.notna(known_value):
                fold_matrix.loc[pos, TARGET_COLUMN] = float(known_value)

        compact_matrix, pos_map, _ = _build_compact_fold_matrix(
            fold_matrix=fold_matrix,
            fold_positions=fold_positions,
            max_rows=fold_prompt_rows,
        )

        print("\n" + "-" * 80)
        print(
            f"Fold {fold_idx}/{effective_folds} | eval rows: {len(fold_positions)} | "
            f"prompt rows this fold: {len(compact_matrix)}"
        )

        imputed_matrix, raw_output, parse_error = _impute_matrix_in_batches(
            matrix=compact_matrix,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=max_new_tokens,
            batch_row=batch_row,
            batch_col=batch_col,
            debug_prompts=debug_prompts,
            debug_label=f"fold{fold_idx}_of_{effective_folds}",
        )

        raw_output_txt = DATA_RESULTS / (
            f"telco_{model_token}_paper_raw_output_fold{fold_idx}_of_{effective_folds}.txt"
        )
        raw_output_txt.parent.mkdir(parents=True, exist_ok=True)
        raw_output_txt.write_text(raw_output, encoding="utf-8")

        fold_rows = []
        for pos in fold_positions:
            original_idx = original_indices[pos]
            row_for_fallback = prompt_matrix.iloc[pos]
            compact_pos = pos_map.get(pos)

            if compact_pos is None:
                pred_value = pd.NA
            else:
                pred_value = pd.to_numeric(imputed_matrix.loc[compact_pos, TARGET_COLUMN], errors="coerce")
            source = "paper_prompt_fold"

            if pd.isna(pred_value):
                pred_value, source = fallback_totalcharges(
                    row=row_for_fallback,
                    totalcharges_median=totalcharges_median,
                )

            ground_truth = pd.to_numeric(df_full.loc[original_idx, TARGET_COLUMN], errors="coerce")

            fold_rows.append(
                {
                    "row_index": int(original_idx),
                    "prompt_row_position": int(pos),
                    "compact_prompt_row_position": int(compact_pos) if compact_pos is not None else pd.NA,
                    "prediction": float(pred_value),
                    "ground_truth": float(ground_truth) if pd.notna(ground_truth) else pd.NA,
                    "source": source,
                    "parse_error": parse_error,
                    "fold_id": fold_idx,
                    "n_folds_requested": n_folds,
                    "n_folds_effective": effective_folds,
                    "model_name": MODEL_NAME,
                    "prompt_rows": len(compact_matrix),
                    "fold_prompt_rows_cap": fold_prompt_rows if fold_prompt_rows is not None else pd.NA,
                    "n_rows_requested": n_rows if n_rows is not None else len(prompt_matrix),
                    "max_new_tokens": max_new_tokens,
                    "batch_row": batch_row,
                    "batch_col": batch_col,
                    "run_timestamp_utc": run_timestamp,
                    "raw_output_preview": raw_output[:600],
                }
            )

        fold_df = pd.DataFrame(fold_rows)
        all_fold_results.append(fold_df)
        _append_prediction_rows(output_csv=output_csv, rows_df=fold_df)

    results_df = pd.concat(all_fold_results, ignore_index=True)
    print("\n" + "=" * 80)
    print("FOLD RESULT TABLE")
    print("=" * 80)
    print(results_df.head())
    print(f"\nSaved predictions to: {output_csv}")
    return results_df


def evaluate_telco_paper_prompt(
    n_rows: int | None = 40,
    max_new_tokens: int = 4096,
    n_folds: int = 1,
    fold_prompt_rows: int | None = 25,
    batch_row: int = 40,
    batch_col: int = 10,
    debug_prompts: bool = False,
) -> pd.DataFrame:
    if n_folds > 1:
        results_df = run_telco_paper_prompt_folds(
            n_rows=n_rows,
            max_new_tokens=max_new_tokens,
            n_folds=n_folds,
            fold_prompt_rows=fold_prompt_rows,
            batch_row=batch_row,
            batch_col=batch_col,
            debug_prompts=debug_prompts,
        )
    else:
        results_df = run_telco_paper_prompt(
            n_rows=n_rows,
            max_new_tokens=max_new_tokens,
            batch_row=batch_row,
            batch_col=batch_col,
            debug_prompts=debug_prompts,
        )

    work = results_df.copy()
    work["prediction"] = pd.to_numeric(work["prediction"], errors="coerce")
    work["ground_truth"] = pd.to_numeric(work["ground_truth"], errors="coerce")
    work = work.dropna(subset=["prediction", "ground_truth"]).copy()

    if work.empty:
        raise ValueError("No valid numeric prediction/ground_truth rows to evaluate.")

    errors = work["ground_truth"] - work["prediction"]
    mean_rmse = float(((errors ** 2).mean()) ** 0.5)
    std_true = float(work["ground_truth"].std(ddof=0))
    mean_nrmse = float(mean_rmse / (std_true + 1e-8))

    model_short = MODEL_NAME.split("/")[-1]
    effective_folds = int(results_df["n_folds_effective"].max()) if "n_folds_effective" in results_df.columns else 1
    if n_rows is None and effective_folds == 1:
        method_name = f"LLM Paper Prompt ({model_short})"
    elif n_rows is None:
        method_name = f"LLM Paper Prompt folds={effective_folds} ({model_short})"
    elif effective_folds == 1:
        method_name = f"LLM Paper Prompt rows={n_rows} ({model_short})"
    else:
        method_name = (
            f"LLM Paper Prompt rows={n_rows} folds={effective_folds} ({model_short})"
        )

    append_to_global_results(
        dataset="Telco",
        missingness_type="MAR",
        missing_rate="missing totalCharges -> tenure, 10%",
        imputation_method=method_name,
        mean_rmse=mean_rmse,
        mean_nrmse=mean_nrmse,
    )

    summary_df = pd.DataFrame(
        [
            {
                "model_name": MODEL_NAME,
                "imputation_method": method_name,
                "n_predictions": int(len(work)),
                "mean_rmse": mean_rmse,
                "mean_nrmse": mean_nrmse,
            }
        ]
    )

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(summary_df)
    print(f"\nAppended to: {RESULTS_RMSE_PATH}")
    print(f"Appended to: {RESULTS_NRMSE_PATH}")

    return summary_df


if __name__ == "__main__":
    evaluate_telco_paper_prompt(n_rows=40, max_new_tokens=4096, n_folds=5)
