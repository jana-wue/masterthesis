"""Legacy pipeline kept only as a guarded compatibility shim.

This module intentionally no longer runs benchmark experiments because the old
pipeline used stale scenario paths and full-matrix evaluation logic that does
not match the final thesis benchmark. Use `main.py` or
`src/scripts/run_classical_manifest.py` instead.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from src.imputation.base import BaseImputer

LEGACY_PIPELINE_MESSAGE = (
    "src/scripts/imputation_pipeline.py is a legacy pipeline and is disabled for the "
    "final thesis benchmark. Use main.py or src/scripts/run_classical_manifest.py "
    "for classical benchmark runs. Prompt and finetuned LLM results are handled "
    "via their dedicated final result builders."
)


def _raise_legacy_error() -> None:
    """Raise the legacy pipeline error."""
    raise RuntimeError(LEGACY_PIPELINE_MESSAGE)


def get_available_method_keys() -> list[str]:
    """Handle get available method keys."""
    return ["meanmode", "medianmode", "mice", "missforest", "dae"]


def get_available_dataset_keys(
    dataset_configs: Sequence[Mapping[str, str]] | None = None,
) -> list[str]:
    """Handle get available dataset keys."""
    del dataset_configs
    return ["telco", "statlog", "creditcard"]


def get_dataset_configs(
    dataset_keys: list[str] | None = None,
    dataset_configs: Sequence[Mapping[str, str]] | None = None,
) -> None:
    """Handle get dataset configs."""
    del dataset_keys, dataset_configs
    _raise_legacy_error()


def _prepare_dataset_context(dataset: Mapping[str, str]) -> None:
    """Prepare dataset context."""
    del dataset
    _raise_legacy_error()


def _prepare_missing_for_dataset(dataset: Mapping[str, str], scenario: str) -> None:
    """Prepare missing for dataset."""
    del dataset, scenario
    _raise_legacy_error()


def run_imputation_experiment(
    imputer: BaseImputer,
    dataset: Mapping[str, str],
    scenario: str,
) -> None:
    """Run imputation experiment."""
    del imputer, dataset, scenario
    _raise_legacy_error()


def run_imputation_method(
    method_key: str,
    dataset_configs: Sequence[Mapping[str, str]] | None = None,
    dataset_keys: list[str] | None = None,
) -> None:
    """Run imputation method."""
    del method_key, dataset_configs, dataset_keys
    _raise_legacy_error()


def run_all_imputation_methods(
    dataset_configs: Sequence[Mapping[str, str]] | None = None,
    dataset_keys: list[str] | None = None,
    method_keys: list[str] | None = None,
) -> None:
    """Run all imputation methods."""
    del dataset_configs, dataset_keys, method_keys
    _raise_legacy_error()
