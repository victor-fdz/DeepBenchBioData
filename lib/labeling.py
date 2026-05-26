#!/usr/bin/env python3
"""Label gene-pair dataframes using profiling metric rankings."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


EXTERNAL_METRIC_COLUMNS = {
    "cosine_sim",
    "met",
    "ssd",
    "z_score_cosine_sim",
}

LABEL_VALUES = {"P", "N", "U"}


def get_metric_column(profiling_method: str) -> str:
    """Return the dataframe column produced by a profiling method."""

    if profiling_method in EXTERNAL_METRIC_COLUMNS:
        return profiling_method

    return f"{profiling_method}_sim"


def validate_labeling_request(
    df: pd.DataFrame,
    metric_column: str,
    n_pos: int,
    n_neg: int,
) -> None:
    """Validate that a dataframe can be labeled without silent coercion."""

    if metric_column not in df.columns:
        raise ValueError(f"Missing profiling metric column: {metric_column}")

    if n_pos <= 0 or n_neg <= 0:
        raise ValueError("n_pos and n_neg must be positive integers.")

    if n_pos + n_neg > len(df):
        raise ValueError(
            f"Cannot label {n_pos} positives and {n_neg} negatives from only {len(df)} rows."
        )

    if df[metric_column].isna().any():
        raise ValueError(f"Metric column {metric_column!r} contains missing values.")


def rank_labeling(
    df: pd.DataFrame,
    profiling_method: str,
    n_pos: int,
    n_neg: int,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, str]:
    """Assign positives to highest metric values and negatives to lowest values."""

    metric_column = get_metric_column(profiling_method)
    validate_labeling_request(df, metric_column, n_pos, n_neg)

    df_out = df.sort_values(metric_column, ascending=False).reset_index(drop=True).copy()
    df_out["label"] = "U"

    df_out.loc[: n_pos - 1, "label"] = "P"
    df_out.loc[len(df_out) - n_neg :, "label"] = "N"

    output_name = f"{profiling_method}_rank_labeling_{n_pos}_{n_neg}.tsv"
    return df_out, output_name


def random_labeling(
    df: pd.DataFrame,
    profiling_method: str,
    n_pos: int,
    n_neg: int,
    random_state: int,
) -> tuple[pd.DataFrame, str]:
    """Assign positive and negative labels randomly without overlap."""

    metric_column = get_metric_column(profiling_method)
    validate_labeling_request(df, metric_column, n_pos, n_neg)

    df_out = df.copy()
    df_out["label"] = "U"

    sampled_indices = df_out.sample(
        n=n_pos + n_neg,
        random_state=random_state,
        replace=False,
    ).index

    positive_indices = sampled_indices[:n_pos]
    negative_indices = sampled_indices[n_pos:]

    df_out.loc[positive_indices, "label"] = "P"
    df_out.loc[negative_indices, "label"] = "N"

    output_name = f"{profiling_method}_random_labeling_{n_pos}_{n_neg}.tsv"
    return df_out, output_name


def rank_labeling_random(
    df: pd.DataFrame,
    profiling_method: str,
    n_pos: int,
    n_neg: int,
    random_state: int,
) -> tuple[pd.DataFrame, str]:
    """Randomly sample positives from the top-ranked pool and negatives from the bottom-ranked pool."""

    metric_column = get_metric_column(profiling_method)
    validate_labeling_request(df, metric_column, n_pos, n_neg)

    df_out = df.sort_values(metric_column, ascending=False).reset_index(drop=True).copy()
    df_out["label"] = "U"

    positive_pool = df_out.head(n_pos)
    negative_pool = df_out.tail(n_neg)

    positive_indices = positive_pool.sample(
        n=n_pos,
        random_state=random_state,
        replace=False,
    ).index

    negative_indices = negative_pool.sample(
        n=n_neg,
        random_state=random_state + 1,
        replace=False,
    ).index

    df_out.loc[positive_indices, "label"] = "P"
    df_out.loc[negative_indices, "label"] = "N"

    output_name = f"{profiling_method}_rank_labeling_random_{n_pos}_{n_neg}.tsv"
    return df_out, output_name


def sanity_check_boxplot(
    df: pd.DataFrame,
    profiling_method: str,
    output_path: Path,
) -> None:
    """Save a label-vs-metric boxplot and fail on invalid inputs."""

    metric_column = get_metric_column(profiling_method)

    if metric_column not in df.columns:
        raise ValueError(f"Missing profiling metric column: {metric_column}")

    if "label" not in df.columns:
        raise ValueError("Missing label column.")

    invalid_labels = set(df["label"].dropna().unique()) - LABEL_VALUES
    if invalid_labels:
        raise ValueError(f"Invalid label values: {sorted(invalid_labels)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.boxplot(x="label", y=metric_column, data=df)
    plt.title(f"{metric_column} by label")
    plt.xlabel("Label")
    plt.ylabel(metric_column)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


LABELING_FUNCTIONS = [
    rank_labeling,
    random_labeling,
    rank_labeling_random,
]
