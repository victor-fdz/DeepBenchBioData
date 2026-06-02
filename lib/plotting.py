#!/usr/bin/env python3
"""Plotting utilities for normalization benchmarking results."""

expression_unit = "counts"

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# -----------------------------
# Constants
# -----------------------------
# Mapping orthology labels to suffix tags in dataframe columns
ORTHOLOGY_TAG = {
    "orthologs": "ortho",
    "nonorthologs": "nonortho",
    "increment": "increment",
}

# Color per orthology type for scatter plots
SCATTER_COLOR = {
    "orthologs": "blue",
    "nonorthologs": "orange",
    "ortho": "blue",
    "nonortho": "orange",
}

# Plot size presets
FIGSIZE_SCATTER = (3, 3)
FIGSIZE_HEATMAP = (12, 7)


# -----------------------------
# Column helpers
# -----------------------------
def mouse_col(tissue: str, expression_unit: str = "tpm") -> str:
    """
    Return mouse expression column name.

    Args:
        - tissue (str): Tissue name
        - expression_unit (str): Expression unit (e.g., "tpm", "counts")

    Returns:
        str: Mouse expression column name
    """
    return f"{tissue}_{expression_unit}_mouse"


def human_col(tissue: str, expression_unit: str = "tpm") -> str:
    """
    Return human expression column name.

    Args:
        - tissue (str): Tissue name
        - expression_unit (str): Expression unit (e.g., "tpm", "counts")


    Returns:
        str: Human expression column name
    """
    return f"{tissue}_{expression_unit}_human"


# -----------------------------
# Core helpers
# -----------------------------
def _filter_by_orthology(df_r: pd.DataFrame, orthology: str) -> pd.DataFrame:
    """
    Filter stats dataframe by orthology and standardize column names.

    Args:
        - df_r (pd.DataFrame): Correlation statistics dataframe
        - orthology (str): Orthology type (orthologs / nonorthologs / increment)

    Returns:
        pd.DataFrame: Filtered and standardized dataframe
    """

    key = orthology.lower()

    # validate orthology input
    if key not in ORTHOLOGY_TAG:
        raise ValueError(f"Unknown orthology: {orthology}")

    tag = ORTHOLOGY_TAG[key]

    # keep only relevant columns for selected orthology
    cols = ["tissue", "method"] + [
        c for c in df_r.columns if c.endswith(f"_{tag}")
    ]

    # standardize column names across orthologies
    rename_map = {
        f"Spearman_rho_{tag}": "Spearman_rho",
        f"Pearson_R_{tag}": "Pearson_R",
    }

    return df_r[cols].rename(columns=rename_map)


def _get_xy(df: pd.DataFrame, tissues: list[str], tissue: str, expression_unit: str = "tpm") -> tuple[pd.Series, pd.Series]:
    """
    Extract x/y expression values for scatter plots.

    Args:
        - df (pd.DataFrame): Normalized expression dataframe
        - tissues (list[str]): List of tissues
        - tissue (str): Tissue name or 'General'

    Returns:
        tuple: (x, y) expression arrays
    """

    # combine all tissues into one vector
    if tissue == "General":
        x = df[[mouse_col(t, expression_unit) for t in tissues]].to_numpy().ravel()
        y = df[[human_col(t, expression_unit) for t in tissues]].to_numpy().ravel()
    else:
        # single tissue comparison
        x = df[mouse_col(tissue, expression_unit)].to_numpy()
        y = df[human_col(tissue, expression_unit)].to_numpy()

    return x, y


def _get_stats(df_r: pd.DataFrame, tissue: str, method: str, expression_unit: str = "tpm"):
    """
    Extract correlation statistics for a tissue/method pair.

    Args:
        - df_r (pd.DataFrame): Stats dataframe
        - tissue (str): Tissue name
        - method (str): Normalization method
        - expression_unit (str): Expression unit (e.g., "tpm", "counts")

    Returns:
        tuple: (Pearson_R, Spearman_rho)
    """

    row = df_r[(df_r["tissue"] == tissue) & (df_r["method"] == method)]

    return float(row["Pearson_R"].values[0]), float(row["Spearman_rho"].values[0])


# -----------------------------
# Main plots
# -----------------------------
def plot_correlations(
    dfs_norm: dict[str, pd.DataFrame],
    df_r: pd.DataFrame,
    tissues: list[str],
    df_name: str,
    orthology: str,
    output_dir: Path,
    expression_unit: str = "tpm",
) -> None:
    """
    Generate scatter plots of human vs mouse expression.

    Layout:
        - columns: normalization methods
        - rows: tissues + General

    Args:
        - dfs_norm (dict): Method → dataframe mapping
        - df_r (pd.DataFrame): Correlation stats dataframe
        - tissues (list[str]): Tissue list
        - df_name (str): Dataset name
        - orthology (str): Orthology type
        - output_dir (Path): Output directory

    Returns:
        None
    """

    logger.info("Plotting scatter plots for %s / %s.", df_name, orthology)

    df_r = _filter_by_orthology(df_r, orthology)

    color = SCATTER_COLOR.get(orthology.lower(), "gray")

    methods = list(dfs_norm.keys())
    row_labels = tissues + ["General"]

    n_rows, n_cols = len(row_labels), len(methods)

    # create grid of subplots
    _, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 3, n_rows * 3),
        squeeze=False,
    )

    # iterate over methods (columns)
    for col_idx, method_key in enumerate(methods):
        df = dfs_norm[method_key]

        # iterate over tissues (rows)
        for row_idx, tissue in enumerate(row_labels):
            ax = axes[row_idx, col_idx]

            # extract expression values
            x, y = _get_xy(df, tissues, tissue, expression_unit=expression_unit,)

            sns.scatterplot(
                x=x,
                y=y,
                ax=ax,
                alpha=0.5,
                legend=False,
                color=color,
            )

            # compute correlation stats
            r_p, r_s = _get_stats(df_r, tissue, method_key)

            stat_text = f"Rp={r_p:.3f} / Rs={r_s:.3f}"

            ax.set_title(
                f"{method_key}\n{stat_text}" if row_idx == 0 else stat_text,
                fontsize=9,
            )

            ax.set_ylabel(
                f"{tissue}\nHuman" if col_idx == 0 else "",
                fontsize=8,
            )

            ax.set_xlabel(
                "Mouse" if row_idx == n_rows - 1 else "",
                fontsize=8,
            )

    plt.tight_layout()

    # save figure
    out_path = output_dir / df_name / "Normalization" / f"{orthology}_scatter.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_path, format="png")
    plt.close()

    logger.info("Scatter plot saved to %s", out_path)


def sort_methods_by_correlation(
    df_r: pd.DataFrame,
    orthology: str,
    sort_col: str = "Pearson_R",
    tissue: str = "General",
) -> list[str]:
    """
    Rank normalization methods by correlation score.

    Args:
        - df_r (pd.DataFrame): Stats dataframe
        - orthology (str): Orthology type
        - sort_col (str): Metric to sort by
        - tissue (str): Tissue used for ranking

    Returns:
        list[str]: Ordered methods
    """

    df_filtered = _filter_by_orthology(df_r, orthology)

    return (
        df_filtered[df_filtered["tissue"] == tissue]
        .sort_values(sort_col, ascending=False)["method"]
        .tolist()
    )


def plot_heatmap(
    df_r: pd.DataFrame,
    df_name: str,
    sorted_methods: list[str],
    criteria: str,
    orthology: str,
    output_dir: Path,
) -> None:
    """
    Generate correlation heatmap across tissues and methods.

    Args:
        - df_r (pd.DataFrame): Stats dataframe
        - df_name (str): Dataset name
        - sorted_methods (list[str]): Method order
        - criteria (str): Metric to plot
        - orthology (str): Orthology type
        - output_dir (Path): Output directory

    Returns:
        None
    """

    logger.info("Plotting heatmap for %s / %s.", df_name, orthology)

    df_filtered = _filter_by_orthology(df_r, orthology)

    # pivot into matrix form
    df_heat = df_filtered.pivot(
        index="tissue",
        columns="method",
        values=criteria,
    )

    # enforce method ordering
    df_heat = df_heat.reindex(columns=sorted_methods)

    # enforce tissue ordering
    row_order = [t for t in df_heat.index if t != "General"] + ["General"]
    df_heat = df_heat.reindex(row_order)

    is_increment = orthology.lower() == "increment"

    cbar_label = {
        "Pearson_R": "Δ Pearson's R" if is_increment else "Pearson R",
        "Spearman_rho": "Δ Spearman's Rho" if is_increment else "Spearman Rho",
    }.get(criteria, criteria)

    plt.figure(figsize=FIGSIZE_HEATMAP)

    sns.heatmap(
        df_heat,
        annot=True,
        cmap="BuGn",
        fmt=".3f",
        linewidths=0.5,
        vmin=0,
        vmax=1,
        cbar_kws={"label": cbar_label},
    )

    plt.ylabel("Tissue")
    plt.xlabel("Normalization Method")
    plt.tight_layout()

    # save output
    out_path = output_dir / df_name / "Normalization" / f"{orthology}_{criteria.replace('_', '')}_heatmap.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(out_path, format="png")
    plt.close()

    logger.info("Heatmap saved to %s", out_path) 