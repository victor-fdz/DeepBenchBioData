#!/usr/bin/env python3
"""Nextflow-native normalization benchmark driver.

This script writes only process-local, stable relative outputs.
Nextflow owns publication into the final results directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import compute_stats as compst
from lib import normalization as normfx

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--tissue", default="General")
    parser.add_argument(
        "--selection-metric",
        default="pearson_increment",
        choices=[
            "pearson_increment",
            "spearman_increment",
            "pearson_ortho",
            "spearman_ortho",
        ],
    )
    parser.add_argument(
        "--methods",
        default="",
        help="Optional comma-separated normalization methods. If omitted, all registered methods are used.",
    )

    return parser.parse_args()


def get_feature_columns(dataframe: pd.DataFrame) -> list[str]:
    return [
        column
        for column in dataframe.columns
        if len(column.split("_")) >= 3 and column.split("_")[-2] == "tpm"
    ]


def get_tissues(feature_columns: list[str]) -> list[str]:
    return sorted({column.rsplit("_", 2)[0] for column in feature_columns})


def resolve_methods(methods_argument: str) -> list[Callable]:
    available_methods = {
        method.__name__: method
        for method in normfx.NORMALIZATION_METHODS
    }

    if not methods_argument:
        return normfx.NORMALIZATION_METHODS

    requested_methods = [
        method_name.strip()
        for method_name in methods_argument.split(",")
        if method_name.strip()
    ]

    unknown_methods = sorted(set(requested_methods) - set(available_methods))
    if unknown_methods:
        raise ValueError(
            "Unknown normalization method(s): "
            + ", ".join(unknown_methods)
            + ". Available methods: "
            + ", ".join(sorted(available_methods))
        )

    return [available_methods[method_name] for method_name in requested_methods]


def generate_all_nonortholog_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    human_columns = [column for column in dataframe.columns if "human" in column] + ["gene_name"]
    mouse_columns = [column for column in dataframe.columns if "mouse" in column] + ["gene_name"]

    human = dataframe[human_columns].rename(columns={"gene_name": "gene_name_human"})
    mouse = dataframe[mouse_columns].rename(columns={"gene_name": "gene_name_mouse"})

    all_pairs_dataframe = human.merge(mouse, how="cross")

    original_ortholog_pairs = set(
        zip(
            dataframe["gene_id_human"].astype(str),
            dataframe["gene_id_mouse"].astype(str),
        )
    )

    candidate_pairs = list(
        zip(
            all_pairs_dataframe["gene_id_human"].astype(str),
            all_pairs_dataframe["gene_id_mouse"].astype(str),
        )
    )

    nonortholog_mask = ~pd.Series(
        candidate_pairs,
        index=all_pairs_dataframe.index,
    ).isin(original_ortholog_pairs)

    if {"gene_name_human", "gene_name_mouse"}.issubset(all_pairs_dataframe.columns):
        nonortholog_mask = (
            nonortholog_mask
            & (all_pairs_dataframe["gene_name_human"] != all_pairs_dataframe["gene_name_mouse"])
        )

    nonortholog_dataframe = all_pairs_dataframe[nonortholog_mask].reset_index(drop=True)

    meta_human = ["gene_id_human", "gene_name_human"]
    meta_mouse = ["gene_id_mouse", "gene_name_mouse"]

    tpm_human = [
        column
        for column in human.columns
        if column not in meta_human
    ]

    tpm_mouse = [
        column
        for column in mouse.columns
        if column not in meta_mouse
    ]

    column_order = (
        meta_human[:1]
        + meta_mouse[:1]
        + meta_human[1:]
        + meta_mouse[1:]
        + tpm_human
        + tpm_mouse
    )

    return nonortholog_dataframe[column_order]


def prepare_output_directories() -> None:
    for directory_name in [
        "Intermediate_Datasets",
        "Orthologs",
        "NonOrthologs",
        "Normalization",
    ]:
        directory_path = Path(directory_name)
        if directory_path.exists():
            shutil.rmtree(directory_path)
        directory_path.mkdir(parents=True, exist_ok=True)


def apply_and_save_normalizations(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    methods: list[Callable],
    output_directory: Path,
) -> tuple[list[pd.DataFrame], list[str]]:
    output_directory.mkdir(parents=True, exist_ok=True)

    dataframes = [dataframe.copy()]
    method_names = ["original"]

    dataframe.to_csv(output_directory / "original.tsv", sep="\t", index=False)

    for method in methods:
        method_name = method.__name__
        normalized_dataframe = method(dataframe.copy(), numeric_cols=feature_columns)

        normalized_dataframe.to_csv(
            output_directory / f"{method_name}.tsv",
            sep="\t",
            index=False,
        )

        dataframes.append(normalized_dataframe)
        method_names.append(method_name)

    return dataframes, method_names


def merge_and_increment(
    stats_orthologs: pd.DataFrame,
    stats_nonorthologs: pd.DataFrame,
) -> pd.DataFrame:
    stats_all = pd.merge(
        stats_orthologs,
        stats_nonorthologs,
        on=["tissue", "method"],
        suffixes=("_ortho", "_nonortho"),
    )

    stats_all["Pearson_R_increment"] = (
        stats_all["Pearson_R_ortho"]
        - stats_all["Pearson_R_nonortho"].clip(lower=0)
    )
    stats_all["Spearman_rho_increment"] = (
        stats_all["Spearman_rho_ortho"]
        - stats_all["Spearman_rho_nonortho"].clip(lower=0)
    )

    return stats_all


def select_best_method(
    stats_dataframe: pd.DataFrame,
    tissue: str,
    selection_metric: str,
) -> tuple[str, pd.DataFrame]:
    tissue_stats = stats_dataframe[stats_dataframe["tissue"] == tissue].copy()

    if tissue_stats.empty:
        available_tissues = sorted(stats_dataframe["tissue"].dropna().unique())
        raise ValueError(
            f"No normalization statistics found for tissue {tissue!r}. "
            f"Available tissues: {available_tissues}"
        )

    score_column_by_metric = {
        "pearson_increment": "Pearson_R_increment",
        "spearman_increment": "Spearman_rho_increment",
        "pearson_ortho": "Pearson_R_ortho",
        "spearman_ortho": "Spearman_rho_ortho",
    }

    score_column = score_column_by_metric[selection_metric]

    ranking = (
        tissue_stats
        .sort_values(score_column, ascending=False)
        .reset_index(drop=True)
    )

    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    ranking["selection_tissue"] = tissue
    ranking["selection_metric"] = selection_metric
    ranking["selection_score_column"] = score_column

    best_method = str(ranking.loc[0, "method"])

    return best_method, ranking


def get_xy_values(
    dataframe: pd.DataFrame,
    tissues: list[str],
    tissue: str,
) -> tuple[np.ndarray, np.ndarray]:
    if tissue == "General":
        human_values = dataframe[[f"{tissue_name}_tpm_human" for tissue_name in tissues]].to_numpy().ravel()
        mouse_values = dataframe[[f"{tissue_name}_tpm_mouse" for tissue_name in tissues]].to_numpy().ravel()
    else:
        human_values = dataframe[f"{tissue}_tpm_human"].to_numpy()
        mouse_values = dataframe[f"{tissue}_tpm_mouse"].to_numpy()

    return mouse_values, human_values


def plot_scatter_grid(
    normalized_dataframes: list[pd.DataFrame],
    method_names: list[str],
    stats_dataframe: pd.DataFrame,
    tissues: list[str],
    output_path: Path,
    title_prefix: str,
) -> None:
    row_labels = tissues + ["General"]
    number_of_rows = len(row_labels)
    number_of_columns = len(method_names)

    _, axes = plt.subplots(
        number_of_rows,
        number_of_columns,
        figsize=(number_of_columns * 3, number_of_rows * 3),
        squeeze=False,
    )

    for column_index, (method_name, dataframe) in enumerate(zip(method_names, normalized_dataframes)):
        for row_index, tissue in enumerate(row_labels):
            axis = axes[row_index, column_index]
            mouse_values, human_values = get_xy_values(dataframe, tissues, tissue)

            sns.scatterplot(
                x=mouse_values,
                y=human_values,
                ax=axis,
                alpha=0.5,
                legend=False,
            )

            stat_row = stats_dataframe[
                (stats_dataframe["tissue"] == tissue)
                & (stats_dataframe["method"] == method_name)
            ]

            if not stat_row.empty:
                pearson_value = float(stat_row["Pearson_R"].iloc[0])
                spearman_value = float(stat_row["Spearman_rho"].iloc[0])
                stat_text = f"Rp={pearson_value:.3f} / Rs={spearman_value:.3f}"
            else:
                stat_text = "stats unavailable"

            axis.set_title(
                f"{method_name}\n{stat_text}" if row_index == 0 else stat_text,
                fontsize=9,
            )
            axis.set_ylabel(f"{tissue}\nHuman" if column_index == 0 else "", fontsize=8)
            axis.set_xlabel("Mouse" if row_index == number_of_rows - 1 else "", fontsize=8)

    plt.suptitle(title_prefix, y=1.002)
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_increment_heatmap(
    stats_dataframe: pd.DataFrame,
    ranking: pd.DataFrame,
    output_path: Path,
) -> None:
    sorted_methods = ranking["method"].tolist()

    heatmap_dataframe = stats_dataframe.pivot(
        index="tissue",
        columns="method",
        values="Pearson_R_increment",
    )
    heatmap_dataframe = heatmap_dataframe.reindex(columns=sorted_methods)

    row_order = [tissue for tissue in heatmap_dataframe.index if tissue != "General"] + ["General"]
    heatmap_dataframe = heatmap_dataframe.reindex(row_order)

    plt.figure(figsize=(12, 7))
    sns.heatmap(
        heatmap_dataframe,
        annot=True,
        cmap="BuGn",
        fmt=".3f",
        linewidths=0.5,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "R orthologs - max(0, R all non-orthologs)"},
    )
    plt.ylabel("Tissue")
    plt.xlabel("Normalization method")
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def main() -> Path:
    args = parse_args()

    prepare_output_directories()

    logger.info("Loading raw expression dataset: %s", args.input)
    raw_dataframe = pd.read_csv(args.input, sep="\t")

    feature_columns = get_feature_columns(raw_dataframe)
    if not feature_columns:
        raise ValueError("No TPM columns detected. Expected columns like <tissue>_tpm_<species>.")

    tissues = get_tissues(feature_columns)
    logger.info("Detected %d TPM columns across %d tissues.", len(feature_columns), len(tissues))

    methods = resolve_methods(args.methods)
    method_names_selected = [method.__name__ for method in methods]

    logger.info("Generating all non-orthologous human x mouse pairs.")
    nonortholog_dataframe = generate_all_nonortholog_dataframe(raw_dataframe)
    nonortholog_dataframe.to_csv(
        Path("Intermediate_Datasets") / "all_nonorthologs.tsv",
        sep="\t",
        index=False,
    )

    logger.info("Applying normalization methods to orthologous dataset.")
    ortholog_dataframes, ortholog_method_names = apply_and_save_normalizations(
        dataframe=raw_dataframe,
        feature_columns=feature_columns,
        methods=methods,
        output_directory=Path("Orthologs"),
    )

    logger.info("Applying normalization methods to all non-orthologous pairs.")
    nonortholog_dataframes, nonortholog_method_names = apply_and_save_normalizations(
        dataframe=nonortholog_dataframe,
        feature_columns=feature_columns,
        methods=methods,
        output_directory=Path("NonOrthologs"),
    )

    logger.info("Computing normalization correlation statistics.")
    stats_orthologs = compst.compute_stats(
        ortholog_dataframes,
        ortholog_method_names,
        feature_columns,
        tissues,
        "Orthologs",
    )

    stats_nonorthologs = compst.compute_stats(
        nonortholog_dataframes,
        nonortholog_method_names,
        feature_columns,
        tissues,
        "NonOrthologs",
    )

    stats_all = merge_and_increment(
        stats_orthologs=stats_orthologs,
        stats_nonorthologs=stats_nonorthologs,
    )

    stats_path = Path("Normalization") / "stats.tsv"
    stats_all.to_csv(stats_path, sep="\t", index=False)

    best_method, ranking = select_best_method(
        stats_dataframe=stats_all,
        tissue=args.tissue,
        selection_metric=args.selection_metric,
    )

    logger.info("Best normalization method: %s", best_method)

    ranking_path = Path("Normalization") / "normalization_method_ranking.tsv"
    ranking.to_csv(ranking_path, sep="\t", index=False)

    best_method_path = Path("Normalization") / "best_method.txt"
    best_method_path.write_text(best_method + "\n")

    manifest = {
        "input": str(args.input),
        "dataset_name": args.dataset_name,
        "tissues": tissues,
        "feature_columns": feature_columns,
        "normalization_methods": method_names_selected,
        "nonortholog_strategy": "all human x mouse pairs minus true ortholog pairs",
        "increment_formula": "R_orthologs - max(0, R_all_nonorthologs)",
        "selection_tissue": args.tissue,
        "selection_metric": args.selection_metric,
        "best_method": best_method,
        "output_policy": "process-local outputs only; Nextflow publishes declared outputs",
    }

    manifest_path = Path("Normalization") / "normalization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    logger.info("Generating normalization plots.")
    plot_scatter_grid(
        normalized_dataframes=ortholog_dataframes,
        method_names=ortholog_method_names,
        stats_dataframe=stats_orthologs,
        tissues=tissues,
        output_path=Path("Normalization") / "orthologs_scatter.svg",
        title_prefix="Orthologs",
    )

    plot_scatter_grid(
        normalized_dataframes=nonortholog_dataframes,
        method_names=nonortholog_method_names,
        stats_dataframe=stats_nonorthologs,
        tissues=tissues,
        output_path=Path("Normalization") / "nonorthologs_scatter.svg",
        title_prefix="All non-orthologs",
    )

    plot_increment_heatmap(
        stats_dataframe=stats_all,
        ranking=ranking,
        output_path=Path("Normalization") / "increment_pearson_heatmap.svg",
    )

    logger.info("Normalization outputs generated in process working directory.")

    return best_method_path


if __name__ == "__main__":
    main()
