#!/usr/bin/env python3
"""Initial data exploration plots for paired human and mouse expression datasets.

Main outputs:
- pca_by_tissue.png
- pca_species_and_arrows.png
- pca_scores.tsv
- pca_loadings.tsv
- exploration_manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SPECIES_NAMES = ["human", "mouse"]
DEFAULT_OUTPUT_DIRECTORY = Path("results")
STALE_PLOT_FILENAMES = [
    "pca_species_and_loadings.png",
    "pca_biplot.png",
    "pca_loadings_weights.png",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input TSV with paired human and mouse expression columns.",
    )

    parser.add_argument(
        "--name",
        "--dataset-name",
        dest="dataset_name",
        default=None,
        help="Dataset name used for the output folder. Defaults to the input filename stem.",
    )

    parser.add_argument(
        "--output-dir",
        "--outdir",
        dest="output_dir",
        default=DEFAULT_OUTPUT_DIRECTORY,
        type=Path,
        help="Root output directory.",
    )

    parser.add_argument(
        "--expression-unit",
        default="tpm",
        help="Expression unit in columns named <tissue>_<expression_unit>_<species>.",
    )

    parser.add_argument(
        "--dpi",
        default=300,
        type=int,
        help="Resolution for saved PNG figures.",
    )

    parser.add_argument(
        "--no-log-transform",
        action="store_true",
        help="Disable log2(x + 1) transformation before principal component analysis.",
    )

    parser.add_argument(
        "--loading-arrow-scale",
        default=3.5,
        type=float,
        help="Scale factor for loading arrows in the biplot.",
    )

    return parser.parse_args()




def get_expression_columns(
    dataframe: pd.DataFrame,
    species: str,
    expression_unit: str,
) -> list[str]:
    """Return expression columns for one species."""

    suffix = f"_{expression_unit}_{species}"
    expression_columns = [
        column
        for column in dataframe.columns
        if column.endswith(suffix)
    ]

    if not expression_columns:
        raise ValueError(
            f"No expression columns found for species={species!r}. "
            f"Expected columns ending in {suffix!r}."
        )

    return expression_columns


def get_tissue_name(
    expression_column: str,
    species: str,
    expression_unit: str,
) -> str:
    """Extract tissue name from an expression column."""

    suffix = f"_{expression_unit}_{species}"
    return expression_column[: -len(suffix)]


def build_species_dataframe(
    dataframe: pd.DataFrame,
    species: str,
    gene_name_column: str,
    expression_unit: str,
) -> pd.DataFrame:
    """Build one gene-level dataframe for one species."""

    gene_id_column = f"gene_id_{species}"

    if gene_id_column not in dataframe.columns:
        raise ValueError(f"Missing gene identifier column: {gene_id_column}")

    expression_columns = get_expression_columns(
        dataframe=dataframe,
        species=species,
        expression_unit=expression_unit,
    )

    selected_columns = [gene_id_column, gene_name_column, *expression_columns]

    output_dataframe = dataframe[selected_columns].copy()
    output_dataframe = output_dataframe.rename(
        columns={
            gene_id_column: "gene_id",
            gene_name_column: "gene_name",
            **{
                column: get_tissue_name(
                    expression_column=column,
                    species=species,
                    expression_unit=expression_unit,
                )
                for column in expression_columns
            },
        }
    )

    output_dataframe["species"] = species

    return output_dataframe


def build_gene_level_dataframe(
    dataframe: pd.DataFrame,
    expression_unit: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Convert paired ortholog rows into one row per gene."""

    gene_name_column = "gene_name"

    species_dataframes = [
        build_species_dataframe(
            dataframe=dataframe,
            species=species,
            gene_name_column=gene_name_column,
            expression_unit=expression_unit,
        )
        for species in SPECIES_NAMES
    ]

    long_dataframe = pd.concat(species_dataframes, ignore_index=True)

    identifier_columns = ["gene_id", "gene_name", "species"]
    tissue_columns = [
        column
        for column in long_dataframe.columns
        if column not in identifier_columns
    ]

    long_dataframe = long_dataframe[identifier_columns + tissue_columns].copy()

    if long_dataframe[tissue_columns].isna().any().any():
        missing_count = int(long_dataframe[tissue_columns].isna().sum().sum())
        raise ValueError(f"Expression table contains {missing_count} missing expression values.")

    return long_dataframe, tissue_columns


def run_principal_component_analysis(
    gene_level_dataframe: pd.DataFrame,
    tissue_columns: list[str],
    use_log_transform: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, PCA]:
    """Run two-component principal component analysis."""

    analysis_dataframe = gene_level_dataframe.copy()

    expression_values = analysis_dataframe[tissue_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if expression_values.isna().any().any():
        missing_count = int(expression_values.isna().sum().sum())
        raise ValueError(f"Expression table contains {missing_count} non-numeric values.")

    if use_log_transform:
        if (expression_values < 0).any().any():
            raise ValueError("Cannot apply log2(x + 1) because negative values were found.")

        expression_values = np.log2(expression_values + 1)

    scaled_values = StandardScaler().fit_transform(expression_values)

    principal_component_analysis = PCA(n_components=2)
    components = principal_component_analysis.fit_transform(scaled_values)

    variance_explained = principal_component_analysis.explained_variance_ratio_ * 100
    first_component_label = f"PC1 ({variance_explained[0]:.2f}%)"
    second_component_label = f"PC2 ({variance_explained[1]:.2f}%)"

    scores_dataframe = pd.DataFrame(
        data=components,
        columns=[first_component_label, second_component_label],
    )

    for column in gene_level_dataframe.columns:
        scores_dataframe[column] = gene_level_dataframe[column].values

    loadings_dataframe = pd.DataFrame(
        data=principal_component_analysis.components_.T,
        columns=[first_component_label, second_component_label],
        index=tissue_columns,
    )

    loadings_dataframe.index.name = "tissue"
    loadings_dataframe = loadings_dataframe.reset_index()

    return scores_dataframe, loadings_dataframe, principal_component_analysis


def set_plot_style() -> None:
    """Set common figure typography without grid lines."""

    sns.set_theme(
        style="white",
        rc={
            "axes.grid": False,
            "grid.alpha": 0.0,
        },
    )

    plt.rcParams.update(
        {
            "axes.grid": False,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _plot_species_pca(
    scores_dataframe: pd.DataFrame,
    axis: plt.Axes,
    first_component_label: str,
    second_component_label: str,
    title: str,
) -> None:
    """Draw a species-colored principal component analysis scatter plot."""

    sns.scatterplot(
        data=scores_dataframe,
        x=first_component_label,
        y=second_component_label,
        hue="species",
        palette="viridis",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
        ax=axis,
    )

    axis.set_title(title)
    axis.set_xlabel(first_component_label)
    axis.set_ylabel(second_component_label)
    axis.legend(title="Species")
    axis.grid(visible=False)
    sns.despine(ax=axis)


def plot_species_and_arrows_grid(
    scores_dataframe: pd.DataFrame,
    loadings_dataframe: pd.DataFrame,
    output_path: Path,
    dpi: int,
    loading_arrow_scale: float,
) -> None:
    """Plot species-colored PCA alone and with tissue loading arrows."""

    first_component_label, second_component_label = list(scores_dataframe.columns[:2])

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(11, 4.5),
    )

    _plot_species_pca(
        scores_dataframe=scores_dataframe,
        axis=axes[0],
        first_component_label=first_component_label,
        second_component_label=second_component_label,
        title="PCA colored by species",
    )

    _plot_species_pca(
        scores_dataframe=scores_dataframe,
        axis=axes[1],
        first_component_label=first_component_label,
        second_component_label=second_component_label,
        title="PCA colored by species with tissue loadings",
    )

    for _, loading_row in loadings_dataframe.iterrows():
        x_arrow = loading_row[first_component_label] * loading_arrow_scale
        y_arrow = loading_row[second_component_label] * loading_arrow_scale

        axes[1].arrow(
            0,
            0,
            x_arrow,
            y_arrow,
            color="black",
            alpha=0.9,
            head_width=0.15,
            linewidth=1.5,
            length_includes_head=True,
        )

        axes[1].text(
            x_arrow * 1.15,
            y_arrow * 1.15,
            loading_row["tissue"],
            color="black",
            fontweight="bold",
            fontsize=10,
            ha="center",
            va="center",
        )

    axes[1].axvline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    axes[1].grid(visible=False)
    sns.despine(ax=axes[1])

    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_tissue_principal_component_grid(
    scores_dataframe: pd.DataFrame,
    tissue_columns: list[str],
    output_path: Path,
    dpi: int,
    number_of_columns: int = 2,
) -> None:
    """Plot principal component analysis colored by each tissue."""

    component_columns = list(scores_dataframe.columns[:2])
    first_component_label, second_component_label = component_columns

    number_of_features = len(tissue_columns)
    number_of_rows = math.ceil(number_of_features / number_of_columns)

    figure, axes = plt.subplots(
        number_of_rows,
        number_of_columns,
        figsize=(number_of_columns * 5.5, number_of_rows * 4),
        squeeze=False,
    )

    flattened_axes = axes.flatten()

    for index, tissue_column in enumerate(tissue_columns):
        axis = flattened_axes[index]

        scatter = axis.scatter(
            scores_dataframe[first_component_label],
            scores_dataframe[second_component_label],
            c=scores_dataframe[tissue_column],
            cmap="viridis",
            alpha=0.8,
            edgecolors="white",
            linewidth=0.3,
        )

        colorbar = figure.colorbar(scatter, ax=axis)
        colorbar.set_label(f"expression value in {tissue_column}", fontsize=12)
        colorbar.ax.tick_params(labelsize=10)

        axis.set_xlabel(first_component_label)
        axis.set_ylabel(second_component_label)
        axis.set_title(f"PCA colored by {tissue_column}")
        axis.grid(visible=False)
        sns.despine(ax=axis)

    for empty_axis in flattened_axes[number_of_features:]:
        figure.delaxes(empty_axis)

    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)



def remove_stale_plot_outputs(output_directory: Path) -> None:
    """Remove old plot names that can make reruns look unchanged."""

    for filename in STALE_PLOT_FILENAMES:
        stale_path = output_directory / filename
        if stale_path.exists():
            stale_path.unlink()


def write_manifest(
    output_path: Path,
    args: argparse.Namespace,
    output_directory: Path,
    gene_level_dataframe: pd.DataFrame,
    tissue_columns: list[str],
    scores_dataframe: pd.DataFrame,
) -> None:
    """Write a reproducibility manifest."""

    manifest = {
        "script_version": SCRIPT_VERSION,
        "input": str(args.input),
        "dataset_name": args.dataset_name,
        "output_directory": str(output_directory),
        "expression_unit": args.expression_unit,
        "log2_x_plus_1_transform": not args.no_log_transform,
        "n_input_rows": int(len(gene_level_dataframe) // len(SPECIES_NAMES)),
        "n_gene_rows_after_species_split": int(len(gene_level_dataframe)),
        "n_tissues": int(len(tissue_columns)),
        "tissues": tissue_columns,
        "principal_component_columns": list(scores_dataframe.columns[:2]),
        "outputs": {
            "pca_scores": "pca_scores.tsv",
            "pca_loadings": "pca_loadings.tsv",
            "tissue_grid_plot": "pca_by_tissue.png",
            "species_and_arrows_plot": "pca_species_and_arrows.png",
        },
    }

    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> Path:
    """Run data exploration."""

    configure_logging()
    set_plot_style()

    args = parse_args()

    if args.dataset_name is None:
        args.dataset_name = args.input.stem

    output_directory = args.output_dir / args.dataset_name / "Data_Exploration"
    output_directory.mkdir(parents=True, exist_ok=True)
    remove_stale_plot_outputs(output_directory)

    LOGGER.info("Loading input dataset: %s", args.input)
    input_dataframe = pd.read_csv(args.input, sep="\t")

    LOGGER.info("Building one-row-per-gene expression table.")
    gene_level_dataframe, tissue_columns = build_gene_level_dataframe(
        dataframe=input_dataframe,
        expression_unit=args.expression_unit,
    )

    gene_level_dataframe.to_csv(
        output_directory / "gene_level_expression.tsv",
        sep="\t",
        index=False,
    )

    LOGGER.info("Running principal component analysis with %d tissues.", len(tissue_columns))
    scores_dataframe, loadings_dataframe, _ = run_principal_component_analysis(
        gene_level_dataframe=gene_level_dataframe,
        tissue_columns=tissue_columns,
        use_log_transform=not args.no_log_transform,
    )

    scores_dataframe.to_csv(
        output_directory / "pca_scores.tsv",
        sep="\t",
        index=False,
    )

    loadings_dataframe.to_csv(
        output_directory / "pca_loadings.tsv",
        sep="\t",
        index=False,
    )

    LOGGER.info("Saving tissue-colored principal component grid.")
    plot_tissue_principal_component_grid(
        scores_dataframe=scores_dataframe,
        tissue_columns=tissue_columns,
        output_path=output_directory / "pca_by_tissue.png",
        dpi=args.dpi,
    )

    LOGGER.info("Saving species-colored principal component grid with loading arrows.")
    plot_species_and_arrows_grid(
        scores_dataframe=scores_dataframe,
        loadings_dataframe=loadings_dataframe,
        output_path=output_directory / "pca_species_and_arrows.png",
        dpi=args.dpi,
        loading_arrow_scale=args.loading_arrow_scale,
    )

    write_manifest(
        output_path=output_directory / "exploration_manifest.json",
        args=args,
        output_directory=output_directory,
        gene_level_dataframe=gene_level_dataframe,
        tissue_columns=tissue_columns,
        scores_dataframe=scores_dataframe,
    )

    LOGGER.info("Data exploration outputs saved to %s", output_directory)

    return output_directory


if __name__ == "__main__":
    main()
