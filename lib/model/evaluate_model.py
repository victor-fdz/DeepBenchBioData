#!/usr/bin/env python3
"""Evaluate Siamese DNA embedding models."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.model.datasets import DNAPairDatasetWithMeta
from lib.model.model import SiameseCNN

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import ttest_ind
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="\033[1;32m%(levelname)s\033[0m | "
           "\033[1;36m%(name)s\033[0m | %(message)s",
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results")

DEFAULT_BATCH_SIZE = 32
DEFAULT_DROPOUT = 0
DEFAULT_KERNEL_SIZE_SMALL = 6
DEFAULT_KERNEL_SIZE_MEDIUM = 10
DEFAULT_KERNEL_SIZE_LARGE = 18
DEFAULT_ATTENTION_HEADS = 4
DEFAULT_EMBEDDING_DIM = 16


# -----------------------------
# CLI
# -----------------------------
def parse_args(cli_args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)

    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Optional JSON model config. Defaults to model_config.json next to --weights when present.",
    )

    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--small-kernel-size", type=int, default=DEFAULT_KERNEL_SIZE_SMALL)
    parser.add_argument("--medium-kernel-size", type=int, default=DEFAULT_KERNEL_SIZE_MEDIUM)
    parser.add_argument("--large-kernel-size", type=int, default=DEFAULT_KERNEL_SIZE_LARGE)
    parser.add_argument("--attention-heads", type=int, default=DEFAULT_ATTENTION_HEADS)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)

    return parser.parse_args(cli_args)


def load_model_config(args: argparse.Namespace) -> dict:
    """Load model config, preferring JSON saved during training."""

    default_config_path = args.weights.parent / "model_config.json"
    config_path = args.model_config or default_config_path

    fallback_config = {
        "dropout_rate": args.dropout,
        "small_kernel_size": args.small_kernel_size,
        "medium_kernel_size": args.medium_kernel_size,
        "large_kernel_size": args.large_kernel_size,
        "attention_heads": args.attention_heads,
        "embedding_dim": args.embedding_dim,
    }

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            loaded_config = json.load(handle)

        model_config = {
            "dropout_rate": loaded_config.get("dropout_rate", fallback_config["dropout_rate"]),
            "small_kernel_size": loaded_config.get("small_kernel_size", fallback_config["small_kernel_size"]),
            "medium_kernel_size": loaded_config.get("medium_kernel_size", fallback_config["medium_kernel_size"]),
            "large_kernel_size": loaded_config.get("large_kernel_size", fallback_config["large_kernel_size"]),
            "attention_heads": loaded_config.get("attention_heads", fallback_config["attention_heads"]),
            "embedding_dim": loaded_config.get("embedding_dim", fallback_config["embedding_dim"]),
        }

        logger.info("Loaded model config from %s", config_path)
        return model_config

    logger.info("No model config found at %s. Using CLI/default model parameters.", config_path)
    return fallback_config


# -----------------------------
# Inference
# -----------------------------
def generate_embeddings(
    model: SiameseCNN,
    loader: DataLoader,
) -> pd.DataFrame:
    """Generate embedding cosine similarities."""

    logger.info("Running model inference.")

    model.eval()
    rows = []

    with torch.no_grad():
        for seq1, seq2, labels, pair_row_id, gene_id_1, gene_id_2 in loader:
            out1, out2 = model(seq1, seq2)

            embedding_cosine_similarity = F.cosine_similarity(
                out1,
                out2,
            ).cpu().numpy()

            for row_id, first_gene_id, second_gene_id, similarity in zip(
                pair_row_id,
                gene_id_1,
                gene_id_2,
                embedding_cosine_similarity,
            ):
                rows.append(
                    {
                        "pair_row_id": int(row_id),
                        "gene_id_1": first_gene_id,
                        "gene_id_2": second_gene_id,
                        "embedding_cosine_sim": float(similarity),
                    }
                )

    return pd.DataFrame(rows)


# -----------------------------
# Metrics
# -----------------------------
def compute_correlations(
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """Compute correlation statistics."""

    if len(x) < 2:
        raise ValueError("At least two rows are required to compute correlations.")

    if np.unique(x).size < 2:
        raise ValueError("Embedding similarities are constant. Cannot compute correlations.")

    if np.unique(y).size < 2:
        raise ValueError("Expression similarities are constant. Cannot compute correlations.")

    pearson_r, _ = pearsonr(x, y)
    spearman_r, _ = spearmanr(x, y)
    kendall_r, _ = kendalltau(x, y)

    return {
        "pearson": float(pearson_r),
        "spearman": float(spearman_r),
        "kendall": float(kendall_r),
    }


# -----------------------------
# Plotting
# -----------------------------
def plot_embedding_vs_expression(
    df: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    """Generate embedding-vs-expression scatter plot with label-wise distances."""

    x = df["embedding_cosine_sim"].to_numpy(dtype=float)
    y = df[metric].to_numpy(dtype=float)

    stats = compute_correlations(x, y)

    label_column = None
    if "original_label" in df.columns:
        label_column = "original_label"
    elif "label" in df.columns:
        label_column = "label"

    distance_text = "Mean distance: unavailable"

    if label_column is not None:
        plot_dataframe = df.copy()

        plot_dataframe["label_normalized"] = plot_dataframe[label_column].replace(
            {
                "1": "P",
                1: "P",
                1.0: "P",
                "0": "N",
                0: "N",
                0.0: "N",
            }
        )

        plot_dataframe["embedding_distance"] = (
            1.0 - plot_dataframe["embedding_cosine_sim"].astype(float)
        )

        mean_distances = (
            plot_dataframe
            .groupby("label_normalized")["embedding_distance"]
            .mean()
            .to_dict()
        )

        distance_text = (
            "Mean distance: "
            f"P={mean_distances.get('P', np.nan):.3f} | "
            f"N={mean_distances.get('N', np.nan):.3f} | "
            f"U={mean_distances.get('U', np.nan):.3f}"
        )

    plt.figure(figsize=(3.8, 3.1))

    if label_column is None:
        plt.scatter(
            x,
            y,
            alpha=0.18,
            s=4,
        )
    else:
        plot_specs = [
            ("U", "blue", "Undefined"),
            ("P", "green", "Positive"),
            ("N", "red", "Negative"),
        ]

        for label_value, colour, label_name in plot_specs:
            subset = plot_dataframe[
                plot_dataframe["label_normalized"] == label_value
            ]

            if subset.empty:
                continue

            plt.scatter(
                subset["embedding_cosine_sim"],
                subset[metric],
                alpha=0.18,
                s=4,
                c=colour,
                label=f"{label_name} (n={len(subset)})",
            )

        plt.legend(
            fontsize=7,
            frameon=False,
        )

    plt.xlabel(
        "Embedding cosine similarity",
        fontsize=8,
    )

    plt.ylabel(
        f"Expression similarity ({metric})",
        fontsize=8,
    )

    plt.title(
        (
            "Embedding vs Expression Similarity\n"
            f"Pearson={stats['pearson']:.3f} | "
            f"Spearman={stats['spearman']:.3f} | "
            f"Kendall={stats['kendall']:.3f}\n"
            f"{distance_text}"
        ),
        fontsize=8,
    )

    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        "Evaluation plot saved to %s",
        output_path,
    )
    
    
def _normalise_distance_labels(dataframe: pd.DataFrame) -> pd.Series:
    """Return labels normalized to P/N/U when available."""

    label_column = "original_label" if "original_label" in dataframe.columns else "label"

    return dataframe[label_column].replace(
        {
            "1": "P",
            1: "P",
            1.0: "P",
            "0": "N",
            0: "N",
            0.0: "N",
        }
    )


def _prepare_distance_scatter_dataframe(
    dataframe: pd.DataFrame,
    stage: str,
) -> pd.DataFrame:
    """Normalize labels and keep P/N embedding distances for one stage."""

    if "embedding_distance" not in dataframe.columns:
        raise ValueError("Missing required column: embedding_distance")

    prepared = dataframe.copy()
    prepared["label_normalized"] = _normalise_distance_labels(prepared)
    prepared = prepared[prepared["label_normalized"].isin(["P", "N"])].copy()

    prepared["stage"] = stage
    prepared["embedding_distance"] = pd.to_numeric(
        prepared["embedding_distance"],
        errors="coerce",
    )
    prepared = prepared.dropna(subset=["embedding_distance"])

    return prepared[["stage", "label_normalized", "embedding_distance"]].copy()


def compute_embedding_distance_statistics(
    distance_dataframe: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Run one-sided Student t-tests: negative distance > positive distance."""

    rows = []

    
    stage_dataframe = distance_dataframe[distance_dataframe["stage"] == "Trained"]
    positive = stage_dataframe.loc[
        stage_dataframe["label_normalized"] == "P",
        "embedding_distance",
    ].to_numpy(dtype=float)
    negative = stage_dataframe.loc[
        stage_dataframe["label_normalized"] == "N",
        "embedding_distance",
    ].to_numpy(dtype=float)

    if len(positive) < 2 or len(negative) < 2:
        statistic = np.nan
        p_value = np.nan
    else:
        test_result = ttest_ind(
            positive,
            negative,
            equal_var=True,
            alternative="less",
            nan_policy="omit",
        )
        statistic = float(test_result.statistic)
        p_value = float(test_result.pvalue)

    rows.append(
        {
            "stage": "Trained",
            "test": "Student t-test",
            "alternative": "mean_negative_distance > mean_positive_distance",
            "n_positive": int(len(positive)),
            "n_negative": int(len(negative)),
            "mean_positive_distance": float(np.mean(positive)) if len(positive) else np.nan,
            "mean_negative_distance": float(np.mean(negative)) if len(negative) else np.nan,
            "mean_difference_negative_minus_positive": (
                float(np.mean(negative) - np.mean(positive))
                if len(positive) and len(negative)
                else np.nan
            ),
            "t_statistic_positive_less_negative": statistic,
            "p_value_one_sided": p_value,
        }
    )

    statistics_dataframe = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    statistics_dataframe.to_csv(output_path, sep="\t", index=False)

    logger.info("Embedding distance statistical tests saved to %s", output_path)

    return statistics_dataframe

def scatter_embedding_distance(
    trained_dataframe: pd.DataFrame,
    output_plot_path: Path,
    output_stats_path: Path,
) -> None:
    """Plot trained-model embedding distances for positive and negative test pairs."""

    plot_dataframe = _prepare_distance_scatter_dataframe(
        trained_dataframe,
        "Trained",
    )

    compute_embedding_distance_statistics(
        distance_dataframe=plot_dataframe,
        output_path=output_stats_path,
    )

    plot_order = ["P", "N"]

    x_positions = {
        "P": 1,
        "N": 2,
    }

    x_labels = ["Positive", "Negative"]

    label_colors = {
        "P": "green",
        "N": "red",
    }

    rng = np.random.default_rng(42)

    figure, axis = plt.subplots(figsize=(2.7, 2.6))

    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")

    for label in plot_order:
        values = plot_dataframe.loc[
            plot_dataframe["label_normalized"] == label,
            "embedding_distance",
        ].to_numpy(dtype=float)

        if values.size == 0:
            continue

        x_center = x_positions[label]
        x_jitter = rng.normal(
            loc=0,
            scale=0.045,
            size=len(values),
        )

        axis.scatter(
            np.full(len(values), x_center) + x_jitter,
            values,
            s=7,
            alpha=0.28,
            color=label_colors[label],
            edgecolors="none",
            rasterized=True,
        )

        axis.hlines(
            float(np.mean(values)),
            x_center - 0.22,
            x_center + 0.22,
            color="black",
            linewidth=1.4,
            zorder=4,
        )

    axis.set_xticks([1, 2])
    axis.set_xticklabels(x_labels, fontsize=10)

    axis.set_ylabel("Embedding distance", fontsize=12)
    axis.set_title("Test embedding distances", fontsize=12)

    axis.tick_params(axis="both", labelsize=10)

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    axis.grid(
        True,
        axis="y",
        linestyle="--",
        linewidth=0.55,
        color="0.82",
        alpha=0.65,
    )

    axis.set_axisbelow(True)

    figure.tight_layout()

    output_plot_path.parent.mkdir(parents=True, exist_ok=True)

    figure.savefig(
        output_plot_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(figure)

    logger.info("Trained embedding distance scatter plot saved to %s", output_plot_path)

def _normalise_pair_labels(dataframe: pd.DataFrame) -> pd.Series | None:
    """Return P/N labels when a label column is available."""

    label_column = None
    if "original_label" in dataframe.columns:
        label_column = "original_label"
    elif "label" in dataframe.columns:
        label_column = "label"

    if label_column is None:
        return None

    return dataframe[label_column].replace(
        {
            "1": "P",
            1: "P",
            1.0: "P",
            "0": "N",
            0: "N",
            0.0: "N",
        }
    )


def select_consistency_gene_columns(dataframe: pd.DataFrame) -> tuple[str, str]:
    """Select the best available human and mouse gene columns for consistency."""

    candidate_column_pairs = [
        ("gene_name_human", "gene_name_mouse"),
        ("human_gene_name", "mouse_gene_name"),
        ("gene_name_1", "gene_name_2"),
        ("gene_id_human", "gene_id_mouse"),
        ("human_gene_id", "mouse_gene_id"),
        ("gene_id_1", "gene_id_2"),
    ]

    for human_gene_column, mouse_gene_column in candidate_column_pairs:
        if human_gene_column in dataframe.columns and mouse_gene_column in dataframe.columns:
            return human_gene_column, mouse_gene_column

    raise ValueError(
        "Could not find suitable human/mouse gene columns for triplet consistency. "
        "Expected one of: gene_name_human/gene_name_mouse, "
        "gene_id_human/gene_id_mouse, or gene_id_1/gene_id_2."
    )


def build_reference_gene_pairs(
    dataframe: pd.DataFrame,
    human_gene_column: str,
    mouse_gene_column: str,
) -> tuple[pd.DataFrame, str]:
    """Build reference one-to-one pairs for triplet consistency."""

    labels = _normalise_pair_labels(dataframe)

    if labels is not None:
        positive_reference_pairs = (
            dataframe.loc[
                labels == "P",
                [human_gene_column, mouse_gene_column],
            ]
            .dropna()
            .drop_duplicates()
            .rename(
                columns={
                    human_gene_column: "reference_human_gene",
                    mouse_gene_column: "reference_mouse_gene",
                }
            )
        )

        if not positive_reference_pairs.empty:
            return positive_reference_pairs, "positive_labeled_pairs"

    reference_human_genes = (
        dataframe[[human_gene_column]]
        .drop_duplicates()
        .assign(
            ortholog_key=lambda table: table[human_gene_column]
            .astype(str)
            .str.lower()
        )
    )

    reference_mouse_genes = (
        dataframe[[mouse_gene_column]]
        .drop_duplicates()
        .assign(
            ortholog_key=lambda table: table[mouse_gene_column]
            .astype(str)
            .str.lower()
        )
    )

    reference_gene_pairs = (
        reference_human_genes
        .merge(reference_mouse_genes, on="ortholog_key", how="inner")
        .rename(
            columns={
                human_gene_column: "reference_human_gene",
                mouse_gene_column: "reference_mouse_gene",
            }
        )
        [["reference_human_gene", "reference_mouse_gene"]]
        .drop_duplicates()
    )

    return reference_gene_pairs, "matching_lowercase_gene_names"


def add_triplet_consistency(
    dataframe: pd.DataFrame,
    human_gene_column: str,
    mouse_gene_column: str,
    expression_similarity_column: str,
    embedding_similarity_column: str = "embedding_cosine_sim",
    exclude_pair_genes_as_references: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Add triplet consistency for every human-mouse pair.

    A reference triplet is consistent when expression similarity and embedding
    similarity choose the same closest side around a reference ortholog pair C:
    sim(A_human, C_mouse) versus sim(C_human, B_mouse).
    """

    required_columns = {
        human_gene_column,
        mouse_gene_column,
        expression_similarity_column,
        embedding_similarity_column,
    }
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"Missing columns for triplet consistency: {sorted(missing_columns)}"
        )

    working_dataframe = dataframe.copy()

    reference_gene_pairs, reference_source = build_reference_gene_pairs(
        dataframe=working_dataframe,
        human_gene_column=human_gene_column,
        mouse_gene_column=mouse_gene_column,
    )

    if reference_gene_pairs.empty:
        working_dataframe["consistent_triplet_count"] = 0
        working_dataframe["total_triplet_count"] = 0
        working_dataframe["triplet_consistency"] = np.nan

        metadata = {
            "computed": False,
            "reason": "No reference one-to-one pairs were found.",
            "reference_source": reference_source,
            "reference_pair_count": 0,
            "human_gene_column": human_gene_column,
            "mouse_gene_column": mouse_gene_column,
            "expression_similarity_column": expression_similarity_column,
            "embedding_similarity_column": embedding_similarity_column,
        }
        return working_dataframe, metadata

    expression_similarity_matrix = working_dataframe.pivot_table(
        index=human_gene_column,
        columns=mouse_gene_column,
        values=expression_similarity_column,
        aggfunc="mean",
    )

    embedding_similarity_matrix = working_dataframe.pivot_table(
        index=human_gene_column,
        columns=mouse_gene_column,
        values=embedding_similarity_column,
        aggfunc="mean",
    )

    consistency_rows = []
    skipped_triplet_count = 0
    tied_triplet_count = 0

    for pair in working_dataframe[[human_gene_column, mouse_gene_column]].itertuples(index=False):
        pair_human_gene = pair[0]
        pair_mouse_gene = pair[1]

        consistent_triplet_count = 0
        total_triplet_count = 0

        for reference in reference_gene_pairs.itertuples(index=False):
            reference_human_gene = reference.reference_human_gene
            reference_mouse_gene = reference.reference_mouse_gene

            if exclude_pair_genes_as_references:
                if reference_human_gene == pair_human_gene:
                    continue
                if reference_mouse_gene == pair_mouse_gene:
                    continue

            try:
                expression_similarity_to_pair_human = expression_similarity_matrix.at[
                    pair_human_gene,
                    reference_mouse_gene,
                ]
                expression_similarity_to_pair_mouse = expression_similarity_matrix.at[
                    reference_human_gene,
                    pair_mouse_gene,
                ]
                embedding_similarity_to_pair_human = embedding_similarity_matrix.at[
                    pair_human_gene,
                    reference_mouse_gene,
                ]
                embedding_similarity_to_pair_mouse = embedding_similarity_matrix.at[
                    reference_human_gene,
                    pair_mouse_gene,
                ]
            except KeyError:
                skipped_triplet_count += 1
                continue

            values = [
                expression_similarity_to_pair_human,
                expression_similarity_to_pair_mouse,
                embedding_similarity_to_pair_human,
                embedding_similarity_to_pair_mouse,
            ]

            if any(pd.isna(value) for value in values):
                skipped_triplet_count += 1
                continue

            expression_difference = (
                expression_similarity_to_pair_human
                - expression_similarity_to_pair_mouse
            )
            embedding_difference = (
                embedding_similarity_to_pair_human
                - embedding_similarity_to_pair_mouse
            )

            if expression_difference == 0 or embedding_difference == 0:
                tied_triplet_count += 1
                continue

            total_triplet_count += 1

            if np.sign(expression_difference) == np.sign(embedding_difference):
                consistent_triplet_count += 1

        triplet_consistency = (
            consistent_triplet_count / total_triplet_count
            if total_triplet_count > 0
            else np.nan
        )

        consistency_rows.append(
            {
                human_gene_column: pair_human_gene,
                mouse_gene_column: pair_mouse_gene,
                "consistent_triplet_count": consistent_triplet_count,
                "total_triplet_count": total_triplet_count,
                "triplet_consistency": triplet_consistency,
            }
        )

    consistency_dataframe = pd.DataFrame(consistency_rows)

    scored_dataframe = working_dataframe.merge(
        consistency_dataframe,
        on=[human_gene_column, mouse_gene_column],
        how="left",
    )

    total_valid_triplets = int(scored_dataframe["total_triplet_count"].sum())
    total_consistent_triplets = int(scored_dataframe["consistent_triplet_count"].sum())

    metadata = {
        "computed": True,
        "reference_source": reference_source,
        "reference_pair_count": int(len(reference_gene_pairs)),
        "human_gene_column": human_gene_column,
        "mouse_gene_column": mouse_gene_column,
        "expression_similarity_column": expression_similarity_column,
        "embedding_similarity_column": embedding_similarity_column,
        "evaluated_pair_count": int(len(scored_dataframe)),
        "pairs_with_valid_triplets": int(scored_dataframe["triplet_consistency"].notna().sum()),
        "total_valid_triplets": total_valid_triplets,
        "total_consistent_triplets": total_consistent_triplets,
        "skipped_missing_triplets": int(skipped_triplet_count),
        "skipped_tied_triplets": int(tied_triplet_count),
        "global_triplet_consistency": (
            total_consistent_triplets / total_valid_triplets
            if total_valid_triplets > 0
            else np.nan
        ),
        "mean_pair_triplet_consistency": float(scored_dataframe["triplet_consistency"].mean()),
        "median_pair_triplet_consistency": float(scored_dataframe["triplet_consistency"].median()),
        "std_pair_triplet_consistency": float(scored_dataframe["triplet_consistency"].std()),
    }

    labels = _normalise_pair_labels(scored_dataframe)
    if labels is not None:
        scored_dataframe["label_normalized_for_consistency"] = labels
        for label_value, label_name in [("P", "positive"), ("N", "negative")]:
            label_subset = scored_dataframe[labels == label_value]
            metadata[f"mean_{label_name}_triplet_consistency"] = float(
                label_subset["triplet_consistency"].mean()
            )
            metadata[f"n_{label_name}_pairs_for_consistency"] = int(
                label_subset["triplet_consistency"].notna().sum()
            )

    return scored_dataframe, metadata


def save_triplet_consistency_summary(
    metadata: dict[str, object],
    output_path: Path,
) -> None:
    """Save triplet consistency summary as a human-readable text file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "Triplet consistency summary",
        "===========================",
        "",
        "Definition:",
        "For each pair (A_human, B_mouse), each reference ortholog C compares",
        "expression sim(A_human, C_mouse) vs sim(C_human, B_mouse) and",
        "embedding sim(A_human, C_mouse) vs sim(C_human, B_mouse).",
        "A triplet is consistent when both spaces choose the same closest side.",
        "",
    ]

    for key, value in metadata.items():
        lines.append(f"{key}: {value}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Triplet consistency summary saved to %s", output_path)

def build_target_dataframe(
    df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Build target columns for merging predictions by pair_row_id."""

    required_columns = {"pair_row_id", metric}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns in evaluation dataframe: {sorted(missing_columns)}")

    target_columns = ["pair_row_id", metric]

    if "original_label" in df.columns:
        target_columns.append("original_label")
    elif "label" in df.columns:
        target_columns.append("label")

    optional_metadata_columns = [
        "gene_name_human",
        "gene_name_mouse",
        "human_gene_name",
        "mouse_gene_name",
        "gene_name_1",
        "gene_name_2",
        "gene_id_human",
        "gene_id_mouse",
        "human_gene_id",
        "mouse_gene_id",
    ]

    for column in optional_metadata_columns:
        if column in df.columns and column not in target_columns:
            target_columns.append(column)

    return df[target_columns].copy()


# -----------------------------
# Main workflow
# -----------------------------
def main(cli_args: list[str] | None = None) -> Path:
    """Run evaluation workflow."""

    args = parse_args(cli_args)

    logger.info("Loading labeled dataframe.")
    df = pd.read_csv(args.pairs, sep="\t")

    loader = DataLoader(
        DNAPairDatasetWithMeta(df),
        batch_size=args.batch_size,
        shuffle=False,
    )

    logger.info("Loading trained model.")
    model_config = load_model_config(args)
    model = SiameseCNN(**model_config)

    model.load_state_dict(
        torch.load(
            args.weights,
            map_location="cpu",
        )
    )

    pred_df = generate_embeddings(model=model, loader=loader)
    target_df = build_target_dataframe(df=df, metric=args.metric)

    aligned_df = pred_df.merge(
        target_df,
        on="pair_row_id",
        how="inner",
    )

    if len(aligned_df) != len(pred_df):
        raise ValueError("Prediction-to-target merge changed the number of rows.")

    out_dir = OUTPUT_DIR / args.name / "Model"
    out_dir.mkdir(parents=True, exist_ok=True)

    aligned_df["embedding_distance"] = (
        1.0 - aligned_df["embedding_cosine_sim"].astype(float)
    )

    aligned_path = out_dir / f"{args.name}_embedding_similarity.tsv"
    aligned_df.to_csv(aligned_path, sep="\t", index=False)

    distance_path = out_dir / f"{args.name}_trained_test_embedding_distances.tsv"
    aligned_df.to_csv(
        distance_path,
        sep="\t",
        index=False,
    )


    logger.info("Similarity dataframe saved to %s", aligned_path)
    logger.info("Trained test embedding distances saved to %s", distance_path)

    consistency_human_gene_column, consistency_mouse_gene_column = select_consistency_gene_columns(
        aligned_df
    )

    consistency_dataframe, consistency_metadata = add_triplet_consistency(
        dataframe=aligned_df,
        human_gene_column=consistency_human_gene_column,
        mouse_gene_column=consistency_mouse_gene_column,
        expression_similarity_column=args.metric,
        embedding_similarity_column="embedding_cosine_sim",
    )

    consistency_table_path = out_dir / f"{args.name}_triplet_consistency.tsv"
    consistency_summary_path = out_dir / f"{args.name}_triplet_consistency_summary.txt"

    consistency_dataframe.to_csv(
        consistency_table_path,
        sep="	",
        index=False,
    )

    save_triplet_consistency_summary(
        metadata=consistency_metadata,
        output_path=consistency_summary_path,
    )

    logger.info("Triplet consistency table saved to %s", consistency_table_path)

    scatter_embedding_distance(
        trained_dataframe=aligned_df,
        output_plot_path=out_dir / f"{args.name}_test_embedding_distance_scatter.png",
        output_stats_path=out_dir / f"{args.name}_test_embedding_distance_statistics.tsv",
    )

    plot_embedding_vs_expression(
        df=aligned_df,
        metric=args.metric,
        output_path=out_dir / f"{args.name}_embedding_vs_expression.png",
    )

    return aligned_path


if __name__ == "__main__":
    main()
