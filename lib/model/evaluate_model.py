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

    plt.figure(figsize=(6, 5))

    if label_column is None:
        plt.scatter(
            x,
            y,
            alpha=0.2,
            s=6,
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
                alpha=0.2,
                s=6,
                c=colour,
                label=f"{label_name} (n={len(subset)})",
            )

        plt.legend(
            fontsize=9,
            frameon=False,
        )

    plt.xlabel(
        "Embedding cosine similarity",
        fontsize=13,
    )

    plt.ylabel(
        f"Expression similarity ({metric})",
        fontsize=13,
    )

    plt.title(
        (
            "Embedding vs Expression Similarity\n"
            f"Pearson={stats['pearson']:.3f} | "
            f"Spearman={stats['spearman']:.3f} | "
            f"Kendall={stats['kendall']:.3f}\n"
            f"{distance_text}"
        ),
        fontsize=13,
    )

    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    logger.info(
        "Evaluation plot saved to %s",
        output_path,
    )
    
    
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

    aligned_path = out_dir / f"{args.name}_embedding_similarity.tsv"
    aligned_df.to_csv(aligned_path, sep="\t", index=False)

    logger.info("Similarity dataframe saved to %s", aligned_path)

    plot_embedding_vs_expression(
        df=aligned_df,
        metric=args.metric,
        output_path=out_dir / f"{args.name}_embedding_vs_expression.png",
    )

    return aligned_path


if __name__ == "__main__":
    main()
