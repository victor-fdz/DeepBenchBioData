#!/usr/bin/env python3
"""Training utilities for Siamese promoter models."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from lib.model.datasets import DNAPairDatasetWithMeta
from lib.model.loss_function import ContrastiveLossCosine
from lib.model.model import SiameseCNN

logging.basicConfig(
    level=logging.INFO,
    format="\033[1;32m%(levelname)s\033[0m | "
           "\033[1;36m%(name)s\033[0m | %(message)s",
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results")

DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_EPOCHS = 6
DEFAULT_LEARNING_RATE = 0.032
DEFAULT_MARGIN = 1.41
DEFAULT_DROPOUT = 0.26
DEFAULT_WEIGHT_DECAY = 7.66e-5
DEFAULT_KERNEL_SIZE_SMALL = 6
DEFAULT_KERNEL_SIZE_MEDIUM = 12
DEFAULT_KERNEL_SIZE_LARGE = 20
DEFAULT_ATTENTION_HEADS = 4
DEFAULT_EMBEDDING_DIM = 16

DEFAULT_OPTUNA_TRIALS = 20
DEFAULT_OPTUNA_JOBS = 1
DEFAULT_OPTUNA_EPOCHS = 8
DEFAULT_RANDOM_STATE = 42


# -----------------------------
# CLI
# -----------------------------
def parse_args(cli_args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--metric", default=None)

    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)

    parser.add_argument("--small-kernel-size", type=int, default=DEFAULT_KERNEL_SIZE_SMALL)
    parser.add_argument("--medium-kernel-size", type=int, default=DEFAULT_KERNEL_SIZE_MEDIUM)
    parser.add_argument("--large-kernel-size", type=int, default=DEFAULT_KERNEL_SIZE_LARGE)
    parser.add_argument("--attention-heads", type=int, default=DEFAULT_ATTENTION_HEADS)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)

    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--optuna-trials", type=int, default=DEFAULT_OPTUNA_TRIALS)
    parser.add_argument("--optuna-jobs", type=int, default=DEFAULT_OPTUNA_JOBS)
    parser.add_argument("--optuna-epochs", type=int, default=DEFAULT_OPTUNA_EPOCHS)

    return parser.parse_args(cli_args)


# -----------------------------
# Validation helpers
# -----------------------------
def _validate_loader_not_empty(loader: DataLoader, name: str) -> None:
    if len(loader) == 0:
        raise ValueError(f"{name} dataloader is empty.")


def _prepare_binary_labels_for_stratification(df_train: pd.DataFrame) -> pd.Series:
    label_map = {
        "P": 1,
        "N": 0,
        "1": 1,
        "0": 0,
        1: 1,
        0: 0,
        1.0: 1,
        0.0: 0,
    }

    if "label" not in df_train.columns:
        raise ValueError("df_train is missing required column 'label'.")

    labels_for_stratification = df_train["label"].map(label_map)

    if labels_for_stratification.isna().any():
        bad_labels = df_train.loc[
            labels_for_stratification.isna(),
            "label",
        ].drop_duplicates().tolist()

        raise ValueError(
            f"Invalid labels found before Optuna stratification: {bad_labels}"
        )

    labels_for_stratification = labels_for_stratification.astype(int)
    label_counts = labels_for_stratification.value_counts()

    if len(label_counts) != 2:
        raise ValueError(
            f"Optuna requires both labels 0 and 1. Observed labels: {dict(label_counts)}"
        )

    return labels_for_stratification


# -----------------------------
# Diagnostics
# -----------------------------
def collect_epoch_embedding_distances(
    model: torch.nn.Module,
    loader: DataLoader,
    split_name: str,
    epoch: int,
) -> list[dict[str, Any]]:
    """Compute mean cosine distance between pair embeddings by label."""

    model.eval()

    distance_sum = {1: 0.0, 0: 0.0}
    distance_count = {1: 0, 0: 0}

    with torch.no_grad():
        for seq1, seq2, labels, *_ in loader:
            output1, output2 = model(seq1, seq2)
            distances = 1 - F.cosine_similarity(output1, output2)
            labels_int = labels.to(torch.int64)

            for label_value in (1, 0):
                mask = labels_int == label_value

                if mask.any():
                    distance_sum[label_value] += float(distances[mask].sum().item())
                    distance_count[label_value] += int(mask.sum().item())

    rows = []

    for label_value in (1, 0):
        rows.append(
            {
                "epoch": epoch,
                "split": split_name,
                "label": "P" if label_value == 1 else "N",
                "label_value": label_value,
                "mean_cosine_distance": (
                    distance_sum[label_value] / distance_count[label_value]
                    if distance_count[label_value] > 0
                    else np.nan
                ),
                "n_pairs": distance_count[label_value],
            }
        )

    return rows


# -----------------------------
# Core training
# -----------------------------
def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    model_path: Path,
    summary_path: Path,
    training_summary: str,
    diagnostics_path: Path | None = None,
) -> torch.nn.Module:
    """Train Siamese model."""

    if num_epochs <= 0:
        raise ValueError("num_epochs must be a positive integer.")

    _validate_loader_not_empty(train_loader, "Training")
    _validate_loader_not_empty(val_loader, "Validation")

    logger.info("Starting model training (%d epochs).", num_epochs)

    diagnostic_rows = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for seq1, seq2, labels, *_ in train_loader:
            optimizer.zero_grad()
            output1, output2 = model(seq1, seq2)

            loss = criterion(
                output1,
                output2,
                labels,
            )

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for seq1, seq2, labels, *_ in val_loader:
                output1, output2 = model(seq1, seq2)

                val_loss += criterion(
                    output1,
                    output2,
                    labels,
                ).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        logger.info(
            "Epoch %d | train=%.4f | val=%.4f",
            epoch,
            avg_train_loss,
            avg_val_loss,
        )

        if diagnostics_path is not None:
            diagnostic_rows.extend(
                collect_epoch_embedding_distances(
                    model=model,
                    loader=train_loader,
                    split_name="train",
                    epoch=epoch,
                )
            )

            diagnostic_rows.extend(
                collect_epoch_embedding_distances(
                    model=model,
                    loader=val_loader,
                    split_name="validation",
                    epoch=epoch,
                )
            )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(training_summary)

    if diagnostics_path is not None:
        pd.DataFrame(diagnostic_rows).to_csv(
            diagnostics_path,
            sep="\t",
            index=False,
        )

        logger.info("Epoch embedding distances saved to %s", diagnostics_path)

    logger.info("Model weights saved to %s", model_path)

    return model


# -----------------------------
# Hyperparameter optimization
# -----------------------------
def optimize_hyperparameters(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_trials: int = DEFAULT_OPTUNA_TRIALS,
    n_jobs: int = DEFAULT_OPTUNA_JOBS,
    trial_epochs: int = DEFAULT_OPTUNA_EPOCHS,
    fixed_margin: float = DEFAULT_MARGIN,
    attention_heads: int = DEFAULT_ATTENTION_HEADS,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    optuna_output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run Optuna hyperparameter search using the external validation set."""

    if n_trials <= 0:
        raise ValueError("n_trials must be a positive integer.")

    if n_jobs <= 0:
        raise ValueError("n_jobs must be a positive integer.")

    if trial_epochs <= 0:
        raise ValueError("trial_epochs must be a positive integer.")

    _prepare_binary_labels_for_stratification(df_train)
    _prepare_binary_labels_for_stratification(df_val)

    logger.info(
        "Running Optuna optimization with external validation: "
        "trials=%d | jobs=%d | trial_epochs=%d.",
        n_trials,
        n_jobs,
        trial_epochs,
    )

    if n_jobs > 1:
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            logger.warning("Could not reset PyTorch interop threads. Continuing.")
        logger.info("Parallel Optuna enabled. Forced PyTorch threads per trial to 1.")

    optuna_epoch_loss_rows: list[dict[str, Any]] = []

    def objective(trial: optuna.trial.Trial) -> float:
        learning_rate = trial.suggest_categorical(
            "learning_rate",
            [1e-5, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
        )

        weight_decay = trial.suggest_categorical(
            "weight_decay",
            [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        )

        dropout_rate = trial.suggest_categorical(
            "dropout_rate",
            [0.0, 0.1, 0.3, 0.5, 0.6, 0.8],
        )

        small_kernel_size = trial.suggest_categorical(
            "small_kernel_size",
            [4, 6, 8],
        )

        medium_kernel_size = trial.suggest_categorical(
            "medium_kernel_size",
            [10, 12, 14],
        )

        large_kernel_size = trial.suggest_categorical(
            "large_kernel_size",
            [18, 20, 24, 26, 30],
        )

        trial_seed = DEFAULT_RANDOM_STATE + trial.number * 1000

        train_loader_generator = torch.Generator()
        train_loader_generator.manual_seed(trial_seed)

        train_loader = DataLoader(
            DNAPairDatasetWithMeta(df_train),
            batch_size=batch_size,
            shuffle=True,
            generator=train_loader_generator,
            num_workers=0,
        )

        val_loader = DataLoader(
            DNAPairDatasetWithMeta(df_val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        _validate_loader_not_empty(train_loader, "Optuna training")
        _validate_loader_not_empty(val_loader, "Optuna validation")

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(trial_seed)

            model = SiameseCNN(
                dropout_rate=dropout_rate,
                small_kernel_size=small_kernel_size,
                medium_kernel_size=medium_kernel_size,
                large_kernel_size=large_kernel_size,
                attention_heads=attention_heads,
                embedding_dim=embedding_dim,
            )

        criterion = ContrastiveLossCosine(
            margin=fixed_margin,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        final_train_loss = np.nan
        final_validation_loss = np.nan
        best_validation_loss = np.inf

        for epoch in range(1, trial_epochs + 1):
            model.train()
            train_loss = 0.0

            for seq1, seq2, labels, *_ in train_loader:
                optimizer.zero_grad()

                out1, out2 = model(seq1, seq2)

                loss = criterion(
                    out1,
                    out2,
                    labels,
                )

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            mean_train_loss = train_loss / len(train_loader)

            model.eval()
            validation_loss = 0.0

            with torch.no_grad():
                for seq1, seq2, labels, *_ in val_loader:
                    out1, out2 = model(seq1, seq2)

                    validation_loss += criterion(
                        out1,
                        out2,
                        labels,
                    ).item()

            mean_validation_loss = validation_loss / len(val_loader)

            final_train_loss = mean_train_loss
            final_validation_loss = mean_validation_loss
            best_validation_loss = min(best_validation_loss, mean_validation_loss)

            optuna_epoch_loss_rows.append(
                {
                    "trial_number": trial.number,
                    "epoch": epoch,
                    "train_loss": float(mean_train_loss),
                    "validation_loss": float(mean_validation_loss),
                    "learning_rate": float(learning_rate),
                    "weight_decay": float(weight_decay),
                    "dropout_rate": float(dropout_rate),
                    "small_kernel_size": int(small_kernel_size),
                    "medium_kernel_size": int(medium_kernel_size),
                    "large_kernel_size": int(large_kernel_size),
                    "margin": float(fixed_margin),
                    "attention_heads": int(attention_heads),
                    "embedding_dim": int(embedding_dim),
                    "trial_seed": int(trial_seed),
                }
            )

            logger.info(
                "Optuna trial %d | epoch %d/%d | train=%.6f | val=%.6f",
                trial.number,
                epoch,
                trial_epochs,
                mean_train_loss,
                mean_validation_loss,
            )

        trial.set_user_attr("final_train_loss", float(final_train_loss))
        trial.set_user_attr("final_validation_loss", float(final_validation_loss))
        trial.set_user_attr("best_validation_loss", float(best_validation_loss))
        trial.set_user_attr("trial_seed", int(trial_seed))

        return float(final_validation_loss)

    study = optuna.create_study(
        direction="minimize",
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=(n_jobs == 1),
    )

    logger.info("Best hyperparameters: %s", study.best_params)

    if optuna_output_dir is not None:
        optuna_output_dir.mkdir(parents=True, exist_ok=True)

        with (optuna_output_dir / "optuna_best_params.json").open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(study.best_params, handle, indent=2)

        study.trials_dataframe().to_csv(
            optuna_output_dir / "optuna_trials.tsv",
            sep="\t",
            index=False,
        )

        pd.DataFrame(optuna_epoch_loss_rows).sort_values(
            ["trial_number", "epoch"]
        ).to_csv(
            optuna_output_dir / "optuna_epoch_losses.tsv",
            sep="\t",
            index=False,
        )

    return study.best_params

# -----------------------------
# Plots
# -----------------------------
def plot_epoch_embedding_distances(
    distances_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Plot mean embedding cosine distance for train and validation splits."""

    distance_dataframe = pd.read_csv(distances_path, sep="\t")

    required_columns = {
        "epoch",
        "split",
        "label",
        "mean_cosine_distance",
    }

    missing_columns = required_columns - set(distance_dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {distances_path}: {sorted(missing_columns)}"
        )

    if output_path is None:
        output_path = distances_path.parent / "embedding_distance_dynamics.png"

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_splits = [
        ("train", "Training"),
        ("validation", "Validation"),
    ]

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(7.2, 2.8),
        sharey=True,
    )

    for axis, (split_name, plot_title) in zip(axes, plot_splits):
        split_dataframe = distance_dataframe[
            distance_dataframe["split"] == split_name
        ].copy()

        if split_dataframe.empty:
            raise ValueError(
                f"No rows found for split={split_name!r} in {distances_path}"
            )

        positive_dataframe = split_dataframe[
            split_dataframe["label"].isin(["P", 1, "1"])
        ].sort_values("epoch")

        negative_dataframe = split_dataframe[
            split_dataframe["label"].isin(["N", 0, "0"])
        ].sort_values("epoch")

        if positive_dataframe.empty or negative_dataframe.empty:
            raise ValueError(
                f"Both positive and negative rows are required for split={split_name!r}."
            )

        axis.plot(
            positive_dataframe["epoch"],
            positive_dataframe["mean_cosine_distance"],
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            label="Positive pairs",
        )

        axis.plot(
            negative_dataframe["epoch"],
            negative_dataframe["mean_cosine_distance"],
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            label="Negative pairs",
        )

        axis.set_title(plot_title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Mean embedding distance")

        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(
            True,
            axis="y",
            linestyle="--",
            linewidth=0.6,
            alpha=0.35,
        )

    axes[1].set_ylabel("")
    axes[0].legend(frameon=False, loc="best")

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(figure)

    logger.info("Embedding distance dynamics plot saved to %s", output_path)

    return output_path

def _guess_metric_column(dataframe: pd.DataFrame) -> str:
    candidates = [
        "cosine_sim",
        "met",
        "ssd",
        "net_sim",
        "shannon_sim",
        "tau_sim",
        "z_score_cosine_sim",
        "gini_sim",
    ]

    for column in candidates:
        if column in dataframe.columns:
            return column

    raise ValueError("Could not infer expression similarity metric column.")


def save_embedding_similarity_dataframe(
    model: torch.nn.Module,
    loader: DataLoader,
    source_dataframe: pd.DataFrame,
    output_path: Path,
    metric_column: str,
) -> pd.DataFrame:
    """Save embedding similarities aligned to the source dataframe by pair_row_id."""

    if metric_column not in source_dataframe.columns:
        raise ValueError(f"Missing metric column in source dataframe: {metric_column}")

    model.eval()
    rows = []

    with torch.no_grad():
        for seq1, seq2, labels, pair_row_id, gene_id_1, gene_id_2 in loader:
            output1, output2 = model(seq1, seq2)

            embedding_similarity = F.cosine_similarity(
                output1,
                output2,
            ).cpu().numpy()

            for row_id, first_gene, second_gene, similarity in zip(
                pair_row_id,
                gene_id_1,
                gene_id_2,
                embedding_similarity,
            ):
                rows.append(
                    {
                        "pair_row_id": int(row_id),
                        "gene_id_1": first_gene,
                        "gene_id_2": second_gene,
                        "embedding_cosine_sim": float(similarity),
                    }
                )

    prediction_dataframe = pd.DataFrame(rows)

    target_columns = ["pair_row_id", metric_column]
    if "original_label" in source_dataframe.columns:
        target_columns.append("original_label")
    elif "label" in source_dataframe.columns:
        target_columns.append("label")

    target_dataframe = source_dataframe[target_columns].copy()

    output_dataframe = prediction_dataframe.merge(
        target_dataframe,
        on="pair_row_id",
        how="inner",
    )

    if len(output_dataframe) != len(prediction_dataframe):
        raise ValueError("Embedding similarity merge changed the number of rows.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dataframe.to_csv(output_path, sep="\t", index=False)

    return output_dataframe


def plot_embedding_vs_expression_similarity(
    dataframe: pd.DataFrame,
    metric_column: str,
    output_path: Path,
    title: str,
) -> None:
    """Plot embedding similarity against expression similarity."""

    x = dataframe["embedding_cosine_sim"].to_numpy(dtype=float)
    y = dataframe[metric_column].to_numpy(dtype=float)

    if len(dataframe) < 2:
        raise ValueError(f"Need at least two rows to plot {title}.")

    if np.unique(x).size < 2 or np.unique(y).size < 2:
        raise ValueError(f"Cannot plot correlations for {title}: one axis is constant.")

    pearson_value, _ = pearsonr(x, y)
    spearman_value, _ = spearmanr(x, y)
    kendall_value, _ = kendalltau(x, y)

    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, alpha=0.2, s=6)
    plt.xlabel("Embedding cosine similarity", fontsize=13)
    plt.ylabel(f"Expression similarity ({metric_column})", fontsize=13)
    plt.title(
        (
            f"{title}\n"
            f"Pearson={pearson_value:.3f} | "
            f"Spearman={spearman_value:.3f} | "
            f"Kendall={kendall_value:.3f}"
        ),
        fontsize=14,
    )
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_model_config(
    output_path: Path,
    model_config: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(model_config, handle, indent=2)


# -----------------------------
# Full training pipeline
# -----------------------------
def run_training(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    model_output_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    margin: float = DEFAULT_MARGIN,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    dropout_rate: float = DEFAULT_DROPOUT,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    small_kernel_size: int = DEFAULT_KERNEL_SIZE_SMALL,
    medium_kernel_size: int = DEFAULT_KERNEL_SIZE_MEDIUM,
    large_kernel_size: int = DEFAULT_KERNEL_SIZE_LARGE,
    attention_heads: int = DEFAULT_ATTENTION_HEADS,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    optimize_hparams: bool = False,
    optuna_trials: int = DEFAULT_OPTUNA_TRIALS,
    optuna_jobs: int = DEFAULT_OPTUNA_JOBS,
    optuna_epochs: int = DEFAULT_OPTUNA_EPOCHS,
    metric_column: str | None = None,
) -> torch.nn.Module:
    """Full Siamese training workflow."""

    model_output_dir.mkdir(parents=True, exist_ok=True)

    if optimize_hparams:
        best = optimize_hyperparameters(
            df_train=df_train,
            df_val=df_val,
            batch_size=batch_size,
            n_trials=optuna_trials,
            n_jobs=optuna_jobs,
            trial_epochs=optuna_epochs,
            fixed_margin=margin,
            attention_heads=attention_heads,
            embedding_dim=embedding_dim,
            optuna_output_dir=model_output_dir,
        )

        learning_rate = best["learning_rate"]
        dropout_rate = best["dropout_rate"]
        weight_decay = best["weight_decay"]
        small_kernel_size = best["small_kernel_size"]
        medium_kernel_size = best["medium_kernel_size"]
        large_kernel_size = best["large_kernel_size"]

    train_loader_generator = torch.Generator()
    train_loader_generator.manual_seed(DEFAULT_RANDOM_STATE)

    train_loader = DataLoader(
        DNAPairDatasetWithMeta(df_train),
        batch_size=batch_size,
        shuffle=True,
        generator=train_loader_generator,
    )

    val_loader = DataLoader(
        DNAPairDatasetWithMeta(df_val),
        batch_size=batch_size,
        shuffle=False,
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(DEFAULT_RANDOM_STATE)
        model = SiameseCNN(
            dropout_rate=dropout_rate,
            small_kernel_size=small_kernel_size,
            medium_kernel_size=medium_kernel_size,
            large_kernel_size=large_kernel_size,
            attention_heads=attention_heads,
            embedding_dim=embedding_dim,
        )

    criterion = ContrastiveLossCosine(margin=margin)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    model_config = {
        "dropout_rate": float(dropout_rate),
        "small_kernel_size": int(small_kernel_size),
        "medium_kernel_size": int(medium_kernel_size),
        "large_kernel_size": int(large_kernel_size),
        "attention_heads": int(attention_heads),
        "embedding_dim": int(embedding_dim),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "margin": float(margin),
        "batch_size": int(batch_size),
        "epochs": int(num_epochs),
        "optimized_hyperparameters": bool(optimize_hparams),
    }

    save_model_config(model_output_dir / "model_config.json", model_config)

    training_summary = f"""
Training Parameters
-------------------------
Batch size: {batch_size}
Epochs: {num_epochs}

Optimizer: AdamW
Learning rate: {learning_rate}
Weight decay: {weight_decay}

Loss function: ContrastiveLossCosine
Margin: {margin}

Small kernel size: {small_kernel_size}
Medium kernel size: {medium_kernel_size}
Large kernel size: {large_kernel_size}
Attention heads: {attention_heads}
Embedding dimension: {embedding_dim}

Dropout rate: {dropout_rate}
Optimized hyperparameters: {optimize_hparams}
"""

    logger.info(training_summary)

    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        model_path=model_output_dir / "model_weights.pth",
        summary_path=model_output_dir / "training_summary.txt",
        training_summary=training_summary,
        diagnostics_path=model_output_dir / "epoch_embedding_distances.tsv",
    )

    plot_epoch_embedding_distances(
        distances_path=model_output_dir / "epoch_embedding_distances.tsv",
        output_path=model_output_dir / "embedding_distance_dynamics.png",
    )

    if metric_column is None:
        metric_column = _guess_metric_column(df_train)

    train_similarity_dataframe = save_embedding_similarity_dataframe(
        model=model,
        loader=train_loader,
        source_dataframe=df_train,
        output_path=model_output_dir / "train_embedding_similarity.tsv",
        metric_column=metric_column,
    )

    validation_similarity_dataframe = save_embedding_similarity_dataframe(
        model=model,
        loader=val_loader,
        source_dataframe=df_val,
        output_path=model_output_dir / "validation_embedding_similarity.tsv",
        metric_column=metric_column,
    )

    plot_embedding_vs_expression_similarity(
        dataframe=train_similarity_dataframe,
        metric_column=metric_column,
        output_path=model_output_dir / "train_embedding_vs_expression.png",
        title="Training: Embedding vs Expression Similarity",
    )

    plot_embedding_vs_expression_similarity(
        dataframe=validation_similarity_dataframe,
        metric_column=metric_column,
        output_path=model_output_dir / "validation_embedding_vs_expression.png",
        title="Validation: Embedding vs Expression Similarity",
    )

    return model


# -----------------------------
# Main workflow
# -----------------------------
def main(cli_args: list[str] | None = None) -> Path:
    """Run training workflow."""

    args = parse_args(cli_args)

    logger.info("Loading datasets.")
    df_train = pd.read_csv(args.train, sep="\t")
    df_val = pd.read_csv(args.validation, sep="\t")

    logger.info("Train=%d | Validation=%d", len(df_train), len(df_val))

    if args.output_dir is None:
        model_output_dir = OUTPUT_DIR / args.name / "Model"
    else:
        model_output_dir = args.output_dir

    model_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting model training.")

    run_training(
        df_train=df_train,
        df_val=df_val,
        model_output_dir=model_output_dir,
        batch_size=args.batch_size,
        margin=args.margin,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout,
        weight_decay=args.weight_decay,
        small_kernel_size=args.small_kernel_size,
        medium_kernel_size=args.medium_kernel_size,
        large_kernel_size=args.large_kernel_size,
        attention_heads=args.attention_heads,
        embedding_dim=args.embedding_dim,
        optimize_hparams=args.optimize,
        optuna_trials=args.optuna_trials,
        optuna_jobs=args.optuna_jobs,
        optuna_epochs=args.optuna_epochs,
        metric_column=args.metric,
    )

    logger.info("Training completed.")

    return model_output_dir / "model_weights.pth"


if __name__ == "__main__":
    main()
