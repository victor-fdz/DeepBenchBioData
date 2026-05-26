#!/usr/bin/env python3
"""Train and evaluate Siamese promoter models."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from torch.utils.data import DataLoader

from lib.model.datasets import DNAPairDatasetWithMeta
from lib.model.evaluate_model import generate_embeddings
from lib.model.evaluate_model import plot_embedding_vs_expression
from lib.model.train_model import run_training

logging.basicConfig(
    level=logging.INFO,
    format="\033[1;32m%(levelname)s\033[0m | "
           "\033[1;36m%(name)s\033[0m | %(message)s",
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results")

DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_EPOCHS = 15
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_MARGIN = 1.0
DEFAULT_DROPOUT = 0.3
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_KERNEL_SIZE_SMALL = 6
DEFAULT_KERNEL_SIZE_MEDIUM = 12
DEFAULT_KERNEL_SIZE_LARGE = 20
DEFAULT_ATTENTION_HEADS = 4
DEFAULT_EMBEDDING_DIM = 16
DEFAULT_OPTUNA_TRIALS = 20
DEFAULT_OPTUNA_JOBS = 1
DEFAULT_OPTUNA_FOLDS = 5
DEFAULT_OPTUNA_EPOCHS = 8


# -----------------------------
# CLI
# -----------------------------
def parse_args(cli_args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--metric", default="cosine_sim")

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
    parser.add_argument("--optuna-folds", type=int, default=DEFAULT_OPTUNA_FOLDS)
    parser.add_argument("--optuna-epochs", type=int, default=DEFAULT_OPTUNA_EPOCHS)

    return parser.parse_args(cli_args)


def _load_split_table(data_dir: Path, split_name: str) -> pd.DataFrame:
    path = data_dir / f"{split_name}.tsv"

    if not path.exists():
        raise FileNotFoundError(f"Missing required split file: {path}")

    return pd.read_csv(path, sep="\t")


def build_evaluation_dataframe(
    prediction_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    metric_column: str,
) -> pd.DataFrame:
    """Align test predictions with expression similarity and labels."""

    required_columns = {"pair_row_id", metric_column}
    missing_columns = required_columns - set(test_dataframe.columns)

    if missing_columns:
        raise ValueError(f"Test dataframe is missing columns: {sorted(missing_columns)}")

    target_columns = ["pair_row_id", metric_column]

    if "original_label" in test_dataframe.columns:
        target_columns.append("original_label")
    elif "label" in test_dataframe.columns:
        target_columns.append("label")

    target_dataframe = test_dataframe[target_columns].copy()

    evaluation_dataframe = prediction_dataframe.merge(
        target_dataframe,
        on="pair_row_id",
        how="inner",
    )

    if len(evaluation_dataframe) != len(prediction_dataframe):
        raise ValueError("Prediction-to-target merge changed the number of rows.")

    return evaluation_dataframe


# -----------------------------
# Main
# -----------------------------
def main(cli_args: list[str] | None = None) -> Path:
    """Run training and evaluation workflow."""

    args = parse_args(cli_args)

    logger.info("Loading datasets.")

    df_train = _load_split_table(args.data, "train")
    df_val = _load_split_table(args.data, "val")
    df_test = _load_split_table(args.data, "test")

    logger.info(
        "Train=%d | Validation=%d | Test=%d",
        len(df_train),
        len(df_val),
        len(df_test),
    )

    model_dir = OUTPUT_DIR / args.name / "Model"
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting model training.")

    model = run_training(
        df_train=df_train,
        df_val=df_val,
        model_output_dir=model_dir,
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
        optuna_folds=args.optuna_folds,
        optuna_epochs=args.optuna_epochs,
        metric_column=args.metric,
    )

    logger.info("Training completed.")
    logger.info("Running evaluation.")

    loader = DataLoader(
        DNAPairDatasetWithMeta(df_test),
        batch_size=args.batch_size,
        shuffle=False,
    )

    prediction_dataframe = generate_embeddings(
        model=model,
        loader=loader,
    )

    evaluation_dataframe = build_evaluation_dataframe(
        prediction_dataframe=prediction_dataframe,
        test_dataframe=df_test,
        metric_column=args.metric,
    )

    plot_embedding_vs_expression(
        df=evaluation_dataframe,
        metric=args.metric,
        output_path=model_dir / "evaluation_plot.png",
    )

    evaluation_path = model_dir / "evaluation_predictions.tsv"
    evaluation_dataframe.to_csv(evaluation_path, sep="\t", index=False)

    logger.info("Evaluation predictions saved to %s", evaluation_path)
    logger.info("Workflow completed successfully.")

    return evaluation_path


if __name__ == "__main__":
    main()
