#!/usr/bin/env python3
"""Train and evaluate Siamese promoter models in a Nextflow process.

This is intentionally close to bin/model_processing.py.

Nextflow-related differences:
- output directory is process-local: Model/
- split files can be provided either as --data or as explicit --train/--validation/--test
- evaluation output is written into Model/ so Nextflow can publish it
"""

from __future__ import annotations

import argparse
import json
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

OUTPUT_DIR = Path(".")

DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_EPOCHS = 15
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_MARGIN = 1.0
DEFAULT_DROPOUT = 0
DEFAULT_WEIGHT_DECAY = 0.001
DEFAULT_KERNEL_SIZE_SMALL = 6
DEFAULT_KERNEL_SIZE_MEDIUM = 10
DEFAULT_KERNEL_SIZE_LARGE = 18
DEFAULT_ATTENTION_HEADS = 4
DEFAULT_EMBEDDING_DIM = 16
DEFAULT_OPTUNA_TRIALS = 20
DEFAULT_OPTUNA_JOBS = 1
DEFAULT_OPTUNA_EPOCHS = 8


def parse_args(cli_args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    This keeps the original model_processing.py interface and adds explicit
    split-file arguments for Nextflow process inputs.
    """

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Original interface
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Directory containing train.tsv, val.tsv and test.tsv.",
    )
    parser.add_argument("--name", "--dataset-name", dest="name", required=True)
    parser.add_argument("--metric", default="cosine_sim")

    # Nextflow explicit inputs
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--validation", type=Path, default=None)
    parser.add_argument("--test", type=Path, default=None)

    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)

    # Architecture hyperparameters
    parser.add_argument("--small-kernel-size", type=int, default=DEFAULT_KERNEL_SIZE_SMALL)
    parser.add_argument("--medium-kernel-size", type=int, default=DEFAULT_KERNEL_SIZE_MEDIUM)
    parser.add_argument("--large-kernel-size", type=int, default=DEFAULT_KERNEL_SIZE_LARGE)
    parser.add_argument("--attention-heads", type=int, default=DEFAULT_ATTENTION_HEADS)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)

    # Optuna
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--optuna-trials", type=int, default=DEFAULT_OPTUNA_TRIALS)
    parser.add_argument("--optuna-jobs", type=int, default=DEFAULT_OPTUNA_JOBS)
    parser.add_argument("--optuna-epochs", type=int, default=DEFAULT_OPTUNA_EPOCHS)

    return parser.parse_args(cli_args)


def _load_split_table(path: Path, split_name: str) -> pd.DataFrame:
    """Load one split table."""

    if not path.exists():
        raise FileNotFoundError(f"Missing required {split_name} split file: {path}")

    dataframe = pd.read_csv(path, sep="	")

    if dataframe.empty:
        raise ValueError(f"{split_name} split is empty: {path}")

    logger.info("%s=%d rows", split_name, len(dataframe))

    return dataframe


def _resolve_split_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Resolve train, validation and test split paths."""

    if args.data is not None:
        return (
            args.data / "train.tsv",
            args.data / "val.tsv",
            args.data / "test.tsv",
        )

    explicit_paths = [args.train, args.validation, args.test]

    if any(path is None for path in explicit_paths):
        raise ValueError(
            "Provide either --data or all explicit split paths: "
            "--train, --validation and --test."
        )

    return args.train, args.validation, args.test


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

    if "gene_id_human" in test_dataframe.columns and "gene_id_mouse" in test_dataframe.columns:
        target_columns.extend(["gene_id_human", "gene_id_mouse"])
    elif "gene_id_1" in test_dataframe.columns and "gene_id_2" in test_dataframe.columns:
        target_columns.extend(["gene_id_1", "gene_id_2"])

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


def write_manifest(
    output_path: Path,
    args: argparse.Namespace,
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> None:
    """Write a run manifest for reproducibility."""

    manifest = {
        "name": args.name,
        "data": str(args.data) if args.data is not None else None,
        "train": str(train_path),
        "validation": str(validation_path),
        "test": str(test_path),
        "metric": args.metric,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "margin": args.margin,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "small_kernel_size": args.small_kernel_size,
        "medium_kernel_size": args.medium_kernel_size,
        "large_kernel_size": args.large_kernel_size,
        "attention_heads": args.attention_heads,
        "embedding_dim": args.embedding_dim,
        "optimize": args.optimize,
        "optuna_trials": args.optuna_trials,
        "optuna_jobs": args.optuna_jobs,
        "optuna_epochs": args.optuna_epochs,
        "train_rows": len(df_train),
        "validation_rows": len(df_val),
        "test_rows": len(df_test),
        "output_policy": "Nextflow process-local output directory: Model/",
    }

    output_path.write_text(json.dumps(manifest, indent=2) + "")


def resolve_metric_column(metric_name: str, dataframes: list[pd.DataFrame]) -> str:
    candidate_columns = [metric_name]

    if not metric_name.endswith("_sim"):
        candidate_columns.append(f"{metric_name}_sim")

    for candidate_column in candidate_columns:
        if all(candidate_column in dataframe.columns for dataframe in dataframes):
            return candidate_column

    available_columns = sorted(set().union(*(set(dataframe.columns) for dataframe in dataframes)))
    raise ValueError(
        f"Could not resolve metric column for '{metric_name}'. "
        f"Tried {candidate_columns}. Available columns: {available_columns}"
    )


def main(cli_args: list[str] | None = None) -> Path:
    """Run training and evaluation workflow."""

    args = parse_args(cli_args)

    logger.info("Loading datasets.")

    train_path, validation_path, test_path = _resolve_split_paths(args)

    df_train = _load_split_table(train_path, "Train")
    df_val = _load_split_table(validation_path, "Validation")
    df_test = _load_split_table(test_path, "Test")

    logger.info(
        "Train=%d | Validation=%d | Test=%d",
        len(df_train),
        len(df_val),
        len(df_test),
    )

    model_dir = OUTPUT_DIR / "Model"
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting model training.")

    metric_column = resolve_metric_column(args.metric, [df_train, df_val, df_test])

    model = run_training(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
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
        optuna_epochs=args.optuna_epochs,
        metric_column=metric_column,
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
        metric_column=metric_column,
    )

    plot_embedding_vs_expression(
        df=evaluation_dataframe,
        metric=metric_column,
        output_path=model_dir / "evaluation_plot.png",
    )

    evaluation_path = model_dir / "evaluation_predictions.tsv"
    evaluation_dataframe.to_csv(evaluation_path, sep="	", index=False)

    write_manifest(
        output_path=model_dir / "model_manifest.json",
        args=args,
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path,
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
    )

    logger.info("Evaluation predictions saved to %s", evaluation_path)
    logger.info("Workflow completed successfully.")

    return evaluation_path


if __name__ == "__main__":
    main()
