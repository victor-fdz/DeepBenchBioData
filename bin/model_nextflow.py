#!/usr/bin/env python3
"""Nextflow wrapper for Siamese model training and evaluation.

This script reuses the project model modules:
- lib.model.datasets
- lib.model.train_model
- lib.model.evaluate_model

It only changes what is required for Nextflow:
- explicit train/validation/test inputs
- process-local Model/ outputs
- stable output folder declared by Nextflow
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.model.datasets import DNAPairDatasetWithMeta
from lib.model.evaluate_model import generate_embeddings
from lib.model.evaluate_model import plot_embedding_vs_expression
from lib.model.train_model import run_training

logging.basicConfig(
    level=logging.INFO,
    format="\033[1;32m%(levelname)s\033[0m | \033[1;36m%(name)s\033[0m | %(message)s",
)

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_EPOCHS = 15
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_MARGIN = 1.0
DEFAULT_DROPOUT = 0.3
DEFAULT_WEIGHT_DECAY = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)

    parser.add_argument("--dataset-name", "--name", dest="dataset_name", required=True)

    parser.add_argument(
        "--metric",
        default="cosine_sim",
        help="Expression similarity metric column.",
    )

    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)

    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run Optuna hyperparameter optimization.",
    )

    return parser.parse_args()


def read_split(path: Path, split_name: str) -> pd.DataFrame:
    logger.info("Loading %s split from %s", split_name, path)

    dataframe = pd.read_csv(path, sep="\t")

    if dataframe.empty:
        raise ValueError(f"{split_name} split is empty: {path}")

    logger.info("%s rows: %d", split_name, len(dataframe))

    return dataframe


def build_evaluation_dataframe(
    model,
    test_dataframe: pd.DataFrame,
    batch_size: int,
    metric: str,
) -> pd.DataFrame:
    loader = DataLoader(
        DNAPairDatasetWithMeta(test_dataframe),
        batch_size=batch_size,
        shuffle=False,
    )

    predictions = generate_embeddings(
        model=model,
        loader=loader,
    )

    target_columns = [
        "pair_row_id",
        metric,
    ]

    if "gene_id_human" in test_dataframe.columns and "gene_id_mouse" in test_dataframe.columns:
        target_columns.extend(["gene_id_human", "gene_id_mouse"])
        merge_columns = ["pair_row_id", "gene_id_human", "gene_id_mouse"]
    elif "gene_id_1" in test_dataframe.columns and "gene_id_2" in test_dataframe.columns:
        target_columns.extend(["gene_id_1", "gene_id_2"])
        predictions = predictions.rename(
            columns={
                "gene_id_human": "gene_id_1",
                "gene_id_mouse": "gene_id_2",
            }
        )
        merge_columns = ["pair_row_id", "gene_id_1", "gene_id_2"]
    else:
        raise ValueError(
            "Test split must contain either gene_id_human/gene_id_mouse "
            "or gene_id_1/gene_id_2."
        )

    if "original_label" in test_dataframe.columns:
        target_columns.append("original_label")
    elif "label" in test_dataframe.columns:
        target_columns.append("label")

    missing_columns = sorted(set(target_columns) - set(test_dataframe.columns))
    if missing_columns:
        raise ValueError(
            f"Missing required columns in test split for evaluation: {missing_columns}"
        )

    target_dataframe = test_dataframe[target_columns].copy()

    evaluation_dataframe = predictions.merge(
        target_dataframe,
        on="pair_row_id",
        how="inner",
    )

    if len(evaluation_dataframe) != len(predictions):
        logger.warning(
            "Evaluation merge kept %d/%d prediction rows.",
            len(evaluation_dataframe),
            len(predictions),
        )

    return evaluation_dataframe


def main() -> Path:
    args = parse_args()

    train_dataframe = read_split(args.train, "train")
    validation_dataframe = read_split(args.validation, "validation")
    test_dataframe = read_split(args.test, "test")

    model_output_dir = Path("Model")
    model_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting model training.")

    model = run_training(
        df_train=train_dataframe,
        df_val=validation_dataframe,
        df_test=test_dataframe,
        model_output_dir=model_output_dir,
        batch_size=args.batch_size,
        margin=args.margin,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout,
        weight_decay=args.weight_decay,
        optimize_hparams=args.optimize,
        metric_column=args.metric,
    )

    logger.info("Training completed. Starting evaluation.")

    evaluation_dataframe = build_evaluation_dataframe(
        model=model,
        test_dataframe=test_dataframe,
        batch_size=args.batch_size,
        metric=args.metric,
    )

    evaluation_path = model_output_dir / "evaluation_predictions.tsv"
    evaluation_dataframe.to_csv(
        evaluation_path,
        sep="\t",
        index=False,
    )

    plot_embedding_vs_expression(
        df=evaluation_dataframe,
        metric=args.metric,
        output_path=model_output_dir / "evaluation_plot.png",
    )

    manifest = {
        "dataset_name": args.dataset_name,
        "train": str(args.train),
        "validation": str(args.validation),
        "test": str(args.test),
        "metric": args.metric,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "margin": args.margin,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "optimize": args.optimize,
        "train_rows": len(train_dataframe),
        "validation_rows": len(validation_dataframe),
        "test_rows": len(test_dataframe),
        "output_policy": "all model outputs are written process-locally under Model/",
    }

    manifest_path = model_output_dir / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    logger.info("Evaluation predictions saved to %s", evaluation_path)
    logger.info("Model outputs generated in %s", model_output_dir)

    return model_output_dir


if __name__ == "__main__":
    main()
