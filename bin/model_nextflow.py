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

import numpy as np
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
DEFAULT_WEIGHT_DECAY = 1e-6
DEFAULT_KERNEL_SIZE_SMALL = 4
DEFAULT_KERNEL_SIZE_MEDIUM = 10
DEFAULT_KERNEL_SIZE_LARGE = 24
DEFAULT_ATTENTION_HEADS = 4
DEFAULT_EMBEDDING_DIM = 16
DEFAULT_OPTUNA_TRIALS = 20
DEFAULT_OPTUNA_JOBS = 2
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
    """Resolve the metric column name in the provided dataframes."""

    candidate_columns = [metric_name]

    if not metric_name.endswith("_sim"):
        candidate_columns.append(f"{metric_name}_sim")

    for candidate_column in candidate_columns:
        if all(candidate_column in dataframe.columns for dataframe in dataframes):
            return candidate_column

    available_columns = sorted(
        set().union(*(set(dataframe.columns) for dataframe in dataframes))
    )
    raise ValueError(
        f"Could not resolve metric column for {metric_name!r}. "
        f"Tried {candidate_columns}. Available columns: {available_columns}"
    )


def resolve_gene_pair_columns(dataframe: pd.DataFrame) -> tuple[str, str]:
    """Resolve human and mouse gene identifier columns for consistency analysis."""

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
        "Could not resolve human/mouse gene columns for triplet consistency. "
        f"Available columns: {sorted(dataframe.columns)}"
    )


def normalise_binary_label_values(label_series: pd.Series) -> pd.Series:
    """Normalize pair labels to P, N or U."""

    return label_series.replace(
        {
            "P": "P",
            "N": "N",
            "U": "U",
            "1": "P",
            1: "P",
            1.0: "P",
            "0": "N",
            0: "N",
            0.0: "N",
            "-1": "U",
            -1: "U",
            -1.0: "U",
        }
    )


def build_reference_ortholog_pairs(
    dataframe: pd.DataFrame,
    human_gene_column: str,
    mouse_gene_column: str,
) -> pd.DataFrame:
    """Build reference one-to-one pairs used as C genes in triplet consistency."""

    label_column = None
    if "original_label" in dataframe.columns:
        label_column = "original_label"
    elif "label" in dataframe.columns:
        label_column = "label"

    if label_column is None:
        logger.warning(
            "No label/original_label column found. Using all test pairs as reference pairs."
        )
        reference_dataframe = dataframe[[human_gene_column, mouse_gene_column]].copy()
    else:
        labelled_dataframe = dataframe.copy()
        labelled_dataframe["label_normalized"] = normalise_binary_label_values(
            labelled_dataframe[label_column]
        )
        reference_dataframe = labelled_dataframe.loc[
            labelled_dataframe["label_normalized"] == "P",
            [human_gene_column, mouse_gene_column],
        ].copy()

    reference_dataframe = (
        reference_dataframe
        .dropna(subset=[human_gene_column, mouse_gene_column])
        .drop_duplicates()
        .rename(
            columns={
                human_gene_column: "reference_human_gene",
                mouse_gene_column: "reference_mouse_gene",
            }
        )
        .reset_index(drop=True)
    )

    if reference_dataframe.empty:
        raise ValueError(
            "No reference ortholog pairs found for triplet consistency. "
            "Expected positive rows in original_label or label."
        )

    return reference_dataframe


def add_triplet_consistency(
    dataframe: pd.DataFrame,
    human_gene_column: str,
    mouse_gene_column: str,
    expression_similarity_column: str,
    embedding_similarity_column: str = "embedding_cosine_sim",
    exclude_pair_genes_as_references: bool = True,
) -> pd.DataFrame:
    """Add triplet consistency for every human-mouse pair in the dataframe."""

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
    working_dataframe[expression_similarity_column] = pd.to_numeric(
        working_dataframe[expression_similarity_column],
        errors="coerce",
    )
    working_dataframe[embedding_similarity_column] = pd.to_numeric(
        working_dataframe[embedding_similarity_column],
        errors="coerce",
    )

    reference_gene_pairs = build_reference_ortholog_pairs(
        dataframe=working_dataframe,
        human_gene_column=human_gene_column,
        mouse_gene_column=mouse_gene_column,
    )

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

    pair_columns = ["pair_row_id", human_gene_column, mouse_gene_column]
    if "pair_row_id" not in working_dataframe.columns:
        pair_columns = [human_gene_column, mouse_gene_column]

    for pair in working_dataframe[pair_columns].itertuples(index=False):
        pair_human_gene = getattr(pair, human_gene_column)
        pair_mouse_gene = getattr(pair, mouse_gene_column)
        pair_row_id = getattr(pair, "pair_row_id", None)

        consistent_triplet_count = 0
        total_triplet_count = 0
        skipped_missing_triplet_count = 0
        skipped_tied_triplet_count = 0

        for reference in reference_gene_pairs.itertuples(index=False):
            reference_human_gene = reference.reference_human_gene
            reference_mouse_gene = reference.reference_mouse_gene

            if exclude_pair_genes_as_references:
                if reference_human_gene == pair_human_gene:
                    continue
                if reference_mouse_gene == pair_mouse_gene:
                    continue

            try:
                expression_similarity_to_pair_human = expression_similarity_matrix.loc[
                    pair_human_gene,
                    reference_mouse_gene,
                ]
                expression_similarity_to_pair_mouse = expression_similarity_matrix.loc[
                    reference_human_gene,
                    pair_mouse_gene,
                ]
                embedding_similarity_to_pair_human = embedding_similarity_matrix.loc[
                    pair_human_gene,
                    reference_mouse_gene,
                ]
                embedding_similarity_to_pair_mouse = embedding_similarity_matrix.loc[
                    reference_human_gene,
                    pair_mouse_gene,
                ]
            except KeyError:
                skipped_missing_triplet_count += 1
                continue

            if pd.isna(expression_similarity_to_pair_human):
                skipped_missing_triplet_count += 1
                continue
            if pd.isna(expression_similarity_to_pair_mouse):
                skipped_missing_triplet_count += 1
                continue
            if pd.isna(embedding_similarity_to_pair_human):
                skipped_missing_triplet_count += 1
                continue
            if pd.isna(embedding_similarity_to_pair_mouse):
                skipped_missing_triplet_count += 1
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
                skipped_tied_triplet_count += 1
                continue

            total_triplet_count += 1

            if np.sign(expression_difference) == np.sign(embedding_difference):
                consistent_triplet_count += 1

        triplet_consistency = (
            consistent_triplet_count / total_triplet_count
            if total_triplet_count > 0
            else np.nan
        )

        output_row = {
            human_gene_column: pair_human_gene,
            mouse_gene_column: pair_mouse_gene,
            "consistent_triplet_count": consistent_triplet_count,
            "total_triplet_count": total_triplet_count,
            "skipped_missing_triplet_count": skipped_missing_triplet_count,
            "skipped_tied_triplet_count": skipped_tied_triplet_count,
            "triplet_consistency": triplet_consistency,
        }
        if pair_row_id is not None:
            output_row["pair_row_id"] = int(pair_row_id)

        consistency_rows.append(output_row)

    consistency_dataframe = pd.DataFrame(consistency_rows)

    merge_columns = [human_gene_column, mouse_gene_column]
    if "pair_row_id" in consistency_dataframe.columns and "pair_row_id" in working_dataframe.columns:
        merge_columns = ["pair_row_id", human_gene_column, mouse_gene_column]

    return working_dataframe.merge(
        consistency_dataframe,
        on=merge_columns,
        how="left",
    )


def write_triplet_consistency_summary(
    consistency_dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write a text summary for triplet consistency results."""

    total_consistent_triplets = int(
        consistency_dataframe["consistent_triplet_count"].sum()
    )
    total_valid_triplets = int(
        consistency_dataframe["total_triplet_count"].sum()
    )
    total_missing_triplets = int(
        consistency_dataframe["skipped_missing_triplet_count"].sum()
    )
    total_tied_triplets = int(
        consistency_dataframe["skipped_tied_triplet_count"].sum()
    )

    global_consistency = (
        total_consistent_triplets / total_valid_triplets
        if total_valid_triplets > 0
        else np.nan
    )

    valid_pair_consistency = consistency_dataframe["triplet_consistency"].dropna()

    lines = [
        "Triplet consistency summary",
        "===========================",
        "",
        f"Pairs evaluated: {len(consistency_dataframe)}",
        f"Pairs with at least one valid triplet: {len(valid_pair_consistency)}",
        f"Total valid triplets: {total_valid_triplets}",
        f"Total consistent triplets: {total_consistent_triplets}",
        f"Total skipped missing triplets: {total_missing_triplets}",
        f"Total skipped tied triplets: {total_tied_triplets}",
        "",
        f"Global triplet consistency: {global_consistency:.6f}",
        f"Mean pair triplet consistency: {valid_pair_consistency.mean():.6f}",
        f"Median pair triplet consistency: {valid_pair_consistency.median():.6f}",
        f"Standard deviation pair triplet consistency: {valid_pair_consistency.std():.6f}",
    ]

    label_column = None
    if "original_label" in consistency_dataframe.columns:
        label_column = "original_label"
    elif "label" in consistency_dataframe.columns:
        label_column = "label"

    if label_column is not None:
        labelled_dataframe = consistency_dataframe.copy()
        labelled_dataframe["label_normalized"] = normalise_binary_label_values(
            labelled_dataframe[label_column]
        )

        lines.extend(["", "By label", "--------"])

        for label_value, label_name in [("P", "Positive"), ("N", "Negative"), ("U", "Undefined")]:
            subset = labelled_dataframe.loc[
                labelled_dataframe["label_normalized"] == label_value,
                "triplet_consistency",
            ].dropna()

            lines.append(
                f"{label_name}: n={len(subset)} | mean={subset.mean():.6f} | "
                f"median={subset.median():.6f}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_triplet_consistency_outputs(
    evaluation_dataframe: pd.DataFrame,
    metric_column: str,
    output_table_path: Path,
    output_summary_path: Path,
) -> pd.DataFrame:
    """Compute and save triplet consistency outputs."""

    try:
        human_gene_column, mouse_gene_column = resolve_gene_pair_columns(evaluation_dataframe)

        consistency_dataframe = add_triplet_consistency(
            dataframe=evaluation_dataframe,
            human_gene_column=human_gene_column,
            mouse_gene_column=mouse_gene_column,
            expression_similarity_column=metric_column,
            embedding_similarity_column="embedding_cosine_sim",
        )

        output_table_path.parent.mkdir(parents=True, exist_ok=True)
        consistency_dataframe.to_csv(output_table_path, sep="\t", index=False)

        write_triplet_consistency_summary(
            consistency_dataframe=consistency_dataframe,
            output_path=output_summary_path,
        )

        logger.info("Triplet consistency table saved to %s", output_table_path)
        logger.info("Triplet consistency summary saved to %s", output_summary_path)

        return consistency_dataframe

    except Exception as error:
        logger.warning("Triplet consistency could not be computed: %s", error)
        output_summary_path.parent.mkdir(parents=True, exist_ok=True)
        output_summary_path.write_text(
            "Triplet consistency summary\n"
            "===========================\n\n"
            "Triplet consistency could not be computed.\n"
            f"Reason: {error}\n",
            encoding="utf-8",
        )
        return evaluation_dataframe.copy()



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

    save_triplet_consistency_outputs(
        evaluation_dataframe=evaluation_dataframe,
        metric_column=metric_column,
        output_table_path=model_dir / "triplet_consistency.tsv",
        output_summary_path=model_dir / "triplet_consistency_summary.txt",
    )

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
