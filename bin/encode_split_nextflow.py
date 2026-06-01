#!/usr/bin/env python3
"""Nextflow wrapper for promoter sequence encoding and split generation.

This script reuses the project encoding/splitting module:
- lib.encoding_splitting

It only changes what is required for Nextflow:
- process-local outputs
- stable output folders declared by Nextflow
- TSV outputs keep raw sequence columns, not serialized tensor columns
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import encoding_splitting as enc_splt

logging.basicConfig(
    level=logging.INFO,
    format="\033[1;32m%(levelname)s\033[0m | \033[1;36m%(name)s\033[0m | %(message)s",
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(".")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Labeled pairs TSV produced by the labeling step.",
    )

    parser.add_argument(
        "--dataset-name",
        "--name",
        dest="dataset_name",
        required=True,
        help="Dataset name used for metadata.",
    )

    parser.add_argument(
        "--human-fasta",
        required=True,
        type=Path,
        help="Human promoter FASTA file.",
    )

    parser.add_argument(
        "--mouse-fasta",
        required=True,
        type=Path,
        help="Mouse promoter FASTA file.",
    )

    parser.add_argument(
        "--profiling",
        required=True,
        help="Profiling metric column to retain.",
    )

    parser.add_argument(
        "--split-mode",
        required=True,
        choices=[
            "leaked",
            "semi_leakage",
            "anti_leakage",
        ],
        help="Train/validation/test split strategy.",
    )

    parser.add_argument(
        "--gene-list",
        default=None,
        type=Path,
        help="Original paired dataset TSV. Required for anti_leakage split.",
    )

    parser.add_argument(
        "--random-seqs",
        action="store_true",
        help="Replace real sequences with random sequences.",
    )

    parser.add_argument(
        "--val-frac",
        default=0.1,
        type=float,
        help="Fraction of genes used for validation.",
    )

    parser.add_argument(
        "--test-frac",
        default=0.05,
        type=float,
        help="Fraction of genes used for testing.",
    )

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed.",
    )

    return parser.parse_args()


def remove_serialized_tensor_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Drop in-memory tensor columns before writing TSV splits.

    The model dataset encodes raw sequence columns at runtime. Keeping tensor
    objects in TSV files creates huge and fragile serialized string columns.
    """

    columns_to_drop = [
        column
        for column in [
            "human_encoded",
            "mouse_encoded",
            "encoded_1",
            "encoded_2",
        ]
        if column in dataframe.columns
    ]

    if columns_to_drop:
        logger.info(
            "Dropping encoded tensor columns before TSV writing: %s",
            ", ".join(columns_to_drop),
        )

        return dataframe.drop(columns=columns_to_drop)

    return dataframe


def copy_split_outputs(split_mode: str) -> None:
    """Expose split output directory at a stable Nextflow-declared path."""

    source_dir = OUTPUT_DIR / "splits" / split_mode
    target_dir = Path("splits") / split_mode

    if source_dir.resolve() == target_dir.resolve():
        return

    if target_dir.exists():
        shutil.rmtree(target_dir)

    shutil.copytree(source_dir, target_dir)


def main() -> Path:
    args = parse_args()

    if args.split_mode == "anti_leakage" and args.gene_list is None:
        raise ValueError("--gene-list is required when --split-mode is anti_leakage.")

    Path("sequences").mkdir(parents=True, exist_ok=True)

    logger.info("Loading and encoding labeled pairs: %s", args.input)

    encoded_pairs = enc_splt.load_and_encode(
        pairs_file_path=args.input,
        human_fasta=args.human_fasta,
        mouse_fasta=args.mouse_fasta,
        profiling=args.profiling,
        randomize=args.random_seqs,
        seed=args.seed,
        output_dir=OUTPUT_DIR if args.random_seqs else None,
    )

    encoded_pairs = remove_serialized_tensor_columns(encoded_pairs)

    logger.info("Generating %s train/validation/test splits.", args.split_mode)

    train_dataframe, validation_dataframe, test_dataframe = enc_splt.split_pairs(
        pairs_df=encoded_pairs,
        mode=args.split_mode,
        output_dir=OUTPUT_DIR,
        gene_list_path=args.gene_list,
        seed=args.seed,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )

    split_dir = OUTPUT_DIR / "splits" / args.split_mode

    manifest = {
        "input": str(args.input),
        "dataset_name": args.dataset_name,
        "human_fasta": str(args.human_fasta),
        "mouse_fasta": str(args.mouse_fasta),
        "profiling": args.profiling,
        "split_mode": args.split_mode,
        "gene_list": str(args.gene_list) if args.gene_list else None,
        "random_sequences": args.random_seqs,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "seed": args.seed,
        "train_rows": len(train_dataframe),
        "validation_rows": len(validation_dataframe),
        "test_rows": len(test_dataframe),
        "output_policy": "raw sequence columns are kept; tensor columns are not serialized to TSV",
    }

    manifest_path = Path("encoding_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    copy_split_outputs(args.split_mode)

    logger.info("Split files saved to %s", split_dir)
    logger.info("Encoding/splitting outputs generated in process working directory.")

    return split_dir


if __name__ == "__main__":
    main()
