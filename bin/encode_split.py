#!/usr/bin/env python3
"""Encode promoter sequences and split labelled gene pairs into train/val/test sets.

Reads the labelled pairs TSV produced by the labeling step, maps each gene ID
to its promoter sequence from FASTA files, applies one-hot encoding, and
produces stratified train/val/test splits.

Three split strategies are available:
  - leaked:       Random pair-level split (ceiling baseline, not realistic).
  - semi_leakage: Gene-level split per species independently.
  - anti_leakage: Strict gene-level split, no gene shared across sets.

Optionally replaces real sequences with random ones to verify
that learned signals are sequence-derived and not artefactual.

Usage:
    python bin/encode_split.py \\
        --input  \\
        --name    \\
        --human-fasta  \\
        --mouse-fasta \\
        --profiling  \\
        --split-mode  \\
        --gene-list 
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to Python path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import encoding_splitting as enc_splt

logging.basicConfig(level=logging.INFO, format="\033[1;32m%(levelname)s\033[0m | \033[1;36m%(name)s\033[0m | %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",            required=True, type=Path, help="Labelled pairs TSV (output of the labeling step).")
    parser.add_argument("--name",             required=True,            help="Dataset name (used for output naming).")
    parser.add_argument("--human-fasta",      required=True, type=Path, help="Human promoter FASTA file.")
    parser.add_argument("--mouse-fasta",      required=True, type=Path, help="Mouse promoter FASTA file.")
    parser.add_argument("--profiling", required=True,            help="Profiling metric column to retain (e.g. cosine_sim).")
    parser.add_argument("--split-mode",       required=True, choices=["leaked", "semi_leakage", "anti_leakage", "species_holdout","species_transfer_antileakage"],
                                                                         help="Train/val/test split strategy.")
    parser.add_argument("--train-species",    default="human", choices=["human", "mouse"],
                                                                         help="Species used for training when --split-mode species_holdout or species_transfer_antileakage.")
    parser.add_argument("--gene-list",        default=None,  type=Path, help="Original paired dataset TSV. Required for anti_leakage split.")
    parser.add_argument("--random-seqs",          action="store_true",      help="Replace real sequences with random ones (sanity check).")
    parser.add_argument("--val-frac",         default=0.1, type=float,help="Fraction of genes used for validation (default: 0.2).")
    parser.add_argument("--test-frac",        default=0.05, type=float,help="Fraction of genes used for testing (default: 0.2).")
    parser.add_argument("--seed",             default=0,    type=int,  help="Random seed for reproducibility (default: 42).")
    return parser.parse_args()


def main() -> tuple:
    args = parse_args()

    if args.split_mode == "anti_leakage" and args.gene_list is None:
        raise ValueError("--gene-list is required when --split-mode is anti_leakage.")

    out_dir = OUTPUT_DIR / args.name

    # 1. Load sequences, encode pairs, and optionally apply sequence randomization.
    pairs_df = enc_splt.load_and_encode(
        pairs_file_path=args.input,
        human_fasta=args.human_fasta,
        mouse_fasta=args.mouse_fasta,
        profiling=args.profiling,
        randomize=args.random_seqs,
        seed=args.seed,
        output_dir=out_dir if args.random_seqs else None,
    )

    # 2. Split into train / val / test.
    df_train, df_val, df_test = enc_splt.split_pairs(
        pairs_df=pairs_df,
        mode=args.split_mode,
        output_dir=out_dir,
        gene_list_path=args.gene_list,
        seed=args.seed,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        # train_species=args.train_species,
    )

    return df_train, df_val, df_test


if __name__ == "__main__":
    main()
