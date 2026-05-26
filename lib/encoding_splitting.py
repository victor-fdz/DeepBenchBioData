#!/usr/bin/env python3
"""Sequence loading, one-hot encoding, and train/val/test splitting for gene pair datasets."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO

logger = logging.getLogger(__name__)

# ---------------------------------------------------------- #
# CONSTANTS                                                  #
# ---------------------------------------------------------- #

SEQ_LENGTH = 500  # Default promoter sequence length

# External profiling metrics keep their raw column names.
_EXTERNAL_METRICS = {"cosine_sim", "met", "ssd", "z_score_cosine_sim"}

# One-hot encoding lookup: rows = nucleotide index, cols = A C G T
# Index 4 (unknown/N) maps to all-zero row.
_NUCLEOTIDE_IDX: dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}
_ENCODING_MATRIX = np.eye(4, dtype=np.float32)[[0, 1, 2, 3, 0]]
_ENCODING_MATRIX[4] = 0  # N / unknown → zero vector


# ---------------------------------------------------------- #
# SEQUENCE I/O                                               #
# ---------------------------------------------------------- #

def load_fasta_sequences(fasta_path: Path) -> dict[str, str]:
    """Parse a FASTA file and return a ``{ENSEMBL_ID: sequence}`` mapping.

    The ENSEMBL ID is extracted as the first ``|``-delimited token of the
    record ID. Sequences are uppercased.

    Args:
        fasta_path: Path to the FASTA file.

    Returns:
        Dictionary mapping ENSEMBL ID to uppercase DNA sequence.
    """
    sequences = {
        record.id.split("|")[0]: str(record.seq).upper()
        for record in SeqIO.parse(fasta_path, "fasta")
    }
    logger.info("Loaded %d sequences from %s", len(sequences), fasta_path)
    return sequences


def save_fasta(sequences: dict[str, str], out_path: Path) -> None:
    """Save a ``{ID: sequence}`` dictionary as a FASTA file.

    Args:
        sequences: Mapping of sequence ID to DNA sequence string.
        out_path: Destination file path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for seq_id, seq in sequences.items():
            f.write(f">{seq_id}\n{seq}\n")
    logger.info("Saved %d sequences to %s", len(sequences), out_path)


# ---------------------------------------------------------- #
# ENCODING                                                   #
# ---------------------------------------------------------- #

def one_hot_encode(sequence: str, seq_length: int = SEQ_LENGTH) -> torch.Tensor:
    """One-hot encode a DNA sequence into a ``(4, seq_length)`` float tensor.

    Sequences longer than ``seq_length`` are truncated; shorter ones are
    right-padded with ``N`` (encoded as all-zero column vectors).

    Uses vectorised numpy indexing instead of character-by-character mapping.

    Args:
        sequence: DNA sequence string (any case).
        seq_length: Fixed output length (default: ``SEQ_LENGTH``).

    Returns:
        Float tensor of shape ``(4, seq_length)``.
    """
    padded  = sequence[:seq_length].upper().ljust(seq_length, "N")
    indices = np.array([_NUCLEOTIDE_IDX.get(c, 4) for c in padded], dtype=np.intp)
    encoded = _ENCODING_MATRIX[indices]          # (seq_length, 4)
    return torch.from_numpy(encoded.T.copy())    # (4, seq_length)


def generate_random_sequences(
    seq_ids: list[str],
    seq_length: int = 500,
    seed: int = 42,
) -> dict[str, str]:
    """Generate unique random nucleotide sequences for each gene ID.

    Used as a sanity-check control: replaces real promoter sequences with
    random ones so that no sequence-based signal should be learnable.

    Args:
        seq_ids: Gene IDs for which to generate sequences.
        seq_length: Length of each random sequence (default: 500).
        seed: Random seed for reproducibility (default: 42).

    Returns:
        Dictionary mapping each gene ID to a unique random sequence.
    """
    rng        = np.random.default_rng(seed)
    nucleotides = np.array(["A", "C", "G", "T"])
    generated  = set()
    result     = {}

    for seq_id in seq_ids:
        while True:
            seq = "".join(rng.choice(nucleotides, size=seq_length))
            if seq not in generated:
                break
        generated.add(seq)
        result[seq_id] = seq

    logger.info("Generated %d unique random sequences (length=%d).", len(result), seq_length)
    return result


# ---------------------------------------------------------- #
# PAIR ENCODING                                              #
# ---------------------------------------------------------- #

def _encode_pairs(
    pairs_df: pd.DataFrame,
    profiling: str,
    human_seqs: dict[str, str],
    mouse_seqs: dict[str, str],
) -> pd.DataFrame:
    """Shared encoding logic: map gene IDs to sequences and one-hot encode them.

    Retains only the columns required for downstream model training:
    gene IDs, profiling metric score, label, sequences, and encoded tensors.

    Args:
        pairs_df: Labelled pairs dataframe (output of the labeling step).
        profiling: Name of the profiling metric column to retain.
        human_seqs: ``{ENSEMBL_ID: sequence}`` for human genes.
        mouse_seqs: ``{ENSEMBL_ID: sequence}`` for mouse genes.

    Returns:
        Dataframe with added ``human_seq``, ``mouse_seq``,
        ``human_encoded``, and ``mouse_encoded`` columns.
    """
    # Resolve metric column name (external metrics keep raw names)
    metric_col = (
        profiling
        if profiling in _EXTERNAL_METRICS
        else f"{profiling}_sim"
    )

    df = pairs_df[["gene_id_human", "gene_id_mouse", metric_col, "label"]].copy()

    # Map gene IDs to sequences
    df["human_seq"] = df["gene_id_human"].map(human_seqs)
    df["mouse_seq"] = df["gene_id_mouse"].map(mouse_seqs)

    # Numeric labels; keep "U" (undefined) as-is for downstream handling
    df["label"] = df["label"].map({"P": 1, "N": 0, "U": "U"}).fillna(df["label"])

    # One-hot encode sequences
    df["human_encoded"] = df["human_seq"].apply(one_hot_encode)
    df["mouse_encoded"] = df["mouse_seq"].apply(one_hot_encode)

    return df


def load_and_encode(
    pairs_file_path: Path,
    human_fasta: Path,
    mouse_fasta: Path,
    profiling: str,
    randomize: bool = False,
    seed: int = 42,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Load a labelled pairs file, map promoter sequences, and one-hot encode them.

    Args:
        pairs_file_path: Path to labelled pairs TSV (output of the labeling step).
        human_fasta: Path to human promoter FASTA file.
        mouse_fasta: Path to mouse promoter FASTA file.
        profiling: Profiling metric column to retain (e.g. ``"cosine_sim"``).
        randomize: If ``True``, replace real sequences with random ones (sanity check).
        seed: Random seed used when ``randomize=True`` (default: 42).
        output_dir: If provided and ``randomize=True``, saves randomized FASTA files here.

    Returns:
        Dataframe with encoded sequences, dropping rows with missing sequences.
    """
    human_seqs = load_fasta_sequences(human_fasta)
    mouse_seqs = load_fasta_sequences(mouse_fasta)

    if randomize:
        logger.info("Randomize mode: replacing real sequences with random ones (seed=%d).", seed)
        human_seqs = generate_random_sequences(list(human_seqs), seed=seed)
        mouse_seqs = generate_random_sequences(list(mouse_seqs), seed=seed + 1)

        if output_dir is not None:
            save_fasta(human_seqs, output_dir / "sequences" / "human_randomized.fasta")
            save_fasta(mouse_seqs, output_dir / "sequences" / "mouse_randomized.fasta")

    pairs_df = pd.read_csv(pairs_file_path, sep="\t")
    df = _encode_pairs(pairs_df, profiling, human_seqs, mouse_seqs)

    before = len(df)
    df = df.dropna(subset=["human_seq", "mouse_seq"]).reset_index(drop=True)
    df["pair_row_id"] = np.arange(len(df), dtype=int)
    logger.info("Encoded %d pairs (%d dropped: missing sequences).", len(df), before - len(df))

    return df


# ---------------------------------------------------------- #
# TRAIN / VAL / TEST SPLITTING                               #
# ---------------------------------------------------------- #

def split_pairs(
    pairs_df: pd.DataFrame,
    mode: str,
    output_dir: Path,
    gene_list_path: Path | None = None,
    seed: int = 42,
    val_frac: float = 0.1,
    test_frac: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split encoded gene pairs into train, validation, and test sets.

    Three split strategies are supported:

    - ``"leaked"``: Random pair-level split. Genes are shared across sets.
      Fastest, but inflates performance metrics — use as a ceiling baseline.
    - ``"semi_leakage"``: Genes are split independently per species, then
      cross-species pairs are assigned to val/test. Partial gene leakage.
    - ``"anti_leakage"``: Strict gene-level split. No gene appears in more
      than one set. Requires ``gene_list_path`` to the original dataset.
      Most realistic evaluation scenario.

    In all modes, ``"U"`` (undefined) labels in the test set are randomly
    assigned 0 or 1 to allow evaluation on ambiguous pairs.

    Args:
        pairs_df: Encoded pairs dataframe (output of ``load_and_encode``).
        mode: Split strategy — one of ``"leaked"``, ``"semi_leakage"``,
            ``"anti_leakage"``.
        output_dir: Directory where ``train.tsv``, ``val.tsv``, ``test.tsv``
            are saved.
        gene_list_path: Path to the original paired dataset TSV. Required
            for ``"anti_leakage"`` mode to derive the gene universe.
        seed: Random seed (default: 42).
        val_frac: Fraction of genes assigned to validation (default: 0.1).
        test_frac: Fraction of genes assigned to test (default: 0.05).

    Returns:
        Tuple of (df_train, df_val, df_test).

    Raises:
        ValueError: If ``mode`` is not recognised or ``gene_list_path`` is
            missing for ``"anti_leakage"`` mode.
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # MODE 1: LEAKED — random pair-level split                            #
    # ------------------------------------------------------------------ #
    if mode == "leaked":
        from sklearn.model_selection import train_test_split

        labeled = pairs_df[pairs_df["label"].isin([0, 1])].copy()
        labeled["label"] = labeled["label"].astype(int)

        df_train, df_temp = train_test_split(labeled,  test_size=val_frac + test_frac,                   stratify=labeled ["label"], random_state=seed)
        df_val,   df_rest = train_test_split(df_temp,  test_size=test_frac / (val_frac + test_frac),     stratify=df_temp ["label"], random_state=seed)

        # Test: labeled remainder + all undefined pairs (P, N, U)
        undefined = pairs_df[pairs_df["label"] == "U"].copy()
        df_test   = pd.concat([df_rest, undefined], ignore_index=True)
        df_test["pair_row_id"] = np.arange(len(df_test), dtype=int)

    # ------------------------------------------------------------------ #
    # MODE 2: ANTI_LEAKAGE — strict gene-level split                     #
    # ------------------------------------------------------------------ #
    elif mode == "anti_leakage":
        if gene_list_path is None:
            raise ValueError("anti_leakage mode requires gene_list_path to the original paired dataset.")

        gene_df  = pd.read_csv(gene_list_path, sep="\t")
        n_genes  = len(gene_df)
        n_val    = max(1, round(n_genes * val_frac))
        n_test   = max(1, round(n_genes * test_frac))
        n_train  = n_genes - n_val - n_test

        genes_train = gene_df.sample(n=n_train + n_val, random_state=seed)
        genes_test  = gene_df.drop(genes_train.index)
        genes_val   = genes_train.sample(n=n_val, random_state=seed)
        genes_train = genes_train.drop(genes_val.index)

        train_h = set(genes_train["gene_id_human"])
        train_m = set(genes_train["gene_id_mouse"])
        val_h   = set(genes_val  ["gene_id_human"])
        val_m   = set(genes_val  ["gene_id_mouse"])
        test_h  = set(genes_test ["gene_id_human"])
        test_m  = set(genes_test ["gene_id_mouse"])

        df_train = pairs_df[pairs_df["gene_id_human"].isin(train_h) & pairs_df["gene_id_mouse"].isin(train_m)].copy()
        df_val   = pairs_df[pairs_df["gene_id_human"].isin(val_h)   & pairs_df["gene_id_mouse"].isin(val_m)  ].copy()
        df_test  = pairs_df[pairs_df["gene_id_human"].isin(test_h)  & pairs_df["gene_id_mouse"].isin(test_m) ].copy()

        # Train/val: labeled pairs only. Test: all pairs (P=1, N=0, U kept as-is).
        df_train = df_train[df_train["label"].isin([0, 1])]
        df_val   = df_val  [df_val  ["label"].isin([0, 1])]
        df_test["pair_row_id"] = np.arange(len(df_test), dtype=int)

    # ------------------------------------------------------------------ #
    # MODE 3: SEMI_LEAKAGE — species-independent gene split              #
    # ------------------------------------------------------------------ #
    elif mode == "semi_leakage":
        unique_human = pd.Series(pairs_df["gene_id_human"].dropna().unique())
        unique_mouse = pd.Series(pairs_df["gene_id_mouse"].dropna().unique())

        def _split_gene_series(genes: pd.Series, val_frac: float, test_frac: float, seed: int) -> tuple:
            n = len(genes)
            n_val  = max(1, round(n * val_frac))
            n_test = max(1, round(n * test_frac))
            g_train = genes.sample(n=n - n_val - n_test, random_state=seed)
            remain  = genes[~genes.isin(g_train)]
            g_val   = remain.sample(n=n_val, random_state=seed)
            g_test  = remain[~remain.isin(g_val)]
            return set(g_train), set(g_val), set(g_test)

        train_h, val_h, test_h = _split_gene_series(unique_human, val_frac, test_frac, seed)
        train_m, val_m, test_m = _split_gene_series(unique_mouse, val_frac, test_frac, seed)

        df_train = pairs_df[pairs_df["gene_id_human"].isin(train_h) & pairs_df["gene_id_mouse"].isin(train_m)].copy()
        df_val   = pairs_df[
            (pairs_df["gene_id_human"].isin(val_h)  & pairs_df["gene_id_mouse"].isin(train_m)) |
            (pairs_df["gene_id_human"].isin(train_h) & pairs_df["gene_id_mouse"].isin(val_m))
        ].copy()
        df_test  = pairs_df[
            (pairs_df["gene_id_human"].isin(test_h)  & pairs_df["gene_id_mouse"].isin(train_m)) |
            (pairs_df["gene_id_human"].isin(train_h) & pairs_df["gene_id_mouse"].isin(test_m))
        ].copy()

        # Train/val: labeled pairs only. Test: all pairs (P=1, N=0, U kept as-is).
        df_train = df_train[df_train["label"].isin([0, 1])]
        df_val   = df_val  [df_val  ["label"].isin([0, 1])]
        df_test["pair_row_id"] = np.arange(len(df_test), dtype=int)

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: 'leaked', 'anti_leakage', 'semi_leakage'.")

    # ------------------------------------------------------------------ #
    # SHARED: log summary and save                                        #
    # ------------------------------------------------------------------ #
    # Keep original labels for downstream plotting before randomizing U in test.
    for split in [df_train, df_val, df_test]:
        if "label" in split.columns and "original_label" not in split.columns:
            split["original_label"] = split["label"]

    # Replace undefined labels in test with random 0/1 for evaluation.
    if "label" in df_test.columns:
        mask = df_test["label"] == "U"
        if mask.any():
            df_test.loc[mask, "label"] = rng.integers(0, 2, size=int(mask.sum()))
            df_test["label"] = df_test["label"].astype(int)

    for name, split in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        logger.info("[%s | %s] %d pairs — labels: %s", mode, name, len(split), dict(split["label"].value_counts()))

    out_dir = output_dir / "splits" / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(out_dir / "train.tsv", sep="\t", index=False)
    df_val  .to_csv(out_dir / "val.tsv",   sep="\t", index=False)
    df_test .to_csv(out_dir / "test.tsv",  sep="\t", index=False)
    logger.info("Splits saved to %s", out_dir)

    return df_train, df_val, df_test