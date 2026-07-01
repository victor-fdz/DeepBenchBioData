#!/usr/bin/env python3

import argparse
import pandas as pd
from Bio.Align import PairwiseAligner
from Bio import SeqIO


# ---------------------------------------
# Arguments
# ---------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute pairwise global alignments between human and mouse promoter FASTA files."
    )

    parser.add_argument(
        "--human-fasta",
        required=True,
        help="Path to human promoter FASTA file.",
    )

    parser.add_argument(
        "--mouse-fasta",
        required=True,
        help="Path to mouse promoter FASTA file.",
    )

    return parser.parse_args()


# ---------------------------------------
# Define functions
# ---------------------------------------

def compute_identity(alignment):
    matches = sum(
        a == b
        for a, b in zip(alignment[0], alignment[1])
        if a != "-" and b != "-"
    )
    alignment_length = alignment.length
    return (matches / alignment_length) * 100 if alignment_length > 0 else 0


def compute_alignment_all_promoters(promoters_name, promoters_name_mouse):
    aligner = PairwiseAligner()
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.5
    aligner.mode = "global"

    results = []
    not_in_fasta = 0

    for gene_human in promoters_name.keys():
        for gene_mouse in promoters_name_mouse.keys():
            print(f"aligning {gene_human} and {gene_mouse}")
            alignments = aligner.align(
                promoters_name[gene_human],
                promoters_name_mouse[gene_mouse],
            )
            alignment = alignments[0]
            score = alignment.score
            identity = compute_identity(alignment)
            results.append((gene_human, gene_mouse, score, identity))
        else:
            not_in_fasta += 1

    df = pd.DataFrame(
        results,
        columns=[
            "gene_id_human",
            "gene_id_mouse",
            "alignment_score",
            "promoter_identity",
        ],
    )

    print(f"{not_in_fasta} gene pairs not found in the FASTA file")
    print(f"{len(results)} alignments computed")

    return df


# ---------------------------------------
# Main
# ---------------------------------------

def main():
    args = parse_args()

    sequences = {}
    for record in SeqIO.parse(args.human_fasta, "fasta"):
        gene_id = record.id
        print(gene_id)
        sequences[gene_id] = str(record.seq)

    sequences_mouse = {}
    for record in SeqIO.parse(args.mouse_fasta, "fasta"):
        gene_id = record.id
        print(gene_id)
        sequences_mouse[gene_id] = str(record.seq)

    print(f"Sequence keys match: {sequences.keys() == sequences_mouse.keys()}")

    alignments = compute_alignment_all_promoters(
        sequences,
        sequences_mouse,
    )

    alignments.to_csv(
        "kinases_promoter_alignment.tsv",
        sep="\t",
        index=False,
    )


if __name__ == "__main__":
    main()