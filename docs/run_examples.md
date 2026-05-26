# Nextflow run examples

## Local run with Conda

```bash
nextflow run main.nf \
    -profile local,conda \
    --input data/kinases.tsv \
    --name kinases \
    --human_fasta data/human_promoters.fasta \
    --mouse_fasta data/mouse_promoters.fasta \
    --normalization quantile_norm \
    --pairing cross_species \
    --profiling cosine_sim \
    --labeling rank_labeling_random \
    --split_mode anti_leakage \
    --n_pos 10000 \
    --n_neg 10000 \
    --seed 42
```

## Resume a failed or interrupted run

```bash
nextflow run main.nf -profile local,conda -resume
```

## Run with randomized promoter sequences

```bash
nextflow run main.nf \
    -profile local,conda \
    --input data/kinases.tsv \
    --name kinases_random_sequences \
    --human_fasta data/human_promoters.fasta \
    --mouse_fasta data/mouse_promoters.fasta \
    --normalization quantile_norm \
    --profiling cosine_sim \
    --random_seqs true
```

## Run on a Slurm cluster

```bash
nextflow run main.nf \
    -profile slurm,conda \
    --input data/kinases.tsv \
    --name kinases \
    --human_fasta data/human_promoters.fasta \
    --mouse_fasta data/mouse_promoters.fasta
```
