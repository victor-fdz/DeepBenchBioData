process LABEL_GENE_PAIRS {
    tag { dataset_name }

    publishDir { "${params.outdir}/${dataset_name}/Labeling" }, mode: params.publish_mode

    input:
    tuple val(dataset_name), path(raw_expression_dataset)

    output:
    tuple val(dataset_name), path("results/${dataset_name}/Labeling/${params.profiling}_labeling_*"), emit: labeled_pairs

    script:
    """
    python ${projectDir}/bin/label.py \
        --input ${raw_expression_dataset} \
        --name ${dataset_name} \
        --normalization ${params.normalization} \
        --pairing ${params.pairing} \
        --profiling ${params.profiling} \
        --labeling ${params.labeling} \
        --n-pos ${params.n_pos} \
        --n-neg ${params.n_neg} \
        --seed ${params.seed}
    """
}
