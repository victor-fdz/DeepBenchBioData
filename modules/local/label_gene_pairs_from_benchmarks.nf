process LABEL_GENE_PAIRS_FROM_BENCHMARKS {
    tag { dataset_name }

    publishDir "${params.outdir}/labeling", mode: params.publish_mode

    input:
    tuple val(dataset_name), path(raw_expression_dataset), path(best_normalization_file), path(best_profiling_file)

    output:
    tuple val(dataset_name),
          path("Labeling/*.tsv"),
          path("Labeling/*.png"),
          path("Labeling/labeling_manifest.json"),
          path("Intermediate_Datasets"),
          path("selected_normalization.txt"),
          path("selected_profiling.txt"),
          emit: labeled_pairs

    script:
    """
    selected_normalization=\$(cat ${best_normalization_file} | tr -d '[:space:]')
    selected_profiling=\$(cat ${best_profiling_file} | tr -d '[:space:]')

    echo "\${selected_normalization}" > selected_normalization.txt
    echo "\${selected_profiling}" > selected_profiling.txt

    python ${projectDir}/bin/label_nextflow.py \
        --input ${raw_expression_dataset} \
        --dataset-name ${dataset_name} \
        --normalization "\${selected_normalization}" \
        --pairing ${params.pairing} \
        --profiling "\${selected_profiling}" \
        --labeling ${params.labeling} \
        --n-pos ${params.n_pos} \
        --n-neg ${params.n_neg} \
        --seed ${params.seed}
    """
}
