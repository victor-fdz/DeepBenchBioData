process LABEL_GENE_PAIRS_FROM_BENCHMARKS {
    tag { dataset_name }

    publishDir "${params.outdir}/labeling", mode: params.publish_mode

    input:
    tuple val(dataset_name), path(raw_expression_dataset), path(best_normalization_file), path(best_profiling_file)

    output:
    tuple val(dataset_name),
          path("labeled_pairs.tsv"),
          path("selected_normalization.txt"),
          path("selected_profiling.txt"),
          path("Labeling"),
          path("Intermediate_Datasets"),
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

    labeled_count=\$(find Labeling -maxdepth 1 -type f -name "*.tsv" | wc -l)
    if [ "\${labeled_count}" -ne 1 ]; then
        echo "Expected exactly one labeled TSV in Labeling/, found \${labeled_count}" >&2
        find Labeling -maxdepth 1 -type f -name "*.tsv" >&2
        exit 1
    fi

    cp \$(find Labeling -maxdepth 1 -type f -name "*.tsv") labeled_pairs.tsv
    """
}
