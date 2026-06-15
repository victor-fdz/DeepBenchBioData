process NORMALIZATION_BENCHMARK {
    tag { dataset_name }

    publishDir "${params.outdir}/normalization", mode: params.publish_mode

    input:
    tuple val(dataset_name), path(raw_expression_dataset)

    output:
    tuple val(dataset_name),
          path("Normalization/best_method.txt"),
          path("Normalization"),
          path("Orthologs"),
          path("NonOrthologs"),
          path("Intermediate_Datasets"),
          emit: results

    script:
    """
    python ${projectDir}/bin/normalize_nextflow.py \
        --input ${raw_expression_dataset} \
        --dataset-name ${dataset_name} \
        --tissue ${params.normalization_tissue} \
        --selection-metric ${params.normalization_selection_metric}
    """
}
