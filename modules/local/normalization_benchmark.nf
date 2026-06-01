process NORMALIZATION_BENCHMARK {
    tag { dataset_name }

    publishDir "${params.outdir}/normalization", mode: params.publish_mode

    input:
    tuple val(dataset_name), path(raw_expression_dataset)

    output:
    tuple val(dataset_name),
          path("Normalization/best_method.txt"),
          path("Normalization/normalization_method_ranking.tsv"),
          path("Normalization/normalization_manifest.json"),
          path("Normalization/${dataset_name}_stats.tsv"),
          path("Normalization/Orthologs_scatter.png"),
          path("Normalization/NonOrthologs_scatter.png"),
          path("Normalization/Increment_PearsonR_heatmap.png"),
          path("Orthologs"),
          path("NonOrthologs"),
          path("Intermediate_Datasets"),
          emit: results

    script:
    def selected_methods_argument = params.normalization_methods ? "--methods ${params.normalization_methods}" : ""

    """
    python ${projectDir}/bin/normalize_nextflow.py \
        --input ${raw_expression_dataset} \
        --dataset-name ${dataset_name} \
        --tissue ${params.normalization_tissue} \
        --selection-metric ${params.normalization_selection_metric} \
        ${selected_methods_argument}
    """
}
