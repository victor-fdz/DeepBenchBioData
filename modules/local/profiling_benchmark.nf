process PROFILING_BENCHMARK {
    tag { dataset_name }

    publishDir "${params.outdir}/profiling", mode: params.publish_mode

    input:
    tuple val(dataset_name), path(raw_expression_dataset)
    
    output:
    tuple val(dataset_name),
          path("Profiling/best_metric.txt"),
          path("Profiling/profiling_method_ranking.tsv"),
          path("Profiling/profiling_manifest.json"),
          path("Intermediate_Datasets"),
          path("Benchmarking"),
          path("Expression_vs_Sequence"),
          emit: results

    script:
    def promoter_argument = params.profiling_promoter ? "--promoter ${projectDir}/${params.profiling_promoter}" : ""

    """
    python ${projectDir}/bin/profiling_nextflow.py \
        --input ${raw_expression_dataset} \
        --dataset-name ${dataset_name} \
        ${promoter_argument}
    """
    }