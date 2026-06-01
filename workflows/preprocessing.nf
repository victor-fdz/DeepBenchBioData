include { NORMALIZATION_BENCHMARK } from '../modules/local/normalization_benchmark'
include { PROFILING_BENCHMARK } from '../modules/local/profiling_benchmark'
include { LABEL_GENE_PAIRS_FROM_BENCHMARKS } from '../modules/local/label_gene_pairs_from_benchmarks'

workflow PREPROCESSING_BENCHMARK_AND_LABELING {
    take:
    raw_expression_dataset_channel

    main:
    NORMALIZATION_BENCHMARK(raw_expression_dataset_channel)
    PROFILING_BENCHMARK(raw_expression_dataset_channel)

    selected_normalization_channel = NORMALIZATION_BENCHMARK.out.results.map { result_tuple ->
        tuple(result_tuple[0], result_tuple[1])
    }

    selected_profiling_channel = PROFILING_BENCHMARK.out.results.map { result_tuple ->
        tuple(result_tuple[0], result_tuple[1])
    }

    labeling_input_channel = raw_expression_dataset_channel
        .join(selected_normalization_channel)
        .join(selected_profiling_channel)

    LABEL_GENE_PAIRS_FROM_BENCHMARKS(labeling_input_channel)

    emit:
    normalization = NORMALIZATION_BENCHMARK.out.results
    profiling = PROFILING_BENCHMARK.out.results
    labeled_pairs = LABEL_GENE_PAIRS_FROM_BENCHMARKS.out.labeled_pairs
}
