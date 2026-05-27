include { NORMALIZATION_BENCHMARK } from '../modules/local/normalization_benchmark'

workflow NORMALIZATION_ONLY {
    take:
    raw_expression_dataset_channel

    main:
    NORMALIZATION_BENCHMARK(raw_expression_dataset_channel)

    emit:
    results = NORMALIZATION_BENCHMARK.out.results
}
