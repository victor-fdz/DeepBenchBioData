include { PROFILING_BENCHMARK } from '../modules/local/profiling_benchmark'

workflow PROFILING_ONLY {
    take:
    raw_expression_dataset_channel

    main:
    PROFILING_BENCHMARK(raw_expression_dataset_channel)

    emit:
    results = PROFILING_BENCHMARK.out.results
}
