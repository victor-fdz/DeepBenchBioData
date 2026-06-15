include { NORMALIZATION_BENCHMARK } from '../modules/local/normalization_benchmark'
include { DATA_EXPLORATION_BEFORE_NORMALIZATION; DATA_EXPLORATION_AFTER_NORMALIZATION } from '../modules/local/data_exploration'
include { PROFILING_BENCHMARK } from '../modules/local/profiling_benchmark'
include { LABEL_GENE_PAIRS } from '../modules/local/label_gene_pairs'
include { ENCODE_AND_SPLIT_PAIRS } from '../modules/local/encode_and_split_pairs'
include { TRAIN_AND_EVALUATE_MODEL } from '../modules/local/train_and_evaluate_model'

workflow FULL_PIPELINE {
    take:
    raw_expression_dataset_channel
    human_fasta_channel
    mouse_fasta_channel
    original_gene_list_channel

    main:
    /*
     * Step 0: raw data exploration before normalization.
     */
    DATA_EXPLORATION_BEFORE_NORMALIZATION(raw_expression_dataset_channel)

    /*
     * Step 1: benchmarking.
     * Normalization and profiling start in parallel from the same raw input.
     */
    NORMALIZATION_BENCHMARK(raw_expression_dataset_channel)
    PROFILING_BENCHMARK(raw_expression_dataset_channel)

    /*
     * Step 1b: normalized data exploration.
     * Runs after normalization, using the selected normalization method.
     */
    data_exploration_after_input_channel = NORMALIZATION_BENCHMARK.out.results.map { result_tuple ->
        tuple(result_tuple[0], result_tuple[1], result_tuple[3])
    }

    DATA_EXPLORATION_AFTER_NORMALIZATION(data_exploration_after_input_channel)

    selected_normalization_channel = NORMALIZATION_BENCHMARK.out.results.map { result_tuple ->
        tuple(result_tuple[0], result_tuple[1])
    }

    selected_profiling_channel = PROFILING_BENCHMARK.out.results.map { result_tuple ->
        tuple(result_tuple[0], result_tuple[1])
    }

    /*
     * Step 2: labeling.
     * LABEL_GENE_PAIRS accepts either method names or method text files.
     */
    labeling_input_channel = raw_expression_dataset_channel
        .join(selected_normalization_channel)
        .join(selected_profiling_channel)

    LABEL_GENE_PAIRS(labeling_input_channel)

    /*
     * Step 3: encoding and splitting.
     * ENCODE_AND_SPLIT_PAIRS accepts either a method name or selected_profiling.txt.
     */
    labeled_pairs_channel = LABEL_GENE_PAIRS.out.labeled_pairs.map { result_tuple ->
        tuple(result_tuple[0], result_tuple[1])
    }

    encoding_input_channel = labeled_pairs_channel.join(selected_profiling_channel)

    ENCODE_AND_SPLIT_PAIRS(
        encoding_input_channel,
        human_fasta_channel,
        mouse_fasta_channel,
        original_gene_list_channel
    )

    /*
     * Step 4: model training/evaluation.
     * Uses train/validation/test splits plus selected_profiling.txt written by encoding.
     */
    model_input_channel = ENCODE_AND_SPLIT_PAIRS.out.split_files.map { result_tuple ->
        tuple(
            result_tuple[0],
            result_tuple[1],
            result_tuple[2],
            result_tuple[3],
            result_tuple[4]
        )
    }

    TRAIN_AND_EVALUATE_MODEL(model_input_channel)

    emit:
    data_exploration_before = DATA_EXPLORATION_BEFORE_NORMALIZATION.out.before_outputs
    normalization = NORMALIZATION_BENCHMARK.out.results
    data_exploration_after = DATA_EXPLORATION_AFTER_NORMALIZATION.out.after_outputs
    profiling = PROFILING_BENCHMARK.out.results
    labeling = LABEL_GENE_PAIRS.out.labeled_pairs
    encoding = ENCODE_AND_SPLIT_PAIRS.out.split_files
    model = TRAIN_AND_EVALUATE_MODEL.out.model_outputs
}
