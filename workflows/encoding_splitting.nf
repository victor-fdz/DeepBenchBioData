include { ENCODE_AND_SPLIT_PAIRS } from '../modules/local/encode_and_split_pairs'

workflow ENCODING_SPLITTING_ONLY {
    take:
    labeled_pairs_channel
    human_fasta_channel
    mouse_fasta_channel
    original_gene_list_channel

    main:
    if (!params.profiling) {
        error "Missing required parameter for standalone encoding/splitting: --profiling"
    }

    encoding_input_channel = labeled_pairs_channel.map { dataset_name, labeled_pairs ->
        tuple(dataset_name, labeled_pairs, params.profiling)
    }

    ENCODE_AND_SPLIT_PAIRS(
        encoding_input_channel,
        human_fasta_channel,
        mouse_fasta_channel,
        original_gene_list_channel
    )

    emit:
    split_files = ENCODE_AND_SPLIT_PAIRS.out.split_files
}
