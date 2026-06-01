include { ENCODE_AND_SPLIT_PAIRS } from '../modules/local/encode_and_split_pairs'

workflow ENCODING_SPLITTING_ONLY {
    take:
    labeled_pairs_channel
    human_fasta_channel
    mouse_fasta_channel
    original_gene_list_channel

    main:
    ENCODE_AND_SPLIT_PAIRS(
        labeled_pairs_channel,
        human_fasta_channel,
        mouse_fasta_channel,
        original_gene_list_channel
    )

    emit:
    split_files = ENCODE_AND_SPLIT_PAIRS.out.split_files
}
