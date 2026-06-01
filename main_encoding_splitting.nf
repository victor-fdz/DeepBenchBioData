nextflow.enable.dsl = 2

include { ENCODING_SPLITTING_ONLY } from './workflows/encoding_splitting'

workflow {
    if (!params.input) {
        error "Missing required parameter: --input"
    }

    if (!params.name) {
        error "Missing required parameter: --name"
    }

    if (!params.human_fasta) {
        error "Missing required parameter: --human_fasta"
    }

    if (!params.mouse_fasta) {
        error "Missing required parameter: --mouse_fasta"
    }

    if (!params.gene_list) {
        error "Missing required parameter: --gene_list"
    }

    Channel
        .fromPath(params.input, checkIfExists: true)
        .map { labeled_pairs -> tuple(params.name, labeled_pairs) }
        .set { labeled_pairs_channel }

    Channel
        .fromPath(params.human_fasta, checkIfExists: true)
        .set { human_fasta_channel }

    Channel
        .fromPath(params.mouse_fasta, checkIfExists: true)
        .set { mouse_fasta_channel }

    Channel
        .fromPath(params.gene_list, checkIfExists: true)
        .set { original_gene_list_channel }

    ENCODING_SPLITTING_ONLY(
        labeled_pairs_channel,
        human_fasta_channel,
        mouse_fasta_channel,
        original_gene_list_channel
    )
}
