process ENCODE_AND_SPLIT_PAIRS_FROM_BENCHMARKS {
    tag { dataset_name }

    publishDir "${params.outdir}/encoding", mode: params.publish_mode

    input:
    tuple val(dataset_name), path(labeled_pairs), path(selected_profiling_file)
    path human_fasta
    path mouse_fasta
    path original_gene_list

    output:
    tuple val(dataset_name),
          path("splits/${params.split_mode}/train.tsv"),
          path("splits/${params.split_mode}/val.tsv"),
          path("splits/${params.split_mode}/test.tsv"),
          path("selected_profiling.txt"),
          path("encoding_manifest.json"),
          path("sequences"),
          emit: split_files

    script:
    def random_sequences_argument = params.random_seqs ? "--random-seqs" : ""

    """
    selected_profiling=\$(cat ${selected_profiling_file} | tr -d '[:space:]')
    echo "\${selected_profiling}" > selected_profiling.txt

    python ${projectDir}/bin/encode_split_nextflow.py \
        --input ${labeled_pairs} \
        --dataset-name ${dataset_name} \
        --human-fasta ${human_fasta} \
        --mouse-fasta ${mouse_fasta} \
        --profiling "\${selected_profiling}" \
        --split-mode ${params.split_mode} \
        --gene-list ${original_gene_list} \
        --val-frac ${params.val_frac} \
        --test-frac ${params.test_frac} \
        --seed ${params.seed} \
        ${random_sequences_argument}
    """
}
