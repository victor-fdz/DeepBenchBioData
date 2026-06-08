process TRAIN_AND_EVALUATE_MODEL {
    tag { dataset_name }

    publishDir "${params.outdir}/model", mode: params.publish_mode

    input:
    tuple val(dataset_name), path(train_split), path(validation_split), path(test_split), val(selected_metric)

    output:
    tuple val(dataset_name),
          path("Model"),
          path("selected_profiling.txt"),
          emit: model_outputs

    script:
    def optimize_argument = params.optimize ? "--optimize" : ""

    """
    selected_metric_input='${selected_metric}'

    if [ -f "\${selected_metric_input}" ]; then
        selected_metric=\$(cat "\${selected_metric_input}" | tr -d '[:space:]')
    else
        selected_metric="\${selected_metric_input}"
    fi

    if [ -z "\${selected_metric}" ] || [ "\${selected_metric}" = "null" ]; then
        echo "Missing model metric. Either run the profiling benchmark before model training or pass --metric." >&2
        exit 1
    fi

    echo "\${selected_metric}" > selected_profiling.txt

    python ${projectDir}/bin/model_nextflow.py \
        --train ${train_split} \
        --validation ${validation_split} \
        --test ${test_split} \
        --name ${dataset_name} \
        --metric "\${selected_metric}" \
        --batch-size ${params.batch_size} \
        --epochs ${params.epochs} \
        --learning-rate ${params.learning_rate} \
        --margin ${params.margin} \
        --dropout ${params.dropout} \
        --weight-decay ${params.weight_decay} \
        --small-kernel-size ${params.small_kernel_size} \
        --medium-kernel-size ${params.medium_kernel_size} \
        --large-kernel-size ${params.large_kernel_size} \
        --attention-heads ${params.attention_heads} \
        --embedding-dim ${params.embedding_dim} \
        --optuna-trials ${params.optuna_trials} \
        --optuna-jobs ${params.optuna_jobs} \
        --optuna-epochs ${params.optuna_epochs} \
        ${optimize_argument}
    """
}
