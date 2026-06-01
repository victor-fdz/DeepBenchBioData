include { TRAIN_AND_EVALUATE_MODEL } from '../modules/local/train_and_evaluate_model'

workflow MODEL_TRAINING_ONLY {
    take:
    split_files_channel

    main:
    TRAIN_AND_EVALUATE_MODEL(split_files_channel)

    emit:
    model_outputs = TRAIN_AND_EVALUATE_MODEL.out.model_outputs
}
