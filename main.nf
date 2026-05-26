#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

include { PROMOTER_SEQUENCE_PIPELINE } from './workflows/main.nf'

workflow {
    PROMOTER_SEQUENCE_PIPELINE()
}
