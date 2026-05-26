#!/usr/bin/env nextflow

include { PROMOTER_SEQUENCE_PIPELINE } from './workflows/main.nf'

workflow {
    PROMOTER_SEQUENCE_PIPELINE()
}
