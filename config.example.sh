#!/bin/bash
# Copy to config.sh and edit paths for your environment
# Usage: source config.sh && ./run_phase1.sh

export PROJECT_ROOT="${PROJECT_ROOT:-/path/to/ems_call}"
export HUMAN_CSV="${PROJECT_ROOT}/vb_ems_anotation/human_anotation_vb.csv"
export AUDIO_DIRS="${PROJECT_ROOT}/random_samples_1 ${PROJECT_ROOT}/random_samples_2"
export NOISE_DIR="${PROJECT_ROOT}/ems_audio_syn_pipeline/noise_samples/ems"
