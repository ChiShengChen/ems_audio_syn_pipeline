#!/bin/bash
# Phase 2: Speech Synthesis (70% XTTS, 20% Bark, 10% Edge)
set -e
cd "$(dirname "$0")"
PROJECT_ROOT="${PROJECT_ROOT:-/media/meow/One Touch/ems_call}"
CORPUS="${PROJECT_ROOT}/ems_audio_syn_pipeline/phase1_output/combined_corpus.jsonl"
OUTPUT="${PROJECT_ROOT}/ems_audio_syn_pipeline/phase2_output"
AUDIO_DIRS=(
    "${PROJECT_ROOT}/random_samples_1"
    "${PROJECT_ROOT}/random_samples_2"
)

if [ ! -f "$CORPUS" ]; then
    echo "Run Phase 1 first. Expected: $CORPUS"
    exit 1
fi

echo "=== Phase 2: Speech Synthesis ==="
python3 phase2_speech_synthesis/tts_orchestrator.py \
    --corpus "$CORPUS" \
    --output "$OUTPUT" \
    --human_csv "${PROJECT_ROOT}/vb_ems_anotation/human_anotation_vb.csv" \
    --audio_dirs "${AUDIO_DIRS[@]}" \
    --device cuda

echo "Phase 2 done. Manifest: ${OUTPUT}/merged_manifest.jsonl"
