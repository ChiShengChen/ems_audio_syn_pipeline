#!/bin/bash
# Phase 3: Noise Augmentation
set -e
cd "$(dirname "$0")"
PROJECT_ROOT="${PROJECT_ROOT:-/media/meow/One Touch/ems_call}"
MANIFEST="${PROJECT_ROOT}/ems_audio_syn_pipeline/phase2_output/merged_manifest.jsonl"
OUTPUT="${PROJECT_ROOT}/ems_audio_syn_pipeline/phase3_output"
NOISE_DIR="${PROJECT_ROOT}/ems_audio_syn_pipeline/noise_samples/ems"

if [ ! -f "$MANIFEST" ]; then
    echo "Run Phase 2 first. Expected: $MANIFEST"
    exit 1
fi

echo "=== Phase 3: Noise Augmentation ==="
python3 phase3_noise_augmentation/noise_augmentation.py \
    --manifest "$MANIFEST" \
    --output_dir "$OUTPUT" \
    --num_variants 4 \
    --noise_dir "$NOISE_DIR"

echo "Phase 3 done. Manifest: ${OUTPUT}/augmented_manifest.jsonl"
