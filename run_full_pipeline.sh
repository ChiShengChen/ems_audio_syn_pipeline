#!/bin/bash
# Full Synthetic Data Pipeline for EMS ASR Fine-Tuning
# Phase 1 -> Phase 2 -> Phase 3
set -e
cd "$(dirname "$0")"

echo "=========================================="
echo "EMS Synthetic Data Pipeline"
echo "=========================================="

./run_phase1.sh
./run_phase2.sh
./run_phase3.sh

echo ""
echo "Pipeline complete!"
echo "Final augmented manifest: phase3_output/augmented_manifest.jsonl"
echo "Use with prepare_whisper_training_data.py for Whisper fine-tuning."
