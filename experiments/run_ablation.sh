#!/bin/bash
# =============================================================
# Speaker Profile + Augmentation Ablation Experiments
# =============================================================
#
# Experiment Design (2x2 factorial):
# ┌─────────────────┬──────────────────────┬──────────────────────┐
# │                  │ Original Augment     │ Enhanced Augment     │
# │                  │ (current pipeline)   │ (+ColorNoise+Gain)   │
# ├─────────────────┼──────────────────────┼──────────────────────┤
# │  5 speakers     │ A_baseline           │ C_enhanced_aug       │
# │  (current)      │                      │                      │
# ├─────────────────┼──────────────────────┼──────────────────────┤
# │ 20 speakers     │ B_more_speakers      │ D_combined           │
# │  (improved)     │                      │                      │
# └─────────────────┴──────────────────────┴──────────────────────┘
#
# Scale: 50 utterances x 4 variants = 200 clips per group
# Total: 800 augmented clips across 4 experiments
#
# Data sources:
#   - Speaker refs: random_samples_1/ + random_samples_2/ (50 real EMS calls)
#   - Text corpus:  phase1_output/ems_radio_500.jsonl (50 randomly selected)
#
# Usage:
#   ./run_ablation.sh              # Full pipeline (synth + augment + eval)
#   ./run_ablation.sh --skip_eval  # Generate audio only (no Whisper eval)
#   ./run_ablation.sh --phase augment  # Run only augmentation phase
# =============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate the conda environment with all dependencies
CONDA_ENV="${CONDA_ENV:-pytorch291}"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
echo "Using conda env: $CONDA_ENV ($(python3 --version))"

DEVICE="${DEVICE:-cuda}"
PHASE="${1:---phase}"
PHASE_VAL="${2:-all}"
EXTRA_ARGS="${@:3}"

# If first arg is --skip_eval, run all with skip_eval
if [ "$PHASE" = "--skip_eval" ]; then
    echo "Running full pipeline (audio only, skip evaluation)..."
    python3 speaker_augment_ablation.py --phase all --device "$DEVICE" --skip_eval
    exit 0
fi

# If first arg is --phase, use specified phase
if [ "$PHASE" = "--phase" ]; then
    echo "Running phase: $PHASE_VAL"
    python3 speaker_augment_ablation.py --phase "$PHASE_VAL" --device "$DEVICE" $EXTRA_ARGS
    exit 0
fi

# Default: run everything
echo "============================================================"
echo "  Speaker + Augmentation Ablation Experiments"
echo "  Device: $DEVICE"
echo "============================================================"
echo ""
echo "Step 1/4: Extracting 20 speaker references from real EMS audio..."
python3 speaker_augment_ablation.py --phase extract_refs --device "$DEVICE"

echo ""
echo "Step 2/4: Synthesizing with XTTS (50 utts x 4 groups)..."
python3 speaker_augment_ablation.py --phase synthesize --device "$DEVICE"

echo ""
echo "Step 3/4: Applying augmentation pipelines..."
python3 speaker_augment_ablation.py --phase augment --device "$DEVICE"

echo ""
echo "Step 4/4: Evaluating with Whisper..."
python3 speaker_augment_ablation.py --phase evaluate --device "$DEVICE"

echo ""
echo "============================================================"
echo "  All experiments complete!"
echo "  Results: experiments/ablation_results/experiment_summary.json"
echo "============================================================"
