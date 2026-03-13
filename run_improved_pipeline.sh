#!/bin/bash
# ============================================================
# Improved Synthetic EMS Pipeline v2
# Enhancements over v1:
#   1. Text disfluency injection (uh, um, repetitions, partial words)
#   2. Wider speed perturbation (0.8x–1.4x) + pitch shift
#   3. AMR / GSM codec simulation in radio channel
#   4. Cross-talk overlay (~20% of clips)
#   5. All radio sim effects enabled (dropout, reverb, codec)
# ============================================================
set -e

PIPELINE_DIR="/media/meow/One Touch/ems_call/ems_audio_syn_pipeline"
PROJECT_DIR="/media/meow/One Touch/ems_call"

PHASE1_DIR="${PIPELINE_DIR}/phase1_output"
PHASE2_DIR="${PIPELINE_DIR}/phase2_output"
PHASE3_DIR="${PIPELINE_DIR}/phase3_output_v2"

CORPUS_ORIG="${PHASE1_DIR}/combined_corpus.jsonl"
CORPUS_DISFLUENT="${PHASE1_DIR}/combined_corpus_disfluent.jsonl"
PHASE2_MANIFEST="${PHASE2_DIR}/merged_manifest.jsonl"

echo "========================================================"
echo "  Improved Synthetic EMS Pipeline v2"
echo "========================================================"

# ----------------------------------------------------------
# Step 1: Disfluency injection on Phase 1 text corpus
# ----------------------------------------------------------
echo ""
echo "Step 1: Inject disfluencies into text corpus"
echo "--------------------------------------------------------"

CORPUS_INPUT="$CORPUS_ORIG"
if [ ! -f "$CORPUS_INPUT" ]; then
    CORPUS_INPUT="${PHASE1_DIR}/ems_radio_500.jsonl"
fi

if [ ! -f "$CORPUS_INPUT" ]; then
    echo "ERROR: No corpus found at ${CORPUS_ORIG} or ems_radio_500.jsonl"
    exit 1
fi

cd "$PIPELINE_DIR"
python3 phase1_text_generation/disfluency_injector.py \
    --input "$CORPUS_INPUT" \
    --output "$CORPUS_DISFLUENT" \
    --disfluent_ratio 0.5

echo "Done: $(wc -l < "$CORPUS_DISFLUENT") lines in disfluent corpus"

# ----------------------------------------------------------
# Step 2: TTS on disfluent corpus (reuse Phase 2 if exists)
# ----------------------------------------------------------
echo ""
echo "Step 2: TTS synthesis"
echo "--------------------------------------------------------"

if [ -f "$PHASE2_MANIFEST" ]; then
    echo "Phase 2 manifest already exists: $PHASE2_MANIFEST"
    echo "  $(wc -l < "$PHASE2_MANIFEST") clips"
    echo "  Skipping TTS (reusing existing audio)."
    echo "  To regenerate with disfluent text, delete phase2_output/ and re-run."
else
    echo "No Phase 2 output found. Please run Phase 2 TTS first:"
    echo "  cd $PIPELINE_DIR && python3 phase2_tts/tts_synthesizer.py \\"
    echo "    --corpus $CORPUS_DISFLUENT \\"
    echo "    --output_dir phase2_output"
    exit 1
fi

# ----------------------------------------------------------
# Step 3: Improved noise augmentation (v2)
# ----------------------------------------------------------
echo ""
echo "Step 3: Noise augmentation v2 (wider speed, codec, cross-talk)"
echo "--------------------------------------------------------"

python3 phase3_noise_augmentation/noise_augmentation.py \
    --manifest "$PHASE2_MANIFEST" \
    --output_dir "$PHASE3_DIR" \
    --num_variants 4 \
    --crosstalk_p 0.2

echo ""
echo "Phase 3 v2 done: $(wc -l < "${PHASE3_DIR}/augmented_manifest.jsonl") augmented clips"

# ----------------------------------------------------------
# Step 4: Prepare Whisper dataset
# ----------------------------------------------------------
echo ""
echo "Step 4: Prepare HuggingFace dataset for Whisper"
echo "--------------------------------------------------------"

WHISPER_DATA_DIR="${PROJECT_DIR}/whisper_synthetic_data_v2"

python3 prepare_for_whisper.py \
    --manifest "${PHASE3_DIR}/augmented_manifest.jsonl" \
    --output_dir "$WHISPER_DATA_DIR" \
    --train_split 0.8 \
    --val_split 0.1

echo ""
echo "========================================================"
echo "  Pipeline v2 complete!"
echo "========================================================"
echo ""
echo "Output:"
echo "  Disfluent corpus: $CORPUS_DISFLUENT"
echo "  Phase 3 v2:       $PHASE3_DIR"
echo "  Whisper dataset:  $WHISPER_DATA_DIR"
echo ""
echo "Next: fine-tune with:"
echo "  cd \"$PROJECT_DIR\""
echo "  python3 finetune_whisper.py \\"
echo "    --dataset_path $WHISPER_DATA_DIR \\"
echo "    --output_dir whisper_finetuned_v2 \\"
echo "    --num_train_epochs 3 --fp16"
