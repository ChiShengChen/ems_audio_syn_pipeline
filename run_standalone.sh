#!/bin/bash
# =============================================================================
# Standalone Demo: Run pipeline with ONLY in-repo data (no external files)
# - Uses sample_data/ems_radio_500.jsonl as corpus
# - XTTS skipped (needs human_csv + audio_dirs)
# - Bark + Edge TTS synthesize 30% of corpus
# - Phase 3, prepare, finetune, evaluate all run
# =============================================================================
set -e

PIPELINE_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PIPELINE_ROOT"

PHASE1_OUT="${PIPELINE_ROOT}/phase1_output"
PHASE2_OUT="${PIPELINE_ROOT}/phase2_output"
PHASE3_OUT="${PIPELINE_ROOT}/phase3_output"
WHISPER_DATA="${PIPELINE_ROOT}/whisper_training_data"
MODEL_OUT="${PIPELINE_ROOT}/whisper_finetuned"

echo "=============================================="
echo "  EMS Pipeline — Standalone Demo"
echo "  (No external data required)"
echo "=============================================="

# --- Phase 1: Use sample corpus (skip LLM) ---
echo ""
echo "=== Phase 1: Prepare corpus from sample_data ==="
mkdir -p "$PHASE1_OUT"
python3 phase1_text_generation/corpus_collector.py \
    --llm_jsonl "${PIPELINE_ROOT}/sample_data/ems_radio_500.jsonl" \
    --output "${PHASE1_OUT}/combined_corpus.jsonl"
echo "Corpus: $(wc -l < "${PHASE1_OUT}/combined_corpus.jsonl") lines"

# --- Phase 2: TTS (Bark + Edge only, no XTTS) ---
echo ""
echo "=== Phase 2: Speech synthesis (Bark + Edge, XTTS skipped) ==="
python3 phase2_speech_synthesis/tts_orchestrator.py \
    --corpus "${PHASE1_OUT}/combined_corpus.jsonl" \
    --output "$PHASE2_OUT" \
    --device cuda \
    --skip_bark

if [ ! -f "${PHASE2_OUT}/merged_manifest.jsonl" ]; then
    echo "❌ Phase 2 failed. Ensure edge-tts is installed: pip install edge-tts"
    exit 1
fi
echo "Manifest: $(wc -l < "${PHASE2_OUT}/merged_manifest.jsonl") clips"

# --- Phase 3: Augmentation ---
echo ""
echo "=== Phase 3: Noise augmentation ==="
python3 phase3_noise_augmentation/noise_augmentation.py \
    --manifest "${PHASE2_OUT}/merged_manifest.jsonl" \
    --output_dir "$PHASE3_OUT" \
    --num_variants 2
echo "Augmented: $(wc -l < "${PHASE3_OUT}/augmented_manifest.jsonl") clips"

# --- Prepare Whisper dataset ---
echo ""
echo "=== Prepare Whisper dataset ==="
python3 prepare_for_whisper.py \
    --manifest "${PHASE3_OUT}/augmented_manifest.jsonl" \
    --output_dir "$WHISPER_DATA" \
    --train_split 0.8 \
    --val_split 0.1

# --- Fine-tune (short run for demo) ---
echo ""
echo "=== Fine-tune Whisper (2 epochs demo) ==="
python3 finetune_whisper.py \
    --dataset_path "$WHISPER_DATA" \
    --output_dir "$MODEL_OUT" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --eval_steps 50 \
    --save_steps 100 \
    --fp16

# --- Evaluate ---
echo ""
echo "=== Evaluate ==="
python3 evaluate_finetuned_whisper.py \
    --model_path "$MODEL_OUT" \
    --test_dataset_path "${WHISPER_DATA}/test" \
    --output_csv "${MODEL_OUT}/eval_results.csv"

echo ""
echo "=============================================="
echo "  ✅ Standalone demo complete!"
echo "  Model: $MODEL_OUT"
echo "=============================================="
</think>

<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>
StrReplace