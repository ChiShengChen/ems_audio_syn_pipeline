#!/bin/bash
# Whisper fine-tuning 使用 Synthetic EMS 資料集（Phase 3 輸出）
# 3508 筆 synthetic clips → train/val/test → LoRA fine-tuning

set -e

PIPELINE_DIR="/media/meow/One Touch/ems_call/ems_audio_syn_pipeline"
PROJECT_DIR="/media/meow/One Touch/ems_call"
MANIFEST="${PIPELINE_DIR}/phase3_output/augmented_manifest.jsonl"
SYNTHETIC_DATA_DIR="${PROJECT_DIR}/whisper_synthetic_data"
OUTPUT_MODEL_DIR="${PROJECT_DIR}/whisper_finetuned_synthetic"
MODEL_NAME="openai/whisper-large-v3"

# 訓練參數（3508 樣本）
NUM_EPOCHS=3
BATCH_SIZE=4
GRADIENT_ACCUMULATION=4
LEARNING_RATE=5e-5
WARMUP_STEPS=100
LORA_R=16
LORA_ALPHA=32

echo "========================================"
echo "Whisper Fine-tuning (Synthetic EMS Data)"
echo "========================================"

# Step 1: 轉換 manifest → HuggingFace Dataset
echo ""
echo "步驟 1: 準備 Whisper 訓練資料"
echo "----------------------------------------"
if [ ! -f "$MANIFEST" ]; then
    echo "❌ Manifest 不存在: $MANIFEST"
    echo "   請先執行 Phase 3: python3 phase3_noise_augmentation/noise_augmentation.py ..."
    exit 1
fi

cd "$PIPELINE_DIR"
python3 prepare_for_whisper.py \
    --manifest phase3_output/augmented_manifest.jsonl \
    --output_dir "$SYNTHETIC_DATA_DIR" \
    --train_split 0.8 \
    --val_split 0.1

if [ ! -d "${SYNTHETIC_DATA_DIR}/train" ]; then
    echo "❌ 資料準備失敗"
    exit 1
fi

echo "✅ 資料準備完成"

# Step 2: Fine-tuning
echo ""
echo "步驟 2: Fine-tuning Whisper-large-v3"
echo "----------------------------------------"
cd "$PIPELINE_DIR"
python3 finetune_whisper.py \
    --dataset_path "$SYNTHETIC_DATA_DIR" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_MODEL_DIR" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --learning_rate $LEARNING_RATE \
    --warmup_steps $WARMUP_STEPS \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --fp16 \
    --eval_steps 100 \
    --save_steps 200 \
    --logging_steps 20 \
    --load_best_model_at_end \
    --metric_for_best_model wer

# Step 3: 評估 (需上層專案的 evaluate_finetuned_whisper.py)
echo ""
echo "步驟 3: 評估模型"
echo "----------------------------------------"
cd "$PROJECT_DIR"
python3 evaluate_finetuned_whisper.py \
    --model_path "$OUTPUT_MODEL_DIR" \
    --test_dataset_path "${SYNTHETIC_DATA_DIR}/test" \
    --output_csv "${OUTPUT_MODEL_DIR}/evaluation_results.csv"

echo ""
echo "========================================"
echo "✅ 完成！"
echo "========================================"
echo "模型: $OUTPUT_MODEL_DIR"
echo "評估: ${OUTPUT_MODEL_DIR}/evaluation_results.csv"
