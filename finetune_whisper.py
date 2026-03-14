#!/usr/bin/env python3
"""
Fine-tune Whisper-large-v3 for EMS ASR
Uses LoRA (Low-Rank Adaptation) for efficient training.
Compatible with prepare_for_whisper.py output (HuggingFace Dataset format).
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add parent dir to path for ems_eval.spec_augment (when pipeline is inside ems_call)
_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from datasets import load_from_disk, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    GenerationConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union
import warnings

warnings.filterwarnings("ignore")

# 載入 WER 評估指標
wer_metric = evaluate.load("wer")

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for Whisper training（支持延遲加載音頻、可選 SpecAugment）"""
    processor: WhisperProcessor
    spec_augment: bool = False
    _training: list = field(default_factory=lambda: [True], repr=False)

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # 處理音頻：從 features 中提取 audio 字段並轉換為特徵
        input_features_list = []
        labels_list = []
        
        for feature in features:
            # 提取音頻
            audio = feature.get("audio")
            
            # 處理 audio 字段（可能是 dict 或 path）
            if isinstance(audio, dict):
                if "array" in audio:
                    audio_array = audio["array"]
                    sampling_rate = audio.get("sampling_rate", 16000)
                elif "path" in audio:
                    import soundfile as sf
                    audio_array, sampling_rate = sf.read(audio["path"])
                else:
                    raise ValueError(f"無法處理 audio 格式: {audio.keys()}")
            elif isinstance(audio, str):
                # 如果是路徑字符串
                import soundfile as sf
                audio_array, sampling_rate = sf.read(audio)
            else:
                raise ValueError(f"無法處理 audio 類型: {type(audio)}")
            
            # 提取音頻特徵
            input_features = self.processor.feature_extractor(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt"
            ).input_features[0]
            input_features_list.append(input_features)
            
            # 處理轉錄標籤
            transcript = feature.get("transcript", "")
            labels = self.processor.tokenizer(transcript).input_ids
            labels_list.append(labels)
        
        # 填充音頻特徵
        input_features_batch = [{"input_features": feat} for feat in input_features_list]
        batch = self.processor.feature_extractor.pad(input_features_batch, return_tensors="pt")
        
        # 確保輸入特徵的數據類型
        if hasattr(self, '_model_dtype'):
            if self._model_dtype == torch.float16:
                batch["input_features"] = batch["input_features"].half()
            elif self._model_dtype == torch.bfloat16:
                batch["input_features"] = batch["input_features"].bfloat16()
        
        # 填充標籤
        label_features = [{"input_ids": label} for label in labels_list]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # 將 padding token 替換為 -100
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        # 如果 decoder_input_ids 是 padding token，也替換為 -100
        if (labels[:, 0] == self.processor.tokenizer.pad_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels

        # SpecAugment（僅訓練時，F=27, T=100）
        if self.spec_augment and self._training[0] and np.random.rand() < 0.5:
            try:
                from ems_eval.spec_augment import spec_augment
                batch["input_features"] = spec_augment(
                    batch["input_features"],
                    freq_mask_param=27,
                    time_mask_param=100,
                    num_freq_masks=2,
                    num_time_masks=2,
                )
            except ImportError:
                pass

        return batch

def prepare_dataset(dataset_path: str, processor: WhisperProcessor, split: str = "train"):
    """準備訓練資料集（優化版本：延遲加載，在 DataCollator 中處理）"""
    print(f"📖 載入 {split} 資料集...")
    dataset = load_from_disk(os.path.join(dataset_path, split))
    
    # 確保音頻欄位正確（延遲加載）
    if "audio" in dataset.column_names:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # 不進行預處理，直接返回原始數據集
    # 音頻加載和特徵提取將在 DataCollator 中按需進行
    print(f"✅ {split} 資料集已載入（{len(dataset)} 個樣本）")
    print(f"   音頻將在訓練時按需加載和處理")
    
    return dataset

def compute_metrics(pred):
    """計算評估指標"""
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # 將 -100 替換為 pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    # 解碼預測和標籤
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    # 計算 WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer}

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper-large-v3 for EMS ASR")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="whisper_training_data",
        help="訓練資料集路徑 (prepare_for_whisper.py 輸出)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/whisper-large-v3",
        help="基礎模型名稱"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="whisper_finetuned",
        help="模型輸出目錄"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5,
        help="訓練輪數"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="每個設備的訓練批次大小"
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="每個設備的評估批次大小"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="梯度累積步數"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="學習率"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Warmup 步數"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="使用 FP16 混合精度訓練"
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="使用 BF16 混合精度訓練"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=50,
        help="評估間隔步數"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="儲存間隔步數"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="日誌間隔步數"
    )
    parser.add_argument(
        "--load_best_model_at_end",
        action="store_true",
        help="訓練結束時載入最佳模型"
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="wer",
        help="最佳模型指標"
    )
    parser.add_argument(
        "--greater_is_better",
        action="store_false",
        help="指標是否越大越好（WER 越小越好）"
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="啟用 EarlyStoppingCallback"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="EarlyStopping patience"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="從指定的 checkpoint 恢復訓練"
    )
    parser.add_argument(
        "--spec_augment",
        action="store_true",
        help="啟用 SpecAugment (F=27, T=100)，需 ems_eval.spec_augment"
    )
    parser.add_argument(
        "--use_dora",
        action="store_true",
        help="使用 DoRA 取代 LoRA"
    )
    
    args = parser.parse_args()
    
    # 檢查 CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  使用設備: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 載入處理器
    print(f"\n📦 載入處理器: {args.model_name}")
    global processor
    processor = WhisperProcessor.from_pretrained(args.model_name)
    
    # 載入模型
    print(f"📦 載入模型: {args.model_name}")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    
    if args.fp16:
        model = model.half()
        print(f"   模型轉換為 FP16")
    elif args.bf16:
        model = model.bfloat16()
        print(f"   模型轉換為 BF16")
    
    # 設置 LoRA/DoRA 配置
    adapter_name = "DoRA" if args.use_dora else "LoRA"
    print(f"\n🔧 設置 {adapter_name} 配置 (r={args.lora_r}, alpha={args.lora_alpha})")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        use_dora=args.use_dora,
    )
    
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    model.generation_config = GenerationConfig(
        max_length=225,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
    )
    print(f"   已設置 hallucination 抑制: no_repeat_ngram_size=3, repetition_penalty=1.2")

    # 準備資料集
    train_dataset = prepare_dataset(args.dataset_path, processor, "train")
    eval_dataset = prepare_dataset(args.dataset_path, processor, "validation")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        fp16=args.fp16,
        bf16=args.bf16,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        push_to_hub=False,
        report_to="tensorboard",
        save_total_limit=3,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        spec_augment=args.spec_augment,
    )
    
    model_dtype = next(model.parameters()).dtype
    data_collator._model_dtype = model_dtype
    print(f"   模型數據類型: {model_dtype}")
    
    callbacks = []
    if args.early_stopping:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
    if args.spec_augment:
        from transformers import TrainerCallback
        class SpecAugmentEvalCallback(TrainerCallback):
            def on_evaluate(self, args, state, control, **kwargs):
                data_collator._training[0] = False
                return control
            def on_train_begin(self, args, state, control, **kwargs):
                data_collator._training[0] = True
                return control
        callbacks.append(SpecAugmentEvalCallback())

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks if callbacks else None,
    )
    
    import signal
    interrupted = False
    
    def signal_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\n\n⚠️  收到中斷信號 (Ctrl+C)")
        print("   正在保存當前狀態...")
        trainer.save_model()
        processor.save_pretrained(args.output_dir)
        print(f"   ✅ 已保存到: {args.output_dir}")
        sys.exit(130)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("\n🚀 開始訓練...")
    target_epochs = args.num_train_epochs
    
    print(f"📊 訓練配置:")
    print(f"   - 目標 Epochs: {target_epochs}")
    print(f"   - 訓練樣本數: {len(train_dataset)}")
    print(f"   - 評估樣本數: {len(eval_dataset)}")
    print(f"   - Batch size: {args.per_device_train_batch_size}")
    print(f"   - Gradient accumulation: {args.gradient_accumulation_steps}")
    
    try:
        if args.resume_from_checkpoint:
            if not os.path.isabs(args.resume_from_checkpoint):
                checkpoint_path = os.path.join(args.output_dir, args.resume_from_checkpoint)
            else:
                checkpoint_path = args.resume_from_checkpoint
            
            if not os.path.exists(checkpoint_path):
                raise ValueError(f"❌ Checkpoint 不存在: {checkpoint_path}")
            
            training_args_bin = os.path.join(checkpoint_path, "training_args.bin")
            if os.path.exists(training_args_bin):
                os.remove(training_args_bin)
            
            training_args.save_steps = args.save_steps
            training_args.eval_steps = args.eval_steps
            training_args.logging_steps = args.logging_steps
            
            trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=callbacks,
            )
            trainer.args.save_steps = args.save_steps
            trainer.args.eval_steps = args.eval_steps
            trainer.args.logging_steps = args.logging_steps
            
            trainer.train(resume_from_checkpoint=checkpoint_path)
        else:
            trainer.train()
        
        final_step = getattr(trainer.state, 'global_step', None) or 0
        final_epoch = getattr(trainer.state, 'epoch', None) or 0.0
        print(f"\n📊 訓練返回: Step {final_step}, Epoch {final_epoch:.4f}/{target_epochs}")
        
        if final_epoch < target_epochs - 0.01:
            print(f"\n⚠️  警告: 訓練提前返回")
        
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        try:
            trainer.save_model()
            processor.save_pretrained(args.output_dir)
            print(f"   ✅ 已保存到: {args.output_dir}")
        except Exception:
            pass
        raise
    
    final_epoch = getattr(trainer.state, 'epoch', None) or 0.0
    is_completed = final_epoch >= target_epochs - 0.05
    
    if is_completed:
        print(f"\n💾 儲存模型到: {args.output_dir}")
        trainer.save_model()
        processor.save_pretrained(args.output_dir)
        print(f"\n✅ 訓練完成！({final_epoch:.2f}/{target_epochs} epochs)")
    else:
        print(f"\n⚠️  訓練未完成 ({final_epoch:.2f}/{target_epochs})")
        trainer.save_model()
        processor.save_pretrained(args.output_dir)
        sys.exit(1)

if __name__ == "__main__":
    main()
