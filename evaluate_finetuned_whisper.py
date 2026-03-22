#!/usr/bin/env python3
"""
評估 Fine-tuned Whisper 模型
計算 WER, CER, mWER, BLEU, TF-IDF cosine 等指標
支援 abbreviation 正規化
"""

import os
import sys
from pathlib import Path

# Add parent dir for ems_eval (when pipeline is inside ems_call)
_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

import torch
import argparse
import warnings
from datasets import load_from_disk
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import evaluate
from tqdm import tqdm
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module='transformers')
warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")
warnings.filterwarnings("ignore", message=".*attention mask.*")
warnings.filterwarnings("ignore", message=".*logits processor.*")
warnings.filterwarnings("ignore", message=".*multilingual Whisper.*")

try:
    from peft import PeftModel
    import peft
    PEFT_AVAILABLE = True
    try:
        from packaging import version
        peft_version = version.parse(peft.__version__)
        if peft_version < version.parse("0.18.0"):
            print(f"⚠️  PEFT {peft.__version__} 建議升級: pip install --upgrade peft")
    except Exception:
        pass
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠️  PEFT 未安裝，無法載入 LoRA: pip install peft")

try:
    from ems_eval.preprocessing import normalize_ems_text
    from ems_eval.metrics import STTMetrics
    EMS_EVAL_AVAILABLE = True
except ImportError:
    EMS_EVAL_AVAILABLE = False

wer_metric = evaluate.load("wer")

def evaluate_model(
    model_path: str,
    test_dataset_path: str,
    output_csv: str = None,
    medical_vocab_path: str = None,
    normalize: bool = True,
):
    """評估 fine-tuned Whisper 模型"""

    print(f"📦 載入模型: {model_path}")
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    is_lora_model = os.path.exists(adapter_config_path)

    if is_lora_model:
        if not PEFT_AVAILABLE:
            raise ImportError("需要 PEFT: pip install peft")
        print("  檢測到 LoRA 適配器，載入基礎模型...")
        import json
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path", "openai/whisper-large-v3")
        print(f"  從基礎模型載入 processor: {base_model_name}")
        processor = WhisperProcessor.from_pretrained(base_model_name)
        use_fp16 = adapter_config.get("torch_dtype") == "float16" or adapter_config.get("fp16", False)
        base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
        if use_fp16:
            base_model = base_model.half()
        print("  載入 LoRA 適配器...")
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        if use_fp16:
            model = model.half()
    else:
        processor = WhisperProcessor.from_pretrained(model_path)
        model = WhisperForConditionalGeneration.from_pretrained(model_path)

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model_dtype = next(model.parameters()).dtype
    print(f"🖥️  設備: {device}, 數據類型: {model_dtype}")

    print(f"📖 載入測試集: {test_dataset_path}")
    test_dataset = load_from_disk(test_dataset_path)
    print(f"🔍 評估 {len(test_dataset)} 個樣本...")

    predictions, references, results = [], [], []
    with torch.no_grad():
        for idx, example in enumerate(tqdm(test_dataset, desc="評估中")):
            audio = example["audio"]
            audio_array = audio["array"]
            sampling_rate = audio["sampling_rate"]
            input_features = processor.feature_extractor(
                audio_array, sampling_rate=sampling_rate, return_tensors="pt"
            ).input_features.to(device)
            if model_dtype == torch.float16:
                input_features = input_features.half()
            elif model_dtype == torch.bfloat16:
                input_features = input_features.bfloat16()
            gen_kwargs = {"language": "en", "task": "transcribe", "no_repeat_ngram_size": 3, "repetition_penalty": 1.2}
            generated_ids = model.generate(input_features, **gen_kwargs)
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            reference = example["transcript"]
            predictions.append(transcription)
            references.append(reference)
            results.append({"index": idx, "reference": reference, "prediction": transcription})

    refs_for_metrics = list(references)
    preds_for_metrics = list(predictions)
    if normalize and EMS_EVAL_AVAILABLE:
        refs_for_metrics = [normalize_ems_text(r) for r in references]
        preds_for_metrics = [normalize_ems_text(p) for p in predictions]

    wer = wer_metric.compute(predictions=preds_for_metrics, references=refs_for_metrics)
    print(f"\n📊 WER: {wer:.4f} ({wer*100:.2f}%)")

    if EMS_EVAL_AVAILABLE:
        base_dir = Path(__file__).resolve().parent
        vocab_path = Path(medical_vocab_path) if medical_vocab_path else base_dir.parent / "ems_eval" / "data" / "medical_vocab.csv"
        stt_metrics = STTMetrics(medical_vocab_path=vocab_path if vocab_path.exists() else None)
        cer_sum = mwer_sum = bleu_sum = tfidf_sum = 0.0
        for ref, pred in zip(refs_for_metrics, preds_for_metrics):
            m = stt_metrics.compute_all(ref, pred)
            cer_sum += m["cer"]
            mwer_sum += m["mwer"]
            bleu_sum += m["bleu"]
            tfidf_sum += m["tfidf_cosine"]
        n = len(refs_for_metrics) or 1
        print(f"   CER: {cer_sum/n:.4f}  mWER: {mwer_sum/n:.4f}  BLEU: {bleu_sum/n:.4f}  TF-IDF: {tfidf_sum/n:.4f}")

    if output_csv:
        pd.DataFrame(results).to_csv(output_csv, index=False)
        print(f"\n💾 結果: {output_csv}")
    return wer, results

def main():
    parser = argparse.ArgumentParser(description="評估 Fine-tuned Whisper 模型")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_dataset_path", type=str, default="whisper_training_data/test")
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--medical_vocab_path", type=str, default=None)
    parser.add_argument("--no_normalize", action="store_true")
    args = parser.parse_args()
    evaluate_model(
        model_path=args.model_path,
        test_dataset_path=args.test_dataset_path,
        output_csv=args.output_csv,
        medical_vocab_path=args.medical_vocab_path,
        normalize=not args.no_normalize,
    )

if __name__ == "__main__":
    main()
