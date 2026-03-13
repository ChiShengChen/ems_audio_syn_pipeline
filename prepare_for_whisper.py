#!/usr/bin/env python3
"""
將 Phase 3 的 augmented_manifest.jsonl 轉換為 Whisper fine-tuning 用的 HuggingFace Dataset
格式：train / validation / test 分割，與 finetune_whisper.py 相容
"""

import json
import argparse
from pathlib import Path
from typing import Optional

from datasets import Dataset, Audio


def prepare_whisper_dataset(
    manifest_path: str,
    output_dir: str,
    base_dir: Optional[str] = None,
    train_split: float = 0.8,
    val_split: float = 0.1,
    target_sr: int = 16000,
    seed: int = 42,
) -> str:
    """
    從 JSONL manifest 建立 Whisper 訓練資料集。

    Args:
        manifest_path: augmented_manifest.jsonl 路徑
        output_dir: 輸出目錄（會建立 train/validation/test 子目錄）
        base_dir: manifest 中音頻路徑的基底目錄（預設為 manifest 所在目錄的父目錄）
        train_split: 訓練集比例
        val_split: 驗證集比例
        target_sr: 目標採樣率
        seed: 隨機種子
    """
    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    base_dir = Path(base_dir) if base_dir else manifest_path.parent.parent

    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            audio_path = rec.get("audio", "")
            text = rec.get("text", rec.get("transcript", ""))
            if not audio_path or not text:
                continue
            # 轉為絕對路徑
            if not Path(audio_path).is_absolute():
                audio_path = str((base_dir / audio_path).resolve())
            if not Path(audio_path).exists():
                print(f"⚠️ 跳過不存在的檔案: {audio_path}")
                continue
            records.append({"audio": audio_path, "transcript": text})

    if not records:
        raise ValueError(f"沒有有效資料，請檢查 manifest: {manifest_path}")

    print(f"📖 讀取 {len(records)} 筆資料")
    dataset = Dataset.from_dict({"audio": [r["audio"] for r in records], "transcript": [r["transcript"] for r in records]})
    dataset = dataset.cast_column("audio", Audio(sampling_rate=target_sr))

    # 分割
    total = len(dataset)
    test_split = 1 - train_split - val_split
    ds = dataset.train_test_split(test_size=1 - train_split, seed=seed)
    train_ds = ds["train"]
    val_test = ds["test"].train_test_split(test_size=test_split / (1 - train_split), seed=seed)
    val_ds = val_test["train"]
    test_ds = val_test["test"]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_ds.save_to_disk(str(output_dir / "train"))
    val_ds.save_to_disk(str(output_dir / "validation"))
    test_ds.save_to_disk(str(output_dir / "test"))

    stats = {
        "total": total,
        "train": len(train_ds),
        "validation": len(val_ds),
        "test": len(test_ds),
    }
    with open(output_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"✅ 已儲存至 {output_dir}")
    print(f"   訓練: {len(train_ds)}, 驗證: {len(val_ds)}, 測試: {len(test_ds)}")
    return str(output_dir)


def main():
    parser = argparse.ArgumentParser(description="將 synthetic manifest 轉為 Whisper 訓練格式")
    parser.add_argument("--manifest", default="phase3_output/augmented_manifest.jsonl", help="augmented_manifest.jsonl 路徑")
    parser.add_argument("--output_dir", default="whisper_synthetic_data", help="輸出目錄")
    parser.add_argument("--base_dir", default=None, help="音頻路徑基底（預設為 manifest 父目錄的父目錄）")
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--target_sr", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_whisper_dataset(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        base_dir=args.base_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        target_sr=args.target_sr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
