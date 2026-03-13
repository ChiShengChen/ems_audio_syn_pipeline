#!/usr/bin/env python3
"""
Corpus Collector: 整合多來源 EMS 文本
- 現有 human_anotation_vb.csv 的 transcript
- LLM 生成的 text_corpus.jsonl
- 可選：分段長 transcript 為多個 utterance
"""

import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd


def load_human_transcripts(csv_path: str) -> List[Dict]:
    """從 human_anotation_vb.csv 載入 transcript，可選分段."""
    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        text = str(row.get("transcript", "")).strip()
        if not text or len(text) < 15:
            continue
        # 可選：依句號或長度分段
        segments = split_transcript(text, max_len=200)
        for seg in segments:
            seg = normalize_text(seg)
            if len(seg) >= 15:
                records.append({
                    "text": seg,
                    "source": "human",
                    "scenario": str(row.get("Call Type", "medical")),
                    "tags": str(row.get("Tags", "")),
                })
    return records


def split_transcript(text: str, max_len: int = 200) -> List[str]:
    """將長 transcript 切成較短 utterance（保留語意邊界）."""
    text = " ".join(text.split())
    if len(text) <= max_len:
        return [text]
    # 依 . 或 2+ 空格切
    parts = re.split(r"\.\s+|\s{2,}", text)
    out = []
    buf = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(buf) + len(p) + 1 <= max_len:
            buf = f"{buf} {p}".strip() if buf else p
        else:
            if buf:
                out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out


def normalize_text(t: str) -> str:
    """簡單正規化：去 [x]、多餘空白."""
    t = re.sub(r"\[x\]", "", t, flags=re.I)
    t = " ".join(t.split())
    return t.strip()


def load_llm_corpus(jsonl_path: str) -> List[Dict]:
    """載入 LLM 生成的 JSONL."""
    records = []
    path = Path(jsonl_path)
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def deduplicate(records: List[Dict], key_field: str = "text") -> List[Dict]:
    """依 text 去重."""
    seen = set()
    out = []
    for r in records:
        k = r.get(key_field, "").strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(r)
    return out


def collect_corpus(
    human_csv: Optional[str] = None,
    llm_jsonl: Optional[str] = None,
    output_path: str = "phase1_output/combined_corpus.jsonl",
    dedup: bool = True,
) -> int:
    """整合所有來源，輸出 combined_corpus.jsonl."""
    all_records = []
    if human_csv:
        all_records.extend(load_human_transcripts(human_csv))
    if llm_jsonl:
        all_records.extend(load_llm_corpus(llm_jsonl))
    if dedup:
        all_records = deduplicate(all_records)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(all_records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_csv", type=str, help="Path to human_anotation_vb.csv")
    parser.add_argument("--llm_jsonl", type=str, help="Path to text_corpus.jsonl")
    parser.add_argument("--output", type=str, default="phase1_output/combined_corpus.jsonl")
    parser.add_argument("--no_dedup", action="store_true")
    args = parser.parse_args()
    n = collect_corpus(
        human_csv=args.human_csv,
        llm_jsonl=args.llm_jsonl,
        output_path=args.output,
        dedup=not args.no_dedup,
    )
    print(f"Collected {n} utterances -> {args.output}")


if __name__ == "__main__":
    main()
