#!/usr/bin/env python3
"""
Edge TTS for EMS Synthetic Audio (10% of corpus)
- Free Microsoft API, many voices
- Output: 16kHz mono WAV
"""

import asyncio
import json
import argparse
from pathlib import Path
from typing import Optional, List

try:
    import edge_tts
    HAS_EDGE = True
except ImportError:
    HAS_EDGE = False

import re

# Diverse US English voices
EDGE_VOICES = [
    "en-US-JennyNeural",
    "en-US-GuyNeural",
    "en-US-AriaNeural",
    "en-US-DavisNeural",
    "en-US-SaraNeural",
    "en-US-TonyNeural",
    "en-US-NancyNeural",
    "en-US-AndrewNeural",
    "en-US-AmberNeural",
    "en-US-AshleyNeural",
]


def _sanitize_text_for_edge(text: str, max_len: int = 500) -> str:
    """Sanitize text for Edge TTS - remove chars that cause 'No audio received'."""
    # Remove/replace problematic chars
    text = re.sub(r"[\[\]<>{}|\\^~`]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0]  # cut at word boundary
    return text or "copy"


async def _synthesize_one(text: str, voice: str, output_path: str) -> bool:
    """Single async synthesis."""
    if not HAS_EDGE:
        raise ImportError("Install edge-tts: pip install edge-tts")
    text = _sanitize_text_for_edge(text)
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)
    return True


def synthesize_edge(text: str, output_path: str, voice: str = "en-US-GuyNeural") -> str:
    """Synthesize with Edge TTS. Returns path."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(_synthesize_one(text, voice, output_path))
    # Edge TTS outputs 24kHz mp3 by default - convert to 16kHz wav if needed
    out_path = Path(output_path)
    if out_path.suffix.lower() == ".mp3":
        # Save as mp3, pipeline can convert later; or convert now
        pass
    return output_path


def run_edge_batch(
    corpus_path: str,
    output_dir: str = "phase2_output/edge",
    max_items: Optional[int] = None,
    voices: Optional[List[str]] = None,
) -> str:
    """Batch synthesize with Edge TTS."""
    voices = voices or EDGE_VOICES
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(output_dir) / "manifest.jsonl"
    with open(corpus_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if max_items:
        lines = lines[:max_items]
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = rec.get("text", "").strip()
        if len(text) < 10:
            continue
        voice = voices[i % len(voices)]
        out_path = Path(output_dir) / f"edge_{i:06d}.mp3"
        try:
            asyncio.run(_synthesize_one(text, voice, str(out_path)))
            with open(manifest_path, "a", encoding="utf-8") as mf:
                mf.write(json.dumps({"audio": str(out_path), "text": text}, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Edge skip {i}: {e}")
    return str(manifest_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output_dir", default="phase2_output/edge")
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()
    run_edge_batch(args.corpus, args.output_dir, args.max)
    print(f"Done. Manifest: {args.output_dir}/manifest.jsonl")


if __name__ == "__main__":
    main()
