#!/usr/bin/env python3
"""
Bark TTS for EMS Synthetic Audio (20% of corpus)
- Supports non-verbal sounds, diverse speakers
- Output: 16kHz mono WAV
"""

import json
import argparse
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from bark import SAMPLE_RATE as BARK_SR, preload_models, generate_audio
    HAS_BARK = True
except ImportError:
    try:
        from bark.api import semantic_to_waveform
        from bark.generation import load_codec_model, codec_decode
        HAS_BARK = True
    except ImportError:
        HAS_BARK = False


def ensure_16k(audio: "np.ndarray", orig_sr: int) -> "np.ndarray":
    """Resample to 16kHz if needed."""
    if orig_sr == 16000:
        return audio
    import librosa
    return librosa.resample(audio.astype(np.float32), orig_sr=orig_sr, target_sr=16000)


def synthesize_bark(
    text: str,
    output_path: str,
    voice_preset: str = "v2/en_speaker_6",
    temp: float = 0.7,
) -> str:
    """
    Synthesize with Bark. Voice presets: v2/en_speaker_0 .. v2/en_speaker_9
    """
    if not HAS_BARK:
        raise ImportError("Install bark: pip install git+https://github.com/suno-ai/bark")
    preload_models()
    # Bark uses different API - check actual bark package
    try:
        audio_array = generate_audio(text, history_prompt=voice_preset, temp=temp)
    except Exception:
        # Fallback for different bark API
        from bark import generate_audio as gen
        audio_array = gen(text, history_prompt=voice_preset)
    sr = getattr(__import__("bark", fromlist=["SAMPLE_RATE"]), "SAMPLE_RATE", 24000)
    audio_16k = ensure_16k(np.array(audio_array), sr)
    import soundfile as sf
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio_16k, 16000)
    return output_path


def run_bark_batch(
    corpus_path: str,
    output_dir: str = "phase2_output/bark",
    max_items: Optional[int] = None,
    voice_rotation: Optional[list] = None,
) -> str:
    """Batch synthesize with Bark, rotate voice presets."""
    voice_rotation = voice_rotation or [f"v2/en_speaker_{i}" for i in range(10)]
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
        voice = voice_rotation[i % len(voice_rotation)]
        out_path = Path(output_dir) / f"bark_{i:06d}.wav"
        try:
            synthesize_bark(text, str(out_path), voice_preset=voice)
            with open(manifest_path, "a", encoding="utf-8") as mf:
                mf.write(json.dumps({"audio": str(out_path), "text": text}, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Bark skip {i}: {e}")
    return str(manifest_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output_dir", default="phase2_output/bark")
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()
    run_bark_batch(args.corpus, args.output_dir, args.max)
    print(f"Done. Manifest: {args.output_dir}/manifest.jsonl")


if __name__ == "__main__":
    main()
