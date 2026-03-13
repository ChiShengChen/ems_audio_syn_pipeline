#!/usr/bin/env python3
"""
XTTS v2 Voice Cloning for EMS Synthetic Audio
- Extract 20 diverse speaker references with RMS quality filtering
- Auto-detect and remove hallucination-prone speakers via Whisper probe
- Synthesize corpus text with cloned voices (rotating through top speakers)
- Output: 16kHz mono WAV with speaker_id metadata
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np

import soundfile as sf

# Patch torchaudio for torchcodec ABI issues (torchaudio 2.9+)
_patch_path = Path(__file__).resolve().parent.parent / "utils" / "torchaudio_patch.py"
if _patch_path.exists():
    import importlib.util
    _spec = importlib.util.spec_from_file_location("torchaudio_patch", str(_patch_path))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

try:
    import torch
    HAS_TORCH = True
    _orig_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
except ImportError:
    HAS_TORCH = False

try:
    from TTS.api import TTS
    HAS_TTS = True
except ImportError:
    HAS_TTS = False

SEED = 42
MIN_RMS_ENERGY = 0.005
HALLUCINATION_WER_THRESHOLD = 1.5


def load_audio(path: str, sr: int = 16000) -> Optional[np.ndarray]:
    """Load audio as mono float32, resample to sr."""
    import librosa
    wav, orig_sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if orig_sr != sr:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=orig_sr, target_sr=sr)
    return wav.astype(np.float32)


def extract_speaker_reference(
    audio_path: str,
    output_path: str,
    start_sec: float = 0.0,
    duration_sec: float = 5.0,
    sr: int = 16000,
) -> str:
    """Extract a speaker reference clip, peak-normalized."""
    audio = load_audio(audio_path, sr)
    start = int(start_sec * sr)
    end = int(min(start_sec + duration_sec, len(audio) / sr) * sr)
    clip = audio[start:end].astype(np.float32)
    clip = clip / (np.max(np.abs(clip)) + 1e-8) * 0.95
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, clip, sr)
    return output_path


def _compute_rms(path: str) -> float:
    audio, _ = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return float(np.sqrt(np.mean(audio ** 2)))


def init_xtts(device: str = "cuda") -> "TTS":
    """Initialize XTTS v2 model."""
    if not HAS_TTS:
        raise ImportError("Install TTS: pip install TTS")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    return tts


def synthesize_xtts(
    tts: "TTS",
    text: str,
    speaker_wav: str,
    output_path: str,
    language: str = "en",
    device: str = "cuda",
) -> str:
    """Synthesize text with XTTS using cloned speaker reference."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tts.tts_to_file(
        text=text,
        file_path=output_path,
        speaker_wav=speaker_wav,
        language=language,
    )
    return output_path


# ---------------------------------------------------------------------------
# Speaker extraction (20 speakers, diverse offsets, RMS-filtered)
# ---------------------------------------------------------------------------

def create_speaker_references_from_csv(
    csv_path: str,
    audio_dirs: List[str],
    output_dir: str = "speaker_references",
    duration_sec: float = 5.0,
    max_speakers: int = 20,
) -> List[Dict]:
    """
    Extract up to max_speakers reference clips from annotated EMS audio.
    Uses diverse time offsets and filters silent clips by RMS energy.
    Returns list of dicts with speaker_id, ref_path, source_file, rms_energy.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    audio_dirs_p = [Path(d) for d in audio_dirs]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    candidates = []
    for _, row in df.iterrows():
        fname = row.get("Filename", "")
        if not fname:
            continue
        for adir in audio_dirs_p:
            cand = adir / fname
            if cand.exists():
                info = sf.info(str(cand))
                candidates.append({
                    "filename": fname, "path": str(cand),
                    "duration": info.duration,
                })
                break

    random.seed(SEED)
    random.shuffle(candidates)

    offsets = [3.0, 8.0, 15.0, 25.0]
    refs = []
    ref_id = 0
    for cand in candidates:
        if ref_id >= max_speakers:
            break
        offset = offsets[ref_id % len(offsets)]
        if cand["duration"] < offset + duration_sec:
            offset = max(1.0, cand["duration"] / 2 - duration_sec / 2)
        out_path = str(Path(output_dir) / f"ref_{ref_id:02d}.wav")
        try:
            extract_speaker_reference(
                cand["path"], out_path, offset, duration_sec)
            rms = _compute_rms(out_path)
            if rms < MIN_RMS_ENERGY:
                print(f"  Skip {cand['filename']}@{offset}s (silent rms={rms:.4f})")
                continue
            refs.append({
                "speaker_id": f"spk_{ref_id:02d}",
                "ref_path": out_path,
                "source_file": cand["filename"],
                "offset_sec": offset,
                "rms_energy": round(rms, 4),
            })
            ref_id += 1
        except Exception as e:
            print(f"  Skip {cand['filename']}: {e}")

    profiles_path = str(Path(output_dir) / "speaker_profiles.json")
    with open(profiles_path, "w") as f:
        json.dump(refs, f, indent=2)
    print(f"Extracted {len(refs)} speaker references -> {profiles_path}")
    return refs


# ---------------------------------------------------------------------------
# Speaker quality probe (detect hallucination-prone references)
# ---------------------------------------------------------------------------

def _word_error_rate(ref_words: List[str], hyp_words: List[str]) -> float:
    r, h = len(ref_words), len(hyp_words)
    d = np.zeros((r + 1, h + 1), dtype=int)
    for i in range(r + 1):
        d[i][0] = i
    for j in range(h + 1):
        d[0][j] = j
    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
    return d[r][h] / max(r, 1)


def probe_speaker_quality(
    tts: "TTS",
    refs: List[Dict],
    probe_texts: Optional[List[str]] = None,
    device: str = "cuda",
    wer_threshold: float = HALLUCINATION_WER_THRESHOLD,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Synthesize a short probe sentence per speaker, transcribe with Whisper,
    and filter out speakers whose WER exceeds threshold (hallucination).
    Returns (good_refs, bad_refs).
    """
    if probe_texts is None:
        probe_texts = [
            "rescue 10 on scene 50 year old male chest pain",
            "medic 4 transporting code 3 to sentara princess anne",
            "dispatch show engine 21 responding to the structure fire",
        ]
    try:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        import librosa as _lr
    except ImportError:
        print("  Whisper not available, skipping speaker probe")
        return refs, []

    print("Probing speaker quality with Whisper ...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small.en")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small.en").to(device)
    model.eval()

    import tempfile
    good, bad = [], []
    for ref in refs:
        text = random.choice(probe_texts)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            synthesize_xtts(tts, text, ref["ref_path"], tmp_path, device=device)
            audio, _ = _lr.load(tmp_path, sr=16000)
            feats = processor(audio, sampling_rate=16000,
                              return_tensors="pt").input_features.to(device)
            with torch.no_grad():
                ids = model.generate(feats)
            hyp = processor.batch_decode(ids, skip_special_tokens=True)[0].strip().lower()
            wer = _word_error_rate(text.lower().split(), hyp.split())
            ref["probe_wer"] = round(wer, 3)
            if wer <= wer_threshold:
                good.append(ref)
                status = "OK"
            else:
                bad.append(ref)
                status = "FILTERED"
            print(f"  {ref['speaker_id']}: WER={wer:.0%} {status}  hyp='{hyp[:60]}'")
        except Exception as e:
            print(f"  {ref['speaker_id']}: probe failed ({e}), keeping")
            good.append(ref)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    print(f"  Speaker probe: {len(good)} good, {len(bad)} filtered")
    return good, bad


def select_top_speakers(
    refs: List[Dict],
    top_n: int = 10,
) -> List[Dict]:
    """Select top_n speakers by lowest probe WER, falling back to RMS energy."""
    scored = [r for r in refs if "probe_wer" in r]
    unscored = [r for r in refs if "probe_wer" not in r]
    scored.sort(key=lambda r: r["probe_wer"])
    selected = scored[:top_n]
    if len(selected) < top_n:
        unscored.sort(key=lambda r: -r["rms_energy"])
        selected.extend(unscored[:top_n - len(selected)])
    return selected


# ---------------------------------------------------------------------------
# Synthesis (with speaker_id metadata in manifest)
# ---------------------------------------------------------------------------

def run_xtts_synthesis(
    corpus_path: str,
    speaker_refs: List[str],
    output_dir: str = "phase2_output/xtts",
    device: str = "cuda",
    max_items: Optional[int] = None,
) -> str:
    """
    Synthesize corpus with XTTS. Rotates through speaker refs.
    Manifest includes speaker_id for downstream analysis.
    """
    tts = init_xtts(device)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(output_dir) / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()
    ref_idx = 0
    count = 0
    with open(corpus_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if max_items:
        lines = lines[:max_items]
    n_spk = len(speaker_refs)
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
        ref = speaker_refs[ref_idx % n_spk]
        spk_id = f"spk_{ref_idx % n_spk:02d}"
        out_name = f"xtts_{i:06d}.wav"
        out_path = Path(output_dir) / out_name
        try:
            synthesize_xtts(tts, text, ref, str(out_path), device=device)
            with open(manifest_path, "a", encoding="utf-8") as mf:
                mf.write(json.dumps({
                    "audio": str(out_path), "text": text,
                    "speaker_id": spk_id,
                }, ensure_ascii=False) + "\n")
            count += 1
        except Exception as e:
            print(f"Skip {i}: {e}")
        ref_idx += 1
    print(f"Synthesized {count} clips ({n_spk} speakers) -> {manifest_path}")
    return str(manifest_path)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_extract = sub.add_parser("extract_refs",
        help="Extract diverse speaker references with quality filtering")
    p_extract.add_argument("--csv", required=True, help="human_anotation_vb.csv")
    p_extract.add_argument("--audio_dirs", nargs="+", required=True)
    p_extract.add_argument("--output_dir", default="speaker_references")
    p_extract.add_argument("--duration", type=float, default=5.0)
    p_extract.add_argument("--max_speakers", type=int, default=20)
    p_extract.add_argument("--probe", action="store_true",
        help="Run Whisper probe to detect hallucination-prone speakers")
    p_extract.add_argument("--top_n", type=int, default=10,
        help="Select top N speakers after probing")
    p_extract.add_argument("--device", default="cuda")

    p_syn = sub.add_parser("synthesize")
    p_syn.add_argument("--corpus", required=True, help="combined_corpus.jsonl")
    p_syn.add_argument("--refs", nargs="+", required=True, help="Speaker reference WAVs")
    p_syn.add_argument("--output_dir", default="phase2_output/xtts")
    p_syn.add_argument("--device", default="cuda")
    p_syn.add_argument("--max", type=int, default=None)

    args = parser.parse_args()

    if args.cmd == "extract_refs":
        refs = create_speaker_references_from_csv(
            args.csv, args.audio_dirs,
            output_dir=args.output_dir,
            duration_sec=args.duration,
            max_speakers=args.max_speakers,
        )
        if args.probe and refs:
            tts = init_xtts(args.device)
            good, bad = probe_speaker_quality(tts, refs, device=args.device)
            top = select_top_speakers(good, top_n=args.top_n)
            profiles_path = str(Path(args.output_dir) / "speaker_profiles.json")
            with open(profiles_path, "w") as f:
                json.dump(top, f, indent=2)
            print(f"\nTop {len(top)} speakers saved to {profiles_path}")
            for r in top:
                wer_s = f"WER={r['probe_wer']:.0%}" if "probe_wer" in r else ""
                print(f"  {r['speaker_id']}: rms={r['rms_energy']:.4f} {wer_s}")
        else:
            ref_paths = [r["ref_path"] for r in refs]
            print(f"Created {len(refs)} speaker refs: {ref_paths}")

    elif args.cmd == "synthesize":
        manifest = run_xtts_synthesis(
            args.corpus, args.refs,
            output_dir=args.output_dir,
            device=args.device,
            max_items=args.max,
        )
        print(f"Manifest: {manifest}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
