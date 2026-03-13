#!/usr/bin/env python3
"""
Speaker Profile + Augmentation Ablation Experiments
====================================================
4 experiment groups to measure the impact of:
  A) Baseline:  5 speakers, original augmentation
  B) +Speakers: 20 speakers, original augmentation
  C) +Augment:  5 speakers, enhanced augmentation (AddColorNoise+GainTransition)
  D) Combined:  20 speakers, enhanced augmentation

Each group: 50 utterances x XTTS synthesis x 4 augment variants = 200 clips,
then evaluates with Whisper to measure WER.

Usage:
  python speaker_augment_ablation.py --phase extract_refs
  python speaker_augment_ablation.py --phase synthesize
  python speaker_augment_ablation.py --phase augment
  python speaker_augment_ablation.py --phase evaluate
  python speaker_augment_ablation.py --phase all
"""

import json
import os
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

# Patch torchaudio.load/save to use soundfile instead of torchcodec
# (torchaudio 2.9+ removed all backends except torchcodec which has FFmpeg ABI issues)
import importlib.util
_patch_path = str(Path(__file__).resolve().parent / "torchaudio_patch.py")
if os.path.exists(_patch_path):
    _spec = importlib.util.spec_from_file_location("torchaudio_patch", _patch_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

try:
    import soundfile as sf
except ImportError:
    raise ImportError("pip install soundfile")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_ROOT = Path(__file__).resolve().parent

AUDIO_DIRS = [
    "/media/meow/One Touch/ems_call/random_samples_1",
    "/media/meow/One Touch/ems_call/random_samples_2",
]
CSV_PATH = "/media/meow/One Touch/ems_call/vb_ems_anotation/human_anotation_vb.csv"
CORPUS_PATH = str(PIPELINE_ROOT / "phase1_output" / "ems_radio_500.jsonl")

EXPERIMENT_CORPUS_SIZE = 50
NUM_AUGMENT_VARIANTS = 4
SEED = 42


# ---------------------------------------------------------------------------
# Speaker Reference Extraction
# ---------------------------------------------------------------------------

def _find_audio(filename):
    for d in AUDIO_DIRS:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return None


def _extract_clip(audio_path, output_path, start_sec, duration_sec, sr=16000):
    audio, file_sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        import librosa
        audio = librosa.resample(
            audio.astype(np.float32), orig_sr=file_sr, target_sr=sr)
    start = int(start_sec * sr)
    end = int(min(start_sec + duration_sec, len(audio) / sr) * sr)
    clip = audio[start:end].astype(np.float32)
    clip = clip / (np.max(np.abs(clip)) + 1e-8) * 0.95
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, clip, sr)
    return output_path


def _compute_rms(audio_path):
    audio, _ = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return float(np.sqrt(np.mean(audio ** 2)))


def extract_diverse_speaker_refs(output_dir, max_speakers=20, duration_sec=5.0):
    """Extract speaker refs from real EMS audio with RMS energy filtering."""
    import pandas as pd
    df = pd.read_csv(CSV_PATH)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    candidates = []
    for _, row in df.iterrows():
        fname = row["Filename"]
        path = _find_audio(fname)
        if not path:
            continue
        info = sf.info(path)
        candidates.append({
            "filename": fname, "path": path,
            "duration": info.duration,
            "tags": str(row.get("Tags", "")),
        })

    random.seed(SEED)
    random.shuffle(candidates)

    refs = []
    offsets = [3.0, 8.0, 15.0, 25.0]
    ref_id = 0
    for cand in candidates:
        if ref_id >= max_speakers:
            break
        offset = offsets[ref_id % len(offsets)]
        if cand["duration"] < offset + duration_sec:
            offset = max(1.0, cand["duration"] / 2 - duration_sec / 2)
        out_path = os.path.join(output_dir, f"spk_{ref_id:02d}.wav")
        try:
            _extract_clip(cand["path"], out_path, offset, duration_sec)
            energy = _compute_rms(out_path)
            if energy < 0.005:
                print(f"  Skip {cand['filename']}@{offset}s (silent rms={energy:.4f})")
                continue
            refs.append({
                "speaker_id": f"spk_{ref_id:02d}",
                "ref_path": out_path,
                "source_file": cand["filename"],
                "offset_sec": offset,
                "duration_sec": duration_sec,
                "rms_energy": round(energy, 4),
                "tags": cand["tags"],
            })
            ref_id += 1
        except Exception as e:
            print(f"  Skip {cand['filename']}: {e}")

    manifest = os.path.join(output_dir, "speaker_profiles.json")
    with open(manifest, "w") as f:
        json.dump(refs, f, indent=2)
    print(f"Extracted {len(refs)} speaker refs -> {manifest}")
    return refs


# ---------------------------------------------------------------------------
# Corpus Preparation
# ---------------------------------------------------------------------------

def prepare_experiment_corpus(output_path, n=EXPERIMENT_CORPUS_SIZE):
    random.seed(SEED)
    with open(CORPUS_PATH) as f:
        lines = [l.strip() for l in f if l.strip()]
    random.shuffle(lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for line in lines[:n]:
            f.write(line + "\n")
    print(f"Corpus: {n} utterances -> {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# XTTS Synthesis
# ---------------------------------------------------------------------------

def run_synthesis_for_group(corpus_path, speaker_refs, output_dir, device="cuda"):
    import sys
    if str(PIPELINE_ROOT) not in sys.path:
        sys.path.insert(0, str(PIPELINE_ROOT))
    from phase2_speech_synthesis.xtts_voice_cloning import init_xtts, synthesize_xtts

    tts = init_xtts(device)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.jsonl")
    if os.path.exists(manifest_path):
        os.remove(manifest_path)

    with open(corpus_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    n_spk = len(speaker_refs)
    count = 0
    for i, line in enumerate(tqdm(lines, desc=f"Synth ({n_spk} spk)")):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = rec.get("text", "").strip()
        if len(text) < 10:
            continue
        ref = speaker_refs[i % n_spk]
        spk_id = f"spk_{i % n_spk:02d}"
        out_path = os.path.join(output_dir, f"syn_{i:04d}.wav")
        try:
            synthesize_xtts(tts, text, ref, out_path, device=device)
            with open(manifest_path, "a") as mf:
                mf.write(json.dumps({
                    "audio": out_path, "text": text, "speaker_id": spk_id,
                }, ensure_ascii=False) + "\n")
            count += 1
        except Exception as e:
            print(f"  Synth skip {i}: {e}")
    print(f"Synthesized {count} clips -> {manifest_path}")
    return manifest_path


# ---------------------------------------------------------------------------
# Augmentation Pipelines
# ---------------------------------------------------------------------------

def build_original_pipeline():
    """Current Phase 3 augmentation (baseline)."""
    import audiomentations as A
    return A.Compose([
        A.BandPassFilter(min_center_freq=1850, max_center_freq=1850,
                         min_bandwidth_fraction=1.65, max_bandwidth_fraction=1.7, p=0.9),
        A.ClippingDistortion(min_percentile_threshold=0,
                             max_percentile_threshold=10, p=0.3),
        A.AddGaussianSNR(min_snr_db=10, max_snr_db=30, p=0.5),
        A.TimeStretch(min_rate=0.8, max_rate=1.4, p=0.6),
        A.PitchShift(min_semitones=-3, max_semitones=3, p=0.4),
        A.Gain(min_gain_db=-12, max_gain_db=6, p=0.5),
        A.TimeMask(min_band_part=0.0, max_band_part=0.15, p=0.4),
    ])


def build_enhanced_pipeline():
    """Enhanced: + AddColorNoise (pink/brown) + GainTransition."""
    import audiomentations as A
    return A.Compose([
        A.BandPassFilter(min_center_freq=1850, max_center_freq=1850,
                         min_bandwidth_fraction=1.65, max_bandwidth_fraction=1.7, p=0.9),
        A.ClippingDistortion(min_percentile_threshold=0,
                             max_percentile_threshold=10, p=0.3),
        # Colored noise: f_decay -4.0=brown, -2.0=pink, -1.0=blue-ish
        A.AddColorNoise(min_snr_db=8, max_snr_db=30,
                        min_f_decay=-4.0, max_f_decay=-1.0, p=0.6),
        A.AddGaussianSNR(min_snr_db=12, max_snr_db=35, p=0.35),
        # Gradual gain change simulating mic distance variation
        A.GainTransition(min_gain_db=-12, max_gain_db=6,
                         min_duration=0.2, max_duration=0.8, p=0.45),
        A.TimeStretch(min_rate=0.8, max_rate=1.4, p=0.6),
        A.PitchShift(min_semitones=-3, max_semitones=3, p=0.4),
        A.Gain(min_gain_db=-12, max_gain_db=6, p=0.5),
        A.TimeMask(min_band_part=0.0, max_band_part=0.15, p=0.4),
    ])


def run_augmentation_for_group(manifest_path, output_dir, pipeline_fn,
                               num_variants=NUM_AUGMENT_VARIANTS, sr=16000):
    import sys
    p3 = str(PIPELINE_ROOT / "phase3_noise_augmentation")
    if p3 not in sys.path:
        sys.path.insert(0, p3)
    from radio_channel_simulator import simulate_radio_channel

    pipeline = pipeline_fn()
    audio_dir = os.path.join(output_dir, "audio")
    Path(audio_dir).mkdir(parents=True, exist_ok=True)
    out_manifest = os.path.join(output_dir, "augmented_manifest.jsonl")
    if os.path.exists(out_manifest):
        os.remove(out_manifest)

    with open(manifest_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    count = 0
    for line in tqdm(lines, desc="Augmenting"):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        audio_path = rec["audio"]
        if not os.path.exists(audio_path):
            continue
        audio, _ = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        stem = Path(audio_path).stem
        for v in range(num_variants):
            try:
                out = audio.copy()
                if pipeline is not None:
                    out = pipeline(samples=out, sample_rate=sr)
                out = simulate_radio_channel(out, sr)
                out = np.clip(out.astype(np.float32), -1, 1)
                out_path = os.path.join(audio_dir, f"{stem}_v{v}.wav")
                sf.write(out_path, out, sr)
                with open(out_manifest, "a") as mf:
                    mf.write(json.dumps({
                        "audio": out_path,
                        "text": rec["text"],
                        "speaker_id": rec.get("speaker_id", "unknown"),
                    }, ensure_ascii=False) + "\n")
                count += 1
            except Exception as e:
                print(f"  Aug skip {stem}_v{v}: {e}")
    print(f"Augmented {count} clips -> {out_manifest}")
    return out_manifest


# ---------------------------------------------------------------------------
# Whisper Evaluation
# ---------------------------------------------------------------------------

def _word_error_rate(ref, hyp):
    r, h = len(ref), len(hyp)
    d = np.zeros((r + 1, h + 1), dtype=int)
    for i in range(r + 1):
        d[i][0] = i
    for j in range(h + 1):
        d[0][j] = j
    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
    return d[r][h] / max(r, 1)


def evaluate_with_whisper(manifest_path, model_name="openai/whisper-small.en",
                          device="cuda"):
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import librosa

    print(f"Loading {model_name} ...")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()

    with open(manifest_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    results = []
    for line in tqdm(lines, desc="Evaluating"):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        audio_path = rec["audio"]
        ref_text = rec["text"].strip().lower()
        if not os.path.exists(audio_path):
            continue
        audio, _ = librosa.load(audio_path, sr=16000)
        feats = processor(audio, sampling_rate=16000,
                          return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            ids = model.generate(feats)
        hyp = processor.batch_decode(ids, skip_special_tokens=True)[0].strip().lower()
        wer = _word_error_rate(ref_text.split(), hyp.split())
        results.append({
            "audio": audio_path, "ref": ref_text, "hyp": hyp,
            "wer": wer, "speaker_id": rec.get("speaker_id", "unknown"),
        })

    avg_wer = np.mean([r["wer"] for r in results]) if results else 0.0
    return {"avg_wer": avg_wer, "n_samples": len(results), "details": results}


# ---------------------------------------------------------------------------
# Experiment Definitions
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "A_baseline": {
        "desc": "5 speakers + original augmentation",
        "n_speakers": 5,
        "pipeline_fn": build_original_pipeline,
    },
    "B_more_speakers": {
        "desc": "20 speakers + original augmentation",
        "n_speakers": 20,
        "pipeline_fn": build_original_pipeline,
    },
    "C_enhanced_aug": {
        "desc": "5 spk + enhanced (ColorNoise+GainTransition)",
        "n_speakers": 5,
        "pipeline_fn": build_enhanced_pipeline,
    },
    "D_combined": {
        "desc": "20 speakers + enhanced augmentation",
        "n_speakers": 20,
        "pipeline_fn": build_enhanced_pipeline,
    },
}


# ---------------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------------

def run_all(device="cuda", skip_eval=False):
    base = EXPERIMENT_ROOT / "ablation_results"
    base.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Extract 20 speaker references")
    print("=" * 60)
    refs = extract_diverse_speaker_refs(str(base / "speaker_refs"), max_speakers=20)
    ref_paths = [r["ref_path"] for r in refs]

    print("\n" + "=" * 60)
    print("STEP 2: Prepare experiment corpus (50 utterances)")
    print("=" * 60)
    corpus = prepare_experiment_corpus(str(base / "experiment_corpus.jsonl"))

    summary = {}
    for name, cfg in EXPERIMENTS.items():
        print("\n" + "=" * 60)
        print(f"EXP {name}: {cfg['desc']}")
        print("=" * 60)
        d = base / name
        rr = ref_paths[:cfg["n_speakers"]]
        print(f"  Using {len(rr)} speaker references")
        sm = run_synthesis_for_group(corpus, rr, str(d / "synth"), device)
        am = run_augmentation_for_group(sm, str(d / "augmented"), cfg["pipeline_fn"])
        if not skip_eval:
            ev = evaluate_with_whisper(am, device=device)
            summary[name] = {
                "desc": cfg["desc"], "n_speakers": cfg["n_speakers"],
                "avg_wer": ev.get("avg_wer", -1),
                "n_samples": ev.get("n_samples", 0),
            }
            with open(d / "eval_results.json", "w") as f:
                json.dump(ev, f, indent=2)
        else:
            summary[name] = {
                "desc": cfg["desc"], "n_speakers": cfg["n_speakers"],
                "status": "audio generated, eval skipped",
            }

    with open(base / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _print_summary(summary)
    return summary


def _print_summary(summary):
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    for n, s in summary.items():
        w = f"WER={s['avg_wer']:.1%}" if "avg_wer" in s else s.get("status", "")
        print(f"  {n:20s} | {s['desc']:45s} | {w}")


def phase_extract_refs():
    base = EXPERIMENT_ROOT / "ablation_results"
    refs = extract_diverse_speaker_refs(str(base / "speaker_refs"), max_speakers=20)
    print("\nSpeaker profiles:")
    for r in refs:
        print(f"  {r['speaker_id']}: {r['source_file']} "
              f"@{r['offset_sec']}s rms={r['rms_energy']:.4f}")


def phase_synthesize(device="cuda"):
    base = EXPERIMENT_ROOT / "ablation_results"
    cp = str(base / "experiment_corpus.jsonl")
    if not os.path.exists(cp):
        prepare_experiment_corpus(cp)
    with open(base / "speaker_refs" / "speaker_profiles.json") as f:
        all_refs = json.load(f)
    rp = [r["ref_path"] for r in all_refs]
    for name, cfg in EXPERIMENTS.items():
        print(f"\n--- Synth {name} ({cfg['n_speakers']} spk) ---")
        run_synthesis_for_group(cp, rp[:cfg["n_speakers"]],
                                str(base / name / "synth"), device)


def phase_augment():
    base = EXPERIMENT_ROOT / "ablation_results"
    for name, cfg in EXPERIMENTS.items():
        sm = str(base / name / "synth" / "manifest.jsonl")
        if not os.path.exists(sm):
            print(f"Skip {name}: synthesize first.")
            continue
        print(f"\n--- Augment {name} ---")
        run_augmentation_for_group(sm, str(base / name / "augmented"),
                                   cfg["pipeline_fn"])


def phase_evaluate(device="cuda"):
    base = EXPERIMENT_ROOT / "ablation_results"
    summary = {}
    for name, cfg in EXPERIMENTS.items():
        am = str(base / name / "augmented" / "augmented_manifest.jsonl")
        if not os.path.exists(am):
            print(f"Skip {name}: no augmented manifest.")
            continue
        print(f"\n--- Eval {name} ---")
        result = evaluate_with_whisper(am, device=device)
        summary[name] = {
            "desc": cfg["desc"],
            "avg_wer": result.get("avg_wer", -1),
            "n_samples": result.get("n_samples", 0),
        }
        with open(base / name / "eval_results.json", "w") as f:
            json.dump(result, f, indent=2)
    with open(base / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for n, s in summary.items():
        print(f"  {n:20s} | WER={s['avg_wer']:.1%} (n={s['n_samples']})")


def main():
    p = argparse.ArgumentParser(description="Speaker + Augmentation Ablation")
    p.add_argument("--phase", required=True,
                   choices=["extract_refs", "synthesize", "augment", "evaluate", "all"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip_eval", action="store_true")
    a = p.parse_args()
    dispatch = {
        "extract_refs": phase_extract_refs,
        "synthesize": lambda: phase_synthesize(a.device),
        "augment": phase_augment,
        "evaluate": lambda: phase_evaluate(a.device),
        "all": lambda: run_all(a.device, a.skip_eval),
    }
    dispatch[a.phase]()


if __name__ == "__main__":
    main()
