#!/usr/bin/env python3
"""
Noise Augmentation Pipeline for EMS Synthetic Audio
Uses audiomentations + radio_channel_simulator
Each clip -> 3-5 augmented variants
"""

import json
import random
import argparse
from pathlib import Path
from typing import Optional, List

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    class _NoOpBar:
        def __init__(self, *a, **kw): pass
        def update(self, n=1): pass
    def tqdm(x=None, total=None, **kwargs):
        return _NoOpBar()

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False

try:
    import audiomentations as A
    HAS_AUDIOMENTATIONS = True
except (ImportError, AttributeError, OSError):
    HAS_AUDIOMENTATIONS = False  # NumPy 2.x / scipy conflicts

try:
    import sys
    from pathlib import Path
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    from radio_channel_simulator import simulate_radio_channel
except (ImportError, AttributeError, OSError):
    def simulate_radio_channel(audio: np.ndarray, sr: int = 16000, **kwargs) -> np.ndarray:
        """Fallback when scipy unavailable: add PTT click + gain + dropout only."""
        out = audio.astype(np.float32)
        if random.random() < 0.7:
            n = int(sr * 0.02)
            click = np.random.uniform(-0.2, 0.2, n).astype(np.float32) * np.exp(-np.linspace(0, 5, n))
            out = np.concatenate([click, out])
        out += np.random.randn(len(out)).astype(np.float32) * 0.01
        if random.random() < 0.3:
            block = int(sr * 0.05)
            for i in range(0, len(out) - block, block):
                if random.random() < 0.02:
                    out[i : i + block] = 0
        return np.clip(out, -1, 1).astype(np.float32)


def _resample_numpy(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Pure NumPy resampling (avoids librosa/scipy NumPy 2.x conflicts)."""
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    new_len = int(duration * target_sr)
    indices = np.linspace(0, len(audio) - 1, new_len, dtype=np.float32)
    return np.interp(indices, np.arange(len(audio), dtype=np.float32), audio.astype(np.float32)).astype(np.float32)


def load_audio(path: str, sr: int = 16000) -> tuple:
    """Load audio, return (samples, sr). Uses pure NumPy resampling to avoid NumPy 2.x / scipy conflicts."""
    if not HAS_SF:
        raise ImportError("pip install soundfile")
    audio, file_sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        audio = _resample_numpy(audio.astype(np.float32), file_sr, sr)
    return audio.astype(np.float32), sr


def save_audio(audio: np.ndarray, path: str, sr: int = 16000) -> None:
    """Save audio to WAV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr)


def build_augmentation_pipeline(noise_dir: Optional[str] = None,
                                enhanced: bool = True) -> "A.Compose":
    """Build audiomentations pipeline.

    Args:
        noise_dir: path to background noise samples (optional)
        enhanced: if True, add conservative AddColorNoise + GainTransition
                  (validated by ablation experiment: pink/brown noise better
                  matches real EMS radio characteristics)
    """
    if not HAS_AUDIOMENTATIONS:
        return None
    transforms = [
        A.BandPassFilter(min_center_freq=1850, max_center_freq=1850,
                         min_bandwidth_fraction=1.65, max_bandwidth_fraction=1.7, p=0.9),
        A.ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=10, p=0.3),
    ]
    if enhanced:
        # Colored noise (f_decay: -4=brown, -2=pink) — more realistic than pure Gaussian
        # Conservative: p=0.35, higher min_snr_db=12 (ablation showed aggressive settings
        # can trigger Whisper hallucination on weaker speakers)
        transforms.append(
            A.AddColorNoise(min_snr_db=12, max_snr_db=30,
                            min_f_decay=-4.0, max_f_decay=-1.0, p=0.35))
        transforms.append(
            A.AddGaussianSNR(min_snr_db=12, max_snr_db=35, p=0.35))
        # Gradual gain change simulating speaker-to-mic distance variation
        transforms.append(
            A.GainTransition(min_gain_db=-10, max_gain_db=4,
                             min_duration=0.2, max_duration=0.8, p=0.35))
    else:
        transforms.append(
            A.AddGaussianSNR(min_snr_db=10, max_snr_db=30, p=0.5))
    transforms.extend([
        A.TimeStretch(min_rate=0.8, max_rate=1.4, p=0.6),
        A.PitchShift(min_semitones=-3, max_semitones=3, p=0.4),
        A.Gain(min_gain_db=-12, max_gain_db=6, p=0.5),
        A.TimeMask(min_band_part=0.0, max_band_part=0.15, p=0.4),
    ])
    try:
        import fast_mp3_augment  # noqa: F401
        transforms.insert(3, A.Mp3Compression(min_bitrate=8, max_bitrate=32, p=0.7))
    except ImportError:
        pass
    if noise_dir and Path(noise_dir).exists():
        transforms.insert(
            2,
            A.AddBackgroundNoise(
                sounds_path=noise_dir,
                min_snr_db=3,
                max_snr_db=15,
                p=0.85,
            ),
        )
    return A.Compose(transforms)


def overlay_crosstalk(
    audio: np.ndarray,
    pool: List[np.ndarray],
    sr: int = 16000,
    snr_db: float = 10.0,
) -> np.ndarray:
    """Mix a random clip from pool on top of audio at a given SNR to simulate cross-talk."""
    if not pool:
        return audio
    other = random.choice(pool)
    if len(other) < sr * 0.3:
        return audio
    start = random.randint(0, max(0, len(audio) - len(other)))
    rms_sig = np.sqrt(np.mean(audio ** 2)) + 1e-8
    rms_other = np.sqrt(np.mean(other ** 2)) + 1e-8
    gain = rms_sig / (rms_other * (10 ** (snr_db / 20)))
    out = audio.copy()
    end = min(start + len(other), len(out))
    out[start:end] += gain * other[: end - start]
    return out


def augment_one(
    audio: np.ndarray,
    sr: int,
    pipeline: Optional["A.Compose"],
    radio_sim: bool = True,
    crosstalk_pool: Optional[List[np.ndarray]] = None,
    crosstalk_p: float = 0.2,
) -> np.ndarray:
    """Apply augmentation to one audio array."""
    out = audio.copy()
    if pipeline:
        out = pipeline(samples=out, sample_rate=sr)
        if isinstance(out, dict):
            out = out.get("samples", out)
    if crosstalk_pool and random.random() < crosstalk_p:
        snr = random.uniform(5, 15)
        out = overlay_crosstalk(out, crosstalk_pool, sr, snr)
    if radio_sim:
        out = simulate_radio_channel(out, sr)
    return np.clip(out.astype(np.float32), -1, 1)


def _build_crosstalk_pool(
    manifest_path: str, sr: int, max_pool: int = 100,
) -> List[np.ndarray]:
    """Pre-load a random subset of clips for cross-talk overlay."""
    pool: List[np.ndarray] = []
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            all_lines = [l.strip() for l in f if l.strip()]
        random.shuffle(all_lines)
        for line in all_lines[:max_pool]:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = rec.get("audio", "")
            if Path(p).exists():
                a, _ = load_audio(p, sr)
                pool.append(a)
            if len(pool) >= max_pool:
                break
    except Exception:
        pass
    return pool


def run_augmentation(
    manifest_path: str,
    output_dir: str = "phase3_output",
    num_variants: int = 4,
    noise_dir: Optional[str] = None,
    sr: int = 16000,
    max_items: Optional[int] = None,
    crosstalk_p: float = 0.2,
    enhanced_aug: bool = True,
) -> str:
    """
    Augment each clip in manifest. Creates num_variants per clip.
    Returns path to augmented manifest.
    """
    pipeline = build_augmentation_pipeline(noise_dir, enhanced=enhanced_aug)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_manifest = Path(output_dir) / "augmented_manifest.jsonl"
    if out_manifest.exists():
        out_manifest.unlink()

    print("Building cross-talk pool ...")
    ct_pool = _build_crosstalk_pool(manifest_path, sr)
    print(f"  cross-talk pool: {len(ct_pool)} clips")

    count = 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if max_items:
        lines = lines[:max_items]
    total = len(lines) * num_variants
    print(f"Phase 3: augmenting {len(lines)} clips x {num_variants} variants = {total} total")
    pbar = tqdm(total=total, desc="Augmenting", unit="clip")
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            pbar.update(num_variants)
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            pbar.update(num_variants)
            continue
        audio_path = rec.get("audio", "")
        text = rec.get("text", "")
        if not Path(audio_path).exists():
            pbar.update(num_variants)
            continue
        try:
            audio, _ = load_audio(audio_path, sr)
        except Exception as e:
            print(f"Skip {audio_path}: {e}")
            pbar.update(num_variants)
            continue
        base_name = Path(audio_path).stem
        for v in range(num_variants):
            try:
                aug = augment_one(
                    audio, sr, pipeline, radio_sim=True,
                    crosstalk_pool=ct_pool, crosstalk_p=crosstalk_p,
                )
                out_path = Path(output_dir) / "audio" / f"{base_name}_v{v}.wav"
                save_audio(aug, str(out_path), sr)
                with open(out_manifest, "a", encoding="utf-8") as mf:
                    mf.write(json.dumps({"audio": str(out_path), "text": text}, ensure_ascii=False) + "\n")
                count += 1
            except Exception as e:
                print(f"Augment skip {base_name} v{v}: {e}")
            pbar.update(1)
    return str(out_manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="phase2 merged_manifest.jsonl")
    parser.add_argument("--output_dir", default="phase3_output")
    parser.add_argument("--num_variants", type=int, default=4)
    parser.add_argument("--noise_dir", type=str, default=None, help="noise_samples/ems/")
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--crosstalk_p", type=float, default=0.2, help="cross-talk overlay probability")
    parser.add_argument("--no_enhanced", action="store_true",
                        help="Disable enhanced augmentation (AddColorNoise+GainTransition)")
    args = parser.parse_args()
    run_augmentation(
        args.manifest,
        args.output_dir,
        args.num_variants,
        args.noise_dir,
        max_items=args.max,
        crosstalk_p=args.crosstalk_p,
        enhanced_aug=not args.no_enhanced,
    )
    print(f"Done. Manifest: {args.output_dir}/augmented_manifest.jsonl")


if __name__ == "__main__":
    main()
