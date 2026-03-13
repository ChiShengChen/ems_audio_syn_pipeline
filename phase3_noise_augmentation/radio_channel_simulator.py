#!/usr/bin/env python3
"""
EMS Radio Channel Simulator
Simulates land mobile radio characteristics:
- PTT click at start/end
- Bandpass 300-3400 Hz
- Resample degradation (8k->16k)
- Signal dropout
"""

import random
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import Optional

try:
    import scipy.signal as signal
    HAS_SCIPY = True
except (ImportError, AttributeError, OSError):
    HAS_SCIPY = False

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False

try:
    subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
    HAS_FFMPEG = True
except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
    HAS_FFMPEG = False


def bandpass_filter(audio: np.ndarray, low: float, high: float, sr: int) -> np.ndarray:
    """Butterworth bandpass 300-3400 Hz (land mobile radio)."""
    if not HAS_SCIPY:
        return audio
    nyq = sr / 2
    low_n = max(low / nyq, 0.01)
    high_n = min(high / nyq, 0.99)
    b, a = signal.butter(4, [low_n, high_n], btype="band")
    return signal.filtfilt(b, a, audio.astype(np.float64)).astype(np.float32)


def add_ptt_click(audio: np.ndarray, sr: int, duration_ms: float = 20) -> np.ndarray:
    """Add brief PTT (push-to-talk) click at start and optionally end."""
    n = int(sr * duration_ms / 1000)
    click = np.random.uniform(-0.3, 0.3, n).astype(np.float32)
    # Exponential decay
    click *= np.exp(-np.linspace(0, 5, n))
    out = np.concatenate([click, audio])
    if random.random() < 0.5:
        out = np.concatenate([out, click * 0.5])
    return out.astype(np.float32)


def resample_degrade(audio: np.ndarray, orig_sr: int, target_sr: int = 8000) -> np.ndarray:
    """Downsample then upsample to simulate codec artifacts."""
    if not HAS_SCIPY:
        return audio
    n = int(len(audio) * target_sr / orig_sr)
    audio_ds = signal.resample(audio, n)
    audio_us = signal.resample(audio_ds, len(audio))
    return audio_us.astype(np.float32)


def simulate_signal_dropout(audio: np.ndarray, sr: int, dropout_prob: float = 0.02) -> np.ndarray:
    """Randomly zero out short segments (signal dropout)."""
    out = audio.copy()
    block = int(sr * 0.05)  # 50ms blocks
    for i in range(0, len(out) - block, block):
        if random.random() < dropout_prob:
            out[i : i + block] = 0
    return out


def add_light_reverb(audio: np.ndarray, sr: int, room_size: float = 0.2) -> np.ndarray:
    """Simple reverb simulation (ambulance cabin)."""
    if not HAS_SCIPY:
        return audio
    delay = int(sr * 0.03 * room_size)
    decay = 0.3
    out = audio.copy()
    if len(audio) > delay:
        out[delay:] += decay * audio[:-delay]
    return np.clip(out, -1, 1).astype(np.float32)


def codec_degrade(audio: np.ndarray, sr: int = 16000, codec: str = "amr") -> np.ndarray:
    """Encode/decode through a lossy codec via ffmpeg to add realistic artifacts.
    Supported codecs: 'amr' (AMR-NB 4.75-12.2 kbps) and 'gsm'."""
    if not HAS_FFMPEG or not HAS_SF:
        return audio
    bitrate = random.choice(["4.75k", "5.9k", "7.4k", "12.2k"]) if codec == "amr" else None
    try:
        with tempfile.TemporaryDirectory() as td:
            in_wav = Path(td) / "in.wav"
            coded = Path(td) / ("coded.amr" if codec == "amr" else "coded.gsm")
            out_wav = Path(td) / "out.wav"
            sf.write(str(in_wav), audio, sr)
            if codec == "amr":
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(in_wav),
                    "-ar", "8000", "-ac", "1", "-ab", bitrate,
                    str(coded),
                ], capture_output=True, timeout=15)
            else:
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(in_wav),
                    "-ar", "8000", "-ac", "1", "-c:a", "libgsm",
                    str(coded),
                ], capture_output=True, timeout=15)
            subprocess.run([
                "ffmpeg", "-y", "-i", str(coded),
                "-ar", str(sr), "-ac", "1",
                str(out_wav),
            ], capture_output=True, timeout=15)
            if out_wav.exists():
                degraded, _ = sf.read(str(out_wav))
                return degraded.astype(np.float32)
    except Exception:
        pass
    return audio


def simulate_radio_channel(
    audio: np.ndarray,
    sr: int = 16000,
    add_ptt: bool = True,
    do_bandpass: bool = True,
    do_resample_degrade: bool = True,
    do_dropout: bool = True,
    do_reverb: bool = True,
    do_codec: bool = True,
) -> np.ndarray:
    """Full EMS radio simulation with codec degradation."""
    out = audio.astype(np.float32)
    if add_ptt and random.random() < 0.7:
        out = add_ptt_click(out, sr)
    if do_bandpass:
        out = bandpass_filter(out, 300, 3400, sr)
    if do_codec and random.random() < 0.5:
        codec = random.choice(["amr", "gsm"])
        out = codec_degrade(out, sr, codec)
    elif do_resample_degrade and random.random() < 0.5:
        out = resample_degrade(out, sr, 8000)
    if do_dropout and random.random() < 0.3:
        out = simulate_signal_dropout(out, sr)
    if do_reverb and random.random() < 0.4:
        out = add_light_reverb(out, sr)
    return np.clip(out, -1, 1).astype(np.float32)


def process_file(
    input_path: str,
    output_path: str,
    sr: int = 16000,
    **kwargs,
) -> str:
    """Process audio file through radio simulator."""
    if not HAS_SF:
        raise ImportError("Install soundfile: pip install soundfile")
    audio, file_sr = sf.read(input_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        if HAS_SCIPY:
            from scipy.signal import resample
            audio = resample(audio, int(len(audio) * sr / file_sr))
        else:
            import librosa
            audio = librosa.resample(audio.astype(np.float32), orig_sr=file_sr, target_sr=sr)
    audio = audio.astype(np.float32) / (np.max(np.abs(audio)) + 1e-8) * 0.9
    out = simulate_radio_channel(audio, sr, **kwargs)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, out, sr)
    return output_path
