"""Audio utilities: load, resample, save to 16kHz mono."""

import numpy as np
from pathlib import Path

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import librosa
except ImportError:
    librosa = None


def load_audio(path: str, sr: int = 16000) -> tuple:
    """Load audio as mono float32, resample to sr. Returns (audio, sr)."""
    if not sf:
        raise ImportError("pip install soundfile")
    audio, file_sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != sr and librosa:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=file_sr, target_sr=sr)
    return audio.astype(np.float32), sr


def save_wav(audio: np.ndarray, path: str, sr: int = 16000) -> None:
    """Save to 16kHz mono WAV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.astype(np.float32), sr)
