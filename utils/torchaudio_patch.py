"""
Monkey-patch torchaudio.load/save for torchaudio 2.9+ where torchcodec
is the only backend but fails due to FFmpeg/ABI issues.
Replaces with soundfile-based implementations.

Usage: import this module BEFORE importing TTS or any code that calls torchaudio.load
"""

import torch
import numpy as np


def _load_soundfile(uri, frame_offset=0, num_frames=-1, normalize=True,
                    channels_first=True, format=None, buffer_size=4096,
                    backend=None):
    import soundfile as sf
    audio, sr = sf.read(str(uri), start=frame_offset,
                        stop=None if num_frames == -1 else frame_offset + num_frames,
                        dtype="float32", always_2d=True)
    # audio shape: [time, channels]
    tensor = torch.from_numpy(audio)
    if channels_first:
        tensor = tensor.T  # [channels, time]
    return tensor, sr


def _save_soundfile(uri, src, sample_rate, channels_first=True,
                    format=None, encoding=None, bits_per_sample=None,
                    buffer_size=4096, backend=None, compression=None):
    import soundfile as sf
    if isinstance(src, torch.Tensor):
        src = src.detach().cpu().numpy()
    if channels_first and src.ndim == 2:
        src = src.T  # [time, channels]
    sf.write(str(uri), src, sample_rate)


def patch():
    import torchaudio
    torchaudio.load = _load_soundfile
    torchaudio.save = _save_soundfile


patch()
