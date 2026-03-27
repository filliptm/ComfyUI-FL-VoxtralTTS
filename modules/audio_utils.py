"""Audio format conversion utilities for ComfyUI."""

import torch
import numpy as np


def numpy_to_comfyui_audio(audio_array: np.ndarray, sample_rate: int) -> dict:
    """Convert numpy audio to ComfyUI AUDIO dict.

    Args:
        audio_array: 1D float32 numpy array
        sample_rate: Sample rate in Hz

    Returns:
        {"waveform": Tensor[1, 1, samples], "sample_rate": int}
    """
    if audio_array.ndim == 1:
        tensor = torch.from_numpy(audio_array).float().unsqueeze(0).unsqueeze(0)
    elif audio_array.ndim == 2:
        tensor = torch.from_numpy(audio_array).float().unsqueeze(0)
    else:
        tensor = torch.from_numpy(audio_array).float()

    return {"waveform": tensor, "sample_rate": sample_rate}


def empty_audio(sample_rate: int = 24000) -> dict:
    """Return 1 second of silence as a ComfyUI AUDIO dict."""
    silence = torch.zeros(1, 1, sample_rate, dtype=torch.float32)
    return {"waveform": silence, "sample_rate": sample_rate}
