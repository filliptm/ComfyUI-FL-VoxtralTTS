"""FL Voxtral TTS Generate node."""

import logging
from comfy.utils import ProgressBar

logger = logging.getLogger(__name__)


class FL_VoxtralTTS_Generate:
    """Generate speech from text using the Voxtral-4B TTS model."""

    CATEGORY = "FL/TTS"
    FUNCTION = "generate"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)

    @classmethod
    def INPUT_TYPES(cls):
        from ..modules.model_info import VOICES
        return {
            "required": {
                "model": ("VOXTRAL_MODEL",),
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello, this is a test of the Voxtral text to speech system.",
                }),
                "voice": (VOICES, {"default": "casual_male"}),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**63 - 1,
                }),
                "max_frames": ("INT", {
                    "default": 2048,
                    "min": 128,
                    "max": 4096,
                    "step": 64,
                }),
                "cfg_alpha": ("FLOAT", {
                    "default": 1.2,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.05,
                }),
                "noise_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                }),
                "euler_steps": ("INT", {
                    "default": 8,
                    "min": 2,
                    "max": 32,
                    "step": 1,
                }),
            },
        }

    def generate(self, model, text, voice, seed, max_frames,
                 cfg_alpha, noise_scale, euler_steps):
        from ..modules.audio_utils import numpy_to_comfyui_audio, empty_audio

        pipeline = model["pipeline"]

        pbar = ProgressBar(max_frames)

        def on_progress(current, total):
            pbar.update(1)

        try:
            audio_np, sample_rate = pipeline.generate(
                text=text,
                voice=voice,
                max_frames=max_frames,
                seed=seed,
                cfg_alpha=cfg_alpha,
                noise_scale=noise_scale,
                euler_steps=euler_steps,
                progress_callback=on_progress,
            )
            audio = numpy_to_comfyui_audio(audio_np, sample_rate)
        except Exception as e:
            logger.error(f"Voxtral TTS generation failed: {e}", exc_info=True)
            audio = empty_audio()

        return (audio,)
