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
                "max_frames": ("INT", {
                    "default": 2048,
                    "min": 128,
                    "max": 4096,
                    "step": 64,
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**63 - 1,
                }),
            },
        }

    def generate(self, model, text, voice, max_frames, seed):
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
                progress_callback=on_progress,
            )
            audio = numpy_to_comfyui_audio(audio_np, sample_rate)
        except Exception as e:
            logger.error(f"Voxtral TTS generation failed: {e}", exc_info=True)
            audio = empty_audio()

        return (audio,)
