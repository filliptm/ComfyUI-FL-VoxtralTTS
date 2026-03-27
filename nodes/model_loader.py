"""FL Voxtral TTS Model Loader node."""

import logging
from comfy.utils import ProgressBar

logger = logging.getLogger(__name__)


class FL_VoxtralTTS_ModelLoader:
    """Downloads and loads the Voxtral-4B TTS model for text-to-speech generation."""

    CATEGORY = "FL/TTS"
    FUNCTION = "load_model"
    RETURN_TYPES = ("VOXTRAL_MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "force_reload": ("BOOLEAN", {"default": False}),
            },
        }

    def load_model(self, device, dtype, force_reload):
        from modules.loader import VoxtralLoader

        pbar = ProgressBar(3)
        pbar.update(1)

        logger.info(f"Loading Voxtral TTS on {device} ({dtype})...")
        pipeline = VoxtralLoader.load(
            device=device,
            dtype=dtype,
            force_reload=force_reload,
        )
        pbar.update(1)

        result = {
            "pipeline": pipeline,
            "device": device,
            "dtype": dtype,
        }
        pbar.update(1)
        return (result,)
