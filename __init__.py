"""ComfyUI-FL-VoxtralTTS — Mistral Voxtral-4B Text-to-Speech for ComfyUI."""

import os
import sys
import logging

logger = logging.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add src to sys.path so voxtral_tts package is importable
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Register TTS model folder with ComfyUI
try:
    import folder_paths

    tts_base = os.path.join(folder_paths.models_dir, "tts")
    voxtral_dir = os.path.join(tts_base, "VoxtralTTS")
    os.makedirs(tts_base, exist_ok=True)
    os.makedirs(voxtral_dir, exist_ok=True)

    supported_ext = {".safetensors", ".bin", ".pt", ".pth", ".json"}
    if "tts" not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths["tts"] = ([tts_base], supported_ext)
except Exception as e:
    logger.warning(f"Could not register TTS folder: {e}")

# Import nodes
from .nodes.model_loader import FL_VoxtralTTS_ModelLoader
from .nodes.generate import FL_VoxtralTTS_Generate

NODE_CLASS_MAPPINGS = {
    "FL_VoxtralTTS_ModelLoader": FL_VoxtralTTS_ModelLoader,
    "FL_VoxtralTTS_Generate": FL_VoxtralTTS_Generate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_VoxtralTTS_ModelLoader": "FL Voxtral TTS Model Loader",
    "FL_VoxtralTTS_Generate": "FL Voxtral TTS Generate",
}

WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Banner
BANNER = """
╔══════════════════════════════════════════════╗
║     FL Voxtral TTS — Mistral 4B             ║
║     20 voices • 9 languages • 24kHz         ║
║     Direct PyTorch inference                 ║
╚══════════════════════════════════════════════╝
"""
logger.info(BANNER)
