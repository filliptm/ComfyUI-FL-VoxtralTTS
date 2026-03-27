"""Model download, weight loading, and caching for Voxtral TTS."""

from __future__ import annotations
import logging
import sys
import os
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

# Global model cache
_MODEL_CACHE = {}


def get_voxtral_models_dir() -> Path:
    """Get or create the VoxtralTTS model storage directory."""
    try:
        import folder_paths
        try:
            tts_paths = folder_paths.get_folder_paths("tts")
        except (KeyError, AttributeError):
            tts_paths = None
        if tts_paths:
            base = Path(tts_paths[0])
        else:
            base = Path(folder_paths.models_dir) / "tts"
    except ImportError:
        base = Path("models") / "tts"

    voxtral_dir = base / "VoxtralTTS"
    voxtral_dir.mkdir(parents=True, exist_ok=True)
    return voxtral_dir


def download_model(repo_id: str, local_dir: Path) -> Path:
    """Download model from HuggingFace Hub if not already present."""
    marker = local_dir / "consolidated.safetensors"
    if marker.exists():
        logger.info(f"Model already downloaded at {local_dir}")
        return local_dir

    logger.info(f"Downloading {repo_id} to {local_dir}...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    logger.info(f"Download complete: {local_dir}")
    return local_dir


def _remap_weights(state_dict: dict) -> dict:
    """Remap Mistral weight names to our module structure.

    Returns dict with keys:
        "backbone": {name: tensor}
        "acoustic_transformer": {name: tensor}
        "codec_decoder": {name: tensor}
        "audio_embeddings": {name: tensor}
    """
    parts = {
        "backbone": {},
        "acoustic_transformer": {},
        "codec_decoder": {},
        "audio_embeddings": {},
    }

    for key, value in state_dict.items():
        if key.startswith("acoustic_transformer."):
            new_key = key[len("acoustic_transformer."):]
            parts["acoustic_transformer"][new_key] = value
        elif key.startswith("audio_tokenizer."):
            new_key = key[len("audio_tokenizer."):]
            parts["codec_decoder"][new_key] = value
        elif key.startswith("mm_audio_embeddings.audio_codebook_embeddings."):
            new_key = key[len("mm_audio_embeddings.audio_codebook_embeddings."):]
            parts["audio_embeddings"][new_key] = value
        elif key.startswith("mm_audio_embeddings.tok_embeddings."):
            # Shared with backbone tok_embeddings
            new_key = key[len("mm_audio_embeddings."):]
            parts["backbone"][new_key] = value
        else:
            # Everything else is backbone (layers, norm, tok_embeddings, output)
            parts["backbone"][key] = value

    return parts


class VoxtralLoader:
    """Handles downloading, loading, and caching the Voxtral TTS pipeline."""

    @staticmethod
    def load(device: str = "auto", dtype: str = "bfloat16",
             force_reload: bool = False) -> "VoxtralTTSPipeline":
        """Load the full Voxtral TTS pipeline.

        Args:
            device: "auto", "cuda", "mps", or "cpu"
            dtype: "bfloat16", "float16", or "float32"
            force_reload: Force re-download and reload

        Returns:
            VoxtralTTSPipeline instance ready for generation
        """
        from .model_info import MODEL_REPO_ID, get_default_device

        # Resolve device
        if device == "auto":
            device = get_default_device()
        torch_device = torch.device(device)

        # Resolve dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        # MPS doesn't support bfloat16 well — fall back to float16
        if device == "mps" and torch_dtype == torch.bfloat16:
            logger.info("MPS device detected — using float16 instead of bfloat16")
            torch_dtype = torch.float16

        cache_key = f"{device}_{dtype}"
        if not force_reload and cache_key in _MODEL_CACHE:
            logger.info("Using cached Voxtral TTS pipeline")
            return _MODEL_CACHE[cache_key]

        # Download
        models_dir = get_voxtral_models_dir()
        model_dir = models_dir / "Voxtral-4B-TTS-2603"
        download_model(MODEL_REPO_ID, model_dir)

        # Add src to path for imports
        src_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        from voxtral_tts.config import VoxtralConfig
        from voxtral_tts.backbone import MistralBackbone
        from voxtral_tts.acoustic_transformer import FlowMatchingAcousticTransformer
        from voxtral_tts.codec_decoder import VoxtralCodecDecoder
        from voxtral_tts.embeddings import MultiVocabEmbeddings
        from voxtral_tts.tokenizer import VoxtralTokenizer
        from voxtral_tts.pipeline import VoxtralTTSPipeline

        # Parse config
        config = VoxtralConfig.from_params_json(model_dir / "params.json")

        # Load weights
        logger.info("Loading model weights...")
        raw_weights = load_file(str(model_dir / "consolidated.safetensors"))
        parts = _remap_weights(raw_weights)
        del raw_weights  # Free memory

        # Build backbone
        logger.info("Building LLM backbone...")
        backbone = MistralBackbone(config.transformer)
        backbone.load_state_dict(parts["backbone"], strict=False)
        backbone.to(device=torch_device, dtype=torch_dtype)
        backbone.eval()

        # Build acoustic transformer
        logger.info("Building acoustic transformer...")
        acoustic_tf = FlowMatchingAcousticTransformer(config.acoustic_transformer)
        acoustic_tf.load_state_dict(parts["acoustic_transformer"], strict=False)
        acoustic_tf.to(device=torch_device, dtype=torch_dtype)
        acoustic_tf.eval()

        # Build codec decoder
        logger.info("Building codec decoder...")
        codec = VoxtralCodecDecoder(config.codec_decoder)
        codec.load_state_dict(parts["codec_decoder"], strict=False)
        codec.to(device=torch_device, dtype=torch_dtype)
        codec.eval()

        # Build audio embeddings — size from checkpoint
        logger.info("Building audio embeddings...")
        emb_weight = parts["audio_embeddings"].get("embeddings.weight")
        total_entries = emb_weight.shape[0] if emb_weight is not None else 9088
        audio_emb = MultiVocabEmbeddings.from_config(
            total_entries=total_entries, embedding_dim=config.transformer.dim
        )
        audio_emb.load_state_dict(parts["audio_embeddings"], strict=False)
        audio_emb.to(device=torch_device, dtype=torch_dtype)
        audio_emb.eval()

        del parts  # Free memory

        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = VoxtralTokenizer.from_model_dir(model_dir)

        # Build pipeline
        pipeline = VoxtralTTSPipeline(
            backbone=backbone,
            acoustic_transformer=acoustic_tf,
            codec_decoder=codec,
            audio_embeddings=audio_emb,
            tokenizer=tokenizer,
            config=config,
            voice_embeddings_dir=model_dir / "voice_embedding",
            device=torch_device,
            dtype=torch_dtype,
        )

        _MODEL_CACHE[cache_key] = pipeline
        logger.info(f"Voxtral TTS pipeline ready on {device} ({dtype})")
        return pipeline

    @staticmethod
    def unload():
        """Clear the model cache and free memory."""
        _MODEL_CACHE.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Voxtral TTS pipeline unloaded")
