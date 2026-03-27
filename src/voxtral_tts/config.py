"""Parse Mistral-format params.json into typed dataclasses."""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class TransformerArgs:
    dim: int = 3072
    n_layers: int = 26
    head_dim: int = 128
    hidden_dim: int = 9216
    n_heads: int = 32
    n_kv_heads: int = 8
    vocab_size: int = 131_072
    norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    max_seq_len: int = 65_536
    tied_embeddings: bool = True


@dataclass
class AcousticTransformerArgs:
    input_dim: int = 3072
    dim: int = 3072
    n_layers: int = 3
    head_dim: int = 128
    hidden_dim: int = 9216
    n_heads: int = 32
    n_kv_heads: int = 8
    norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    n_acoustic_codebook: int = 36
    acoustic_codebook_size: int = 21
    semantic_codebook_size: int = 8192
    n_special_tokens: int = 2  # EMPTY_AUDIO=0, END_AUDIO=1
    # Flow matching
    acoustic_decode_iters: int = 8
    cfg_alpha: float = 1.2
    noise_scale: float = 1.0


@dataclass
class CodecTransformerStageArgs:
    n_layers: int = 2
    dim: int = 1024
    hidden_dim: int = 4096
    n_heads: int = 8
    n_kv_heads: int = 8
    head_dim: int = 128
    sliding_window: int = 16
    norm_eps: float = 1e-2


@dataclass
class CodecConvStageArgs:
    stride: int = 2
    kernel_size: int = 4


@dataclass
class CodecDecoderArgs:
    sample_rate: int = 24_000
    patch_size: int = 240
    semantic_codebook_size: int = 8192
    semantic_codebook_dim: int = 256
    acoustic_codebook_size: int = 21
    n_acoustic_codebook: int = 36
    n_special_tokens: int = 2
    input_dim: int = 292  # 256 semantic + 36 acoustic
    hidden_dim: int = 1024
    output_dim: int = 240  # patch_size
    norm_eps: float = 1e-2
    # Decoder has 4 transformer stages and 3 upsample convs
    decoder_strides: List[int] = field(default_factory=lambda: [1, 2, 2, 2])
    n_transformer_stages: int = 4
    n_conv_upsample_stages: int = 3


@dataclass
class VoxtralConfig:
    transformer: TransformerArgs = field(default_factory=TransformerArgs)
    acoustic_transformer: AcousticTransformerArgs = field(default_factory=AcousticTransformerArgs)
    codec_decoder: CodecDecoderArgs = field(default_factory=CodecDecoderArgs)

    # Special token IDs
    bos_id: int = 1
    audio_token_id: int = 24
    begin_audio_token_id: int = 25
    text_to_audio_token_id: int = 35
    audio_to_text_token_id: int = 36
    empty_audio_id: int = 0
    end_audio_id: int = 1

    @classmethod
    def from_params_json(cls, path: Path) -> "VoxtralConfig":
        """Load from Mistral-format params.json."""
        with open(path) as f:
            raw = json.load(f)

        cfg = cls()

        # Transformer args
        for k in ("dim", "n_layers", "head_dim", "hidden_dim", "n_heads",
                   "n_kv_heads", "vocab_size", "norm_eps", "rope_theta"):
            if k in raw:
                setattr(cfg.transformer, k, raw[k])
        if "tied_embeddings" in raw:
            cfg.transformer.tied_embeddings = raw["tied_embeddings"]

        # Acoustic transformer args from multimodal section
        mm = raw.get("multimodal", {})
        audio_model = mm.get("audio_model_args", {})
        at_raw = audio_model.get("acoustic_transformer_args", {})
        for k in ("input_dim", "dim", "n_layers", "head_dim", "hidden_dim",
                   "n_heads", "n_kv_heads", "norm_eps", "rope_theta"):
            if k in at_raw:
                setattr(cfg.acoustic_transformer, k, at_raw[k])

        # Codec args from multimodal section
        at_config = mm.get("audio_tokenizer_args", {})
        if "sampling_rate" in at_config:
            cfg.codec_decoder.sample_rate = at_config["sampling_rate"]
        if "patch_size" in at_config:
            cfg.codec_decoder.patch_size = at_config["patch_size"]
            cfg.codec_decoder.output_dim = at_config["patch_size"]

        # Special token IDs
        if "audio_token_id" in raw:
            cfg.audio_token_id = raw["audio_token_id"]
        if "begin_audio_token_id" in raw:
            cfg.begin_audio_token_id = raw["begin_audio_token_id"]

        return cfg
