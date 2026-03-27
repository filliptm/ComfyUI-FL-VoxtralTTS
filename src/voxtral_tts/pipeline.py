"""High-level Voxtral TTS inference pipeline."""

from __future__ import annotations
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Callable

from .config import VoxtralConfig
from .backbone import MistralBackbone
from .acoustic_transformer import FlowMatchingAcousticTransformer
from .codec_decoder import VoxtralCodecDecoder
from .embeddings import MultiVocabEmbeddings
from .tokenizer import VoxtralTokenizer

logger = logging.getLogger(__name__)


class VoxtralTTSPipeline:
    """End-to-end Voxtral TTS: text + voice → 24 kHz audio waveform."""

    def __init__(self, backbone: MistralBackbone,
                 acoustic_transformer: FlowMatchingAcousticTransformer,
                 codec_decoder: VoxtralCodecDecoder,
                 audio_embeddings: MultiVocabEmbeddings,
                 tokenizer: VoxtralTokenizer,
                 config: VoxtralConfig,
                 voice_embeddings_dir: Path,
                 device: torch.device,
                 dtype: torch.dtype):
        self.backbone = backbone
        self.acoustic_transformer = acoustic_transformer
        self.codec_decoder = codec_decoder
        self.audio_embeddings = audio_embeddings
        self.tokenizer = tokenizer
        self.config = config
        self.voice_embeddings_dir = voice_embeddings_dir
        self.device = device
        self.dtype = dtype
        self._voice_cache = {}

    def _load_voice_embedding(self, voice: str) -> torch.Tensor:
        """Load a preset voice embedding from disk.

        Returns:
            [N, dim] voice embedding tensor
        """
        if voice in self._voice_cache:
            return self._voice_cache[voice]

        path = self.voice_embeddings_dir / f"{voice}.pt"
        if not path.exists():
            raise ValueError(f"Voice '{voice}' not found at {path}")

        emb = torch.load(path, map_location=self.device, weights_only=True)
        emb = emb.to(self.dtype)
        self._voice_cache[voice] = emb
        return emb

    @torch.no_grad()
    def generate(self, text: str, voice: str = "casual_male",
                 max_frames: int = 2048, seed: int = -1,
                 progress_callback: Optional[Callable[[int, int], None]] = None
                 ) -> tuple[np.ndarray, int]:
        """Generate speech from text.

        Args:
            text: Text to synthesize
            voice: Preset voice name
            max_frames: Maximum audio frames to generate
            seed: Random seed (-1 for random)
            progress_callback: Optional callback(current_frame, max_frames)

        Returns:
            (audio_np, sample_rate) — mono float32 numpy array at 24 kHz
        """
        # Setup RNG
        generator = None
        if seed >= 0:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

        # Load voice embedding
        voice_emb = self._load_voice_embedding(voice)  # [N, dim]
        n_voice_frames = voice_emb.size(0)

        # Tokenize text and build prompt
        prompt_tokens = self.tokenizer.build_prompt_tokens(text, n_voice_frames)
        token_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)

        # Build input embeddings — replace AUDIO token positions with voice embeddings
        input_embeds = self.backbone.embed_tokens(token_ids)  # [1, L, dim]

        # Positions 1..N (after BOS) are AUDIO placeholders → replace with voice
        audio_start = 1
        audio_end = audio_start + n_voice_frames
        input_embeds[:, audio_start:audio_end, :] = voice_emb.unsqueeze(0)

        # Prefill: run prompt through LLM backbone
        hidden, cache = self.backbone(input_embeds=input_embeds, start_pos=0)

        # The last hidden state is from BEGIN_AUDIO — start generating
        llm_hidden = hidden[:, -1:, :]  # [1, 1, dim]
        pos = input_embeds.size(1)

        # Autoregressive generation loop
        all_codes = []
        for frame_idx in range(max_frames):
            # Get hidden state for current position
            h = llm_hidden[:, -1, :]  # [1, dim]

            # Generate one audio frame (37 codes)
            codes = self.acoustic_transformer.generate_frame(h, generator)

            if codes is None:
                # END_AUDIO predicted
                logger.info(f"Generation stopped at frame {frame_idx} (END_AUDIO)")
                break

            all_codes.append(codes)  # [1, 37]

            if progress_callback is not None:
                progress_callback(frame_idx + 1, max_frames)

            # Embed the 37 codes and feed back to LLM
            frame_embed = self.audio_embeddings(codes)  # [1, dim]
            frame_embed = frame_embed.unsqueeze(1).to(self.dtype)  # [1, 1, dim]

            llm_hidden, cache = self.backbone(
                input_embeds=frame_embed, start_pos=pos, cache=cache
            )
            pos += 1

        if not all_codes:
            logger.warning("No audio frames generated")
            return np.zeros(24000, dtype=np.float32), 24000

        # Stack all codes: [1, T, 37]
        all_codes = torch.stack(all_codes, dim=1)

        # Decode through audio codec
        waveform = self.codec_decoder.decode(all_codes)  # [1, 1, num_samples]

        # Convert to numpy
        audio_np = waveform.squeeze().cpu().float().numpy()
        sample_rate = self.config.codec_decoder.sample_rate

        logger.info(f"Generated {len(audio_np)} samples ({len(audio_np)/sample_rate:.2f}s) "
                     f"from {len(all_codes[0])} frames")

        return audio_np, sample_rate
