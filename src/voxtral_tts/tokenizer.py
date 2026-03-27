"""Tokenizer wrapper for Voxtral TTS using mistral_common's Tekken BPE."""

from __future__ import annotations
import torch
from pathlib import Path
from typing import List

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest


class VoxtralTokenizer:
    """Wraps mistral_common tokenizer for Voxtral TTS prompt construction.

    Prompt format:
        [BOS=1] [AUDIO=24 x N] [/INST=36] text_tokens [INST=35] [BEGIN_AUDIO=25]

    Where N is the number of voice embedding frames.
    """

    BOS_ID = 1
    AUDIO_TOKEN_ID = 24
    BEGIN_AUDIO_ID = 25
    INST_ID = 35       # [INST] = text_to_audio delimiter
    END_INST_ID = 36   # [/INST] = audio_to_text delimiter

    def __init__(self, tokenizer: MistralTokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_model_dir(cls, model_dir: Path) -> "VoxtralTokenizer":
        """Load Tekken tokenizer from the model directory."""
        tekken_path = model_dir / "tekken.json"
        tokenizer = MistralTokenizer.from_file(str(tekken_path))
        return cls(tokenizer)

    def encode_text(self, text: str) -> List[int]:
        """Encode text to token IDs using the Tekken BPE tokenizer."""
        # Use the raw tokenizer encode (without chat template)
        encoded = self.tokenizer.instruct_tokenizer.tokenizer.encode(text, bos=False, eos=False)
        return encoded

    def build_prompt_tokens(self, text: str, n_voice_frames: int) -> List[int]:
        """Build the full TTS prompt token sequence.

        Args:
            text: Text to synthesize
            n_voice_frames: Number of voice embedding frames

        Returns:
            List of token IDs for the prompt
        """
        text_tokens = self.encode_text(text)

        # Build prompt:
        # [BOS] [AUDIO x N] [/INST] text_tokens [INST] [BEGIN_AUDIO]
        prompt = [self.BOS_ID]
        prompt += [self.AUDIO_TOKEN_ID] * n_voice_frames
        prompt += [self.END_INST_ID]
        prompt += text_tokens
        prompt += [self.INST_ID]
        prompt += [self.BEGIN_AUDIO_ID]

        return prompt
