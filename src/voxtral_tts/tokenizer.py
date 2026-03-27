"""Tokenizer wrapper for Voxtral TTS using mistral_common's official SpeechRequest API."""

from __future__ import annotations
import torch
from pathlib import Path
from typing import List

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.speech.request import SpeechRequest


class VoxtralTokenizer:
    """Wraps mistral_common tokenizer for Voxtral TTS.

    Uses the official encode_speech_request API to build the correct prompt.
    Prompt format (from the API):
        [BOS=1] [BEGIN_AUDIO=25] [AUDIO=24 x N] [/INST=36] text_tokens [INST=35] [BEGIN_AUDIO=25]
    """

    AUDIO_TOKEN_ID = 24
    BEGIN_AUDIO_ID = 25

    def __init__(self, tokenizer: MistralTokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_model_dir(cls, model_dir: Path) -> "VoxtralTokenizer":
        tekken_path = model_dir / "tekken.json"
        tokenizer = MistralTokenizer.from_file(str(tekken_path))
        return cls(tokenizer)

    def build_prompt_tokens(self, text: str, voice: str) -> List[int]:
        """Build TTS prompt using the official SpeechRequest API.

        Args:
            text: Text to synthesize
            voice: Voice name (used for token count from voice embedding)

        Returns:
            List of token IDs
        """
        request = SpeechRequest(input=text, voice=voice)
        result = self.tokenizer.encode_speech_request(request)
        return result.tokens

    def find_audio_token_positions(self, tokens: List[int]) -> tuple[int, int]:
        """Find the start and end positions of AUDIO placeholder tokens.

        Returns:
            (start_idx, end_idx) — AUDIO tokens span tokens[start_idx:end_idx]
        """
        start = None
        end = None
        for i, t in enumerate(tokens):
            if t == self.AUDIO_TOKEN_ID:
                if start is None:
                    start = i
                end = i + 1
        return (start or 0, end or 0)
