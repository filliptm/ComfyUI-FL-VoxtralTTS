"""MultiVocabEmbeddings — 37-codebook audio token embedding with offset-based lookup."""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import List


class MultiVocabEmbeddings(nn.Module):
    """Embedding table shared across 37 codebooks (1 semantic + 36 acoustic).

    Each codebook has its own offset into a single flat embedding table.
    At each frame, all 37 codebook embeddings are summed into one vector.
    """

    def __init__(self, codebook_sizes: List[int], embedding_dim: int):
        super().__init__()
        self.codebook_sizes = codebook_sizes
        self.embedding_dim = embedding_dim
        self.n_codebooks = len(codebook_sizes)

        # Compute offsets: cumulative sum of sizes
        offsets = [0]
        for s in codebook_sizes:
            offsets.append(offsets[-1] + s)
        self.register_buffer("offsets", torch.tensor(offsets[:-1], dtype=torch.long))
        total_entries = offsets[-1]

        self.embeddings = nn.Embedding(total_entries, embedding_dim)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """Embed and sum across codebooks.

        Args:
            codes: [B, n_codebooks] or [B, T, n_codebooks] integer codes

        Returns:
            [B, embedding_dim] or [B, T, embedding_dim] summed embeddings
        """
        # Add codebook offsets to get global indices
        global_indices = codes + self.offsets.to(codes.device)
        # Look up each codebook embedding
        all_embeds = self.embeddings(global_indices)  # [..., n_codebooks, dim]
        # Sum across codebooks
        return all_embeds.sum(dim=-2)

    @classmethod
    def from_config(cls, n_special: int = 2, semantic_size: int = 8192,
                    acoustic_size: int = 21, n_acoustic: int = 36,
                    embedding_dim: int = 3072) -> "MultiVocabEmbeddings":
        """Build with standard Voxtral codebook layout."""
        codebook_sizes = [semantic_size + n_special]  # semantic: 8194
        codebook_sizes += [acoustic_size + n_special] * n_acoustic  # acoustic: 23 each
        return cls(codebook_sizes, embedding_dim)
