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

    def __init__(self, total_entries: int, embedding_dim: int,
                 codebook_sizes: List[int] = None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(total_entries, embedding_dim)

        if codebook_sizes is not None:
            offsets = [0]
            for s in codebook_sizes:
                offsets.append(offsets[-1] + s)
            self.register_buffer("offsets", torch.tensor(offsets[:-1], dtype=torch.long))
            self.n_codebooks = len(codebook_sizes)
        else:
            # Default Voxtral layout: offsets will be set from config
            # semantic(8224) + 36 * acoustic(24) = 9088
            semantic_size = 8224
            acoustic_size = 24
            n_acoustic = 36
            sizes = [semantic_size] + [acoustic_size] * n_acoustic
            offsets = [0]
            for s in sizes:
                offsets.append(offsets[-1] + s)
            self.register_buffer("offsets", torch.tensor(offsets[:-1], dtype=torch.long))
            self.n_codebooks = len(sizes)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """Embed and sum across codebooks.

        Args:
            codes: [B, n_codebooks] or [B, T, n_codebooks] integer codes

        Returns:
            [B, embedding_dim] or [B, T, embedding_dim] summed embeddings
        """
        global_indices = codes + self.offsets.to(codes.device)
        # Clamp to avoid OOB
        global_indices = global_indices.clamp(0, self.embeddings.num_embeddings - 1)
        all_embeds = self.embeddings(global_indices)
        return all_embeds.sum(dim=-2)

    @classmethod
    def from_config(cls, total_entries: int = 9088,
                    embedding_dim: int = 3072) -> "MultiVocabEmbeddings":
        """Build with standard Voxtral codebook layout."""
        return cls(total_entries, embedding_dim)
