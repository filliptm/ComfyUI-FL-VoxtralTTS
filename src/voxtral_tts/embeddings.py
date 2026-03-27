"""MultiVocabEmbeddings — 37-codebook audio token embedding with offset-based lookup."""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import List


class MultiVocabEmbeddings(nn.Module):
    """Embedding table shared across 37 codebooks (1 semantic + 36 acoustic).

    Each codebook has its own offset into a single flat embedding table.
    At each frame, all 37 codebook embeddings are summed into one vector.

    The checkpoint has 9088 entries (padded), but the actual codebook layout
    uses only 9022 entries:
      semantic: 8192 + 2 special = 8194 entries
      acoustic (x36): 21 + 2 special = 23 entries each = 828
      Total used: 8194 + 828 = 9022
    """

    def __init__(self, total_entries: int, embedding_dim: int,
                 n_special: int = 2, semantic_size: int = 8192,
                 acoustic_size: int = 21, n_acoustic: int = 36):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(total_entries, embedding_dim)

        # Compute offsets from ACTUAL codebook sizes (not padded)
        sizes = [semantic_size + n_special]  # 8194
        sizes += [acoustic_size + n_special] * n_acoustic  # 23 each
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
        global_indices = global_indices.clamp(0, self.embeddings.num_embeddings - 1)
        all_embeds = self.embeddings(global_indices)
        return all_embeds.sum(dim=-2)

    @classmethod
    def from_config(cls, total_entries: int = 9088,
                    embedding_dim: int = 3072) -> "MultiVocabEmbeddings":
        """Build with standard Voxtral codebook layout."""
        return cls(total_entries, embedding_dim)
