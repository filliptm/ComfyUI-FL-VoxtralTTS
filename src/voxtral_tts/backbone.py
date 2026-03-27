"""Mistral LLM backbone for Voxtral TTS.

Standard Mistral architecture (GQA, RoPE, SiLU MLP) with the key difference
that during audio generation we expose hidden states rather than projecting
to vocabulary logits.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

from .config import TransformerArgs


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 1_000_000.0,
                         device: torch.device = None) -> torch.Tensor:
    """Precompute RoPE complex frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(end, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor,
                     freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    # Reshape to pairs of 2 for complex multiplication
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs = freqs_cis.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim//2]
    xq_out = torch.view_as_real(xq_ * freqs).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs).flatten(-2)
    return xq_out.to(xq.dtype), xk_out.to(xk.dtype)


class Attention(nn.Module):
    def __init__(self, args: TransformerArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.head_dim
        self.n_rep = args.n_heads // args.n_kv_heads

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, L, _ = x.shape

        q = self.wq(x).view(B, L, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        # KV cache for autoregressive generation
        if cache is not None:
            k_cache, v_cache = cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
        new_cache = (k, v)

        # GQA: repeat KV heads
        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(B, k.size(1), self.n_rep, self.n_kv_heads, self.head_dim)
            k = k.reshape(B, k.size(1), self.n_heads, self.head_dim)
            v = v.unsqueeze(2).expand(B, v.size(1), self.n_rep, self.n_kv_heads, self.head_dim)
            v = v.reshape(B, v.size(1), self.n_heads, self.head_dim)

        # Transpose to [B, heads, seq, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=(mask is None and L > 1))
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.wo(out), new_cache


class FeedForward(nn.Module):
    def __init__(self, args: TransformerArgs):
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)  # down
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)  # up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: TransformerArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        h, new_cache = self.attention(self.attention_norm(x), freqs_cis, mask, cache)
        x = x + h
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, new_cache


class MistralBackbone(nn.Module):
    """Mistral LLM backbone that exposes hidden states for TTS audio generation."""

    def __init__(self, args: TransformerArgs):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, args.norm_eps)

        # Output projection (tied with tok_embeddings if configured)
        if args.tied_embeddings:
            self.output = None  # Use tok_embeddings.weight
        else:
            self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Precompute RoPE frequencies
        self._freqs_cis: Optional[torch.Tensor] = None

    def _get_freqs_cis(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self._freqs_cis is None or self._freqs_cis.size(0) < seq_len:
            self._freqs_cis = precompute_freqs_cis(
                self.args.head_dim, max(seq_len, 4096),
                theta=self.args.rope_theta, device=device
            )
        return self._freqs_cis[:seq_len].to(device)

    def forward(self, tokens: Optional[torch.Tensor] = None,
                input_embeds: Optional[torch.Tensor] = None,
                start_pos: int = 0,
                cache: Optional[list] = None
                ) -> Tuple[torch.Tensor, list]:
        """Forward pass returning hidden states.

        Args:
            tokens: [B, L] token IDs (used for text tokens)
            input_embeds: [B, L, dim] pre-computed embeddings (used for audio tokens)
            start_pos: position offset for RoPE
            cache: list of (k, v) tuples per layer

        Returns:
            hidden_states: [B, L, dim] — NOT projected to vocab logits
            new_cache: updated KV cache
        """
        if input_embeds is not None:
            h = input_embeds
        else:
            h = self.tok_embeddings(tokens)

        B, L, _ = h.shape
        freqs_cis = self._get_freqs_cis(start_pos + L, h.device)[start_pos:start_pos + L]

        new_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            h, c = layer(h, freqs_cis, cache=layer_cache)
            new_cache.append(c)

        h = self.norm(h)
        return h, new_cache

    def get_text_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits (for text-mode tokens)."""
        if self.output is not None:
            return self.output(hidden)
        return F.linear(hidden, self.tok_embeddings.weight)

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Get token embeddings without running through layers."""
        return self.tok_embeddings(tokens)
