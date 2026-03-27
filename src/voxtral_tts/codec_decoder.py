"""Voxtral audio codec decoder.

Converts quantized audio codes (1 semantic + 36 acoustic per frame) into
24 kHz mono waveform. Matches the checkpoint structure exactly:
  decoder_blocks[0] = input conv
  decoder_blocks[1] = transformer (2 layers)
  decoder_blocks[2] = upsample conv (stride 2)
  decoder_blocks[3] = transformer (2 layers)
  decoder_blocks[4] = upsample conv (stride 2)
  decoder_blocks[5] = transformer (2 layers)
  decoder_blocks[6] = upsample conv (stride 2)
  decoder_blocks[7] = transformer (2 layers)
  output_proj = final conv
  quantizer.semantic_codebook = VQ dequantization
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from typing import List

from .config import CodecDecoderArgs


# ─── Normalization ──────────────────────────────────────────────────────────


class CodecRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-2):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


# ─── Causal Convolutions (with parametrized weight_norm) ────────────────────


class CausalConv1d(nn.Module):
    """1D convolution with left-padding for causal behavior and weight_norm."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class CausalConvTranspose1d(nn.Module):
    """1D transposed convolution for upsampling with weight_norm."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv = weight_norm(
            nn.ConvTranspose1d(in_ch, out_ch, kernel_size, stride=stride, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        trim = self.kernel_size - self.stride
        if trim > 0:
            y = y[:, :, :-trim]
        return y


# ─── ALiBi Attention ─────────────────────────────────────────────────────────


def _compute_alibi_slopes(n_heads: int) -> torch.Tensor:
    closest_power = 2 ** math.floor(math.log2(n_heads))
    base = 2.0 ** (-(2.0 ** -(math.log2(closest_power) - 3)))
    slopes = torch.pow(base, torch.arange(1, closest_power + 1, dtype=torch.float32))
    if closest_power < n_heads:
        extra_base = 2.0 ** (-(2.0 ** -(math.log2(2 * closest_power) - 3)))
        extra = torch.pow(extra_base, torch.arange(1, 2 * (n_heads - closest_power) + 1, 2, dtype=torch.float32))
        slopes = torch.cat([slopes, extra])
    return slopes[:n_heads]


class ALiBiAttention(nn.Module):
    """Multi-head attention with ALiBi bias and sliding window.

    Weight names match checkpoint: wq, wk, wv, wo, q_norm, k_norm
    """

    def __init__(self, dim: int = 1024, n_heads: int = 8, n_kv_heads: int = 8,
                 head_dim: int = 128, sliding_window: int = 16, norm_eps: float = 1e-2):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads
        self.sliding_window = sliding_window

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

        # QK norm applied to full projection (dim-sized, not head_dim-sized)
        self.q_norm = CodecRMSNorm(dim, norm_eps)
        self.k_norm = CodecRMSNorm(dim, norm_eps)

        slopes = _compute_alibi_slopes(n_heads)
        self.register_buffer("alibi_slopes", slopes)

    def _build_alibi_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(seq_len, device=device)
        rel_pos = pos.unsqueeze(0) - pos.unsqueeze(1)
        causal_mask = rel_pos > 0
        window_mask = rel_pos < -self.sliding_window
        bias = self.alibi_slopes.view(-1, 1, 1) * rel_pos.unsqueeze(0).float()
        bias = bias.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
        bias = bias.masked_fill(window_mask.unsqueeze(0), float("-inf"))
        return bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        # Apply QK norm BEFORE splitting into heads
        q = self.q_norm(self.wq(x)).view(B, L, self.n_heads, self.head_dim)
        k = self.k_norm(self.wk(x)).view(B, L, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim)

        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(B, L, self.n_rep, self.n_kv_heads, self.head_dim)
            k = k.reshape(B, L, self.n_heads, self.head_dim)
            v = v.unsqueeze(2).expand(B, L, self.n_rep, self.n_kv_heads, self.head_dim)
            v = v.reshape(B, L, self.n_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        alibi = self._build_alibi_bias(L, x.device).unsqueeze(0)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale + alibi
        attn = F.softmax(attn.float(), dim=-1).to(x.dtype)
        out = attn @ v
        return self.wo(out.transpose(1, 2).contiguous().view(B, L, -1))


# ─── Codec Transformer Block ────────────────────────────────────────────────


class CodecFeedForward(nn.Module):
    def __init__(self, dim: int = 1024, hidden_dim: int = 4096):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CodecTransformerBlock(nn.Module):
    """Single transformer block matching checkpoint naming:
    attention.{wq,wk,wv,wo,q_norm,k_norm}, attention_norm, attention_scale,
    feed_forward.{w1,w2,w3}, ffn_norm, ffn_scale
    """

    def __init__(self, dim: int = 1024, sliding_window: int = 16,
                 norm_eps: float = 1e-2, layer_scale_init: float = 1.0):
        super().__init__()
        self.attention = ALiBiAttention(dim=dim, sliding_window=sliding_window,
                                         norm_eps=norm_eps)
        self.feed_forward = CodecFeedForward(dim=dim)
        self.attention_norm = CodecRMSNorm(dim, norm_eps)
        self.ffn_norm = CodecRMSNorm(dim, norm_eps)
        # Layer scale as plain parameters (matches checkpoint naming)
        self.attention_scale = nn.Parameter(torch.full((dim,), layer_scale_init))
        self.ffn_scale = nn.Parameter(torch.full((dim,), layer_scale_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention_scale * self.attention(self.attention_norm(x))
        x = x + self.ffn_scale * self.feed_forward(self.ffn_norm(x))
        return x


# ─── Transformer Stage (2 blocks) ───────────────────────────────────────────


class CodecTransformerStage(nn.Module):
    """A stage with 2 transformer blocks, stored as `layers.0` and `layers.1`."""

    def __init__(self, dim: int = 1024, sliding_window: int = 16,
                 norm_eps: float = 1e-2, base_layer_idx: int = 0):
        super().__init__()
        self.layers = nn.ModuleList([
            CodecTransformerBlock(dim=dim, sliding_window=sliding_window,
                                  norm_eps=norm_eps,
                                  layer_scale_init=0.1 / math.sqrt(2 * (base_layer_idx + j + 1)))
            for j in range(2)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ─── Semantic Codebook (VQ) ─────────────────────────────────────────────────


class SemanticCodebook(nn.Module):
    """Vector-quantized semantic codebook for dequantization."""

    def __init__(self, n_entries: int = 8192, dim: int = 256):
        super().__init__()
        self.embedding_sum = nn.Parameter(torch.randn(n_entries, dim))
        self.cluster_usage = nn.Parameter(torch.ones(n_entries))

    def dequantize(self, codes: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding_sum / self.cluster_usage.unsqueeze(-1).clamp(min=1e-8)
        return F.embedding(codes, embeddings)


class Quantizer(nn.Module):
    """Wrapper matching checkpoint path: quantizer.semantic_codebook.*"""

    def __init__(self, n_entries: int = 8192, dim: int = 256):
        super().__init__()
        self.semantic_codebook = SemanticCodebook(n_entries, dim)


# ─── Full Codec Decoder ─────────────────────────────────────────────────────


class VoxtralCodecDecoder(nn.Module):
    """Decodes quantized audio codes into 24 kHz waveform.

    Structure matches checkpoint exactly:
      decoder_blocks[0] = CausalConv1d(292, 1024, k=3)
      decoder_blocks[1] = TransformerStage(window=2)
      decoder_blocks[2] = CausalConvTranspose1d(1024, 1024, k=4, s=2)
      decoder_blocks[3] = TransformerStage(window=4)
      decoder_blocks[4] = CausalConvTranspose1d(1024, 1024, k=4, s=2)
      decoder_blocks[5] = TransformerStage(window=8)
      decoder_blocks[6] = CausalConvTranspose1d(1024, 1024, k=4, s=2)
      decoder_blocks[7] = TransformerStage(window=16)
      output_proj = CausalConv1d(1024, 240, k=7)
      quantizer.semantic_codebook = VQ
    """

    def __init__(self, args: CodecDecoderArgs):
        super().__init__()
        self.args = args
        H = args.hidden_dim  # 1024
        eps = args.norm_eps

        # Quantizer for semantic dequantization
        self.quantizer = Quantizer(args.semantic_codebook_size, args.semantic_codebook_dim)

        # Flat decoder_blocks matching checkpoint indices
        windows = [2, 4, 8, 16]
        self.decoder_blocks = nn.ModuleList([
            # 0: input conv
            CausalConv1d(args.input_dim, H, kernel_size=3),
            # 1: transformer stage (window=2)
            CodecTransformerStage(H, windows[0], eps, base_layer_idx=0),
            # 2: upsample conv (stride=2)
            CausalConvTranspose1d(H, H, kernel_size=4, stride=2),
            # 3: transformer stage (window=4)
            CodecTransformerStage(H, windows[1], eps, base_layer_idx=2),
            # 4: upsample conv (stride=2)
            CausalConvTranspose1d(H, H, kernel_size=4, stride=2),
            # 5: transformer stage (window=8)
            CodecTransformerStage(H, windows[2], eps, base_layer_idx=4),
            # 6: upsample conv (stride=2)
            CausalConvTranspose1d(H, H, kernel_size=4, stride=2),
            # 7: transformer stage (window=16)
            CodecTransformerStage(H, windows[3], eps, base_layer_idx=6),
        ])

        # Output projection
        self.output_proj = CausalConv1d(H, args.output_dim, kernel_size=7)

    def _dequantize_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Dequantize [B, T, 37] codes → [B, T, 292] features."""
        n_special = self.args.n_special_tokens

        # Semantic: VQ lookup (remove special token offset)
        semantic_codes = (codes[:, :, 0] - n_special).clamp(0, self.args.semantic_codebook_size - 1)
        semantic_features = self.quantizer.semantic_codebook.dequantize(semantic_codes)

        # Acoustic: FSQ decode
        acoustic_codes = codes[:, :, 1:].float() - n_special
        acoustic_features = (acoustic_codes / (self.args.acoustic_codebook_size - 1)) * 2.0 - 1.0

        return torch.cat([semantic_features, acoustic_features], dim=-1)

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode audio codes to waveform.

        Args:
            codes: [B, T, 37] quantized codes per frame

        Returns:
            [B, 1, num_samples] audio waveform at 24 kHz
        """
        features = self._dequantize_codes(codes)  # [B, T, 292]
        # Cast to model dtype (dequantize returns float32 from VQ lookup)
        target_dtype = self.output_proj.conv.parametrizations.weight.original1.dtype
        features = features.to(target_dtype)
        x = features.transpose(1, 2)  # [B, 292, T]

        # Run through decoder blocks
        for i, block in enumerate(self.decoder_blocks):
            if isinstance(block, CodecTransformerStage):
                # Transformer operates on [B, T, C]
                x = block(x.transpose(1, 2)).transpose(1, 2)
            else:
                # Conv operates on [B, C, T]
                x = block(x)

        # Output projection
        x = self.output_proj(x)  # [B, 240, T']

        # Reshape patches to waveform
        B, P, T = x.shape
        return x.transpose(1, 2).reshape(B, 1, T * P)
