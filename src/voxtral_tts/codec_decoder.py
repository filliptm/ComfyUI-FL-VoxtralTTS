"""Voxtral audio codec decoder.

Converts quantized audio codes (1 semantic + 36 acoustic per frame) into
24 kHz mono waveform. Architecture: 4-stage decoder with CausalConv1d,
ALiBi-attended transformer blocks, and transposed convolution upsampling.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

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


# ─── Causal Convolutions (with weight norm) ─────────────────────────────────


class CausalConv1d(nn.Module):
    """1D convolution with left-padding for causal behavior."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * 1  # dilation=1
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class CausalConvTranspose1d(nn.Module):
    """1D transposed convolution for upsampling with causal trimming."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv = nn.utils.weight_norm(
            nn.ConvTranspose1d(in_ch, out_ch, kernel_size, stride=stride)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        # Trim right to maintain causal alignment
        trim = self.kernel_size - self.stride
        if trim > 0:
            y = y[:, :, :-trim]
        return y


# ─── ALiBi Attention ─────────────────────────────────────────────────────────


def _compute_alibi_slopes(n_heads: int) -> torch.Tensor:
    """Compute ALiBi attention slopes as a geometric sequence."""
    closest_power = 2 ** math.floor(math.log2(n_heads))
    base = 2.0 ** (-(2.0 ** -(math.log2(closest_power) - 3)))
    slopes = torch.pow(base, torch.arange(1, closest_power + 1, dtype=torch.float32))
    if closest_power < n_heads:
        extra_base = 2.0 ** (-(2.0 ** -(math.log2(2 * closest_power) - 3)))
        extra = torch.pow(extra_base, torch.arange(1, 2 * (n_heads - closest_power) + 1, 2, dtype=torch.float32))
        slopes = torch.cat([slopes, extra])
    return slopes[:n_heads]


class ALiBiAttention(nn.Module):
    """Multi-head attention with ALiBi positional bias and optional sliding window."""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 sliding_window: int = 16, norm_eps: float = 1e-2, qk_norm: bool = True):
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

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = CodecRMSNorm(head_dim, norm_eps)
            self.k_norm = CodecRMSNorm(head_dim, norm_eps)

        # ALiBi slopes
        slopes = _compute_alibi_slopes(n_heads)
        self.register_buffer("alibi_slopes", slopes)

    def _build_alibi_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build causal ALiBi attention bias with sliding window."""
        pos = torch.arange(seq_len, device=device)
        rel_pos = pos.unsqueeze(0) - pos.unsqueeze(1)  # [T, T]
        # Causal mask: future positions get -inf
        causal_mask = rel_pos > 0
        # Sliding window: positions beyond window get -inf
        window_mask = rel_pos < -self.sliding_window

        bias = self.alibi_slopes.view(-1, 1, 1) * rel_pos.unsqueeze(0).float()  # [heads, T, T]
        bias = bias.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
        bias = bias.masked_fill(window_mask.unsqueeze(0), float("-inf"))
        return bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(B, L, self.n_rep, self.n_kv_heads, self.head_dim)
            k = k.reshape(B, L, self.n_heads, self.head_dim)
            v = v.unsqueeze(2).expand(B, L, self.n_rep, self.n_kv_heads, self.head_dim)
            v = v.reshape(B, L, self.n_heads, self.head_dim)

        q = q.transpose(1, 2)  # [B, heads, L, dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        alibi = self._build_alibi_bias(L, x.device).unsqueeze(0)  # [1, heads, L, L]
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale + alibi
        attn = F.softmax(attn, dim=-1)
        out = attn @ v

        return self.wo(out.transpose(1, 2).contiguous().view(B, L, -1))


# ─── Codec Transformer Block ────────────────────────────────────────────────


class CodecFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CodecTransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 hidden_dim: int, sliding_window: int, norm_eps: float,
                 layer_scale_init: float = 1.0):
        super().__init__()
        self.attention = ALiBiAttention(dim, n_heads, n_kv_heads, head_dim,
                                         sliding_window, norm_eps)
        self.feed_forward = CodecFeedForward(dim, hidden_dim)
        self.attention_norm = CodecRMSNorm(dim, norm_eps)
        self.ffn_norm = CodecRMSNorm(dim, norm_eps)
        # Layer scale
        self.attn_scale = nn.Parameter(torch.full((dim,), layer_scale_init))
        self.ffn_scale = nn.Parameter(torch.full((dim,), layer_scale_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_scale * self.attention(self.attention_norm(x))
        x = x + self.ffn_scale * self.feed_forward(self.ffn_norm(x))
        return x


# ─── Semantic Codebook (VQ) ─────────────────────────────────────────────────


class SemanticCodebook(nn.Module):
    """Vector-quantized semantic codebook for dequantization (decode only)."""

    def __init__(self, n_entries: int = 8192, dim: int = 256):
        super().__init__()
        # Stored as embedding_sum and cluster_usage (from VQ training)
        self.embedding_sum = nn.Parameter(torch.randn(n_entries, dim))
        self.cluster_usage = nn.Parameter(torch.ones(n_entries))

    def dequantize(self, codes: torch.Tensor) -> torch.Tensor:
        """Look up semantic embeddings for given code indices.

        Args:
            codes: [B, T] integer codes in range [0, n_entries)

        Returns:
            [B, T, dim] dequantized embeddings
        """
        # Normalize: embedding = embedding_sum / cluster_usage
        embeddings = self.embedding_sum / self.cluster_usage.unsqueeze(-1).clamp(min=1e-8)
        return F.embedding(codes, embeddings)


# ─── Full Codec Decoder ─────────────────────────────────────────────────────


class VoxtralCodecDecoder(nn.Module):
    """Decodes quantized audio codes into 24 kHz waveform.

    Pipeline:
    1. Dequantize semantic codes (VQ lookup) + acoustic codes (FSQ rescale)
    2. Concatenate → [292, T]
    3. Input conv → [1024, T]
    4. 4 stages of [Transformer(2 layers) + ConvTranspose1d upsample]
    5. Output conv → [240, T']
    6. Reshape → waveform [1, T' * 240]
    """

    def __init__(self, args: CodecDecoderArgs):
        super().__init__()
        self.args = args
        hidden = args.hidden_dim  # 1024

        # Semantic codebook for dequantization
        self.semantic_codebook = SemanticCodebook(args.semantic_codebook_size,
                                                   args.semantic_codebook_dim)

        # Input projection: 292 → 1024
        self.input_conv = CausalConv1d(args.input_dim, hidden, kernel_size=3)

        # 4 transformer stages (2 layers each) with increasing window sizes
        windows = [2, 4, 8, 16]
        self.transformer_stages = nn.ModuleList()
        for i in range(args.n_transformer_stages):
            stage = nn.ModuleList([
                CodecTransformerBlock(
                    dim=hidden, n_heads=8, n_kv_heads=8, head_dim=128,
                    hidden_dim=4096, sliding_window=windows[i],
                    norm_eps=args.norm_eps,
                    layer_scale_init=0.1 / math.sqrt(2 * (i * 2 + j + 1))
                ) for j in range(2)
            ])
            self.transformer_stages.append(stage)

        # 3 upsample convolutions (strides [2, 2, 2])
        self.upsample_convs = nn.ModuleList()
        for stride in args.decoder_strides[1:]:  # skip first stride=1
            self.upsample_convs.append(
                CausalConvTranspose1d(hidden, hidden, kernel_size=stride * 2, stride=stride)
            )

        # Output projection: 1024 → 240 (patch_size)
        self.output_conv = CausalConv1d(hidden, args.output_dim, kernel_size=7)

    def _dequantize_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Dequantize full frame codes into continuous features.

        Args:
            codes: [B, T, 37] — first column is semantic, rest are acoustic

        Returns:
            [B, T, 292] — concatenated semantic (256) + acoustic (36) features
        """
        n_special = self.args.n_special_tokens

        # Semantic: VQ lookup (remove special token offset)
        semantic_codes = codes[:, :, 0] - n_special
        semantic_codes = semantic_codes.clamp(0, self.args.semantic_codebook_size - 1)
        semantic_features = self.semantic_codebook.dequantize(semantic_codes)  # [B, T, 256]

        # Acoustic: FSQ decode (remove offset, rescale to [-1, 1])
        acoustic_codes = codes[:, :, 1:].float() - n_special  # [B, T, 36]
        acoustic_features = (acoustic_codes / (self.args.acoustic_codebook_size - 1)) * 2.0 - 1.0

        return torch.cat([semantic_features, acoustic_features], dim=-1)  # [B, T, 292]

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode audio codes to waveform.

        Args:
            codes: [B, T, 37] quantized codes per frame

        Returns:
            [B, 1, num_samples] audio waveform at 24 kHz
        """
        # Dequantize
        features = self._dequantize_codes(codes)  # [B, T, 292]

        # Transpose for conv1d: [B, 292, T]
        x = features.transpose(1, 2)

        # Input conv
        x = self.input_conv(x)  # [B, 1024, T]

        # Decoder stages: transformer + upsample
        for i, (tf_stage, conv) in enumerate(zip(
                self.transformer_stages[:3], self.upsample_convs)):
            # Transformer: [B, C, T] -> [B, T, C] -> transformer -> [B, C, T]
            xt = x.transpose(1, 2)
            for block in tf_stage:
                xt = block(xt)
            x = xt.transpose(1, 2)
            # Upsample
            x = conv(x)

        # Final transformer stage (no upsample after)
        xt = x.transpose(1, 2)
        for block in self.transformer_stages[3]:
            xt = block(xt)
        x = xt.transpose(1, 2)

        # Output projection
        x = self.output_conv(x)  # [B, 240, T']

        # Reshape patches to waveform: [B, 240, T'] → [B, 1, T' * 240]
        B, P, T = x.shape
        waveform = x.transpose(1, 2).reshape(B, 1, T * P)
        return waveform
