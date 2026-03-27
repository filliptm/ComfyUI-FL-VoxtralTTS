"""Flow-matching acoustic transformer for Voxtral TTS.

Takes LLM hidden states and produces 37 audio codes per frame:
- 1 semantic code (greedy argmax over 8192-entry codebook)
- 36 acoustic codes (8-step Euler ODE with classifier-free guidance)
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import AcousticTransformerArgs


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class BidirectionalAttention(nn.Module):
    """Standard multi-head attention WITHOUT causal mask or RoPE."""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.n_rep > 1:
            k = k[:, :, None, :, :].expand(B, self.n_kv_heads, self.n_rep, L, self.head_dim)
            k = k.reshape(B, self.n_heads, L, self.head_dim)
            v = v[:, :, None, :, :].expand(B, self.n_kv_heads, self.n_rep, L, self.head_dim)
            v = v.reshape(B, self.n_heads, L, self.head_dim)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.wo(out.transpose(1, 2).contiguous().view(B, L, -1))


class BidirectionalFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class BidirectionalTransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, hidden_dim: int, norm_eps: float):
        super().__init__()
        self.attention = BidirectionalAttention(dim, n_heads, n_kv_heads, head_dim)
        self.feed_forward = BidirectionalFeedForward(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim, norm_eps)
        self.ffn_norm = RMSNorm(dim, norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


def sinusoidal_time_embedding(t: torch.Tensor, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Sinusoidal time embedding for flow matching timestep conditioning."""
    half_dim = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(half_dim, device=t.device).float() / half_dim))
    args = t.unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([args.cos(), args.sin()], dim=-1)


class FlowMatchingAcousticTransformer(nn.Module):
    """Generates audio codes from LLM hidden states via flow matching.

    Architecture: 3 bidirectional transformer layers operating on a 3-token sequence:
    [acoustic_input, time_embedding, llm_hidden]

    Semantic head: greedy argmax over 8192 codebook entries
    Acoustic head: 8-step Euler ODE with CFG (alpha=1.2) producing 36 FSQ codes
    """

    def __init__(self, args: AcousticTransformerArgs):
        super().__init__()
        self.args = args
        dim = args.dim
        n_acoustic = args.n_acoustic_codebook

        # Padded semantic vocab for alignment: ceil((8192+2)/128)*128 = 8320
        padded_semantic = ((args.semantic_codebook_size + args.n_special_tokens + 127) // 128) * 128
        self.padded_semantic_size = padded_semantic

        # Output heads
        self.semantic_codebook_output = nn.Linear(dim, padded_semantic, bias=False)
        self.acoustic_codebook_output = nn.Linear(dim, n_acoustic, bias=False)

        # Input projections for the 3-token sequence
        self.input_projection = nn.Linear(n_acoustic, dim, bias=False)
        self.time_projection = nn.Linear(dim, dim, bias=False)
        self.llm_projection = nn.Linear(args.input_dim, dim, bias=False)

        # Bidirectional transformer layers
        self.layers = nn.ModuleList([
            BidirectionalTransformerBlock(
                dim=dim, n_heads=args.n_heads, n_kv_heads=args.n_kv_heads,
                head_dim=args.head_dim, hidden_dim=args.hidden_dim,
                norm_eps=args.norm_eps
            ) for _ in range(args.n_layers)
        ])
        self.norm = RMSNorm(dim, args.norm_eps)

    def predict_semantic(self, llm_hidden: torch.Tensor) -> torch.Tensor:
        """Predict semantic codebook code from LLM hidden state.

        Args:
            llm_hidden: [B, dim] hidden state from LLM backbone

        Returns:
            [B] semantic code IDs (including special tokens offset)
        """
        logits = self.semantic_codebook_output(llm_hidden).float()
        n_special = self.args.n_special_tokens
        n_semantic = self.args.semantic_codebook_size
        # Mask invalid positions
        logits[:, 0] = float("-inf")  # EMPTY_AUDIO=0
        logits[:, (n_special + n_semantic):] = float("-inf")  # padding
        return logits.argmax(dim=-1)

    def _predict_velocity(self, x_t: torch.Tensor, llm_hidden: torch.Tensor,
                          t: torch.Tensor) -> torch.Tensor:
        """Single velocity prediction for the Euler ODE.

        Args:
            x_t: [B, 36] current acoustic state
            llm_hidden: [B, dim] LLM hidden state (zeros for uncond)
            t: [B] timestep values in [0, 1]

        Returns:
            [B, 36] predicted velocity
        """
        B = x_t.size(0)

        # Build 3-token sequence
        tok_acoustic = self.input_projection(x_t).unsqueeze(1)  # [B, 1, dim]
        tok_time = self.time_projection(
            sinusoidal_time_embedding(t, self.args.dim)
        ).unsqueeze(1)  # [B, 1, dim]
        tok_llm = self.llm_projection(llm_hidden).unsqueeze(1)  # [B, 1, dim]

        h = torch.cat([tok_acoustic, tok_time, tok_llm], dim=1)  # [B, 3, dim]

        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        # Velocity from the acoustic token position (index 0)
        return self.acoustic_codebook_output(h[:, 0, :])

    @torch.no_grad()
    def decode_acoustic(self, llm_hidden: torch.Tensor,
                        generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Generate 36 acoustic codes via 8-step Euler flow matching with CFG.

        Args:
            llm_hidden: [B, dim] LLM hidden state

        Returns:
            [B, 36] integer acoustic codes (with special token offset applied)
        """
        B = llm_hidden.size(0)
        device = llm_hidden.device
        dtype = llm_hidden.dtype
        n_steps = self.args.acoustic_decode_iters
        alpha = self.args.cfg_alpha

        # Start from noise
        x = torch.randn(B, self.args.n_acoustic_codebook, device=device, dtype=dtype,
                         generator=generator) * self.args.noise_scale
        timesteps = torch.linspace(0, 1, n_steps, device=device, dtype=dtype)

        # Zero LLM hidden for unconditional branch
        llm_uncond = torch.zeros_like(llm_hidden)

        for i in range(n_steps - 1):
            t = timesteps[i].expand(B)
            dt = timesteps[i + 1] - timesteps[i]

            # Classifier-free guidance
            v_cond = self._predict_velocity(x, llm_hidden, t)
            v_uncond = self._predict_velocity(x, llm_uncond, t)
            v = alpha * v_cond + (1.0 - alpha) * v_uncond

            x = x + v * dt

        # FSQ quantization: clamp to [-1, 1], scale to [0, levels-1], round
        levels = self.args.acoustic_codebook_size  # 21
        x = x.clamp(-1.0, 1.0)
        codes = ((x + 1.0) / 2.0 * (levels - 1)).round().long()
        # Add special token offset
        codes = codes + self.args.n_special_tokens
        return codes

    @torch.no_grad()
    def generate_frame(self, llm_hidden: torch.Tensor,
                       generator: Optional[torch.Generator] = None) -> Optional[torch.Tensor]:
        """Generate one complete audio frame (37 codes).

        Args:
            llm_hidden: [B, dim] hidden state from the last LLM position

        Returns:
            [B, 37] codes (semantic + 36 acoustic), or None if END_AUDIO
        """
        # Semantic prediction
        semantic_code = self.predict_semantic(llm_hidden)  # [B]

        # Check for END_AUDIO
        if (semantic_code == 1).all():  # END_AUDIO=1
            return None

        # Acoustic codes via flow matching
        acoustic_codes = self.decode_acoustic(llm_hidden, generator)  # [B, 36]

        # Combine: [semantic, acoustic_0, acoustic_1, ..., acoustic_35]
        return torch.cat([semantic_code.unsqueeze(-1), acoustic_codes], dim=-1)
