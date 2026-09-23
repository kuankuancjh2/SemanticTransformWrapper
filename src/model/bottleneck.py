"""Semantic Bottleneck: K learned latent queries compress encoder states via
cross attention. Output [B, K, D] is independent of input length -- this IS
the fixed interface contract. No stride slicing anywhere.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SemanticBottleneck(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_queries: int, dropout: float,
                 noise_std: float = 0.1, use_noise_train: bool = True) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model
        self.queries = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                                batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.noise_std = noise_std
        self.use_noise_train = use_noise_train

    def forward(self, enc_states: torch.Tensor,
                enc_padding_mask: torch.Tensor | None = None,
                noise_std: float | None = None,
                apply_noise: bool | None = None) -> torch.Tensor:
        """enc_states: [B, T, D]; enc_padding_mask: [B, T] True = PAD.

        Returns semantic tokens [B, K, D].
        """
        B = enc_states.size(0)
        q = self.norm_q(self.queries).unsqueeze(0).expand(B, -1, -1)
        kv = self.norm_kv(enc_states)
        # our mask: True = REAL. PyTorch key_padding_mask: True = PAD -> invert.
        key_padding = ~enc_padding_mask if enc_padding_mask is not None else None
        attn_out, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding,
                                      need_weights=False)
        z = self.norm_out(q + attn_out)
        z = z + self.ffn(z)

        std = self.noise_std if noise_std is None else noise_std
        use = self.use_noise_train if apply_noise is None else apply_noise
        if use and std > 0 and self.training:
            z = z + torch.randn_like(z) * std
        return z
