"""Shared building blocks: sinusoidal positional embedding and embedding stack."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: (d_model + 1) // 2]) if d_model % 2 == 1 else torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return x + self.pe[: x.size(1)].unsqueeze(0)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int, dropout: float,
                 pad_id: int) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = SinusoidalPositionalEmbedding(d_model, max_len + 8)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.drop(self.norm(self.pos(self.tok(ids) * math.sqrt(self.d_model))))
