"""Bidirectional Transformer language encoder (token+pos embedding -> N layers).
Supports the layer-split experiment: expose hidden states after any prefix.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .layers import TokenEmbedding


class LanguageEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int,
                 ffn_dim: int, max_seq_len: int, dropout: float, pad_id: int) -> None:
        super().__init__()
        self.embed = TokenEmbedding(vocab_size, d_model, max_seq_len, dropout, pad_id)
        layer = nn.TransformerEncoderLayer(
            d_model, num_heads, ffn_dim, dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(
                d_model, num_heads, ffn_dim, dropout, batch_first=True,
                norm_first=True, activation="gelu") for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, ids: torch.Tensor, padding_mask: Optional[torch.Tensor] = None,
                return_all_layers: bool = False):
        """ids: [B, T]; padding_mask: [B, T], True = PAD.

        Returns hidden states [B, T, D]; if return_all_layers, a list of
        per-layer states (post each block, last one fully normalized).
        """
        x = self.embed(ids)
        # padding_mask semantics: True = REAL token. PyTorch key_padding_mask
        # expects True = PAD, so invert.
        key_padding = ~padding_mask if padding_mask is not None else None
        all_layers: List[torch.Tensor] = []
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding)
            all_layers.append(x)
        out = self.final_norm(x)
        if return_all_layers:
            all_layers[-1] = out
            return all_layers
        return out
