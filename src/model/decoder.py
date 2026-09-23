"""Autoregressive language decoder.

Cross-attends ONLY to the semantic tokens by default (never to the prompt's
encoder states). `decoder_sees_prompt=True` adds an optional second
cross-attention over prompt states for the prompt-visibility ablation.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import TokenEmbedding


def causal_mask(t: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(t, t, dtype=torch.bool, device=device), diagonal=1)


class LanguageDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int,
                 ffn_dim: int, max_seq_len: int, dropout: float, pad_id: int,
                 decoder_sees_prompt: bool = False) -> None:
        super().__init__()
        self.embed = TokenEmbedding(vocab_size, d_model, max_seq_len, dropout, pad_id)
        self.blocks = nn.ModuleList([
            _DecBlock(d_model, num_heads, ffn_dim, dropout, decoder_sees_prompt)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.pad_id = pad_id
        self.sees_prompt = decoder_sees_prompt

    def forward(self, tgt_ids: torch.Tensor, semantic_tokens: torch.Tensor,
                tgt_padding_mask: Optional[torch.Tensor] = None,
                prompt_states: Optional[torch.Tensor] = None,
                prompt_padding_mask: Optional[torch.Tensor] = None,
                all_layers: Optional[list] = None) -> torch.Tensor:
        """tgt_ids: [B, T] previous tokens (teacher forcing, starts with BOS).

        Returns logits [B, T, V]. semantic_tokens: [B, K, D].
        """
        T = tgt_ids.size(1)
        x = self.embed(tgt_ids)
        cmask = causal_mask(T, tgt_ids.device)
        # our masks: True = REAL. PyTorch key_padding_mask: True = PAD -> invert.
        tkey = ~tgt_padding_mask if tgt_padding_mask is not None else None
        pkey = ~prompt_padding_mask if prompt_padding_mask is not None else None
        for blk in self.blocks:
            x = blk(x, semantic_tokens, cmask, tkey,
                    prompt_states, pkey, all_layers=all_layers)
        x = self.final_norm(x)
        return self.head(x)

    # ---------------------------------------------------------------- decode
    @torch.no_grad()
    def step(self, last_ids: torch.Tensor, semantic_tokens: torch.Tensor,
             prompt_states: Optional[torch.Tensor] = None,
             prompt_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """One incremental decode step for the last token only.

        last_ids: [B, t] tokens generated so far. Returns logits for the
        final position [B, V]. (Re-running the decoder per step keeps memory
        modest at our scale; the semantic representation is computed ONCE
        outside and never re-computed per token.)
        """
        x = self.embed(last_ids)
        cmask = causal_mask(last_ids.size(1), last_ids.device)
        pkey = ~prompt_padding_mask if prompt_padding_mask is not None else None
        for blk in self.blocks:
            x = blk(x, semantic_tokens, cmask, None, prompt_states, pkey)
        x = self.final_norm(x[:, -1:])
        return self.head(x[:, 0])


class _DecBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float,
                 sees_prompt: bool) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.sees_prompt = sees_prompt
        if sees_prompt:
            self.prompt_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            self.norm2b = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, cmask, tkey, prompt_states, pkey, all_layers=None):
        a, _ = self.self_attn(x, x, x, attn_mask=cmask, key_padding_mask=tkey,
                              need_weights=False)
        x = x + self.drop(a)
        x = self.norm1(x)
        m, _ = self.cross_attn(x, memory, memory, need_weights=False)
        x = self.norm2(x + self.drop(m))
        if self.sees_prompt and prompt_states is not None:
            p, _ = self.prompt_attn(x, prompt_states, prompt_states,
                                    key_padding_mask=pkey, need_weights=False)
            x = self.norm2b(x + self.drop(p))
        x = x + self.drop(self.ffn(x))
        x = self.norm3(x)
        if all_layers is not None:
            all_layers.append(x)
        return x
