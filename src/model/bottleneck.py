"""Semantic Bottleneck: K learned latent queries compressed from encoder
states via cross attention. Output [B, K, D] is independent of input length --
this IS the fixed interface contract. No stride slicing anywhere.

Optional VAE mode (config.vae.enabled, Stage-1 only): the bottleneck
additionally predicts a DIAGONAL GAUSSIAN over each semantic token. mu_head /
logvar_head produce (mu, logvar); during training the latent is SAMPLED with
the reparameterization trick

    z = mu + exp(0.5 * logvar) * eps,   eps ~ N(0, I)

and at eval / inference (or when a deterministic anchor is requested) the
mean mu is used. This turns "one encoding point per sentence" into "a
probability region in semantic space" while keeping the [B, K, D] interface
untouched -- Semantic Cores and the decoder need no changes.

Posterior-collapse guards live in the trainer: KL annealing (warmup), free
bits, and an active-units monitor.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SemanticBottleneck(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_queries: int, dropout: float,
                 noise_std: float = 0.1, use_noise_train: bool = True,
                 use_vae: bool = False, logvar_clamp: float = 10.0) -> None:
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
        # ---- VAE heads (only created when enabled -> old checkpoints load as-is)
        self.use_vae = use_vae
        self.logvar_clamp = logvar_clamp
        if use_vae:
            self.mu_head = nn.Linear(d_model, d_model)
            self.logvar_head = nn.Linear(d_model, d_model)
            nn.init.zeros_(self.logvar_head.bias)
            nn.init.zeros_(self.logvar_head.weight)  # start sigma=1: KL small at t=0

    def forward(self, enc_states: torch.Tensor,
                enc_padding_mask: torch.Tensor | None = None,
                noise_std: float | None = None,
                apply_noise: bool | None = None,
                sample: bool | None = None,
                return_dist: bool = False):
        """enc_states: [B, T, D]; enc_padding_mask: [B, T], True = PAD.

        sample: VAE mode only -- None follows training/eval mode, False forces
        the deterministic mu (e.g. paraphrase anchors). return_dist=True makes
        the return value (z, mu, logvar); logvar is zeros in non-VAE mode so
        the KL computed from it is exactly 0.
        """
        B = enc_states.size(0)
        q = self.norm_q(self.queries).unsqueeze(0).expand(B, -1, -1)
        kv = self.norm_kv(enc_states)
        # our mask: True = REAL. PyTorch key_padding_mask: True = PAD -> invert.
        key_padding = ~enc_padding_mask if enc_padding_mask is not None else None
        attn_out, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding,
                                      need_weights=False)
        h = self.norm_out(q + attn_out)
        h = h + self.ffn(h)

        if not self.use_vae:
            std = self.noise_std if noise_std is None else noise_std
            use = self.use_noise_train if apply_noise is None else apply_noise
            if use and std > 0 and self.training:
                h = h + torch.randn_like(h) * std
            if return_dist:
                return h, h, torch.zeros_like(h)
            return h

        # ---- VAE path
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(-self.logvar_clamp, self.logvar_clamp)
        deterministic = (not self.training) or (sample is False)
        if deterministic:
            z = mu
        else:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        if return_dist:
            return z, mu, logvar
        return z
