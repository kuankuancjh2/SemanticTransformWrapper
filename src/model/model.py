"""Top-level model assemblies.

Stage1Model: Language Autoencoder (Encoder -> Bottleneck -> AR Decoder).
Stage2Model: frozen Stage-1 parts + one pluggable SemanticCore.

The semantic interface is fixed at [B, K, D] end to end. DECODER_SEES_PROMPT
(default False) is the only ablation switch on the decoder side.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from ..config import Config
from .bottleneck import SemanticBottleneck
from .decoder import LanguageDecoder
from .encoder import LanguageEncoder


class Stage1Model(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        m = cfg.model
        self.cfg = cfg
        self.encoder = LanguageEncoder(
            m.vocab_size, m.hidden_dim, m.num_heads, m.encoder_layers,
            m.ffn_dim, m.max_seq_len, m.dropout, pad_id=0)
        self.bottleneck = SemanticBottleneck(
            m.hidden_dim, m.num_heads, m.num_semantic_tokens, m.dropout,
            noise_std=cfg.bottleneck.noise_std,
            use_noise_train=cfg.bottleneck.use_noise_train,
            use_vae=cfg.vae.enabled,
            logvar_clamp=cfg.vae.logvar_clamp)
        self.decoder = LanguageDecoder(
            m.vocab_size, m.hidden_dim, m.num_heads, m.decoder_layers,
            m.ffn_dim, m.max_seq_len, m.dropout, pad_id=0,
            decoder_sees_prompt=m.decoder_sees_prompt)

    # ------------------------------------------------------------- encoding
    def encode(self, ids: torch.Tensor, padding_mask: Optional[torch.Tensor] = None,
               noise_std: float | None = None,
               apply_noise: bool | None = None,
               sample: bool | None = None) -> torch.Tensor:
        """text ids [B, T] -> semantic tokens [B, K, D].
        sample: VAE mode -- None follows mode, False forces mu (deterministic)."""
        split = self.cfg.model.encoder_layer_split
        if split is not None:
            layers = self.encoder(ids, padding_mask, return_all_layers=True)
            enc = layers[min(split, len(layers)) - 1]
        else:
            enc = self.encoder(ids, padding_mask)
        return self.bottleneck(enc, padding_mask, noise_std=noise_std,
                               apply_noise=apply_noise, sample=sample)

    # ------------------------------------------------------------- decoding
    def decode_logits(self, tgt_in: torch.Tensor, semantic_tokens: torch.Tensor,
                      tgt_padding_mask: Optional[torch.Tensor] = None,
                      prompt_states: Optional[torch.Tensor] = None,
                      prompt_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.decoder(tgt_in, semantic_tokens, tgt_padding_mask,
                            prompt_states, prompt_padding_mask)

    def forward(self, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor,
                tgt_in: torch.Tensor, tgt_mask: torch.Tensor,
                noise_std: float | None = None) -> dict:
        """Teacher-forced pass. tgt_in starts with BOS, no EOS (labels = shifted).

        VAE mode additionally returns 'mu' / 'logvar' for the KL term
        (both None when the VAE is disabled).
        """
        enc = self.encoder(prompt_ids, prompt_mask)
        if self.bottleneck.use_vae:
            z, mu, logvar = self.bottleneck(enc, prompt_mask, noise_std=noise_std,
                                            return_dist=True)
        else:
            z = self.bottleneck(enc, prompt_mask, noise_std=noise_std)
            mu = logvar = None  # AE mode: no distribution heads
        logits = self.decoder(
            tgt_in, z, tgt_mask,
            enc if self.decoder.sees_prompt else None,
            prompt_mask if self.decoder.sees_prompt else None)
        return {"logits": logits, "semantic": z, "mu": mu, "logvar": logvar,
                "encoder_states": enc}


class Stage2Model(nn.Module):
    """Frozen (encoder, bottleneck, decoder) + trainable SemanticCore."""

    def __init__(self, stage1: Stage1Model, core: nn.Module) -> None:
        super().__init__()
        self.cfg = stage1.cfg
        self.encoder = stage1.encoder
        self.bottleneck = stage1.bottleneck
        self.decoder = stage1.decoder
        self.core = core
        for mod in (self.encoder, self.bottleneck, self.decoder):
            mod.eval()
            mod.requires_grad_(False)

    def train(self, mode: bool = True) -> "Stage2Model":
        super().train(mode)
        if mode:  # frozen modules stay in eval mode
            self.encoder.eval()
            self.bottleneck.eval()
            self.decoder.eval()
        return self

    @torch.no_grad()
    def encode_prompt(self, ids: torch.Tensor, padding_mask: torch.Tensor,
                      apply_noise: bool = False) -> torch.Tensor:
        enc = self.encoder(ids, padding_mask)
        return self.bottleneck(enc, padding_mask, apply_noise=apply_noise)

    @torch.no_grad()
    def encode_target(self, ids: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        return self.encode_prompt(ids, padding_mask, apply_noise=False)

    def transform(self, z_prompt: torch.Tensor) -> torch.Tensor:
        return self.core(z_prompt)

    def decode_logits(self, tgt_in: torch.Tensor, semantic_tokens: torch.Tensor,
                      tgt_padding_mask: Optional[torch.Tensor] = None,
                      prompt_states: Optional[torch.Tensor] = None,
                      prompt_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # grads flow through the frozen decoder back into the core
        return self.decoder(tgt_in, semantic_tokens, tgt_padding_mask)

    def forward(self, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor,
                target_ids: torch.Tensor, target_mask: torch.Tensor) -> dict:
        z_p = self.encode_prompt(prompt_ids, prompt_mask)
        z_t = self.encode_target(target_ids, target_mask)
        z_pred = self.transform(z_p)
        return {"z_pred": z_pred, "z_target": z_t}
