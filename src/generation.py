"""Autoregressive generation utilities.

Guarantees required by the architecture: the semantic representation is
computed EXACTLY ONCE per prompt (Encoder once, SemanticCore once) and the
decoder runs autoregressively against it -- it is never re-computed per token.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F


@torch.no_grad()
def sample_next(logits: torch.Tensor, temperature: float, top_p: float,
                greedy: bool) -> int:
    if greedy or temperature <= 0:
        return int(logits.argmax(dim=-1).item())
    logits = logits / temperature
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        mask = cum - torch.softmax(sorted_logits, dim=-1) > top_p  # keep first above
        sorted_logits[mask] = float("-inf")
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(0, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


class Generator:
    """Wraps a Stage1 or Stage2 model behind one generation interface."""

    def __init__(self, model, tokenizer, max_seq_len: int, gen_cfg, device: torch.device) -> None:
        self.model = model
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        self.gen_cfg = gen_cfg
        self.device = device

    def _encode_semantic(self, prompt: str, apply_noise: bool = False) -> torch.Tensor:
        ids = [self.tok.bos_id] + self.tok.encode(prompt, max_len=self.max_seq_len - 2)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        mask = torch.ones_like(x, dtype=torch.bool)
        if hasattr(self.model, "encode_prompt"):  # Stage2Model
            z = self.model.encode_prompt(x, mask, apply_noise=apply_noise)
            z = self.model.transform(z)  # SemanticCore, once
        else:  # Stage1Model
            z = self.model.encode(x, mask, apply_noise=apply_noise)
        return z

    @torch.no_grad()
    def generate(self, prompt: str, latent: Optional[torch.Tensor] = None,
                 zero_latent: bool = False, temperature: Optional[float] = None,
                 top_p: Optional[float] = None, greedy: Optional[bool] = None,
                 max_gen_len: Optional[int] = None) -> str:
        temp = self.gen_cfg.temperature if temperature is None else temperature
        p = self.gen_cfg.top_p if top_p is None else top_p
        gr = self.gen_cfg.greedy if greedy is None else greedy
        max_len = self.gen_cfg.max_gen_len if max_gen_len is None else max_gen_len

        if latent is None:
            latent = self._encode_semantic(prompt)
        if zero_latent:
            latent = torch.zeros_like(latent)

        ids: List[int] = [self.tok.bos_id] + self.tok.encode(
            prompt, max_len=self.max_seq_len - 2)
        for _ in range(max_len):
            x = torch.tensor([ids], dtype=torch.long, device=self.device)
            logits = self.model.decoder.step(x, latent)
            nxt = sample_next(logits[0], temp, p, gr)
            if nxt == self.tok.eos_id:
                break
            ids.append(nxt)
        return self.tok.decode(ids[1:])

    @torch.no_grad()
    def generate_all_modes(self, prompt: str) -> dict:
        return {
            "greedy": self.generate(prompt, greedy=True),
            "temp0.7": self.generate(prompt, temperature=0.7, top_p=0.9, greedy=False),
            "top_p0.9": self.generate(prompt, temperature=1.0, top_p=0.9, greedy=False),
        }
