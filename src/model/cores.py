"""Pluggable Semantic Core registry.

Interface contract (the ONLY thing trainers may rely on):

    class SemanticCore(nn.Module):
        def forward(self, z: [B, K, D]) -> [B, K, D]: ...

Cores are selected via config (`core.type`), never by importing a concrete
class in the training code. All cores are independently save/load-able.
"""
from __future__ import annotations

from typing import Callable, Dict, Type

import torch
import torch.nn as nn

_REGISTRY: Dict[str, Type["SemanticCore"]] = {}


def register(name: str) -> Callable[[Type["SemanticCore"]], Type["SemanticCore"]]:
    def deco(cls: Type["SemanticCore"]) -> Type["SemanticCore"]:
        _REGISTRY[name] = cls
        cls.name = name
        return cls
    return deco


class SemanticCore(nn.Module):
    """Base class. Subclasses MUST only implement forward([B,K,D]) -> [B,K,D]."""
    name = "base"

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


@register("mlp")
class SemanticMLP(SemanticCore):
    """Position-wise MLP over each semantic token (no cross-token mixing)."""

    def __init__(self, d_model: int, num_semantic_tokens: int,
                 mlp_hidden: int = 2048, dropout: float = 0.1, **kw) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.net(z)


@register("transformer")
class SemanticTransformer(SemanticCore):
    """Bidirectional Transformer encoder over semantic tokens (default core)."""

    def __init__(self, d_model: int, num_semantic_tokens: int, num_layers: int = 2,
                 num_heads: int = 8, ffn_dim: int = 2048, dropout: float = 0.1,
                 **kw) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model, num_heads, ffn_dim, dropout, batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.norm(z + self.enc(z))


@register("bihopfield")
class SemanticBiHopfield(SemanticCore):
    """Minimal modern (energy-based) Hopfield association over semantic tokens:
    repeated pattern-denoising steps using global associative memory
    (beta-scaled softmax attention of the state against stored patterns --
    each token attends to ALL semantic tokens, à la Ramsauer et al. 2020).
    Minimal but runnable; kept intentionally simple.
    """

    def __init__(self, d_model: int, num_semantic_tokens: int, num_layers: int = 2,
                 hopfield_beta: float = 1.0, hopfield_steps: int = 3,
                 dropout: float = 0.1, **kw) -> None:
        super().__init__()
        self.beta = hopfield_beta
        self.steps = hopfield_steps
        self.W = nn.ModuleList([nn.Linear(d_model, d_model, bias=False)
                                for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for W, N in zip(self.W, self.norms):
            state = N(z)
            for _ in range(self.steps):
                # associative recall: each token retrieves from all tokens
                attn = torch.softmax(self.beta * state @ state.transpose(-1, -2), dim=-1)
                state = attn @ state
            z = z + W(state)
        return z


@register("identity")
class IdentityCore(SemanticCore):
    def __init__(self, **kw) -> None:
        super().__init__()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z


@register("random")
class RandomCore(SemanticCore):
    """Frozen random projection -- a destructive-control baseline for ablation."""

    def __init__(self, d_model: int, num_semantic_tokens: int, seed: int = 1234,
                 **kw) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("P", torch.randn(d_model, d_model, generator=g) / d_model ** 0.5)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.P


def available_cores():
    return sorted(_REGISTRY)


def build_semantic_core(cfg, d_model=None, num_semantic_tokens=None) -> SemanticCore:
    """Factory from a SemanticCoreConfig-like object. d_model / K are injected
    from the MODEL config by callers (kept out of the core's own YAML section)."""
    cls = _REGISTRY.get(cfg.type)
    if cls is None:
        raise KeyError(f"Unknown semantic core type '{cfg.type}'. Available: {sorted(_REGISTRY)}")
    return cls(
        d_model=d_model if d_model is not None else getattr(cfg, "d_model", 512),
        num_semantic_tokens=(num_semantic_tokens if num_semantic_tokens is not None
                             else getattr(cfg, "num_semantic_tokens", 16)),
        num_layers=getattr(cfg, "num_layers", 2), num_heads=getattr(cfg, "num_heads", 8),
        ffn_dim=getattr(cfg, "ffn_dim", 2048), dropout=getattr(cfg, "dropout", 0.1),
        mlp_hidden=getattr(cfg, "mlp_hidden", 2048),
        hopfield_beta=getattr(cfg, "hopfield_beta", 1.0),
        hopfield_steps=getattr(cfg, "hopfield_steps", 3),
        seed=getattr(cfg, "seed", 1234),
    )
