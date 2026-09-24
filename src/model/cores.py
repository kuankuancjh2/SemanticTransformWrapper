"""Pluggable Semantic Core registry.

Interface contract (the ONLY thing trainers may rely on):

    class SemanticCore(nn.Module):
        def forward(self, z: [B, K, D]) -> [B, K, D]: ...

All cores are selected via config (`core.type`) or --core CLI, never by
importing a concrete class in training code. Every core is independently
save/load-able (state_dict) and runs in constant memory per step.

Available cores (all parameterizable from `SemanticCoreConfig` / YAML):
  mlp         depth/width/activation controlled position-wise MLP
  transformer bidirectional Transformer encoder over semantic tokens
  bihopfield  modern Hopfield associative memory (layer-count + beta + steps)
  conv        1D convolutional core (kernel / depth / dilation)
  mamba       selective SSM (Mamba-style) in PURE PyTorch (no mamba_ssm CUDA
              dependency -- runs on CPU / Kaggle / Windows)
  diffusion   diffusion-inspired iterative denoising refiner (step count +
              cosine/linear schedule; deterministic z->z at eval)
  identity    pass-through ablation control
  random      frozen random projection (destructive control)
"""
from __future__ import annotations

import math
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


# ----------------------------------------------------------------- activations
_ACTS: Dict[str, Callable[[], nn.Module]] = {
    "gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh,
}


# ------------------------------------------------------------------------- MLP
@register("mlp")
class SemanticMLP(SemanticCore):
    """Position-wise MLP over each semantic token (no cross-token mixing).
    Controllable: mlp_depth (hidden layer count), mlp_hidden (width),
    mlp_activation (gelu|relu|silu|tanh)."""

    def __init__(self, d_model: int, num_semantic_tokens: int,
                 mlp_hidden: int = 2048, mlp_depth: int = 2,
                 mlp_activation: str = "gelu", dropout: float = 0.1,
                 **kw) -> None:
        super().__init__()
        act = _ACTS.get(mlp_activation)
        if act is None:
            raise KeyError(f"unknown mlp_activation '{mlp_activation}'. "
                           f"Available: {sorted(_ACTS)}")
        hidden_dims = [mlp_hidden] * max(1, mlp_depth)
        dims = [d_model] + hidden_dims + [d_model]
        layers: list[nn.Module] = [nn.LayerNorm(d_model)]
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1])]
            if i < len(dims) - 2:  # no activation on the output projection
                layers += [act(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.net(z)


# ----------------------------------------------------------------- Transformer
@register("transformer")
class SemanticTransformer(SemanticCore):
    """Bidirectional Transformer encoder over semantic tokens (default core).
    Controllable: num_layers, num_heads, ffn_dim."""

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


# ----------------------------------------------------------------- BiHopfield
@register("bihopfield")
class SemanticBiHopfield(SemanticCore):
    """Modern (energy-based) Hopfield association over semantic tokens:
    repeated beta-scaled softmax associative recall against ALL semantic
    tokens (Ramsauer et al. 2020 style), with a controllable stack depth.
    Controllable: num_layers (Hopfield block count), hopfield_beta,
    hopfield_steps (inner energy-descent iterations)."""

    def __init__(self, d_model: int, num_semantic_tokens: int, num_layers: int = 2,
                 hopfield_beta: float = 1.0, hopfield_steps: int = 3,
                 dropout: float = 0.1, **kw) -> None:
        super().__init__()
        self.beta = hopfield_beta
        self.steps = hopfield_steps
        self.W = nn.ModuleList([nn.Linear(d_model, d_model, bias=False)
                                for _ in range(max(1, num_layers))])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model)
                                    for _ in range(max(1, num_layers))])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for W, N in zip(self.W, self.norms):
            state = N(z)
            for _ in range(self.steps):
                # associative recall: each token retrieves from all tokens
                attn = torch.softmax(self.beta * state @ state.transpose(-1, -2), dim=-1)
                state = attn @ state
            z = z + W(state)
        return z


# ------------------------------------------------------------------------- Conv
@register("conv")
class SemanticConv(SemanticCore):
    """1D causal-free convolutional core over the K semantic slots.
    Controllable: conv_kernel (kernel size), num_layers (depth),
    conv_dilation (receptive-field growth). Output length is always K
    (asymmetric padding + crop), so the interface contract holds."""

    def __init__(self, d_model: int, num_semantic_tokens: int, num_layers: int = 2,
                 conv_kernel: int = 3, conv_dilation: int = 1,
                 dropout: float = 0.1, **kw) -> None:
        super().__init__()
        k = conv_kernel
        self.kernel = k
        self.dilation = conv_dilation
        self.blocks = nn.ModuleList()
        for _ in range(max(1, num_layers)):
            self.blocks.append(nn.ModuleDict({
                "norm": nn.LayerNorm(d_model),
                "conv": nn.Conv1d(d_model, d_model, kernel_size=k,
                                  dilation=conv_dilation, padding=0),
                "act": nn.GELU(),
                "drop": nn.Dropout(dropout),
            }))
        self.out_norm = nn.LayerNorm(d_model)

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D, K]; asymmetric padding keeps length K for any kernel/dilation
        left = (self.dilation * (self.kernel - 1)) // 2
        right = self.dilation * (self.kernel - 1) - left
        return nn.functional.pad(x, (left, right))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = z
        for blk in self.blocks:
            h = blk["norm"](out)
            h = self._pad(h.transpose(1, 2))
            h = blk["conv"](h)[..., : out.size(1)].transpose(1, 2)
            h = blk["drop"](blk["act"](h))
            out = out + h
        return self.out_norm(out)


# ------------------------------------------------------------------------ Mamba
class _MambaBlock(nn.Module):
    """Selective SSM block (Mamba-style, pure PyTorch).
    Selectivity: per-token input-dependent Delta/B/C; recurrent scan over the
    K semantic slots. K is tiny (e.g. 16) so the python scan loop is cheap and
    memory-constant -- no CUDA kernel or mamba_ssm dependency needed."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: float = 2.0, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_inner = int(d_model * expand)
        self.d_state = d_state
        self.dt_rank = max(8, self.d_inner // 16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                              padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        # S4D-real initialization of A (log-magnitude, negative = decaying)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, K, d_model] -> [B, K, d_model]"""
        B_, K, _ = x.shape
        res = x
        x = self.norm(x)
        xz = self.in_proj(x)                       # [B, K, 2*d_inner]
        h, gate = xz.chunk(2, dim=-1)              # [B, K, d_inner] each
        # depthwise causal-free local conv (crop back to K)
        h = h.transpose(1, 2)
        h = self.conv(h)[..., :K].transpose(1, 2)  # [B, K, d_inner]
        h = torch.nn.functional.silu(h)

        dbc = self.x_proj(h)                       # [B, K, dt_rank + 2*d_state]
        dt = dbc[..., : self.dt_rank]
        Bmat = dbc[..., self.dt_rank: self.dt_rank + self.d_state]
        Cmat = dbc[..., self.dt_rank + self.d_state:]
        delta = torch.nn.functional.softplus(self.dt_proj(dt))   # [B, K, d_inner]
        A = -torch.exp(self.A_log)                               # [d_inner, d_state]

        # selective recurrent scan over K slots (constant memory)
        hstate = torch.zeros(B_, self.d_inner, self.d_state,
                             device=x.device, dtype=x.dtype)
        ys = []
        for t in range(K):
            d_t = delta[:, t].unsqueeze(-1)        # [B, d_inner, 1]
            hstate = torch.exp(d_t * A) * hstate + \
                d_t * Bmat[:, t].unsqueeze(1) * h[:, t].unsqueeze(-1)
            ys.append((hstate * Cmat[:, t].unsqueeze(1)).sum(-1))  # [B, d_inner]
        y = torch.stack(ys, dim=1)                 # [B, K, d_inner]
        y = y + h * self.D                         # skip connection
        y = y * torch.nn.functional.silu(gate)     # gated output
        return res + self.drop(self.out_proj(y))


@register("mamba")
class SemanticMamba(SemanticCore):
    """Mamba-style selective SSM core, pure PyTorch.
    Controllable: num_layers (block depth), ssm_d_state, ssm_d_conv, ssm_expand."""

    def __init__(self, d_model: int, num_semantic_tokens: int, num_layers: int = 2,
                 ssm_d_state: int = 16, ssm_d_conv: int = 4, ssm_expand: float = 2.0,
                 dropout: float = 0.1, **kw) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            _MambaBlock(d_model, ssm_d_state, ssm_d_conv, ssm_expand, dropout)
            for _ in range(max(1, num_layers))
        ])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = z
        for blk in self.blocks:
            out = blk(out)
        return self.out_norm(out)


# -------------------------------------------------------------------- Diffusion
@register("diffusion")
class SemanticDiffusion(SemanticCore):
    """Diffusion-inspired iterative refinement core.

    Honest scope: the Semantic Core contract is a DETERMINISTIC map z->z, so
    this is not a stochastic DDPM sampler. Training samples a timestep t,
    noises the input (sqrt(alpha_bar_t) z + sqrt(1-alpha_bar_t) eps) and learns
    a t-conditioned denoiser whose output the Stage-2 loss pulls toward the
    target semantics. Evaluation runs `diffusion_steps` deterministic
    refinement passes (t = T..1) without noise injection.

    Controllable: diffusion_steps (schedule length / eval passes), num_layers,
    num_heads, ffn_dim, diffusion_schedule (cosine|linear)."""

    def __init__(self, d_model: int, num_semantic_tokens: int, num_layers: int = 2,
                 num_heads: int = 8, ffn_dim: int = 2048, dropout: float = 0.1,
                 diffusion_steps: int = 4, diffusion_schedule: str = "cosine",
                 **kw) -> None:
        super().__init__()
        self.T = max(1, diffusion_steps)
        if diffusion_schedule == "cosine":
            s = 0.008
            f = lambda u: math.cos((u + s) / (1 + s) * math.pi / 2) ** 2
            bar = [f((t + 1) / self.T) / f(1.0) for t in range(self.T)]
        elif diffusion_schedule == "linear":
            bar = list(torch.linspace(0.9, 0.1, self.T).tolist())
        else:
            raise KeyError(f"unknown diffusion_schedule '{diffusion_schedule}'")
        self.register_buffer("alpha_bar", torch.clamp(torch.tensor(bar), 1e-4, 1.0))
        self.temb = nn.Embedding(self.T + 1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, num_heads, ffn_dim, dropout, batch_first=True, norm_first=True)
        self.denoiser = nn.TransformerEncoder(layer, max(1, num_layers))
        self.norm = nn.LayerNorm(d_model)

    def _step(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """One denoising step; t: [B] int in 1..T. Residual parameterization."""
        h = x + self.temb(t).unsqueeze(1)          # broadcast timestep embedding
        return x + self.denoiser(h)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.training:
            B = z.size(0)
            t = torch.randint(1, self.T + 1, (B,), device=z.device)
            abar = self.alpha_bar[t - 1].view(B, 1, 1)
            eps = torch.randn_like(z)
            x_t = abar.sqrt() * z + (1 - abar).sqrt() * eps
            return self.norm(self._step(x_t, t))
        # eval: deterministic reverse refinement t = T..1 (no noise)
        x = z
        for t in range(self.T, 0, -1):
            tt = torch.full((z.size(0),), t, dtype=torch.long, device=z.device)
            x = self._step(x, tt)
        return self.norm(x)


# -------------------------------------------------------------------- controls
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


# --------------------------------------------------------------------- factory
def available_cores():
    return sorted(_REGISTRY)


def build_semantic_core(cfg, d_model=None, num_semantic_tokens=None) -> SemanticCore:
    """Factory from a SemanticCoreConfig-like object. d_model / K are injected
    from the MODEL config by callers (kept out of the core's own YAML section).
    All core knobs are read with getattr + default, so old YAML files (and old
    checkpoints' saved configs) keep working unchanged."""
    cls = _REGISTRY.get(cfg.type)
    if cls is None:
        raise KeyError(f"Unknown semantic core type '{cfg.type}'. "
                       f"Available: {sorted(_REGISTRY)}")
    return cls(
        d_model=d_model if d_model is not None else getattr(cfg, "d_model", 512),
        num_semantic_tokens=(num_semantic_tokens if num_semantic_tokens is not None
                             else getattr(cfg, "num_semantic_tokens", 16)),
        num_layers=getattr(cfg, "num_layers", 2),
        num_heads=getattr(cfg, "num_heads", 8),
        ffn_dim=getattr(cfg, "ffn_dim", 2048),
        dropout=getattr(cfg, "dropout", 0.1),
        mlp_hidden=getattr(cfg, "mlp_hidden", 2048),
        mlp_depth=getattr(cfg, "mlp_depth", 2),
        mlp_activation=getattr(cfg, "mlp_activation", "gelu"),
        conv_kernel=getattr(cfg, "conv_kernel", 3),
        conv_dilation=getattr(cfg, "conv_dilation", 1),
        ssm_d_state=getattr(cfg, "ssm_d_state", 16),
        ssm_d_conv=getattr(cfg, "ssm_d_conv", 4),
        ssm_expand=getattr(cfg, "ssm_expand", 2.0),
        hopfield_beta=getattr(cfg, "hopfield_beta", 1.0),
        hopfield_steps=getattr(cfg, "hopfield_steps", 3),
        diffusion_steps=getattr(cfg, "diffusion_steps", 4),
        diffusion_schedule=getattr(cfg, "diffusion_schedule", "cosine"),
        seed=getattr(cfg, "seed", 1234),
    )
