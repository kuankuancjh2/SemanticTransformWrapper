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
  bihopfield  PERSISTENT-STATE neural dynamics (see class docstring):
              [B, depth, K, D] state surviving across calls, discrete tick
              loop, dual-axis (token + depth) fully-connected MLP mixing,
              tick/depth embeddings, gated-delta updates, collapse
              diagnostic. NO attention anywhere.
  global_mlp  flatten [B,K,D] -> deep fully-connected MLP over ALL tokens
              jointly (every intermediate layer connects every token)
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
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn

_REGISTRY: Dict[str, type] = {}


def register(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        _REGISTRY[name] = cls
        cls.name = name
        return cls
    return deco


class SemanticCore(nn.Module):
    """Base class. Subclasses MUST only implement forward([B,K,D]) -> [B,K,D].
    Memory-capable cores additionally implement forward_with_state(z, state,
    cond) -> (z_out, new_state) and set supports_memory=True."""
    name = "base"
    supports_memory = False
    accepts_cond = False

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def forward_with_state(self, z: torch.Tensor, state=None, cond=None
                           ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        return self.forward(z), None


def core_forward(core: "SemanticCore", z: torch.Tensor, state=None, cond=None):
    """Single dispatch used by trainers/generators: memory cores get
    (state, cond), everything else keeps the plain z -> z contract."""
    if getattr(core, "supports_memory", False):
        return core.forward_with_state(z, state=state, cond=cond)
    return core(z), None


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
        layers: list = [nn.LayerNorm(d_model)]
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1])]
            if i < len(dims) - 2:
                layers += [act(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.net(z)


# ----------------------------------------------------------------- Transformer
@register("transformer")
class SemanticTransformer(SemanticCore):
    """Bidirectional Transformer encoder over semantic tokens.
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
    """BiHopfield: persistent-state neural-dynamics semantic core.

    This is the REAL BiHopfield design (non-autoregressive, persistent
    latent state) -- NOT a transformer-style mixer:

    - persistent state S: [B, depth, K, D] that survives across calls --
      multi-turn context lives in this state, not in a concatenated prompt.
      The state tensor is carried EXPLICITLY by the caller (trainer /
      generator) so batching stays honest and checkpoints stay clean.
    - discrete tick loop: `hopfield_steps` inner time-steps; each tick reads
      the current state and produces the next state (dynamics, not one
      forward pass).
    - mixing is done by FULLY-CONNECTED layers on TWO axes (no attention):
        * token axis (K x K): every semantic slot reads every slot
        * depth axis (depth x depth): state slices interact
        * token-wise MLP (D -> 2D -> D)
    - tick embedding + depth embedding (which tick / which slice),
      pre-LayerNorm, and a GATED DELTA update  S <- S + g * delta.
    - collapse diagnostic: `last_diag` holds batch-variance and mean pairwise
      cosine of the (depth-pooled) state after every call.

    Controllable: num_layers = state depth (P), hopfield_steps = ticks,
    bi_detach_state (detach state between calls: True = truncated dynamics,
    False = BPTT through the whole tick chain)."""

    supports_memory = True
    accepts_cond = False

    def __init__(self, d_model: int, num_semantic_tokens: int, num_layers: int = 8,
                 hopfield_steps: int = 20, dropout: float = 0.1,
                 bi_detach_state: bool = True, **kw) -> None:
        super().__init__()
        self.depth = max(1, num_layers)
        self.steps = max(1, hopfield_steps)
        self.K = num_semantic_tokens
        self.D = d_model
        self.detach_state = bi_detach_state
        # dynamics weights (shared across ticks; tick/depth embeddings give
        # the time/slice specificity)
        self.norm_state = nn.LayerNorm(d_model)
        self.tok_mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(),
            nn.Linear(2 * d_model, d_model))
        self.seq_mix = nn.Linear(self.K, self.K, bias=False)            # token axis
        self.depth_mix = nn.Linear(self.depth, self.depth, bias=False)  # depth axis
        self.tick_emb = nn.Embedding(self.steps + 1, d_model)
        self.depth_emb = nn.Embedding(self.depth, d_model)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.readout = nn.Linear(d_model, d_model)
        self.norm_out = nn.LayerNorm(d_model)
        self.last_diag: Dict[str, float] = {}

    # ------------------------------------------------------------ state utils
    def init_state(self, batch: int, device, dtype=None) -> torch.Tensor:
        dtype = dtype or torch.get_default_dtype()
        return torch.zeros(batch, self.depth, self.K, self.D,
                           device=device, dtype=dtype)

    def _tick(self, S: torch.Tensor, z: torch.Tensor, t: int) -> torch.Tensor:
        """One discrete tick: S [B, P, K, D] + stimulus z [B, K, D] -> S'."""
        P = self.depth
        te = self.tick_emb(torch.tensor(t, device=S.device, dtype=torch.long))
        Sx = self.norm_state(S + te.view(1, 1, 1, -1)
                             + self.depth_emb.weight.view(1, P, 1, -1).to(S.dtype))
        h = self.tok_mlp(Sx)                                      # token-wise MLP
        h = h + self.seq_mix(Sx.transpose(2, 3)).transpose(2, 3)  # K-axis FC
        h = h + self.depth_mix(Sx.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return S + self.gate(Sx) * h                              # gated delta

    def _diag(self, S: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            pooled = S.mean(dim=1).reshape(S.size(0), -1)         # [B, K*D]
            var = float(pooled.var(dim=0).mean())
            pn = torch.nn.functional.normalize(pooled, dim=-1)
            sim = pn @ pn.T
            off = sim[~torch.eye(S.size(0), dtype=torch.bool, device=S.device)]
            cos = float(off.mean()) if off.numel() else 0.0
            return {"state_batch_var": var, "state_pairwise_cos": cos}

    # -------------------------------------------------------------- interface
    def forward_with_state(self, z: torch.Tensor, state: torch.Tensor | None = None,
                           cond: torch.Tensor | None = None):
        """z: [B, K, D] stimulus; state: [B, depth, K, D] or None (= zeros).
        Returns (z_out [B, K, D], new_state). The returned state is detached
        between calls unless bi_detach_state=False (BPTT through ticks)."""
        S = state if state is not None else self.init_state(z.size(0), z.device, z.dtype)
        for t in range(self.steps):
            S = self._tick(S, z, t)
        z_out = self.norm_out(z + self.readout(S.mean(dim=1)))
        new_state = S.detach() if self.detach_state else S
        self.last_diag = self._diag(new_state)
        return z_out, new_state

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Stateless contract call (ablation / single-shot): fresh zero state."""
        z_out, _ = self.forward_with_state(z, state=None)
        return z_out


# ----------------------------------------------------------------- Global MLP
@register("global_mlp")
class SemanticGlobalMLP(SemanticCore):
    """Global fully-connected core: ALL K semantic tokens are jointly
    connected through the intermediate layers -- the [B, K, D] latent is
    flattened to [B, K*D] and pushed through a deep MLP, so every layer
    sees and mixes every token and every dimension (no per-token
    independence, no attention).
    Controllable: mlp_depth (hidden layer count), mlp_hidden (width).
    NOTE: parameter count scales ~ (K*D)^2; at K=32 / D=512 / hidden=2048 /
    depth=8 this is ~96M params (that is the point of the core)."""

    def __init__(self, d_model: int, num_semantic_tokens: int,
                 mlp_hidden: int = 2048, mlp_depth: int = 2,
                 dropout: float = 0.1, **kw) -> None:
        super().__init__()
        self.K = num_semantic_tokens
        self.D = d_model
        self.kd = d_model * num_semantic_tokens
        dims = [self.kd] + [mlp_hidden] * max(1, mlp_depth) + [self.kd]
        layers: list = [nn.LayerNorm(self.kd)]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [nn.GELU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, K, D = z.shape
        if K * D != self.kd:
            raise ValueError(
                f"global_mlp was built for K*D={self.kd} but got {K}*{D}; "
                "rebuild with matching num_semantic_tokens/hidden_dim")
        return z + self.net(z.reshape(B, self.kd)).view(B, K, D)


# ------------------------------------------------------------------------- Conv
@register("conv")
class SemanticConv(SemanticCore):
    """1D causal-free convolutional core over the K semantic slots.
    Controllable: conv_kernel, num_layers, conv_dilation. Output length is
    always K (asymmetric padding + crop), so the interface contract holds."""

    def __init__(self, d_model: int, num_semantic_tokens: int, num_layers: int = 2,
                 conv_kernel: int = 3, conv_dilation: int = 1,
                 dropout: float = 0.1, **kw) -> None:
        super().__init__()
        self.kernel = conv_kernel
        self.dilation = conv_dilation
        self.blocks = nn.ModuleList()
        for _ in range(max(1, num_layers)):
            self.blocks.append(nn.ModuleDict({
                "norm": nn.LayerNorm(d_model),
                "conv": nn.Conv1d(d_model, d_model, kernel_size=conv_kernel,
                                  dilation=conv_dilation, padding=0),
                "act": nn.GELU(),
                "drop": nn.Dropout(dropout),
            }))
        self.out_norm = nn.LayerNorm(d_model)

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
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
    """Selective SSM block (Mamba-style, pure PyTorch). K is tiny (e.g. 16)
    so the python scan loop is cheap and memory-constant."""

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
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, K, _ = x.shape
        res = x
        x = self.norm(x)
        xz = self.in_proj(x)
        h, gate = xz.chunk(2, dim=-1)
        h = self.conv(h.transpose(1, 2))[..., :K].transpose(1, 2)
        h = torch.nn.functional.silu(h)
        dbc = self.x_proj(h)
        dt = dbc[..., : self.dt_rank]
        Bmat = dbc[..., self.dt_rank: self.dt_rank + self.d_state]
        Cmat = dbc[..., self.dt_rank + self.d_state:]
        delta = torch.nn.functional.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        hstate = torch.zeros(B_, self.d_inner, self.d_state,
                             device=x.device, dtype=x.dtype)
        ys = []
        for t in range(K):
            d_t = delta[:, t].unsqueeze(-1)
            hstate = torch.exp(d_t * A) * hstate + \
                d_t * Bmat[:, t].unsqueeze(1) * h[:, t].unsqueeze(-1)
            ys.append((hstate * Cmat[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1)
        y = y + h * self.D
        y = y * torch.nn.functional.silu(gate)
        return res + self.drop(self.out_proj(y))


@register("mamba")
class SemanticMamba(SemanticCore):
    """Mamba-style selective SSM core, pure PyTorch.
    Controllable: num_layers, ssm_d_state, ssm_d_conv, ssm_expand."""

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
    """Diffusion-inspired iterative refinement core (deterministic z->z map;
    NOT a stochastic DDPM sampler). Training: sample timestep t, noise the
    input, learn a t-conditioned denoiser. Eval: `diffusion_steps`
    deterministic refinement passes. Zip-B adds forward_with_state with an
    encoder-conditioned mode (cond = context encoder states).
    Controllable: diffusion_steps, num_layers, num_heads, ffn_dim,
    diffusion_schedule (cosine|linear)."""

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
        h = x + self.temb(t).unsqueeze(1)
        return x + self.denoiser(h)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.training:
            B = z.size(0)
            t = torch.randint(1, self.T + 1, (B,), device=z.device)
            abar = self.alpha_bar[t - 1].view(B, 1, 1)
            eps = torch.randn_like(z)
            x_t = abar.sqrt() * z + (1 - abar).sqrt() * eps
            return self.norm(self._step(x_t, t))
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
    """Frozen random projection -- a destructive-control baseline."""

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
    from the MODEL config by callers. All knobs are read with getattr +
    default, so old YAML files and old checkpoints keep working."""
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
        bi_detach_state=getattr(cfg, "bi_detach_state", True),
        diffusion_steps=getattr(cfg, "diffusion_steps", 4),
        diffusion_schedule=getattr(cfg, "diffusion_schedule", "cosine"),
        seed=getattr(cfg, "seed", 1234),
    )
