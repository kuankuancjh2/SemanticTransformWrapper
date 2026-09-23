"""All training losses, padding-aware.

Stage 1:  L = L_recon + a*L_paraphrase + b*L_var + c*L_cov
Stage 2:  L = w_lat*(set-level latent MSE + cosine) + w_dec*L_decode
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def reconstruction_loss(logits: torch.Tensor, target_ids: torch.Tensor,
                        pad_id: int) -> torch.Tensor:
    """Standard CE over the target sequence; padding ignored via ignore_index."""
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1),
        ignore_index=pad_id,
    )


def token_accuracy(logits: torch.Tensor, target_ids: torch.Tensor,
                   pad_id: int) -> float:
    pred = logits.argmax(-1)
    ok = (pred == target_ids) & (target_ids != pad_id)
    total = (target_ids != pad_id).sum().item()
    return float(ok.sum().item()) / max(1, total)


# --------------------------------------------------------------------- stage 1
def paraphrase_consistency_loss(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """Cosine distance between pooled semantic representations of a pair."""
    za = F.normalize(z_a.mean(dim=1), dim=-1)
    zb = F.normalize(z_b.mean(dim=1), dim=-1)
    return (1.0 - (za * zb).sum(dim=-1)).mean()


def variance_loss(z: torch.Tensor, target: float = 1.0) -> torch.Tensor:
    """VICReg-style: penalize per-dimension std deviating from `target`."""
    zs = z.reshape(-1, z.size(-1))
    std = zs.std(dim=0) + 1e-4
    return F.relu(target - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """Penalize off-diagonal covariance of semantic dimensions (stable, not
    |corr|)."""
    zs = z.reshape(-1, z.size(-1))
    zs = zs - zs.mean(dim=0, keepdim=True)
    n = zs.size(0)
    cov = (zs.T @ zs) / max(1, n - 1)
    off = cov - torch.diag(torch.diag(cov))
    return (off ** 2).sum() / zs.size(-1)


def latent_stats(z: torch.Tensor) -> Dict[str, float]:
    zs = z.reshape(-1, z.size(-1)).detach().float()
    zn = F.normalize(zs, dim=-1)
    sim = (zn @ zn.T)
    k = zs.size(0)
    off = sim[~torch.eye(k, dtype=torch.bool, device=sim.device)]
    cov = torch.cov(zs.T)
    try:
        eig = torch.linalg.eigvalsh(cov)
        eff = float((eig.sum() ** 2) / (eig ** 2).sum().clamp_min(1e-8))
    except Exception:
        eff = 0.0
    return {
        "std": float(zs.std().item()),
        "norm": float(zs.norm(dim=-1).mean().item()),
        "cos_sim": float(off.mean().item()) if off.numel() else 0.0,
        "cov_off": float(((cov - torch.diag(torch.diag(cov))) ** 2).sum().item() / zs.size(-1)),
        "eff_rank": eff,
    }


def stage1_total(losses: Dict[str, torch.Tensor], cfg_loss) -> torch.Tensor:
    return (losses["reconstruction"]
            + cfg_loss.paraphrase_weight * losses["paraphrase"]
            + cfg_loss.variance_weight * losses["variance"]
            + cfg_loss.covariance_weight * losses["covariance"])


# --------------------------------------------------------------------- stage 2
def latent_set_loss(z_pred: torch.Tensor, z_target: torch.Tensor,
                    cosine_weight: float = 0.5) -> Dict[str, torch.Tensor]:
    """Set-level latent distance: pooled MSE (NO position-wise pairing, prompt
    and target lengths may differ) + pooled cosine distance."""
    p = z_pred.mean(dim=1)
    t = z_target.mean(dim=1)
    mse = F.mse_loss(p, t)
    pn = F.normalize(p, dim=-1)
    tn = F.normalize(t, dim=-1)
    cos = (1.0 - (pn * tn).sum(dim=-1)).mean()
    return {"latent_mse": mse, "latent_cosine": cos,
            "latent": mse + cosine_weight * cos}


def stage2_total(latent: torch.Tensor, decode: torch.Tensor,
                 cfg_loss) -> torch.Tensor:
    return cfg_loss.latent_weight * latent + cfg_loss.decode_weight * decode
