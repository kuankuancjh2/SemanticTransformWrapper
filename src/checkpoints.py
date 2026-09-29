"""Checkpoint save/load with full reproducibility payload.

Storage policy (v3):
  best.pt        full state, UNCOMPRESSED (resume-anchored init for later stages)
  latest.pt.gz   full state (incl. optimizer), gzip-compressed
  epoch_N.pt.gz  fp16 weights only (no optimizer/scheduler/random state), gzip
All loaders are gzip-transparent: passing either `x.pt` or `x.pt.gz` works.
"""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from .config import Config
from .model.model import Stage1Model, Stage2Model
from .model.cores import build_semantic_core
from .tokenization import load_tokenizer
from .utils.seed import capture_random_state, restore_random_state  # noqa: F401


def half_state_dict(sd: Dict[str, Any]) -> Dict[str, Any]:
    """fp32 -> fp16 for floating tensors (halves snapshot size; load_state_dict
    casts back to the model's parameter dtypes on load)."""
    return {k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v)
            for k, v in sd.items()}


def save_payload(path: str | Path, payload: Dict[str, Any],
                 compress: bool = False) -> str:
    """torch.save, optionally through gzip (adds .gz when missing)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        if path.suffix != ".gz":
            path = path.with_suffix(path.suffix + ".gz")
        with gzip.open(path, "wb") as f:
            torch.save(payload, f)
    else:
        torch.save(payload, str(path))
    return str(path)


def save_train_checkpoint(out_dir: str | Path, tag: str, model: torch.nn.Module,
                          optimizer: Optional[torch.optim.Optimizer],
                          scheduler, epoch: int, step: int, best_val_loss: float,
                          cfg: Config, tokenizer_path: str,
                          random_state: Optional[Dict[str, Any]] = None, *,
                          stage: Optional[int] = None,
                          core_type: Optional[str] = None) -> str:
    """Trainer-side save policy (see module docstring):
      best.pt       full payload, UNCOMPRESSED (init anchor for the next stage)
      latest.pt.gz  full payload (incl. optimizer, resume-ready), gzip
      epoch_*.pt.gz weights-only fp16 snapshot (no optimizer/scheduler/RNG), gzip
    DDP-safe (unwraps .module) and honors train.compress_checkpoints."""
    m = getattr(model, "module", model)
    payload: Dict[str, Any] = {
        "model_state_dict": m.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "step": step,
        "best_val_loss": best_val_loss,
        "config": cfg.to_dict(),
        "tokenizer": tokenizer_path,
        "random_state": random_state,
        "stage": stage,
        "core_type": core_type,
    }
    compress = bool(getattr(cfg.train, "compress_checkpoints", True))
    if tag == "best.pt":
        return save_payload(Path(out_dir) / tag, payload, compress=False)
    if tag.startswith("epoch_"):
        payload["optimizer_state_dict"] = None
        payload["scheduler_state_dict"] = None
        payload["random_state"] = None
        payload["model_state_dict"] = half_state_dict(payload["model_state_dict"])
        return save_payload(Path(out_dir) / tag, payload, compress=compress)
    return save_payload(Path(out_dir) / tag, payload, compress=compress)


def save_checkpoint(path: str | Path, model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer],
                    scheduler, epoch: int, step: int, best_val_loss: float,
                    cfg: Config, tokenizer_path: str,
                    random_state: Optional[Dict[str, Any]] = None,
                    extra: Optional[Dict[str, Any]] = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "step": step,
        "best_val_loss": best_val_loss,
        "config": cfg.to_dict(),
        "tokenizer": tokenizer_path,
        "random_state": random_state,
        "stage": extra.get("stage") if extra else None,
        "core_type": extra.get("core_type") if extra else None,
    }
    torch.save(payload, str(path))


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    """Gzip-transparent checkpoint loading (.pt and .pt.gz both work)."""
    p = str(path)
    if p.endswith(".gz"):
        with gzip.open(p, "rb") as f:
            return torch.load(f, map_location=map_location, weights_only=False)
    return torch.load(p, map_location=map_location, weights_only=False)


def build_stage1_from_checkpoint(ckpt: Dict[str, Any], device: torch.device
                                 ) -> Tuple[Stage1Model, Config, Any]:
    cfg = Config.from_dict(ckpt["config"])
    tok = load_tokenizer(ckpt["tokenizer"])
    cfg.model.vocab_size = tok.vocab_size  # architecture must match tokenizer
    model = Stage1Model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, cfg, tok


def build_stage2_from_checkpoint(ckpt: Dict[str, Any], device: torch.device
                                 ) -> Tuple[Stage2Model, Config, Any]:
    cfg = Config.from_dict(ckpt["config"])
    tok = load_tokenizer(ckpt["tokenizer"])
    cfg.model.vocab_size = tok.vocab_size  # architecture must match tokenizer
    stage1 = Stage1Model(cfg)
    # Stage2Model.state_dict() includes 'core.*' keys; strip them for Stage1Model
    sd = {k: v for k, v in ckpt["model_state_dict"].items() if not k.startswith("core.")}
    stage1.load_state_dict(sd)
    stage1.to(device)
    core_type = ckpt.get("core_type") or cfg.core.type
    cfg.core.type = core_type
    core = build_semantic_core(cfg.core, d_model=cfg.model.hidden_dim,
                               num_semantic_tokens=cfg.model.num_semantic_tokens)
    if "core_state_dict" in ckpt and ckpt["core_state_dict"] is not None:
        core.load_state_dict(ckpt["core_state_dict"])
    core.to(device)
    model = Stage2Model(stage1, core)
    model.eval()
    tok = load_tokenizer(ckpt["tokenizer"])
    return model, cfg, tok
