"""Shared trainer internals (Zip-B): loaders for Stage 1 (pairs) and
Stage 2/3 (messages conversations), optimizer/schedule, AMP, preview."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..config import Config
from ..generation import Generator
from ..utils.logging_utils import get_logger

log = get_logger("train")


def make_loaders(cfg: Config, tok, stage: int) -> Tuple[DataLoader, DataLoader, dict]:
    """stage=1 -> single-turn reconstruction pairs (no multi-turn anywhere).
    stage in (2, 3) -> messages-format conversation loaders."""
    if cfg.data.max_seq_len > cfg.model.max_seq_len:
        raise ValueError(
            f"data.max_seq_len ({cfg.data.max_seq_len}) exceeds model.max_seq_len "
            f"({cfg.model.max_seq_len}) -- positional embeddings would overflow. "
            "Keep the two keys equal (all shipped configs do).")
    from ..dataset import (MessagesDataset, TextPairDataset, collate_messages,
                           collate_pairs, load_messages, load_pairs)

    base = Path(cfg.data.processed_dir)
    pad = tok.pad_id
    pin = (torch.cuda.is_available() if getattr(cfg.train, "pin_memory", "auto") == "auto"
           else cfg.train.pin_memory == "on")
    common = dict(batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers,
                  pin_memory=pin)
    if stage == 1:
        tr = load_pairs(base / "stage1_train.jsonl")
        va = load_pairs(base / "stage1_val.jsonl")
        train_loader = DataLoader(
            TextPairDataset(tr, tok, cfg.data.max_seq_len), shuffle=True,
            drop_last=True, collate_fn=lambda b: collate_pairs(b, pad), **common)
        val_loader = DataLoader(
            TextPairDataset(va, tok, cfg.data.max_seq_len), shuffle=False,
            collate_fn=lambda b: collate_pairs(b, pad), **common)
    else:
        tr = load_messages(base / "train.jsonl")
        va = load_messages(base / "val.jsonl")
        train_loader = DataLoader(
            MessagesDataset(tr, tok, cfg.data.max_seq_len), shuffle=True,
            drop_last=True, collate_fn=lambda b: collate_messages(b, pad), **common)
        val_loader = DataLoader(
            MessagesDataset(va, tok, cfg.data.max_seq_len), shuffle=False,
            collate_fn=lambda b: collate_messages(b, pad), **common)
    meta = {}
    mp = Path(cfg.data.metadata_path)
    if mp.exists():
        meta = dict(json.loads(mp.read_text(encoding="utf-8")))
    return train_loader, val_loader, meta


import json  # noqa: E402  (used by make_loaders metadata read)


def setup_amp(device: torch.device, mode: str):
    if device.type != "cuda" or mode == "off":
        return None, None
    if mode == "auto":
        mode = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    try:
        if mode == "bf16":
            return torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.amp.GradScaler("cuda")
        return torch.amp.autocast("cuda", dtype=torch.float16), torch.amp.GradScaler("cuda")
    except Exception:
        return None, None


def lr_lambda(warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return f


def save_preview(cfg: Config, model, gen: Generator, step: int, out_dir: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"step_{step:06d}.txt"
    with open(path, "w", encoding="utf-8") as f:
        for p in cfg.train.preview_prompts:
            modes = gen.generate_all_modes(p)
            f.write(f"[prompt] {p}\n")
            for name, g in modes.items():
                f.write(f"[{name}] {g}\n")
            f.write("\n")
    log.info("Preview saved: %s", path)
