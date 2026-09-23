"""Shared trainer internals: loaders, optimizer/scheduler, AMP, TensorBoard,
preview generation, checkpoint bookkeeping. Used by both stage scripts.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..config import Config
from ..dataset import TextPairDataset, collate_pairs
from ..generation import Generator
from ..utils.logging_utils import get_logger
from ..utils.seed import capture_random_state

log = get_logger("train")


def make_loaders(cfg: Config, tok, stage: int) -> Tuple[DataLoader, DataLoader, dict]:
    from ..dataset import load_split

    base = Path(cfg.data.processed_dir)
    pad = tok.pad_id
    common = dict(batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers,
                  collate_fn=lambda b: collate_pairs(b, pad),
                  pin_memory=torch.cuda.is_available())
    if stage == 1:
        tr = load_split(base / "stage1_train.jsonl")
        va = load_split(base / "stage1_val.jsonl")
    else:
        tr = load_split(base / "train.jsonl")
        va = load_split(base / "val.jsonl")
    train_loader = DataLoader(TextPairDataset(tr, tok, cfg.data.max_seq_len),
                              shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(TextPairDataset(va, tok, cfg.data.max_seq_len),
                            shuffle=False, **common)
    meta = {}
    mp = Path(cfg.data.metadata_path)
    if mp.exists():
        meta = dict(__import__("json").loads(mp.read_text(encoding="utf-8")))
    return train_loader, val_loader, meta


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


def tb_scalars(writer: SummaryWriter, prefix: str, scalars: Dict[str, float],
               step: int) -> None:
    for k, v in scalars.items():
        writer.add_scalar(f"{k}", v, step)
