"""Greedy / sampled generation from a trained Stage-1 or Stage-2 checkpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from src.checkpoints import build_stage1_from_checkpoint, build_stage2_from_checkpoint
from src.config import resolve_device
from src.generation import Generator
from src.utils.logging_utils import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--greedy", action="store_true", default=None)
    ap.add_argument("--max-gen-len", type=int, default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = resolve_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ckpt.get("stage") == 2:
        model, cfg, tok = build_stage2_from_checkpoint(ckpt, device)
    else:
        model, cfg, tok = build_stage1_from_checkpoint(ckpt, device)

    gen_cfg = cfg.gen
    if args.temperature is not None:
        gen_cfg.temperature = args.temperature
    if args.top_p is not None:
        gen_cfg.top_p = args.top_p
    if args.greedy:
        gen_cfg.greedy = True
    if args.max_gen_len:
        gen_cfg.max_gen_len = args.max_gen_len

    gen = Generator(model, tok, cfg.data.max_seq_len, gen_cfg, device)
    latent = gen._encode_semantic(args.prompt)
    out = gen.generate(args.prompt)
    print(f"latent shape : {tuple(latent.shape)}")
    print(f"latent norm  : {latent.norm(dim=-1).mean().item():.4f}")
    print(f"generated    : {out}")


if __name__ == "__main__":
    main()
