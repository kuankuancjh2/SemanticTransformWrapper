"""Interactive demo CLI (and --once mode for non-interactive smoke tests)."""
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
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--once", action="store_true", help="run one fixed prompt and exit")
    args = ap.parse_args()

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        cands = [Path("checkpoints/stage2/best.pt"), Path("checkpoints/stage1/best.pt")]
        cands = [c for c in cands if c.exists()]
        if not cands:
            print("no checkpoint found; run train_stage1.py first")
            sys.exit(1)
        ckpt_path = cands[0]

    device = resolve_device(args.device)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if ckpt.get("stage") == 2:
        model, cfg, tok = build_stage2_from_checkpoint(ckpt, device)
        core_name = ckpt.get("core_type", cfg.core.type)
    else:
        model, cfg, tok = build_stage1_from_checkpoint(ckpt, device)
        core_name = "(none - stage1 autoencoder)"
    gen = Generator(model, tok, cfg.data.max_seq_len, cfg.gen, device)

    def turn(prompt: str) -> None:
        latent = gen._encode_semantic(prompt)
        out = gen.generate(prompt)
        print("=" * 50)
        print("Prompt:")
        print(f"> {prompt}")
        print("Semantic Core:")
        print(f"> {core_name}")
        print("Generated:")
        print(f"> {out}")
        print("-" * 50)
        print(f"latent shape       : {tuple(latent.shape)}")
        print(f"latent norm        : {latent.norm(dim=-1).mean().item():.4f}")
        print(f"generation length  : {len(tok.encode(out))}")

    if args.once:
        turn("小明拿起苹果")
        return

    print(f"demo loaded: {ckpt_path} | core: {core_name} | Ctrl-C to quit")
    while True:
        try:
            prompt = input("\nPrompt:\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if prompt:
            turn(prompt)


if __name__ == "__main__":
    main()
