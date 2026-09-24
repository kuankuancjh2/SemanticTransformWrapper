"""Semantic Core ablation experiment: train Stage 2 with each core type and
compare validation loss / perplexity / latent dependency.

python ablation_semantic_core.py --stage1-checkpoint checkpoints/stage1/best.pt \
    --cores identity,mlp,transformer,bihopfield,random --max-steps 300
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from src.config import Config, load_config
from src.training.stage2 import train_stage2, validate_stage2
from src.utils.logging_utils import setup_logging
from src.data import load_split
from torch.utils.data import DataLoader

CORES = ["identity", "random", "mlp", "transformer", "bihopfield", "conv",
         "mamba", "diffusion"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--stage1-checkpoint", required=True)
    ap.add_argument("--cores", default=",".join(CORES))
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    setup_logging("logs")
    cores = [c for c in args.cores.split(",") if c]
    results = {}
    for core in cores:
        cfg = load_config(args.config)
        cfg.train.device = args.device
        cfg.train.max_steps = args.max_steps
        cfg.train.val_interval = max(50, args.max_steps // 3)
        cfg.core.type = core
        cfg.train.log_dir = f"logs/ablation_{core}"
        best = train_stage2(cfg, args.stage1_checkpoint, core_override=core)

        # final validation numbers for the report
        from src.checkpoints import load_checkpoint, build_stage2_from_checkpoint
        ck = load_checkpoint(best, map_location="cpu")
        model, rcfg, tok = build_stage2_from_checkpoint(ck, torch.device(args.device))
        from src.training.common import make_loaders
        _, val_loader, _ = make_loaders(rcfg, tok, stage=2)
        vs = validate_stage2(model, val_loader, rcfg, torch.device(args.device), tok.pad_id)
        results[core] = {
            "val_loss": vs["loss"], "val_decode": vs["decode"],
            "val_latent": vs["latent"], "perplexity": vs["ppl"],
            "latent_dependency": vs["dep"], "token_accuracy": vs["acc"],
            "checkpoint": best,
        }
        print(f"[ablation] {core}: {results[core]}")

    with open("ablation_semantic_core.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n=== Semantic Core Ablation ===")
    for c, r in results.items():
        print(f"{c:12s} val_loss={r['val_loss']:.3f} ppl={r['perplexity']:.2f} "
              f"latent_dep={r['latent_dependency']:.3f}")


if __name__ == "__main__":
    main()
