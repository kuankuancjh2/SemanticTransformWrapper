"""Stage 3 entry: END-TO-END training (prompt -> target, everything unfrozen).

python train_stage3.py [--stage1-checkpoint checkpoints/stage1/best.pt]
                        [--core transformer] [--resume ...]
Lower default LR (stage3.lr), dedicated warmup, grad clipping + grad-norm
logging, retained encoder auxiliary losses.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config
from src.training.stage3 import train_stage3
from src.utils.logging_utils import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--stage1-checkpoint", default=None,
                    help="initialize from stage-1 weights (recommended)")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--core", default=None,
                    choices=["mlp", "transformer", "bihopfield", "global_mlp",
                             "conv", "mamba", "diffusion", "identity", "random"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--val-interval", type=int, default=None)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg.train.device = args.device
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.lr:
        cfg.stage3.lr = args.lr
    if args.epochs:
        cfg.stage3.epochs = args.epochs
    if args.max_steps:
        cfg.stage3.max_steps = args.max_steps
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.val_interval is not None:
        cfg.train.val_interval = args.val_interval

    setup_logging(cfg.train.log_dir)
    train_stage3(cfg, stage1_ckpt=args.stage1_checkpoint, resume=args.resume,
                 core_override=args.core)


if __name__ == "__main__":
    main()
