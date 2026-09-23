"""Stage 1 entry: python train_stage1.py [--config ...] [--resume ...]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config
from src.training.stage1 import train_stage1
from src.utils.logging_utils import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--val-interval", type=int, default=None)
    ap.add_argument("--noise-std", type=float, default=None)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg.train.device = args.device
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.lr:
        cfg.train.lr = args.lr
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.max_steps:
        cfg.train.max_steps = args.max_steps
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.val_interval is not None:
        cfg.train.val_interval = args.val_interval
    if args.noise_std is not None:
        cfg.bottleneck.noise_std = args.noise_std

    setup_logging(cfg.train.log_dir)
    train_stage1(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
