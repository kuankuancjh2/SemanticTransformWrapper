"""Multi-GPU support (DistributedDataParallel via torchrun) + CPU fallback.

Launch:
    torchrun --standalone --nproc_per_node=2 train_stage1.py ...
    # multi-node: torchrun --nnodes=.. --nproc_per_node=.. --rdzv_endpoint=..

Design notes:
- Stage 1 wraps the model in DDP (its training path goes through forward()).
- Stage 2/3 forwards go through module METHODS (encode_prompt / core state
  routing / persistent memory), which bypasses DDP's reducer preparation;
  there we sync gradients MANUALLY after backward (sync_gradients).
- Only rank 0 saves checkpoints / writes TensorBoard / previews.
- Validation runs on every rank over a sharded val set; metrics are
  averaged across ranks (reduce_metrics).
- Single-process runs (no torchrun) are unaffected: every helper no-ops.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def init_distributed() -> tuple[int, int, int, bool]:
    """Init the process group when launched under torchrun. Returns
    (rank, world_size, local_rank, ddp_enabled). Safe on plain CPU runs."""
    rank = int(os.environ.get("RANK", "0") or 0)
    world = int(os.environ.get("WORLD_SIZE", "1") or 1)
    local_rank = int(os.environ.get("LOCAL_RANK", "0") or 0)
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size(), local_rank, True
    if not (dist.is_available() and world > 1):
        return 0, 1, local_rank, False
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    return dist.get_rank(), dist.get_world_size(), local_rank, True


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: Optional[int] = None) -> bool:
    if rank is None:
        rank = dist.get_rank() if dist.is_initialized() else 0
    return rank == 0


def device_for(local_rank: int, base: str = "auto") -> torch.device:
    """Per-rank device: cuda:{local_rank} under DDP+CUDA, else the usual
    resolve_device() logic (cpu / single cuda)."""
    if base == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        if base == "auto":
            return torch.device(f"cuda:{local_rank % max(1, torch.cuda.device_count())}")
        if base.startswith("cuda"):
            return torch.device(base)
        return torch.device("cpu")
    return torch.device("cpu")


def wrap_model(model: torch.nn.Module, ddp: bool, device: torch.device):
    """Stage-1 style: wrap for DDP when the training path uses forward()."""
    if not ddp:
        return model
    return DDP(model,
               device_ids=[device.index] if device.type == "cuda" else None,
               output_device=device.index if device.type == "cuda" else None)


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def sync_gradients(params: Sequence[torch.nn.Parameter]) -> None:
    """Manual gradient all-reduce (average) for Stage 2/3, whose forwards go
    through module methods instead of DDP.forward(). No-op when not
    distributed."""
    if not dist.is_initialized():
        return
    for p in params:
        if p.grad is None:
            continue
        dist.all_reduce(p.grad)
        p.grad /= dist.get_world_size()


def reduce_metrics(tot: Dict[str, float], n: int) -> Dict[str, float]:
    """Average validation metrics across ranks. No-op single-process."""
    if not dist.is_initialized():
        return {k: v / max(1, n) for k, v in tot.items()}
    keys = sorted(tot)
    dev = torch.device("cuda" if dist.get_backend() == "nccl" else "cpu")
    vals = torch.tensor([tot[k] for k in keys], dtype=torch.float64, device=dev)
    count = torch.tensor([float(n)], dtype=torch.float64, device=dev)
    dist.all_reduce(vals)
    dist.all_reduce(count)
    return {k: float(vals[i].item()) / max(1.0, float(count.item()))
            for i, k in enumerate(keys)}


def average_stats(stats: Dict[str, float], n: int) -> Dict[str, float]:
    """Sample-weighted average of per-rank validation stats (validate() already
    divided by its own n; weights = each rank's n). No-op single-process."""
    if not dist.is_initialized():
        return stats
    keys = sorted(k for k, v in stats.items() if isinstance(v, (int, float)))
    dev = torch.device("cuda" if dist.get_backend() == "nccl" else "cpu")
    vals = torch.tensor([float(stats[k]) * n for k in keys],
                        dtype=torch.float64, device=dev)
    w = torch.tensor([float(n)], dtype=torch.float64, device=dev)
    dist.all_reduce(vals)
    dist.all_reduce(w)
    total = max(1.0, float(w.item()))
    return {**stats, **{k: float(vals[i].item()) / total for i, k in enumerate(keys)}}


def sampler_epoch(loaders, epoch: int) -> None:
    """DistributedSampler needs set_epoch per epoch (otherwise every rank sees
    the same shuffling order every epoch)."""
    for loader in loaders:
        s = getattr(loader, "sampler", None)
        if hasattr(s, "set_epoch"):
            s.set_epoch(epoch)
