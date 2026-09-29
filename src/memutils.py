"""Memory utilities: size every buffer from ACTUALLY available memory.

- available_ram_bytes(): psutil when installed, else /proc/meminfo
  (MemAvailable), else sysconf. MemAvailable accounts for reclaimable page
  cache -- the right basis on Linux.
- stream_flush_rows(): rows that fit in a flush buffer of
  available * mem_safety_fraction * buffer_share, using a per-row estimate.
- gpu_free_bytes(): torch.cuda.mem_get_info (None on CPU).
- effective_batch_size(): cap the configured batch so a full batch of
  activations (B * T * D fp32 proxy, x safety factor) fits in the free GPU
  memory share; falls back to the configured value on CPU.
"""
from __future__ import annotations

import os
from typing import Optional

import torch


def available_ram_bytes(fraction: float = 1.0) -> int:
    """(Really available RAM) * fraction. Never trusts total RAM."""
    avail: Optional[int] = None
    try:
        import psutil  # optional

        avail = int(psutil.virtual_memory().available)
    except Exception:
        pass
    if avail is None:
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        avail = int(line.split()[1]) * 1024
                        break
        except Exception:
            pass
    if avail is None:
        try:
            avail = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except Exception:
            avail = 512 * 1024 * 1024  # last-resort floor: 512 MB
    return int(avail * max(0.05, min(1.0, fraction)))


def stream_flush_rows(row_bytes: int, safety_fraction: float,
                      buffer_share: float = 0.05, floor: int = 256,
                      cap: int = 200_000) -> int:
    """How many rows of ~`row_bytes` fit in a write/encode flush buffer of
    (available RAM * safety_fraction * buffer_share)."""
    budget = available_ram_bytes(safety_fraction) * buffer_share
    return max(floor, min(cap, int(budget // max(1, row_bytes))))


def gpu_free_bytes() -> Optional[int]:
    if not torch.cuda.is_available():
        return None
    try:
        free, _total = torch.cuda.mem_get_info()
        return int(free)
    except Exception:
        return None


def effective_batch_size(cfg_batch: int, max_seq_len: int, hidden_dim: int,
                         safety_fraction: float, ddp_world: int = 1) -> int:
    """Cap the batch size by free GPU memory (per rank).

    Activation proxy per row: T * D * 4 bytes * 24 (layers/attn/MLP factor,
    conservative). Only ever SHRINKS the configured batch; CPU keeps the
    configured value (no CUDA to measure)."""
    free = gpu_free_bytes()
    if free is None:
        return cfg_batch
    per_row = max_seq_len * hidden_dim * 4 * 24
    cap = int(free * max(0.05, min(1.0, safety_fraction))) // max(1, per_row)
    return max(1, min(cfg_batch, max(1, cap)) // max(1, ddp_world) or 1)
