"""Shared trainer internals: loaders (Stage 1 pairs / Stage 2-3 messages),
external --data support, distributed samplers, tokenizer resolution,
optimizer/schedule, AMP, preview.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..config import Config
from ..memutils import effective_batch_size
from ..generation import Generator
from ..utils.logging_utils import get_logger

log = get_logger("train")


def tok_path(cfg: Config) -> str:
    p = Path(cfg.data.cache_dir) / ("tokenizer_bpe.json" if cfg.data.tokenizer_type == "bpe"
                                    else "tokenizer_char.json")
    return str(p)


# --------------------------------------------------------------- --data mode
def load_external_items(path: str | Path) -> List[dict]:
    """Load an external data file and normalize every entry to a
    messages-conversation. Accepts .json (list or single object) and
    .jsonl (one JSON object per line). Per-line forms:

      {"messages": [{"role": "user", "content": ...}, ...]}   standard
      {"prompt": ..., "target": ...}                          single exchange
        (also question/answer, input/output, src/tgt)
      {"text": "..."} / "plain string"                        user-only turn
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--data file not found: {p}")
    if p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        raw = data if isinstance(data, list) else [data]
    else:
        raw = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    items: List[dict] = []
    for ex in raw:
        c = _to_conversation(ex)
        if c:
            items.append(c)
    if not items:
        raise ValueError(f"no usable conversations in {p}")
    log.info("--data: loaded %d conversations from %s", len(items), p)
    return items


def _to_conversation(ex) -> Optional[dict]:
    if isinstance(ex, str):
        ex = {"text": ex}
    if not isinstance(ex, dict):
        return None
    if isinstance(ex.get("messages"), list) and ex["messages"]:
        msgs = [{"role": str(m.get("role", "user")),
                 "content": str(m.get("content", "")).strip()}
                for m in ex["messages"] if str(m.get("content", "")).strip()]
        return {"messages": msgs} if msgs else None
    p = ex.get("prompt") or ex.get("question") or ex.get("input") or ex.get("src")
    t = (ex.get("target") or ex.get("answer") or ex.get("response")
         or ex.get("output") or ex.get("tgt"))
    if p is not None and t is not None:
        return {"messages": [{"role": "user", "content": str(p).strip()},
                             {"role": "assistant", "content": str(t).strip()}]}
    text = ex.get("text")
    if text:
        return {"messages": [{"role": "user", "content": str(text).strip()}]}
    return None


def resolve_tokenizer(cfg: Config):
    """Cache-hit load, else train a fresh tokenizer (over --data texts when
    given, else the processed train split) and cache it."""
    from ..dataset import load_messages, utterances_of
    from ..tokenization import CharTokenizer, load_tokenizer

    p = Path(tok_path(cfg))
    if p.exists():
        return load_tokenizer(p)
    if cfg.train.data_path:
        texts = [u for c in load_external_items(cfg.train.data_path)
                 for u in utterances_of([c])]
    else:
        texts = [u for c in load_messages(Path(cfg.data.processed_dir) / "train.jsonl")
                 for u in utterances_of([c])]
    if not texts:
        raise FileNotFoundError("no tokenizer cache and no texts to build one -- "
                                "run prepare_data.py first or pass --data <file>")
    log.info("training a fresh tokenizer over %d texts", len(texts))
    if cfg.data.tokenizer_type == "bpe":
        try:
            from ..tokenization import BPETokenizer
            tok = BPETokenizer.train_from_texts(texts, cfg.data.bpe_vocab_size,
                                                save_path=p)
            tok.save(p)
            return tok
        except ImportError:
            log.warning("`tokenizers` not installed; falling back to char tokenizer")
            cfg.data.tokenizer_type = "char"
    tok = CharTokenizer.train_from_texts(texts)
    tok.save(tok_path(cfg))
    return tok


# ------------------------------------------------------------------ builders
def _pin(cfg: Config) -> bool:
    return (torch.cuda.is_available() if getattr(cfg.train, "pin_memory", "auto") == "auto"
            else cfg.train.pin_memory == "on")


def _build_loaders(tr_ds, va_ds, collate, cfg: Config, ddp: bool
                   ) -> Tuple[DataLoader, DataLoader]:
    common = dict(num_workers=cfg.train.num_workers, pin_memory=_pin(cfg))
    world = 1
    if ddp:
        import torch.distributed as _d
        world = _d.get_world_size()
    # auto_batch: cap the batch by ACTUALLY free GPU memory (activation proxy),
    # divided across DDP ranks; never grows the configured batch, CPU no-op
    bs = cfg.train.batch_size
    if getattr(cfg.train, "auto_batch", True):
        capped = effective_batch_size(cfg.train.batch_size, cfg.data.max_seq_len,
                                      cfg.model.hidden_dim,
                                      cfg.train.mem_safety_fraction,
                                      world if ddp else 1)
        if capped < bs:
            log.info("auto_batch: %d -> %d (free-VRAM cap, world=%d)", bs, capped, world)
            bs = capped
    per_rank = (len(tr_ds) + world - 1) // world
    # tiny external datasets: never let batch_size exceed the per-rank split,
    # and never drop the only (partial) batch -- drop_last=True with 0 full
    # batches would silently skip ALL training
    bs = max(1, min(bs, per_rank))
    drop_last = per_rank >= bs
    if ddp:
        from torch.utils.data.distributed import DistributedSampler
        ts = DistributedSampler(tr_ds, shuffle=True, seed=cfg.train.seed)
        vs = DistributedSampler(va_ds, shuffle=False)
        train_loader = DataLoader(tr_ds, batch_size=bs, sampler=ts,
                                  drop_last=drop_last, collate_fn=collate, **common)
        val_loader = DataLoader(va_ds, batch_size=cfg.train.batch_size, sampler=vs,
                                collate_fn=collate, **common)
    else:
        train_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True,
                                  drop_last=drop_last, collate_fn=collate, **common)
        val_loader = DataLoader(va_ds, batch_size=cfg.train.batch_size, shuffle=False,
                                collate_fn=collate, **common)
    return train_loader, val_loader


def _split_small(items: List, fraction: float = 0.2) -> Tuple[List, List]:
    """Tiny-data split: carve a val slice when there is enough, else reuse all
    (learning/memorization test on a hand-written toy file)."""
    if len(items) >= 8:
        k = max(1, int(len(items) * fraction))
        return items[k:], items[:k]
    return items, items


def make_loaders(cfg: Config, tok, stage: int, ddp: bool = False
                 ) -> Tuple[DataLoader, DataLoader, dict]:
    """stage=1 -> single-turn reconstruction pairs. stage in (2, 3) ->
    messages-format conversation loaders. `cfg.train.data_path` (--data)
    bypasses the processed splits entirely."""
    if cfg.data.max_seq_len > cfg.model.max_seq_len:
        raise ValueError(
            f"data.max_seq_len ({cfg.data.max_seq_len}) exceeds model.max_seq_len "
            f"({cfg.model.max_seq_len}) -- positional embeddings would overflow. "
            "Keep the two keys equal (all shipped configs do).")
    from ..dataset import (LazyMessages, LazyPairs, LazyTokbinPairs,
                           MessagesDataset, TextPairDataset, collate_messages,
                           collate_pairs, load_messages, load_pairs,
                           utterances_of)
    from ..memutils import effective_batch_size

    if cfg.train.data_path:  # ---- external data mode
        convos = load_external_items(cfg.train.data_path)
        meta = {"source": f"external:{cfg.train.data_path}"}
        if stage == 1:
            pairs = [(u, u, "ae") for c in convos for u in utterances_of([c])]
            train, val = _split_small(pairs)
            tr_ds = TextPairDataset(train, tok, cfg.data.max_seq_len)
            va_ds = TextPairDataset(val, tok, cfg.data.max_seq_len)
            tl, vl = _build_loaders(tr_ds, va_ds,
                                    lambda b: collate_pairs(b, tok.pad_id), cfg, ddp)
            return tl, vl, meta
        usable = [c for c in convos
                  if any(m["role"] == "assistant" for m in c["messages"])]
        if len(usable) < len(convos):
            log.info("--data: %d/%d conversations have no assistant turn; "
                     "skipped for stage %d", len(convos) - len(usable),
                     len(convos), stage)
        train, val = _split_small(usable)
        tr_ds = MessagesDataset(train, tok, cfg.data.max_seq_len)
        va_ds = MessagesDataset(val, tok, cfg.data.max_seq_len)
        tl, vl = _build_loaders(tr_ds, va_ds,
                                lambda b: collate_messages(b, tok.pad_id), cfg, ddp)
        return tl, vl, meta

    # ---- processed-split mode (prepare_data.py output)
    base = Path(cfg.data.processed_dir)
    pad = tok.pad_id
    if stage == 1:
        # fast path: pre-tokenized memmap pairs (no text in RAM, no re-tokenize)
        if (base / "stage1_train.tokbin").exists():
            tr_ds = LazyTokbinPairs(base / "stage1_train",
                                    max_len=cfg.data.max_seq_len, eos_id=tok.eos_id)
        else:
            tr_ds = TextPairDataset(LazyPairs(base / "stage1_train.jsonl"), tok,
                                    cfg.data.max_seq_len)
        if (base / "stage1_val.tokbin").exists():
            va_ds = LazyTokbinPairs(base / "stage1_val",
                                    max_len=cfg.data.max_seq_len, eos_id=tok.eos_id)
        else:
            va_ds = TextPairDataset(LazyPairs(base / "stage1_val.jsonl"), tok,
                                    cfg.data.max_seq_len)
        tl, vl = _build_loaders(tr_ds, va_ds, lambda b: collate_pairs(b, pad), cfg, ddp)
    else:
        # disk-backed lazy conversations: one line per __getitem__
        tr_ds = MessagesDataset(LazyMessages(base / "train.jsonl"), tok,
                                cfg.data.max_seq_len)
        va_ds = MessagesDataset(LazyMessages(base / "val.jsonl"), tok,
                                cfg.data.max_seq_len)
        tl, vl = _build_loaders(tr_ds, va_ds, lambda b: collate_messages(b, pad), cfg, ddp)
    meta = {}
    mp = Path(cfg.data.metadata_path)
    if mp.exists():
        meta = dict(json.loads(mp.read_text(encoding="utf-8")))
    return tl, vl, meta


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
