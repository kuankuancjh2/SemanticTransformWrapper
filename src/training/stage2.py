"""Stage 2 trainer (Zip-B): train the Semantic Core over messages data.

Two context-routing modes, selected by the CORE's capability (not by flags
scattered in the trainer):

- latent-memory cores (bihopfield; diffusion with diffusion_conditioned=true):
  history turns are encoded turn-by-turn and ingested into the core's
  PERSISTENT STATE ("equivalent context"); the stimulus is the FINAL USER
  TURN's latent, transformed with that state. For conditioned diffusion the
  condition is the encoder result of the full conversation.

- non-memory cores (mlp / global_mlp / transformer / conv / mamba /
  plain diffusion): the context is CONCATENATED into the prompt text
  (prompt_full = whole conversation) and encoded normally.

Stage-1 parts stay frozen; decode loss gradients flow through the frozen
decoder into the core.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..checkpoints import load_checkpoint, save_train_checkpoint
from ..config import Config, resolve_device
from ..generation import Generator
from ..losses import (latent_set_loss, latent_stats, reconstruction_loss,
                      stage2_total, token_accuracy)
from ..model import Stage2Model
from ..model.cores import build_semantic_core, core_forward
from ..utils.logging_utils import get_logger
from ..utils.seed import capture_random_state, restore_random_state, set_seed
from .common import (lr_lambda, make_loaders, save_preview, setup_amp,
                     tok_path)
from .distributed import (average_stats, cleanup_distributed, device_for,
                          init_distributed, is_main, sampler_epoch,
                          sync_gradients)

log = get_logger("stage2")


def core_is_memory(core) -> bool:
    return bool(getattr(core, "supports_memory", False))


def is_conditioned_diffusion(core) -> bool:
    return core.name == "diffusion" and getattr(core, "conditioned", False)


def build_memory_state(model: Stage2Model, core, batch, device, grad: bool = False):
    """Encode history turns and ingest them into the core's persistent state.

    Returns (state, cond): state for latent-memory cores ([B, depth, K, D] or
    None), cond = pooled encoder states of the full conversation for
    conditioned diffusion (None otherwise). History encoding is no_grad in
    Stage 2 (encoder frozen); Stage 3 passes grad=True.
    """
    state, cond = None, None
    hist_ids = batch["hist_ids"].to(device)
    if core.name == "diffusion" and getattr(core, "conditioned", False):
        _, enc = model.encode_prompt_full(batch["prompt"].to(device),
                                          batch["prompt_mask"].to(device))
        cond = enc
        return state, cond
    if not getattr(core, "ingests_history", False) or hist_ids.numel() == 0:
        return state, cond
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        hz = model.encode_prompt(hist_ids, batch["hist_mask"].to(device))
    counts = batch["hist_counts"].tolist()
    max_turns = max(counts) if counts else 0
    if max_turns == 0:
        return state, cond
    B = batch["prompt"].size(0)
    state = core.init_state(B, device, hz.dtype)
    # autograd-safe turn ingestion: gather + torch.where (no in-place scatter,
    # so Stage-3 gradients flow through every history ingestion)
    prefix = [sum(counts[:i]) for i in range(B)]
    for t in range(max_turns):
        keep = torch.tensor([1.0 if c > t else 0.0 for c in counts],
                            device=device, dtype=hz.dtype)
        idx = torch.tensor([prefix[i] + t if counts[i] > t else 0
                            for i in range(B)], device=device, dtype=torch.long)
        z_t = hz[idx] * keep.view(B, 1, 1)  # inactive rows: zero stimulus
        _, s_new = core_forward(core, z_t, state=state)
        state = torch.where(keep.view(B, 1, 1, 1) > 0, s_new.to(state.dtype), state)
    return state, cond


def forward_stage2(model: Stage2Model, core, batch, device):
    """Common stimulus pass -> (z_pred, z_target).

    Routing: cores that INGEST history into a persistent state
    (bihopfield) and conditioned diffusion get the memory/cond branch;
    everything else (incl. plain diffusion) gets the concat branch."""
    if getattr(core, "ingests_history", False) or is_conditioned_diffusion(core):
        state, cond = build_memory_state(model, core, batch, device)
        z_prompt = model.encode_prompt(batch["prompt_user"].to(device),
                                       batch["prompt_user_mask"].to(device))
        z_pred, _ = core_forward(core, z_prompt, state=state, cond=cond)
    else:  # concat mode: context lives in the prompt text
        z_prompt = model.encode_prompt(batch["prompt"].to(device),
                                       batch["prompt_mask"].to(device))
        z_pred, _ = core_forward(core, z_prompt)
    z_target = model.encode_target(batch["target"].to(device),
                                   batch["target_mask"].to(device))
    return z_pred, z_target


def validate_stage2(model: Stage2Model, loader, cfg: Config, device: torch.device,
                    pad_id: int) -> Dict[str, float]:
    model.eval()
    tot = {k: 0.0 for k in ("loss", "latent", "decode", "ppl", "acc", "dep")}
    n = 0
    for batch in loader:
        target = batch["target"].to(device)
        tmask = batch["target_mask"].to(device)
        tgt_in, labels = target[:, :-1], target[:, 1:]
        lbl_mask = tmask[:, 1:]
        z_pred, z_t = forward_stage2(model, model.core, batch, device)
        lset = latent_set_loss(z_pred, z_t, cfg.loss.latent_cosine_weight)
        logits = model.decode_logits(tgt_in, z_pred, lbl_mask)
        l_dec = reconstruction_loss(logits, labels, pad_id)
        loss = stage2_total(lset["latent"], l_dec, cfg.loss)
        z_zero = torch.zeros_like(z_pred)
        logits0 = model.decode_logits(tgt_in, z_zero, lbl_mask)
        l0 = reconstruction_loss(logits0, labels, pad_id)
        dep = (l0 - l_dec).detach()
        B = target.size(0)
        for k, v in (("loss", loss), ("latent", lset["latent"]),
                     ("decode", l_dec), ("dep", dep)):
            tot[k] += float(v) * B
        tot["ppl"] += math.exp(min(20.0, float(l_dec))) * B
        tot["acc"] += token_accuracy(logits, labels, pad_id) * B
        n += B
    model.train()
    stats = {k: v / max(1, n) for k, v in tot.items()}
    stats["n"] = float(n)  # weight for cross-rank averaging
    return stats


def train_stage2(cfg: Config, stage1_ckpt: str, resume: Optional[str] = None,
                 core_override: Optional[str] = None) -> str:
    set_seed(cfg.train.seed)
    rank, _world, local_rank, ddp = init_distributed()
    device = device_for(local_rank, cfg.train.device)
    if core_override:
        cfg.core.type = core_override
    # per-core checkpoint dir: different cores NEVER overwrite each other
    ckpt_dir = Path(cfg.train.checkpoint_dir) / "stage2" / cfg.core.type
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = (SummaryWriter(str(Path(cfg.train.log_dir) / f"stage2_{cfg.core.type}"))
              if is_main(rank) else None)

    ckpt = load_checkpoint(stage1_ckpt, map_location="cpu")
    s1_cfg = Config.from_dict(ckpt["config"])
    s1_model, _, tok = None, None, None
    from ..checkpoints import build_stage1_from_checkpoint
    s1_model, s1_cfg, tok = build_stage1_from_checkpoint(ckpt, device)
    s1_model.cfg = s1_cfg
    for k, v in dict(vocab_size=s1_cfg.model.vocab_size,
                     hidden_dim=s1_cfg.model.hidden_dim,
                     num_heads=s1_cfg.model.num_heads,
                     encoder_layers=s1_cfg.model.encoder_layers,
                     decoder_layers=s1_cfg.model.decoder_layers,
                     ffn_dim=s1_cfg.model.ffn_dim,
                     num_semantic_tokens=s1_cfg.model.num_semantic_tokens,
                     max_seq_len=s1_cfg.model.max_seq_len).items():
        setattr(cfg.model, k, v)

    core = build_semantic_core(cfg.core, d_model=cfg.model.hidden_dim,
                               num_semantic_tokens=cfg.model.num_semantic_tokens).to(device)
    model = Stage2Model(s1_model, core, frozen=True).to(device)
    model.train()

    opt = torch.optim.AdamW([p for p in core.parameters() if p.requires_grad],
                            lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    train_loader, val_loader, _ = make_loaders(cfg, tok, stage=2, ddp=ddp)
    total_steps = cfg.train.max_steps or cfg.train.epochs * max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda(cfg.train.warmup_steps, total_steps))
    setup_amp(device, cfg.train.amp)

    start_epoch, gstep, best = 0, 0, float("inf")
    if resume:
        r = load_checkpoint(resume, map_location=str(device))
        core.load_state_dict(r["core_state_dict"])
        if r.get("optimizer_state_dict"):
            opt.load_state_dict(r["optimizer_state_dict"])
        start_epoch, gstep = r["epoch"], r["step"]
        best = r.get("best_val_loss", float("inf"))
        if r.get("random_state"):
            restore_random_state(r["random_state"])
        log.info("Resumed stage2 core from %s", resume)

    gen = Generator(model, tok, s1_cfg.data.max_seq_len, cfg.gen, device)

    def save(tag: str, epoch: int, step: int, val: float):
        save_train_checkpoint(ckpt_dir, tag, model, opt, sched, epoch, step, val,
                              cfg, tok_path(cfg), None, stage=2,
                              core_type=cfg.core.type)

    stop = False
    for epoch in range(start_epoch, cfg.train.epochs):
        sampler_epoch([train_loader, val_loader], epoch)
        pbar = tqdm(train_loader, desc=f"stage2[{cfg.core.type}] epoch{epoch}",
                    ncols=110, disable=not is_main(rank))
        for batch in pbar:
            target = batch["target"].to(device)
            tmask = batch["target_mask"].to(device)
            tgt_in, labels = target[:, :-1], target[:, 1:]
            lbl_mask = tmask[:, 1:]

            z_pred, z_t = forward_stage2(model, core, batch, device)
            lset = latent_set_loss(z_pred, z_t, cfg.loss.latent_cosine_weight)
            logits = model.decode_logits(tgt_in, z_pred, lbl_mask)
            l_dec = reconstruction_loss(logits, labels, pad_id=tok.pad_id)
            loss = stage2_total(lset["latent"], l_dec, cfg.loss)

            opt.zero_grad(set_to_none=True)
            (loss / cfg.train.grad_accum).backward()
            # manual grad all-reduce: forward goes through module methods, so
            # plain DDP wrapping would not sync these gradients
            sync_gradients([p for p in core.parameters() if p.requires_grad])
            if cfg.train.grad_clip > 0:
                gnorm = float(torch.nn.utils.clip_grad_norm_(
                    core.parameters(), cfg.train.grad_clip))
            opt.step()
            sched.step()
            gstep += 1

            if writer is not None and gstep % cfg.train.log_interval == 0:
                writer.add_scalar("loss/train", loss.item(), gstep)
                writer.add_scalar("loss/latent", float(lset["latent"]), gstep)
                writer.add_scalar("loss/decode", float(l_dec), gstep)
                writer.add_scalar("lr", sched.get_last_lr()[0], gstep)
                if hasattr(core, "last_diag") and core.last_diag:
                    for k, v in core.last_diag.items():
                        writer.add_scalar(f"core_state/{k}", v, gstep)
                pbar.set_postfix(loss=f"{loss.item():.3f}",
                                 lat=f"{float(lset['latent']):.3f}",
                                 dec=f"{float(l_dec):.3f}")

            if gstep % cfg.train.val_interval == 0 or gstep == total_steps:
                vs = validate_stage2(model, val_loader, cfg, device, tok.pad_id)
                n_rank = vs.pop("n", 1.0)
                vs = average_stats(vs, n_rank)
                if writer is not None:
                    writer.add_scalar("loss/val", vs["loss"], gstep)
                    writer.add_scalar("loss/val_decode", vs["decode"], gstep)
                    writer.add_scalar("loss/val_latent", vs["latent"], gstep)
                    writer.add_scalar("perplexity", vs["ppl"], gstep)
                    writer.add_scalar("token_accuracy", vs["acc"], gstep)
                    writer.add_scalar("latent_dependency", vs["dep"], gstep)
                if is_main(rank):
                    log.info("val @%d: loss=%.3f decode=%.3f latent=%.3f ppl=%.2f dep=%.3f",
                             gstep, vs["loss"], vs["decode"], vs["latent"], vs["ppl"], vs["dep"])
                    save_preview(cfg, model, gen, gstep,
                                 str(Path(cfg.train.samples_dir) / f"stage2_{cfg.core.type}"))
                best_new = vs["loss"] < best
                best = min(best, vs["loss"])
                if is_main(rank):  # only rank 0 touches the filesystem
                    save("latest.pt", epoch, gstep, best)
                    if best_new:
                        save("best.pt", epoch, gstep, best)
                    if cfg.train.save_epoch_checkpoints:
                        save(f"epoch_{epoch}.pt", epoch, gstep, best)
            if cfg.train.max_steps and gstep >= cfg.train.max_steps:
                stop = True
                break
        if stop:
            break
    if writer is not None:
        writer.close()
    if is_main(rank):
        log.info("Stage 2 [%s] done. best=%.4f -> %s", cfg.core.type, best,
                 ckpt_dir / "best.pt")
    cleanup_distributed()
    return str(ckpt_dir / "best.pt")