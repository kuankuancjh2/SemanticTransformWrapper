"""Stage 2 trainer: train a pluggable SemanticCore against frozen Stage-1 parts.

L = latent_weight * (pooled MSE + cosine_weight * cosine)   [set-level]
  + decode_weight * CE( frozen_Decoder(Core(z_prompt)), target )
Gradients of the decode term flow THROUGH the frozen decoder into the core.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..checkpoints import save_checkpoint, load_checkpoint, build_stage1_from_checkpoint
from ..config import Config, resolve_device
from ..generation import Generator
from ..losses import (latent_set_loss, latent_stats, reconstruction_loss,
                      stage2_total, token_accuracy)
from ..model import Stage2Model
from ..model.cores import build_semantic_core as build_core
from ..utils.logging_utils import get_logger
from ..utils.seed import restore_random_state, set_seed
from .common import lr_lambda, make_loaders, save_preview, setup_amp
from .stage1 import tok_path

log = get_logger("stage2")


def validate_stage2(model: Stage2Model, loader, cfg: Config, device: torch.device,
                    pad_id: int) -> Dict[str, float]:
    model.eval()
    tot = {k: 0.0 for k in ("loss", "latent", "decode", "ppl", "acc", "dep")}
    n = 0
    for batch in loader:
        prompt = batch["prompt"].to(device)
        pmask = batch["prompt_mask"].to(device)
        target = batch["target"].to(device)
        tmask = batch["target_mask"].to(device)
        tgt_in, labels = target[:, :-1], target[:, 1:]
        lbl_mask = tmask[:, 1:]
        out = model(prompt, pmask, target, tmask)
        z_pred, z_t = out["z_pred"], out["z_target"]
        lset = latent_set_loss(z_pred, z_t, cfg.loss.latent_cosine_weight)
        logits = model.decode_logits(tgt_in, z_pred, lbl_mask)
        l_dec = reconstruction_loss(logits, labels, pad_id)
        loss = stage2_total(lset["latent"], l_dec, cfg.loss)

        # latent dependency score: does the frozen decoder actually use z?
        z_zero = torch.zeros_like(z_pred)
        logits0 = model.decode_logits(tgt_in, z_zero, lbl_mask)
        l0 = reconstruction_loss(logits0, labels, pad_id)
        dep = (l0 - l_dec).detach()  # >0 means decoder relies on latent

        B = prompt.size(0)
        for k, v in (("loss", loss), ("latent", lset["latent"]),
                     ("decode", l_dec), ("dep", dep)):
            tot[k] += float(v) * B
        tot["ppl"] += math.exp(min(20.0, float(l_dec))) * B
        tot["acc"] += token_accuracy(logits, labels, pad_id) * B
        n += B
    model.train()
    return {k: v / max(1, n) for k, v in tot.items()}


def train_stage2(cfg: Config, stage1_ckpt: str, resume: Optional[str] = None,
                 core_override: Optional[str] = None) -> str:
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    ckpt_dir = Path(cfg.train.checkpoint_dir) / "stage2"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(Path(cfg.train.log_dir) / f"stage2_{cfg.core.type}"))

    if core_override:
        cfg.core.type = core_override

    # ---- frozen stage-1 parts
    ckpt = load_checkpoint(stage1_ckpt, map_location="cpu")
    s1_cfg = Config.from_dict(ckpt["config"])
    s1_model, _, tok = build_stage1_from_checkpoint(ckpt, device)
    s1_model.cfg = s1_cfg
    # the model architecture MUST match stage-1 weights: inherit model dims
    cfg.model.vocab_size = s1_cfg.model.vocab_size
    cfg.model.hidden_dim = s1_cfg.model.hidden_dim
    cfg.model.num_heads = s1_cfg.model.num_heads
    cfg.model.encoder_layers = s1_cfg.model.encoder_layers
    cfg.model.decoder_layers = s1_cfg.model.decoder_layers
    cfg.model.ffn_dim = s1_cfg.model.ffn_dim
    cfg.model.num_semantic_tokens = s1_cfg.model.num_semantic_tokens
    cfg.model.max_seq_len = s1_cfg.model.max_seq_len
    # keep training-time config for data/loader/schedule, stage-1 weights for model
    core = build_core(cfg.core, d_model=s1_cfg.model.hidden_dim,
                      num_semantic_tokens=s1_cfg.model.num_semantic_tokens)
    core = core.to(device)
    model = Stage2Model(s1_model, core).to(device)
    model.train()

    # only the core is optimized
    opt = torch.optim.AdamW(core.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    train_loader, val_loader, _ = make_loaders(cfg, tok, stage=2)
    total_steps = cfg.train.max_steps or cfg.train.epochs * max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda(cfg.train.warmup_steps, total_steps))
    setup_amp(device, cfg.train.amp)  # core is small; plain fp32 math is fine on CPU

    start_epoch, gstep, best = 0, 0, float("inf")
    if resume:
        r = torch.load(resume, map_location=device, weights_only=False)
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
        payload = {
            "model_state_dict": model.state_dict(),
            "core_state_dict": core.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "epoch": epoch, "step": step, "best_val_loss": val,
            "config": cfg.to_dict(), "tokenizer": tok_path(cfg),
            "random_state": None, "stage": 2, "core_type": cfg.core.type,
        }
        torch.save(payload, str(ckpt_dir / tag))

    stop = False
    for epoch in range(start_epoch, cfg.train.epochs):
        pbar = tqdm(train_loader, desc=f"stage2[{cfg.core.type}] epoch{epoch}", ncols=110)
        for batch in pbar:
            prompt = batch["prompt"].to(device)
            pmask = batch["prompt_mask"].to(device)
            target = batch["target"].to(device)
            tmask = batch["target_mask"].to(device)
            tgt_in, labels = target[:, :-1], target[:, 1:]
            lbl_mask = tmask[:, 1:]

            out = model(prompt, pmask, target, tmask)
            lset = latent_set_loss(out["z_pred"], out["z_target"],
                                   cfg.loss.latent_cosine_weight)
            logits = model.decode_logits(tgt_in, out["z_pred"], lbl_mask)
            l_dec = reconstruction_loss(logits, labels, pad_id=tok.pad_id)
            loss = stage2_total(lset["latent"], l_dec, cfg.loss)

            opt.zero_grad(set_to_none=True)
            (loss / cfg.train.grad_accum).backward()
            if cfg.train.grad_clip > 0:
                gnorm = torch.nn.utils.clip_grad_norm_(core.parameters(), cfg.train.grad_clip)
            opt.step()
            sched.step()
            gstep += 1

            if gstep % cfg.train.log_interval == 0:
                writer.add_scalar("loss/train", loss.item(), gstep)
                writer.add_scalar("loss/latent", float(lset["latent"]), gstep)
                writer.add_scalar("loss/decode", float(l_dec), gstep)
                writer.add_scalar("lr", sched.get_last_lr()[0], gstep)
                pbar.set_postfix(loss=f"{loss.item():.3f}",
                                 lat=f"{float(lset['latent']):.3f}",
                                 dec=f"{float(l_dec):.3f}")

            if gstep % cfg.train.val_interval == 0 or gstep == total_steps:
                vs = validate_stage2(model, val_loader, cfg, device, tok.pad_id)
                writer.add_scalar("loss/val", vs["loss"], gstep)
                writer.add_scalar("loss/val_decode", vs["decode"], gstep)
                writer.add_scalar("loss/val_latent", vs["latent"], gstep)
                writer.add_scalar("perplexity", vs["ppl"], gstep)
                writer.add_scalar("token_accuracy", vs["acc"], gstep)
                writer.add_scalar("latent_dependency", vs["dep"], gstep)
                log.info("val @%d: loss=%.3f decode=%.3f latent=%.3f ppl=%.2f dep=%.3f",
                         gstep, vs["loss"], vs["decode"], vs["latent"], vs["ppl"], vs["dep"])
                save_preview(cfg, model, gen, gstep,
                             str(Path(cfg.train.samples_dir) / f"stage2_{cfg.core.type}"))
                best_new = vs["loss"] < best
                best = min(best, vs["loss"])
                save("latest.pt", epoch, gstep, best)
                if best_new:
                    save("best.pt", epoch, gstep, best)
                save(f"epoch_{epoch}.pt", epoch, gstep, best)
            if cfg.train.max_steps and gstep >= cfg.train.max_steps:
                stop = True
                break
        if stop:
            break
    writer.close()
    log.info("Stage 2 [%s] done. best=%.4f -> %s", cfg.core.type, best, ckpt_dir / "best.pt")
    return str(ckpt_dir / "best.pt")
