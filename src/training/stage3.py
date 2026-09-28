"""Stage 3 trainer (Zip-B): END-TO-END training.

prompt -> Encoder -> Bottleneck -> Semantic Core -> AR Decoder -> target,
ALL modules unfrozen, gradients flowing decoder -> core -> encoder
(BPTT is bounded by the core's per-ingest state detach).

Stability measures (per spec):
  - lower default LR (stage3.lr = 1e-4)
  - dedicated warmup (stage3.warmup_steps) + cosine schedule
  - gradient clipping (train.grad_clip), grad-norm logged to TensorBoard
  - auxiliary ENCODER losses retained: variance + covariance regularization
    (+ VAE KL when the VAE is on), with the Stage-1 loss weights
  - loss = L_recon + var/cov (+ KL) + latent alignment (pooled MSE+cosine
    between the core OUTPUT and the encoder latent of the TARGET text --
    keeps the semantic interface meaningful while everything moves)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..checkpoints import load_checkpoint
from ..config import Config, resolve_device
from ..generation import Generator
from ..losses import (covariance_loss, kl_divergence_loss, latent_set_loss,
                      latent_stats, reconstruction_loss, token_accuracy,
                      variance_loss)
from ..model import Stage2Model
from ..model.cores import build_semantic_core, core_forward
from ..utils.logging_utils import get_logger
from ..utils.seed import capture_random_state, restore_random_state, set_seed
from .common import lr_lambda, make_loaders, save_preview, setup_amp
from .stage1 import tok_path
from .stage2 import build_memory_state, core_is_memory, is_conditioned_diffusion

log = get_logger("stage3")


def forward_stage3(model: Stage2Model, batch, device, cfg: Config) -> Dict:
    """Full-graph pass. Returns dict with logits/z_pred/aux losses."""
    core = model.core
    target = batch["target"].to(device)
    tmask = batch["target_mask"].to(device)
    tgt_in, labels = target[:, :-1], target[:, 1:]
    lbl_mask = tmask[:, 1:]

    if getattr(core, "ingests_history", False) or is_conditioned_diffusion(core):
        # history ingestion WITH gradient (encoder unfrozen; per-turn state
        # detach bounds the graph length)
        state, cond = build_memory_state(model, core, batch, device, grad=True)
        z_prompt, enc = model.encode_prompt_full(batch["prompt_user"].to(device),
                                                 batch["prompt_user_mask"].to(device))
        if is_conditioned_diffusion(core):
            z_pred, _ = core_forward(core, z_prompt, cond=cond)
        else:
            z_pred, _ = core_forward(core, z_prompt, state=state)
    else:
        z_prompt, enc = model.encode_prompt_full(batch["prompt"].to(device),
                                                 batch["prompt_mask"].to(device))
        z_pred, _ = core_forward(core, z_prompt)

    logits = model.decode_logits(tgt_in, z_pred, lbl_mask)
    l_recon = reconstruction_loss(logits, labels, pad_id=0)
    # --- retained encoder auxiliary losses (on the stimulus latent)
    l_var = variance_loss(z_prompt, cfg.loss.variance_target)
    l_cov = covariance_loss(z_prompt)
    aux = l_var * cfg.loss.variance_weight + l_cov * cfg.loss.covariance_weight
    kl_raw = 0.0
    if cfg.vae.enabled and model.bottleneck.use_vae:
        # VAE KL from the stimulus distribution params (cheap head re-run on
        # the encoder states already computed above)
        use_user = getattr(core, "ingests_history", False) \
            and not is_conditioned_diffusion(core)
        stim_mask = batch["prompt_user_mask" if use_user else "prompt_mask"].to(device)
        _, mu_s, lv_s = model.bottleneck(enc, stim_mask, return_dist=True)
        kl_for_loss, kl_raw_t = kl_divergence_loss(mu_s, lv_s, cfg.vae.free_bits)
        aux = aux + cfg.vae.beta * kl_for_loss
        kl_raw = float(kl_raw_t)
    # latent alignment: core output vs the ENCODER latent of the target text
    z_tgt, _ = model.encode_prompt_full(target, tmask)
    lset = latent_set_loss(z_pred, z_tgt, cfg.loss.latent_cosine_weight)
    return {"logits": logits, "labels": labels, "z_prompt": z_prompt,
            "z_pred": z_pred, "recon": l_recon, "aux": aux, "align": lset["latent"],
            "kl_raw": kl_raw}


def validate_stage3(model: Stage2Model, loader, cfg: Config, device,
                    pad_id: int) -> Dict[str, float]:
    model.eval()
    tot = {k: 0.0 for k in ("loss", "recon", "aux", "align", "ppl", "acc")}
    n = 0
    with torch.no_grad():
        for batch in loader:
            out = forward_stage3(model, batch, device, cfg)
            loss = out["recon"] + out["aux"] + cfg.loss.latent_weight * out["align"]
            B = out["labels"].size(0)
            tot["loss"] += float(loss) * B
            tot["recon"] += float(out["recon"]) * B
            tot["aux"] += float(out["aux"]) * B
            tot["align"] += float(out["align"]) * B
            tot["ppl"] += math.exp(min(20.0, float(out["recon"]))) * B
            tot["acc"] += token_accuracy(out["logits"], out["labels"], pad_id) * B
            n += B
    model.train()
    return {k: v / max(1, n) for k, v in tot.items()}


def train_stage3(cfg: Config, stage1_ckpt: Optional[str] = None,
                 resume: Optional[str] = None,
                 core_override: Optional[str] = None) -> str:
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    ckpt_dir = Path(cfg.train.checkpoint_dir) / "stage3"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if core_override:
        cfg.core.type = core_override
    writer = SummaryWriter(str(Path(cfg.train.log_dir) / f"stage3_{cfg.core.type}"))

    # ---- assemble the FULL model (all modules), init from stage-1 weights
    if stage1_ckpt:
        from ..checkpoints import build_stage1_from_checkpoint
        ckpt = load_checkpoint(stage1_ckpt, map_location="cpu")
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
    else:
        from ..tokenization import load_tokenizer
        from ..model import Stage1Model
        s1_cfg = cfg
        tok = load_tokenizer(tok_path(cfg))
        cfg.model.vocab_size = tok.vocab_size
        s1_model = Stage1Model(cfg).to(device)

    core = build_semantic_core(cfg.core, d_model=cfg.model.hidden_dim,
                               num_semantic_tokens=cfg.model.num_semantic_tokens).to(device)
    model = Stage2Model(s1_model, core, frozen=False).to(device)
    model.train()  # EVERYTHING trains (no frozen override)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.stage3.lr,
                            weight_decay=cfg.train.weight_decay)
    train_loader, val_loader, _ = make_loaders(cfg, tok, stage=3)
    total_steps = cfg.stage3.max_steps or cfg.stage3.epochs * max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda(cfg.stage3.warmup_steps, total_steps))
    setup_amp(device, cfg.train.amp)

    start_epoch, gstep, best = 0, 0, float("inf")
    if resume:
        r = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(r["model_state_dict"])
        if r.get("optimizer_state_dict"):
            opt.load_state_dict(r["optimizer_state_dict"])
        start_epoch, gstep = r["epoch"], r["step"]
        best = r.get("best_val_loss", float("inf"))
        if r.get("random_state"):
            restore_random_state(r["random_state"])
        log.info("Resumed stage3 from %s", resume)

    gen = Generator(model, tok, cfg.data.max_seq_len, cfg.gen, device)

    def save(tag: str, epoch: int, step: int, val: float):
        torch.save({
            "model_state_dict": model.state_dict(),
            "core_state_dict": core.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "epoch": epoch, "step": step, "best_val_loss": val,
            "config": cfg.to_dict(), "tokenizer": tok_path(cfg),
            "random_state": capture_random_state(), "stage": 3,
            "core_type": cfg.core.type,
        }, str(ckpt_dir / tag))

    stop = False
    for epoch in range(start_epoch, cfg.stage3.epochs):
        pbar = tqdm(train_loader, desc=f"stage3[{cfg.core.type}] epoch{epoch}", ncols=110)
        for batch in pbar:
            out = forward_stage3(model, batch, device, cfg)
            loss = out["recon"] + out["aux"] + cfg.loss.latent_weight * out["align"]

            opt.zero_grad(set_to_none=True)
            (loss / cfg.train.grad_accum).backward()
            if cfg.train.grad_clip > 0:
                # float() immediately (tensor return pins the autograd graph)
                gnorm = float(torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip))
            opt.step()
            sched.step()
            gstep += 1

            if gstep % cfg.train.log_interval == 0:
                writer.add_scalar("loss/train", loss.item(), gstep)
                writer.add_scalar("loss/reconstruction", float(out["recon"]), gstep)
                writer.add_scalar("loss/aux_encoder", float(out["aux"]), gstep)
                writer.add_scalar("loss/align", float(out["align"]), gstep)
                writer.add_scalar("lr", sched.get_last_lr()[0], gstep)
                writer.add_scalar("gradient_norm", gnorm, gstep)
                ls = latent_stats(out["z_pred"].unsqueeze(0))
                for k, v in ls.items():
                    writer.add_scalar(f"latent/{k}", v, gstep)
                pbar.set_postfix(loss=f"{loss.item():.3f}",
                                 recon=f"{float(out['recon']):.3f}",
                                 gn=f"{gnorm:.2f}")

            if gstep % cfg.train.val_interval == 0 or gstep == total_steps:
                vs = validate_stage3(model, val_loader, cfg, device, tok.pad_id)
                for k, v in vs.items():
                    writer.add_scalar(f"val/{k}", v, gstep)
                log.info("val @%d: loss=%.3f recon=%.3f aux=%.3f align=%.3f ppl=%.2f acc=%.3f",
                         gstep, vs["loss"], vs["recon"], vs["aux"], vs["align"],
                         vs["ppl"], vs["acc"])
                save_preview(cfg, model, gen, gstep,
                             str(Path(cfg.train.samples_dir) / f"stage3_{cfg.core.type}"))
                best_new = vs["loss"] < best
                best = min(best, vs["loss"])
                save("latest.pt", epoch, gstep, best)
                if best_new:
                    save("best.pt", epoch, gstep, best)
                if cfg.train.save_epoch_checkpoints:
                    save(f"epoch_{epoch}.pt", epoch, gstep, best)
            if cfg.stage3.max_steps and gstep >= cfg.stage3.max_steps:
                stop = True
                break
        if stop:
            break
    writer.close()
    log.info("Stage 3 [%s] done. best=%.4f -> %s", cfg.core.type, best, ckpt_dir / "best.pt")
    return str(ckpt_dir / "best.pt")