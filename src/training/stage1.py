"""Stage 1 trainer: Language Autoencoder with semantic-bottleneck losses.

Supports an optional Semantic VAE (config.vae.enabled / --vae):
  L = L_recon + beta_eff * KL(q(z|x) || N(0,I)) + paraphrase/variance/covariance
  beta_eff = beta * min(1, step / kl_warmup_steps)        [KL annealing]
Posterior-collapse guards: free bits (per-dim KL floor), logvar clamping,
active-units monitoring. Eval/inference use the deterministic mu.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..checkpoints import save_checkpoint
from ..config import Config, resolve_device
from ..generation import Generator
from ..losses import (active_units, covariance_loss, kl_divergence_loss,
                      latent_stats, paraphrase_consistency_loss,
                      reconstruction_loss, stage1_total, token_accuracy,
                      variance_loss)
from ..model import Stage1Model
from ..utils.logging_utils import get_logger
from ..utils.seed import capture_random_state, restore_random_state, set_seed
from .common import lr_lambda, make_loaders, save_preview, setup_amp

log = get_logger("stage1")


def validate(model: Stage1Model, loader, cfg: Config, device: torch.device,
             pad_id: int) -> Dict[str, float]:
    model.eval()
    tot = {k: 0.0 for k in ("loss", "recon", "paraphrase", "variance", "covariance",
                            "ppl", "acc", "kl")}
    latent_chunks, mu_chunks, lv_chunks = [], [], []
    n = 0
    with torch.no_grad():
        for batch in loader:
            prompt = batch["prompt"].to(device)
            pmask = batch["prompt_mask"].to(device)
            target = batch["target"].to(device)
            tmask = batch["target_mask"].to(device)
            tgt_in, labels = target[:, :-1], target[:, 1:]
            lbl_mask = tmask[:, 1:]
            out = model(prompt, pmask, tgt_in, lbl_mask, noise_std=0.0)
            l_recon = reconstruction_loss(out["logits"], labels, pad_id)
            z = out["semantic"]
            if cfg.vae.enabled:
                kl_loss, kl_raw = kl_divergence_loss(out["mu"], out["logvar"],
                                                     cfg.vae.free_bits)
            else:
                kl_loss = kl_raw = torch.zeros((), device=device)
            # paraphrase consistency applies to true paraphrase rows only
            is_para = (batch["ptype"] == 1).to(device)
            if is_para.any():
                l_para = paraphrase_consistency_loss(z[is_para], z[is_para])
            else:
                l_para = torch.zeros((), device=device)
            l_var = variance_loss(z, cfg.loss.variance_target)
            l_cov = covariance_loss(z)
            # validation reports the loss at the FULL beta (annealing is a
            # training-time schedule only)
            beta_val = cfg.vae.beta if cfg.vae.enabled else 0.0
            loss = l_recon + cfg.loss.paraphrase_weight * l_para + \
                cfg.loss.variance_weight * l_var + cfg.loss.covariance_weight * l_cov \
                + beta_val * kl_loss
            B = prompt.size(0)
            tot["loss"] += loss.item() * B
            tot["recon"] += l_recon.item() * B
            tot["paraphrase"] += float(l_para) * B
            tot["variance"] += float(l_var) * B
            tot["covariance"] += float(l_cov) * B
            tot["kl"] += float(kl_raw) * B
            tot["ppl"] += math.exp(min(20.0, l_recon.item())) * B
            tot["acc"] += token_accuracy(out["logits"], labels, pad_id) * B
            n += B
            # cap accumulated latents: the stats below only need a sample;
            # holding ALL validation latents grows memory with val size.
            if sum(c.shape[0] for c in latent_chunks) < 4096:
                latent_chunks.append(z.reshape(-1, z.size(-1)).cpu())
                if cfg.vae.enabled:
                    mu_chunks.append(out["mu"].reshape(-1, z.size(-1)).cpu())
                    lv_chunks.append(out["logvar"].reshape(-1, z.size(-1)).cpu())
    model.train()
    stats = {k: v / max(1, n) for k, v in tot.items()}
    allz = torch.cat(latent_chunks)[:2048]
    stats.update({f"latent/{k}": v for k, v in latent_stats(allz.unsqueeze(0)).items()})
    if cfg.vae.enabled and mu_chunks:
        stats["vae/active_units"] = float(active_units(
            torch.cat(mu_chunks), torch.cat(lv_chunks),
            cfg.vae.active_unit_threshold))
    return stats


def train_stage1(cfg: Config, resume: Optional[str] = None) -> str:
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    ckpt_dir = Path(cfg.train.checkpoint_dir) / "stage1"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(Path(cfg.train.log_dir) / "stage1"))

    tok, train_loader, val_loader = _data(cfg)
    cfg.model.vocab_size = tok.vocab_size
    model = Stage1Model(cfg).to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    total_steps = cfg.train.max_steps or cfg.train.epochs * max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda(cfg.train.warmup_steps, total_steps))
    scaler_amp = setup_amp(device, cfg.train.amp)

    start_epoch, gstep, best = 0, 0, float("inf")
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict"):
            opt.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            sched.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch, gstep = ckpt["epoch"], ckpt["step"]
        best = ckpt.get("best_val_loss", float("inf"))
        if ckpt.get("random_state"):
            restore_random_state(ckpt["random_state"])
        log.info("Resumed stage1 from %s (epoch %d, step %d)", resume, start_epoch, gstep)

    gen = Generator(model, tok, cfg.data.max_seq_len, cfg.gen, device)
    stop = False
    bad_vals = 0

    for epoch in range(start_epoch, cfg.train.epochs):
        pbar = tqdm(train_loader, desc=f"stage1 epoch{epoch}", ncols=110)
        for batch in pbar:
            prompt = batch["prompt"].to(device)
            pmask = batch["prompt_mask"].to(device)
            target = batch["target"].to(device)
            tmask = batch["target_mask"].to(device)
            tgt_in, labels = target[:, :-1], target[:, 1:]
            lbl_mask = tmask[:, 1:]

            out = model(prompt, pmask, tgt_in, lbl_mask)
            z = out["semantic"]
            l_recon = reconstruction_loss(out["logits"], labels, pad_id=tok.pad_id)
            if cfg.vae.enabled:
                kl_loss, kl_raw = kl_divergence_loss(out["mu"], out["logvar"],
                                                     cfg.vae.free_bits)
                beta_eff = cfg.vae.beta * min(
                    1.0, gstep / max(1, cfg.vae.kl_warmup_steps))
            else:
                kl_loss = kl_raw = torch.zeros((), device=device)
                beta_eff = 0.0
            is_para = (batch["ptype"] == 1)
            if is_para.any():
                # paraphrase rows carry (A, B) as prompt/target; the B side is a
                # TARGET (no gradient needed) -> encode under no_grad. This both
                # halves activation memory on paraphrase batches and matches the
                # loss semantics (consistency toward a fixed anchor).
                # sample=False forces the DETERMINISTIC mu for the anchor so the
                # consistency loss does not chase sampling noise.
                with torch.no_grad():
                    z_b = model.encode(target[is_para], tmask[is_para],
                                       apply_noise=False, sample=False)
                l_para = paraphrase_consistency_loss(z[is_para], z_b)
            else:
                l_para = torch.zeros((), device=device)
            l_var = variance_loss(z, cfg.loss.variance_target)
            l_cov = covariance_loss(z)
            loss = stage1_total({"reconstruction": l_recon, "paraphrase": l_para,
                                 "variance": l_var, "covariance": l_cov}, cfg.loss)
            if cfg.vae.enabled:
                loss = loss + beta_eff * kl_loss

            (loss / cfg.train.grad_accum).backward()
            if (gstep + 1) % cfg.train.grad_accum == 0:
                if cfg.train.grad_clip > 0:
                    # float() immediately: the returned tensor carries the whole
                    # autograd graph; keeping it as a tensor across iterations
                    # pins the previous step's graph in memory (real leak).
                    gnorm = float(torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.train.grad_clip))
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

            gstep += 1
            if gstep % cfg.train.log_interval == 0:
                ls = latent_stats(z.unsqueeze(0))
                writer.add_scalar("loss/train", loss.item(), gstep)
                writer.add_scalar("loss/reconstruction", l_recon.item(), gstep)
                writer.add_scalar("loss/semantic", float(l_para), gstep)
                writer.add_scalar("loss/variance", float(l_var), gstep)
                writer.add_scalar("loss/decorrelation", float(l_cov), gstep)
                if cfg.vae.enabled:
                    writer.add_scalar("loss/kl", float(kl_raw), gstep)
                    writer.add_scalar("loss/kl_weighted", float(kl_loss), gstep)
                    writer.add_scalar("vae/beta_eff", beta_eff, gstep)
                writer.add_scalar("lr", sched.get_last_lr()[0], gstep)
                writer.add_scalar("gradient_norm", gnorm, gstep)
                for k, v in ls.items():
                    writer.add_scalar(f"latent/{k}", v, gstep)
                pbar.set_postfix(loss=f"{loss.item():.3f}",
                                 recon=f"{l_recon.item():.3f}",
                                 kl=f"{float(kl_raw):.2f}")

            if gstep % cfg.train.val_interval == 0 or gstep == total_steps:
                vs = validate(model, val_loader, cfg, device, tok.pad_id)
                for k, v in vs.items():
                    writer.add_scalar(f"val/{k}" if not k.startswith("latent") else k, v, gstep)
                writer.add_scalar("loss/val", vs["loss"], gstep)
                writer.add_scalar("perplexity", vs["ppl"], gstep)
                writer.add_scalar("token_accuracy", vs["acc"], gstep)
                if cfg.vae.enabled:
                    writer.add_scalar("loss/val_kl", vs["kl"], gstep)
                extra = (f" kl={vs['kl']:.3f} au={vs.get('vae/active_units', 0.0):.0f}"
                         if cfg.vae.enabled else "")
                log.info("val @%d: loss=%.3f ppl=%.2f acc=%.3f latent_std=%.3f "
                         "cos=%.3f eff_rank=%.1f%s",
                         gstep, vs["loss"], vs["ppl"], vs["acc"], vs["latent/std"],
                         vs["latent/cos_sim"], vs["latent/eff_rank"], extra)
                save_preview(cfg, model, gen, gstep, cfg.train.samples_dir)
                is_best = vs["loss"] < best
                best = min(best, vs["loss"])
                save_checkpoint(ckpt_dir / "latest.pt", model, opt, sched, epoch,
                                gstep, best, cfg, tok_path(cfg),
                                capture_random_state(), {"stage": 1})
                if is_best:
                    save_checkpoint(ckpt_dir / "best.pt", model, opt, sched, epoch,
                                    gstep, best, cfg, tok_path(cfg),
                                    capture_random_state(), {"stage": 1})
                if cfg.train.save_epoch_checkpoints:
                    save_checkpoint(ckpt_dir / f"epoch_{epoch}.pt", model, opt, sched,
                                    epoch, gstep, best, cfg, tok_path(cfg),
                                    capture_random_state(), {"stage": 1})
                if cfg.train.patience is not None:
                    bad_vals = bad_vals + 1 if vs["loss"] > best else 0
                    if bad_vals >= cfg.train.patience:
                        log.info("Early stopping at step %d", gstep)
                        stop = True
                        break
            if cfg.train.max_steps and gstep >= cfg.train.max_steps:
                stop = True
                break
        if stop:
            break
    writer.close()
    log.info("Stage 1 done. best val loss=%.4f -> %s", best, ckpt_dir / "best.pt")
    return str(ckpt_dir / "best.pt")


def _data(cfg: Config):
    from .common import make_loaders as ml
    from ..tokenization import load_tokenizer
    tok = load_tokenizer(tok_path(cfg))
    tr, va, _ = ml(cfg, tok, stage=1)
    return tok, tr, va


def tok_path(cfg: Config) -> str:
    p = Path(cfg.data.cache_dir) / ("tokenizer_bpe.json" if cfg.data.tokenizer_type == "bpe"
                                    else "tokenizer_char.json")
    return str(p)
