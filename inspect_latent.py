"""Latent visualization: PCA / t-SNE projection, cosine-similarity heatmap,
covariance heatmap, latent-norm histogram -> PNG files.

python inspect_latent.py --checkpoint checkpoints/stage1/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.checkpoints import build_stage1_from_checkpoint, build_stage2_from_checkpoint
from src.config import resolve_device
from src.data import load_split
from src.utils.logging_utils import setup_logging

OUT_DIR = Path("latent_vis")


def get_latents(model, tok, triples, device, max_items=256):
    zs, texts = [], []
    with torch.no_grad():
        for p, t, _ in triples[:max_items]:
            ids = [tok.bos_id] + tok.encode(t, max_len=126)
            x = torch.tensor([ids], dtype=torch.long, device=device)
            m = torch.ones_like(x, dtype=torch.bool)
            if hasattr(model, "encode_prompt"):
                z = model.transform(model.encode_prompt(x, m))
            else:
                z = model.encode(x, m, apply_noise=False)
            zs.append(z.squeeze(0).mean(dim=0).cpu())
            texts.append(t)
    return torch.stack(zs), texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-items", type=int, default=256)
    args = ap.parse_args()

    setup_logging("logs")
    OUT_DIR.mkdir(exist_ok=True)
    device = resolve_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ckpt.get("stage") == 2:
        model, cfg, tok = build_stage2_from_checkpoint(ckpt, device)
    else:
        model, cfg, tok = build_stage1_from_checkpoint(ckpt, device)

    triples = load_split(Path(cfg.data.processed_dir) / "val.jsonl")
    Z, texts = get_latents(model, tok, triples, device, args.max_items)
    X = Z.numpy()
    Zn = F.normalize(Z, dim=-1)

    # ---- 1. PCA
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pca = Xc @ Vt[:2].T
    plt.figure(figsize=(7, 6))
    plt.scatter(pca[:, 0], pca[:, 1], s=14, alpha=0.7)
    ev = S[:2] ** 2 / (S ** 2).sum()
    plt.title(f"Latent PCA (var explained: {ev[0]:.2f}, {ev[1]:.2f})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "latent_pca.png", dpi=140)
    plt.close()

    # ---- 2. t-SNE / UMAP if available
    try:
        from sklearn.manifold import TSNE

        emb = TSNE(n_components=2, init="pca", random_state=0).fit_transform(X)
        plt.figure(figsize=(7, 6))
        plt.scatter(emb[:, 0], emb[:, 1], s=14, alpha=0.7)
        plt.title("Latent t-SNE")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "latent_tsne.png", dpi=140)
        plt.close()
    except ImportError:
        print("sklearn not installed -> skip t-SNE (PCA already produced)")

    # ---- 3. covariance matrix
    Zc = Z - Z.mean(0, keepdim=True)
    cov = (Zc.T @ Zc) / max(1, Z.size(0) - 1)
    plt.figure(figsize=(7, 6))
    plt.imshow(cov.numpy(), cmap="RdBu_r", vmin=-np.abs(cov.numpy()).max(),
               vmax=np.abs(cov.numpy()).max())
    plt.colorbar()
    plt.title("Latent covariance (decorrelation diagnostic)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "latent_covariance.png", dpi=140)
    plt.close()

    # ---- 4. pairwise cosine similarity
    sim = (Zn @ Zn.T).numpy()
    plt.figure(figsize=(7, 6))
    plt.imshow(sim, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar()
    plt.title(f"Pairwise cosine (mean off-diag {sim[~np.eye(len(sim), dtype=bool)].mean():.3f})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "latent_similarity.png", dpi=140)
    plt.close()

    # ---- 5. latent norm distribution
    norms = Z.norm(dim=-1).numpy()
    plt.figure(figsize=(7, 5))
    plt.hist(norms, bins=30, alpha=0.8)
    plt.title(f"Latent norm distribution (mean {norms.mean():.2f})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "latent_norm.png", dpi=140)
    plt.close()

    print("saved:", [p.name for p in sorted(OUT_DIR.glob("*.png"))])


if __name__ == "__main__":
    main()
