"""Latent-space semantics: the tests that decide whether the bottleneck
really encodes MEANING (not surface form).
- semantic_tests: paraphrase / different-meaning / subject-swap / object-swap
  / surface-variation cosine comparisons
- intervention: interpolation, noise, dimension masking, token masking,
  token swapping, zero-latent dependency
- probes: linear surface probes (next-char identity / word order proxy) vs
  semantic probes (subject / object / action / polarity) from frozen latents
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from ..checkpoints import build_stage1_from_checkpoint, load_checkpoint
from ..config import Config
from ..generation import Generator
from ..model.model import Stage1Model
from ..utils.logging_utils import get_logger
log = get_logger("semantic_eval")

@torch.no_grad()
def _semantic_of(model, tok, text: str, device: torch.device,
                 apply_noise: bool = False) -> torch.Tensor:
    ids = [tok.bos_id] + tok.encode(text, max_len=126)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.ones_like(x, dtype=torch.bool)
    if hasattr(model, "encode_prompt"):  # Stage2Model: prompt -> core -> semantic
        z = model.encode_prompt(x, mask, apply_noise=apply_noise)
        z = model.transform(z)
    else:
        z = model.encode(x, mask, apply_noise=apply_noise)
    return z

@torch.no_grad()
def _cos(model: Stage1Model, tok, a: str, b: str, device) -> float:
    za = _semantic_of(model, tok, a, device).mean(dim=1)
    zb = _semantic_of(model, tok, b, device).mean(dim=1)
    return float(F.cosine_similarity(F.normalize(za, dim=-1), F.normalize(zb, dim=-1)).item())


def run_semantic_tests(model: Stage1Model, tok, device: torch.device) -> Dict[str, float]:
    """Five canonical comparisons from the project spec. ENGLISH VERSION"""
    tests = {
        "paraphrase_same_meaning": ("Alice ate an apple.", "An apple was eaten by Alice.", "high"),
        "different_meaning": ("Alice ate an apple.", "Alice bought a car.", "low"),
        "subject_swap": ("Alice eats an apple.", "Bob eats an apple.", "low-mid"),
        "object_swap": ("Alice eats an apple.", "Alice eats a banana.", "low-mid"),
        "surface_variation": ("Alice ate an apple.", "Alice is eating an apple.", "high"),
        "surface_variation2": ("Alice ate an apple.", "An apple was eaten by Alice.", "high"),
    }
    out = {name: _cos(model, tok, a, b, device) for name, (a, b, _) in tests.items()}
    out["contrast_margin"] = out["paraphrase_same_meaning"] - out["different_meaning"]
    return out


@torch.no_grad()
def run_interventions(model: Stage1Model, tok, device, gen: Generator,
                      prompt_a: str = "Alice eats an apple.",
                      prompt_b: str = "Alice eats a banana.") -> Dict[str, object]:
    """Latent intervention suite on the semantic interface. ENGLISH"""
    za = _semantic_of(model, tok, prompt_a, device)
    zb = _semantic_of(model, tok, prompt_b, device)
    res: Dict[str, object] = {}
    # 1) interpolation sweep
    interps = {}
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        zm = (1 - alpha) * za + alpha * zb
        interps[f"alpha={alpha}"] = gen.generate(prompt_a, latent=zm)
    res["interpolation"] = interps
    # 2) latent noise sweep
    noise = {}
    for std in (0.1, 0.5, 1.0):
        zn = za + torch.randn_like(za) * std
        noise[f"std={std}"] = gen.generate(prompt_a, latent=zn)
    res["latent_noise"] = noise
    # 3) semantic-token masking (zero out half the slots)
    zm = za.clone()
    zm[:, zm.size(1) // 2:] = 0.0
    res["token_masking_half"] = gen.generate(prompt_a, latent=zm)
    # 4) semantic-token swapping between A and B
    zs = za.clone()
    zs[:, ::2] = zb[:, ::2]
    res["token_swapping"] = gen.generate(prompt_a, latent=zs)
    # 5) dimension masking
    zd = za.clone()
    zd[..., : zd.size(-1) // 2] = 0.0
    res["dim_masking_half"] = gen.generate(prompt_a, latent=zd)
    # 6) zero-latent dependency (the collapse detector)
    real = gen.generate(prompt_a, latent=za, greedy=True)
    zero = gen.generate(prompt_a, latent=torch.zeros_like(za), greedy=True)
    res["generate_real_latent"] = real
    res["generate_zero_latent"] = zero
    res["latent_dependency_note"] = (
        "if real == zero, the decoder is ignoring the semantic representation")
    return res


def run_probes(model, tok, triples, device, max_items: int = 400
               ) -> Dict[str, float]:
    """Linear probes on frozen latents. ENGLISH vocab
    Semantic probe: predict subject / object word (chosen from the fixed
    vocabularies used to generate the corpus) from pooled latent.
    Surface probe: predict exact last token id and the sentence length bucket.
    Ideal research state: semantic probe acc HIGH, surface probe acc LOW.
    """
    # English subject / object word list, replace Chinese
    subj = ["Alice", "Bob", "Teacher", "Mom", "Grandpa", "Sister", "Boy", "Girl"]
    obj = ["apple", "banana", "car", "book", "cup", "bread", "key", "flower"]

    feats, s_lab, o_lab, surf_last, surf_len = [], [], [], [], []
    for p, t, _pt in triples[:max_items]:
        with torch.no_grad():  # latents are frozen inputs; the PROBE trains
            full = t if len(t) > len(p) else p
            z = _semantic_of(model, tok, full, device).mean(dim=1)[0].cpu()
        feats.append(z)
        s_lab.append(next((i for i, w in enumerate(subj) if w in full), len(subj)))
        o_lab.append(next((i for i, w in enumerate(obj) if w in full), len(obj)))
        ids = tok.encode(full)
        surf_last.append(ids[-1] if ids else tok.unk_id)
        surf_len.append(min(4, len(ids) // 8))
    if len(feats) < 20:
        return {"semantic_subject_acc": 0.0, "semantic_object_acc": 0.0,
                "surface_last_char_acc": 0.0, "surface_length_acc": 0.0,
                "note": "not enough data"}
    X = torch.stack(feats)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)

    def probe_acc(labels: List[int], n_classes: int) -> float:
        y = torch.tensor(labels)
        keep = y < n_classes
        if keep.sum() < 20:
            return 0.0
        Xk, yk = X[keep], y[keep]
        ntr = int(0.8 * len(Xk))
        W = torch.zeros(Xk.size(1), n_classes, requires_grad=True)
        optw = torch.optim.Adam([W], lr=0.05)
        yone = F.one_hot(yk[:ntr], n_classes).float()
        for _ in range(150):
            optw.zero_grad()
            loss = F.cross_entropy(Xk[:ntr] @ W, yone.argmax(-1))
            loss.backward()
            optw.step()
        with torch.no_grad():
            pred = (Xk[ntr:] @ W).argmax(-1)
        return float((pred == yk[ntr:]).float().mean().item())

    return {
        "semantic_subject_acc": probe_acc(s_lab, len(subj)),
        "semantic_object_acc": probe_acc(o_lab, len(obj)),
        "surface_last_char_acc": probe_acc(surf_last, tok.vocab_size),
        "surface_length_bucket_acc": probe_acc(surf_len, 5),
    }
