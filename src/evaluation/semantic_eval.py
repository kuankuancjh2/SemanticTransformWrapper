"""Latent-space semantics: the tests that decide whether the bottleneck
really encodes MEANING (not surface form).

All eval sentences / probe vocabularies are CONFIG-DRIVEN
(`eval:` section in config.yaml, previously hard-coded here). Two built-in
presets -- 'english' (default) and 'chinese' -- plus a JSON custom preset.

- semantic_tests: paraphrase / different-meaning / subject-swap / object-swap
  / surface-variation cosine comparisons
- intervention: interpolation, noise, dimension masking, token masking,
  token swapping, zero-latent dependency
- probes: linear surface probes (last-token identity / length bucket) vs
  semantic probes (subject / object) from frozen latents
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from ..config import Config
from ..generation import Generator
from ..utils.logging_utils import get_logger

log = get_logger("semantic_eval")


# ------------------------------------------------------------------- presets
TEST_PRESETS: Dict[str, Dict[str, object]] = {
    "english": {
        "tests": {
            "paraphrase_same_meaning": ("Alice ate an apple.",
                                        "An apple was eaten by Alice.", "high"),
            "different_meaning": ("Alice ate an apple.",
                                  "Alice bought a car.", "low"),
            "subject_swap": ("Alice eats an apple.",
                             "Bob eats an apple.", "low-mid"),
            "object_swap": ("Alice eats an apple.",
                            "Alice eats a banana.", "low-mid"),
            "surface_variation": ("Alice ate an apple.",
                                  "Alice was eating an apple.", "high"),
            "surface_variation2": ("Alice ate an apple quickly.",
                                   "Alice quickly ate an apple.", "high"),
        },
        "intervention_a": "Alice eats an apple.",
        "intervention_b": "Alice eats a banana.",
        "probe_subjects": ["Alice", "Bob", "Tom", "Mary", "Mom", "Dad",
                           "the boy", "the girl"],
        "probe_objects": ["apple", "banana", "car", "book", "cup", "bread",
                          "key", "flower"],
    },
    "chinese": {
        "tests": {
            "paraphrase_same_meaning": ("小明吃了一个苹果。", "一个苹果被小明吃掉了。", "high"),
            "different_meaning": ("小明吃了一个苹果。", "小明买了一辆汽车。", "low"),
            "subject_swap": ("小明吃苹果", "小红吃苹果", "low-mid"),
            "object_swap": ("小明吃苹果", "小明吃香蕉", "low-mid"),
            "surface_variation": ("小明吃了一个苹果。", "小明正在吃一个苹果。", "high"),
            "surface_variation2": ("小明吃了一个苹果。", "一个苹果被小明吃了。", "high"),
        },
        "intervention_a": "小明吃苹果",
        "intervention_b": "小明吃香蕉",
        "probe_subjects": ["小明", "小红", "老师", "妈妈", "爷爷", "姐姐",
                           "男孩", "女孩"],
        "probe_objects": ["苹果", "香蕉", "汽车", "书", "水杯", "面包",
                          "钥匙", "花"],
    },
}


def get_eval_spec(eval_cfg, max_len: int) -> Dict[str, object]:
    """Resolve the eval spec: preset -> custom JSON -> explicit config fields.

    Returns {tests, intervention_a, intervention_b, probe_subjects,
    probe_objects, max_len, preset}.
    """
    preset = getattr(eval_cfg, "data_preset", "english")
    if preset == "custom":
        path = getattr(eval_cfg, "custom_tests_file", None)
        if not path:
            raise ValueError("eval.data_preset='custom' requires "
                             "eval.custom_tests_file")
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        spec.setdefault("tests", {})
        spec.setdefault("intervention_a", "")
        spec.setdefault("intervention_b", "")
        spec.setdefault("probe_subjects", [])
        spec.setdefault("probe_objects", [])
    else:
        if preset not in TEST_PRESETS:
            raise KeyError(f"unknown eval.data_preset '{preset}'; "
                           f"available: {sorted(TEST_PRESETS)} + 'custom'")
        base = TEST_PRESETS[preset]
        spec = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
                for k, v in base.items()}
        spec["tests"] = dict(base["tests"])
    # explicit config fields override the preset
    for key, attr in (("intervention_a", "intervention_prompt_a"),
                      ("intervention_b", "intervention_prompt_b"),
                      ("probe_subjects", "probe_subjects"),
                      ("probe_objects", "probe_objects")):
        val = getattr(eval_cfg, attr, None)
        if val:
            spec[key] = val
    spec["max_len"] = max_len
    spec["preset"] = preset
    return spec


# ------------------------------------------------------------------- helpers
@torch.no_grad()
def _semantic_of(model, tok, text: str, device: torch.device,
                 apply_noise: bool = False, max_len: int = 126) -> torch.Tensor:
    ids = [tok.bos_id] + tok.encode(text, max_len=max_len)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.ones_like(x, dtype=torch.bool)
    if hasattr(model, "encode_prompt"):  # Stage2Model: prompt -> core -> semantic
        z = model.encode_prompt(x, mask, apply_noise=apply_noise)
        z = model.transform(z)
    else:
        z = model.encode(x, mask, apply_noise=apply_noise)
    return z


@torch.no_grad()
def _cos(model, tok, a: str, b: str, device, max_len: int = 126) -> float:
    za = _semantic_of(model, tok, a, device, max_len=max_len).mean(dim=1)
    zb = _semantic_of(model, tok, b, device, max_len=max_len).mean(dim=1)
    return float(F.cosine_similarity(F.normalize(za, dim=-1),
                                     F.normalize(zb, dim=-1)).item())


# --------------------------------------------------------------------- tests
def run_semantic_tests(model, tok, device: torch.device,
                       spec: Optional[Dict[str, object]] = None,
                       max_len: int = 126) -> Dict[str, float]:
    """Canonical semantic comparisons, from the config-selected preset."""
    if spec is None:
        spec = get_eval_spec(
            type("E", (), {"data_preset": "english", "custom_tests_file": None})(),
            max_len)
    ml = spec.get("max_len", max_len)
    out = {name: _cos(model, tok, a, b, device, max_len=ml)
           for name, (a, b, _exp) in spec["tests"].items()}
    if "paraphrase_same_meaning" in out and "different_meaning" in out:
        out["contrast_margin"] = (out["paraphrase_same_meaning"]
                                  - out["different_meaning"])
    return out


@torch.no_grad()
def run_interventions(model, tok, device, gen: Generator,
                      spec: Optional[Dict[str, object]] = None,
                      prompt_a: Optional[str] = None,
                      prompt_b: Optional[str] = None) -> Dict[str, object]:
    """Latent intervention suite on the semantic interface (config-driven)."""
    if prompt_a is None:
        prompt_a = spec["intervention_a"] if spec else "Alice eats an apple."
    if prompt_b is None:
        prompt_b = spec["intervention_b"] if spec else "Alice eats a banana."
    ml = spec.get("max_len", 126) if spec else 126
    za = _semantic_of(model, tok, prompt_a, device, max_len=ml)
    zb = _semantic_of(model, tok, prompt_b, device, max_len=ml)
    res: Dict[str, object] = {"prompt_a": prompt_a, "prompt_b": prompt_b}

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


def run_probes(model, tok, triples, device, spec: Optional[Dict[str, object]] = None,
               max_items: int = 400) -> Dict[str, float]:
    """Linear probes on frozen latents.

    Semantic probe: predict subject / object word (from the config-selected
    probe vocabulary) from pooled latent. Surface probe: exact last token
    (word-form proxy) and length bucket (word-order proxy).

    Ideal research state: semantic probe acc HIGH, surface probe acc LOW.
    """
    subj = list(spec["probe_subjects"]) if spec else TEST_PRESETS["english"]["probe_subjects"]
    obj = list(spec["probe_objects"]) if spec else TEST_PRESETS["english"]["probe_objects"]
    ml = spec.get("max_len", 126) if spec else 126
    feats, s_lab, o_lab, surf_last, surf_len = [], [], [], [], []
    for p, t, _pt in triples[:max_items]:
        with torch.no_grad():  # latents are frozen inputs; the PROBE trains
            full = t if len(t) > len(p) else p
            z = _semantic_of(model, tok, full, device, max_len=ml).mean(dim=1)[0].cpu()
        feats.append(z)
        s_lab.append(next((i for i, w in enumerate(subj) if w in full), len(subj)))
        o_lab.append(next((i for i, w in enumerate(obj) if w in full), len(obj)))
        ids = tok.encode(full)
        surf_last.append(ids[-1] if ids else tok.unk_id)
        surf_len.append(min(4, len(ids) // 8))
    if len(feats) < 20 or not subj or not obj:
        return {"semantic_subject_acc": 0.0, "semantic_object_acc": 0.0,
                "surface_last_char_acc": 0.0, "surface_length_acc": 0.0,
                "note": ("not enough data / probe vocabulary mismatch -- "
                         "check eval.data_preset against your corpus language")}
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
