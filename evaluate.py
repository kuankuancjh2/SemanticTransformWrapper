"""Full evaluation: val metrics, semantic tests, latent interventions, probes,
zero-latent dependency. Works for Stage-1 or Stage-2 checkpoints.

Eval sentences / probe vocabularies come from CONFIG (`eval:` section):
  eval.data_preset: english (default) | chinese | custom
  eval.custom_tests_file: JSON for preset=custom
CLI: --eval-preset / --custom-tests override the config.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from src.checkpoints import build_stage1_from_checkpoint, build_stage2_from_checkpoint
from src.config import resolve_device
from src.data import load_split
from src.evaluation import run_interventions, run_probes, run_semantic_tests
from src.evaluation.semantic_eval import get_eval_spec
from src.generation import Generator
from src.losses import latent_stats
from src.utils.logging_utils import setup_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="eval_report.json")
    ap.add_argument("--max-probe-items", type=int, default=400)
    ap.add_argument("--eval-preset", default=None,
                    choices=["english", "chinese", "custom"],
                    help="override eval.data_preset from the config")
    ap.add_argument("--custom-tests", default=None,
                    help="JSON file for --eval-preset custom")
    args = ap.parse_args()

    setup_logging("logs")
    device = resolve_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ckpt.get("stage") == 2:
        model, cfg, tok = build_stage2_from_checkpoint(ckpt, device)
    else:
        model, cfg, tok = build_stage1_from_checkpoint(ckpt, device)

    if args.eval_preset:
        cfg.eval.data_preset = args.eval_preset
    if args.custom_tests:
        cfg.eval.custom_tests_file = args.custom_tests
    spec = get_eval_spec(cfg.eval, cfg.data.max_seq_len)

    report = {"checkpoint": args.checkpoint, "eval_preset": spec["preset"]}

    # ---- semantic tests (config-selected sentence set)
    report["semantic_tests"] = run_semantic_tests(model, tok, device, spec=spec)

    # ---- latent statistics
    triples = load_split(Path(cfg.data.processed_dir) / "val.jsonl")
    zs = []
    for p, t, _ in triples[:256]:
        ids = [tok.bos_id] + tok.encode(t, max_len=cfg.data.max_seq_len - 2)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        m = torch.ones_like(x, dtype=torch.bool)
        with torch.no_grad():
            if hasattr(model, "encode_prompt"):
                z = model.encode_prompt(x, m)
                z = model.transform(z)
            else:
                z = model.encode(x, m, apply_noise=False)
        zs.append(z.squeeze(0).cpu())
    Z = torch.stack(zs).unsqueeze(0) if zs else torch.zeros(1, 1, cfg.model.hidden_dim)
    report["latent_stats"] = latent_stats(Z)

    # ---- generation + interventions (needs the generator)
    gen = Generator(model, tok, cfg.data.max_seq_len, cfg.gen, device)
    report["interventions"] = run_interventions(model, tok, device, gen, spec=spec)

    # ---- probes
    report["probes"] = run_probes(model, tok, triples, device, spec=spec,
                                  max_items=args.max_probe_items)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
