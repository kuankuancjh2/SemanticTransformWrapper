"""Full evaluation (Zip-B): val metrics, semantic tests, latent interventions,
probes, zero-latent dependency. val.jsonl is now MESSAGES-format conversations.

Eval sentences / probe vocabularies come from CONFIG (`eval:` section):
  eval.data_preset: english (default) | chinese | custom
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
from src.data import format_conversation, load_messages
from src.evaluation import run_interventions, run_probes, run_semantic_tests
from src.evaluation.semantic_eval import get_eval_spec
from src.generation import Generator
from src.losses import latent_stats
from src.utils.logging_utils import setup_logging


def conversations_to_triples(convos):
    """(context_text, final_target, 'conv') triples for latent stats/probes."""
    triples = []
    for c in convos:
        target = next((m["content"] for m in reversed(c["messages"])
                       if m["role"] == "assistant"), "")
        triples.append((format_conversation(c["messages"]), target, "conv"))
    return triples


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
    if ckpt.get("stage") in (2, 3):
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

    # ---- latent statistics (over conversation targets)
    convos = load_messages(Path(cfg.data.processed_dir) / "val.jsonl")
    triples = conversations_to_triples(convos)
    zs = []
    for _p, t, _pt in triples[:256]:
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

    # ---- probes (probe text = final assistant turn)
    probe_triples = [(t, t, "conv") for _p, t, _ in triples]
    report["probes"] = run_probes(model, tok, probe_triples, device, spec=spec,
                                  max_items=args.max_probe_items)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()