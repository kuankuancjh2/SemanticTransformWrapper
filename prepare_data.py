"""Data preparation pipeline.

First run: downloads (only if online) -> tokenizes -> splits -> writes
data/processed/*.jsonl + data/cache/* + metadata.json. Later runs of the
trainers NEVER touch the network. `--offline` forbids network entirely.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import Config, load_config
from src.corpus import build_bundled_corpus
from src.dataset import save_split
from src.tokenization import CharTokenizer, BPETokenizer
from src.utils.logging_utils import get_logger, setup_logging

log = get_logger("prepare_data")

def maybe_fetch_hf(cfg: Config, offline: bool) -> Optional[list]:
    if offline or not cfg.data.hf_dataset:
        return None

    try:
        from datasets import load_dataset

        # ------------------------------------------------------------
        # 优先尝试 Parquet / Arrow 数据
        #
        # roskoN/dailydialog 原仓库带有旧版 dailydialog.py，
        # 新版 datasets 不再支持 Dataset Script。
        #
        # 因此这里明确指定 refs/convert/parquet。
        # ------------------------------------------------------------

        ds = load_dataset(
            cfg.data.hf_dataset,
            revision="refs/convert/parquet",
            cache_dir=cfg.data.cache_dir,
        )

        out = []

        # ------------------------------------------------------------
        # DailyDialog:
        #
        # {
        #   "id": "...",
        #   "acts": [...],
        #   "emotions": [...],
        #   "utterances": [...]
        # }
        # ------------------------------------------------------------

        valid_splits = (
            "train",
            "validation",
            "test",
        )

        splits = [
            s for s in ds.keys()
            if s in valid_splits
        ]

        if not splits:
            splits = list(ds.keys())

        # 最多保留多少个历史 utterance
        MAX_CONTEXT_TURNS = 3

        for split in splits:

            for ex in ds[split]:

                dialog = ex.get("utterances", [])

                if not dialog:
                    continue

                if not isinstance(dialog, (list, tuple)):
                    continue

                # 清理文本
                dialog = [
                    str(x).strip()
                    for x in dialog
                    if x is not None and str(x).strip()
                ]

                if len(dialog) < 2:
                    continue

                # ----------------------------------------------------
                # 将 multi-turn dialogue 转成：
                #
                # context -> response
                #
                # 例如：
                #
                # A: Hello.
                # B: Hi.
                # A: How are you?
                # B: Fine.
                #
                # 变成：
                #
                # Hello.
                # -> Hi.
                #
                # Hello.
                # Hi.
                # -> How are you?
                #
                # Hi.
                # How are you?
                # -> Fine.
                #
                # ----------------------------------------------------

                for i in range(1, len(dialog)):

                    start = max(
                        0,
                        i - MAX_CONTEXT_TURNS,
                    )

                    context = "\n".join(
                        dialog[start:i]
                    ).strip()

                    response = dialog[i].strip()

                    if not context or not response:
                        continue

                    out.append(
                        (
                            context,
                            response,
                            "cont",
                        )
                    )

        if not out:
            log.warning(
                "HF dataset loaded but produced no usable dialogue pairs"
            )
            return None

        log.info(
            "Loaded %d dialogue pairs from HF dataset %s",
            len(out),
            cfg.data.hf_dataset,
        )

        return out

    except Exception as e:
        log.warning(
            "HF dataset unavailable (%s); "
            "falling back to bundled corpus",
            e,
        )
        return None
            
def prepare(cfg: Config, offline: bool = False) -> dict:
    raw_dir = Path(cfg.data.raw_dir)
    processed = Path(cfg.data.processed_dir)
    cache_dir = Path(cfg.data.cache_dir)
    for d in (raw_dir, processed, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------ 1. raw text acquisition
    hf = maybe_fetch_hf(cfg, offline)
    if hf is None:
        bundled = build_bundled_corpus(seed=cfg.data.seed)
        raw_path = raw_dir / "bundled_corpus.jsonl"
        save_split([t for split in bundled.values() for t in split], raw_path)
        triples = {s: list(bundled[s]) for s in ("train", "val", "test")}
        source = "bundled_synthetic"
    else:
        source = f"hf:{cfg.data.hf_dataset}"
        n = len(hf)
        n_val = max(1, int(n * cfg.data.val_split))
        n_test = max(1, int(n * cfg.data.test_split))
        triples = {
            "train": hf[: n - n_val - n_test],
            "val": hf[n - n_val - n_test : n - n_test],
            "test": hf[n - n_test :],
        }
        raw_path = raw_dir / "hf_corpus.jsonl"
        save_split([t for v in triples.values() for t in v], raw_path)

    # ------------------------------------------------ 2. tokenizer training
    tok_path = cache_dir / ("tokenizer_bpe.json" if cfg.data.tokenizer_type == "bpe"
                            else "tokenizer_char.json")
    if tok_path.exists():
        log.info("Tokenizer cache hit: %s", tok_path)
        from src.tokenization import load_tokenizer as _load_tok
        tok = _load_tok(tok_path)
    elif cfg.data.tokenizer_type == "bpe":
        try:
            texts = [t[1] for v in triples.values() for t in v]
            tok = BPETokenizer.train_from_texts(texts, cfg.data.bpe_vocab_size,
                                                save_path=tok_path)
        except ImportError:
            log.warning("`tokenizers` not installed; using char tokenizer")
            tok = None
        if tok is None:
            tok = CharTokenizer.train_from_texts([t[1] for v in triples.values() for t in v])
            tok_path = cache_dir / "tokenizer_char.json"
            tok.save(tok_path)
        else:
            tok_path = cache_dir / "tokenizer_bpe.json"
            tok.save(tok_path)
    else:
        tok = CharTokenizer.train_from_texts([t[1] for v in triples.values() for t in v])
        tok.save(tok_path)

    # ------------------------------------------------ 3. processed splits
    for name in ("train", "val", "test"):
        save_split(triples[name], processed / f"{name}.jsonl")

    # Stage-1 autoencoding view of train (prompt==target) + paraphrase pairs
    s1 = [(p, p, "ae") for p, _, _ in triples["train"]] + \
         [(p, t, "para") for p, t, pt in triples["train"] if pt == "para"]
    save_split(s1, processed / "stage1_train.jsonl")
    s1v = [(p, p, "ae") for p, _, _ in triples["val"]]
    save_split(s1v, processed / "stage1_val.jsonl")

    # ------------------------------------------------ 4. metadata
    all_texts = [t for v in triples.values() for p, t, _ in v]
    lengths = [len(tok.encode(t)) for t in all_texts[:2000]]
    meta = {
        "source": source,
        "offline_used": hf is None,
        "train_size": len(triples["train"]),
        "val_size": len(triples["val"]),
        "test_size": len(triples["test"]),
        "stage1_train_size": len(s1),
        "vocab_size": tok.vocab_size,
        "tokenizer": tok_path.as_posix(),
        "mean_len_tokens": sum(lengths) / max(1, len(lengths)),
        "max_len_tokens": max(lengths) if lengths else 0,
        "p99_len_tokens": sorted(lengths)[int(0.99 * len(lengths))] if lengths else 0,
    }
    with open(cfg.data.metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # inject vocab size into default config copy saved with data
    cfg.model.vocab_size = tok.vocab_size
    cfg.save("configs/config.yaml")
    log.info("Data ready: %s", json.dumps(meta, ensure_ascii=False))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare data + tokenizer (cached)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--offline", action="store_true",
                    help="never access the network")
    args = ap.parse_args()
    setup_logging("logs")
    cfg = load_config(args.config)
    prepare(cfg, offline=args.offline)


if __name__ == "__main__":
    main()
