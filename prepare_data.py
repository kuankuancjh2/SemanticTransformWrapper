"""Data preparation pipeline (Zip-B: STANDARD MESSAGES FORMAT).

First run: acquires conversations (bundled offline / optional HF), trains the
tokenizer over all utterances, and writes:
  data/processed/{train,val,test}.jsonl   -- messages conversations (Stage 2/3)
  data/processed/stage1_{train,val}.jsonl -- SINGLE-TURN reconstruction pairs
  data/cache/tokenizer_*  +  data/metadata.json
Later training runs never touch the network. `--offline` forbids network.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config
from src.corpus import PARAPHRASE_PAIRS, build_bundled_corpus
from src.dataset import save_messages, save_pairs, utterances_of
from src.tokenization import CharTokenizer
from src.utils.logging_utils import get_logger, setup_logging

log = get_logger("prepare_data")


def maybe_fetch_hf(cfg, offline: bool) -> Optional[List[Dict]]:
    """Optional HuggingFace dataset -> standard messages conversations.

    Every conversation is normalized to:

        {
            "messages": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ]
        }

    Supported dataset formats:
      - prompt / target
      - question / answer
      - dialog / dialogues as a list
      - src / tgt
      - T5 dialogue data stored as JSON inside `text`:
            {"text": "{\"src\": \"<speaker1>...\", \"tgt\": \"...\"}"}

    Multi-turn conversations are preserved as multi-turn messages.
    They are NOT converted into independent single-turn examples.
    """
    if offline or not cfg.data.hf_dataset:
        return None

    try:
        import json
        import re
        from datasets import load_dataset

        ds = load_dataset(
            cfg.data.hf_dataset,
            cache_dir=cfg.data.cache_dir,
        )

        out = []

        def clean(x) -> str:
            if x is None:
                return ""
            return str(x).strip()

        def parse_speaker_dialogue(src: str) -> List[Dict]:
            """Convert <speaker1>/<speaker2> dialogue into messages.

            <speaker1> -> user
            <speaker2> -> assistant
            """
            src = clean(src)
            if not src:
                return []

            # Normalize common variants just in case.
            src = src.replace("<speaker 1>", "<speaker1>")
            src = src.replace("<speaker 2>", "<speaker2>")

            parts = re.split(r"(<speaker[12]>)", src)

            messages = []
            current_role = None
            buffer = []

            for part in parts:
                if not part:
                    continue

                if part == "<speaker1>" or part == "<speaker2>":
                    # Flush previous utterance.
                    if current_role is not None:
                        content = clean("".join(buffer))
                        if content:
                            messages.append({
                                "role": current_role,
                                "content": content,
                            })

                    current_role = (
                        "user"
                        if part == "<speaker1>"
                        else "assistant"
                    )
                    buffer = []

                else:
                    buffer.append(part)

            # Flush final utterance.
            if current_role is not None:
                content = clean("".join(buffer))
                if content:
                    messages.append({
                        "role": current_role,
                        "content": content,
                    })

            return messages

        def parse_dialog_list(dialog) -> Optional[Dict]:
            """Convert a list of utterances into alternating messages."""
            if not isinstance(dialog, list):
                return None

            utterances = [
                clean(x)
                for x in dialog
                if clean(x)
            ]

            if len(utterances) < 2:
                return None

            messages = []

            for i, utterance in enumerate(utterances):
                messages.append({
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": utterance,
                })

            return {"messages": messages}

        def parse_t5_text(text) -> Optional[Dict]:
            """Parse JSON stored inside the dataset's `text` field."""
            text = clean(text)

            if not text:
                return None

            try:
                obj = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                return None

            if not isinstance(obj, dict):
                return None

            src = obj.get("src")
            tgt = obj.get("tgt")

            if src is None or tgt is None:
                return None

            src = clean(src)
            tgt = clean(tgt)

            if not src or not tgt:
                return None

            # Multi-turn source.
            if "<speaker1>" in src or "<speaker2>" in src:
                messages = parse_speaker_dialogue(src)

                if not messages:
                    return None

                # tgt is the response to the entire dialogue context.
                messages.append({
                    "role": "assistant",
                    "content": tgt,
                })

                return {"messages": messages}

            # Single-turn source.
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": src,
                    },
                    {
                        "role": "assistant",
                        "content": tgt,
                    },
                ]
            }

        for split in ("train", "validation", "test"):
            if split not in ds:
                continue

            for ex in ds[split]:
                if not isinstance(ex, dict):
                    continue

                parsed = None

                # =========================================================
                # 1. T5-dialogue-pretrain-data
                #
                #    {
                #        "text":
                #            "{\"src\": \"<speaker1>...\", \"tgt\": \"...\"}"
                #    }
                # =========================================================
                if ex.get("text") is not None:
                    parsed = parse_t5_text(ex["text"])

                    if parsed is not None:
                        out.append(parsed)
                        continue

                # =========================================================
                # 2. Explicit src / tgt
                # =========================================================
                if ex.get("src") is not None and ex.get("tgt") is not None:
                    src = clean(ex["src"])
                    tgt = clean(ex["tgt"])

                    if not src or not tgt:
                        continue

                    if "<speaker1>" in src or "<speaker2>" in src:
                        messages = parse_speaker_dialogue(src)

                        if messages:
                            messages.append({
                                "role": "assistant",
                                "content": tgt,
                            })

                            out.append({
                                "messages": messages
                            })
                    else:
                        out.append({
                            "messages": [
                                {
                                    "role": "user",
                                    "content": src,
                                },
                                {
                                    "role": "assistant",
                                    "content": tgt,
                                },
                            ]
                        })

                    continue

                # =========================================================
                # 3. Generic prompt / target
                # =========================================================
                p = (
                    ex.get("prompt")
                    or ex.get("question")
                    or ex.get("input")
                )

                t = (
                    ex.get("target")
                    or ex.get("answer")
                    or ex.get("response")
                )

                if p is not None and t is not None:
                    # List prompt = multi-turn dialogue.
                    if isinstance(p, list):
                        messages = parse_dialog_list(p)

                        if messages is not None:
                            # If target exists, treat it as the next
                            # assistant response rather than discarding it.
                            target = clean(t)

                            if target:
                                messages["messages"].append({
                                    "role": "assistant",
                                    "content": target,
                                })

                            out.append(messages)

                    else:
                        p = clean(p)
                        t = clean(t)

                        if p and t:
                            out.append({
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": p,
                                    },
                                    {
                                        "role": "assistant",
                                        "content": t,
                                    },
                                ]
                            })

                    continue

                # =========================================================
                # 4. dialog / dialogues field
                # =========================================================
                dialog = (
                    ex.get("dialog")
                    or ex.get("dialogue")
                    or ex.get("dialogues")
                )

                if isinstance(dialog, list):
                    parsed = parse_dialog_list(dialog)

                    if parsed is not None:
                        out.append(parsed)

                    continue

                # =========================================================
                # 5. Plain text fallback
                #
                # Do NOT invent an assistant response.
                # A single text field becomes a one-message conversation.
                # =========================================================
                if ex.get("text") is not None:
                    text = clean(ex["text"])

                    if text:
                        out.append({
                            "messages": [
                                {
                                    "role": "user",
                                    "content": text,
                                }
                            ]
                        })

        log.info(
            "Loaded %d conversations from HF dataset '%s'",
            len(out),
            cfg.data.hf_dataset,
        )

        if out:
            # Useful sanity check.
            total_turns = sum(
                len(c["messages"])
                for c in out
                if isinstance(c, dict) and "messages" in c
            )

            log.info(
                "HF messages: %.2f turns/conversation on average",
                total_turns / max(1, len(out)),
            )

        return out or None

    except Exception as e:
        log.warning(
            "HF dataset unavailable (%s); "
            "falling back to bundled corpus",
            e,
        )
        return None

def prepare(cfg, offline: bool = False) -> dict:
    raw_dir = Path(cfg.data.raw_dir)
    processed = Path(cfg.data.processed_dir)
    cache_dir = Path(cfg.data.cache_dir)
    for d in (raw_dir, processed, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------ 1. conversations
    convos = maybe_fetch_hf(cfg, offline)
    if convos is None:
        bundled = build_bundled_corpus(seed=cfg.data.seed)
        convos = bundled["train"] + bundled["val"] + bundled["test"]
        save_messages(convos, raw_dir / "bundled_corpus_messages.jsonl")
        n_val = len(bundled["val"])
        n_test = len(bundled["test"])
        splits = {"train": convos[: len(convos) - n_val - n_test],
                  "val": convos[len(convos) - n_val - n_test: len(convos) - n_test],
                  "test": convos[len(convos) - n_test:]}
        source = "bundled_synthetic_messages"
    else:
        source = f"hf:{cfg.data.hf_dataset}"
        n = len(convos)
        n_val = max(1, int(n * cfg.data.val_split))
        n_test = max(1, int(n * cfg.data.test_split))
        splits = {"train": convos[: n - n_val - n_test],
                  "val": convos[n - n_val - n_test: n - n_test],
                  "test": convos[n - n_test:]}

    # ------------------------------------------------ 2. tokenizer (all utterances)
    tok_path = cache_dir / ("tokenizer_bpe.json" if cfg.data.tokenizer_type == "bpe"
                            else "tokenizer_char.json")
    if tok_path.exists():
        log.info("Tokenizer cache hit: %s", tok_path)
        from src.tokenization import load_tokenizer as _load_tok
        tok = _load_tok(tok_path)
    elif cfg.data.tokenizer_type == "bpe":
        try:
            from src.tokenization import BPETokenizer
            texts = [u for c in splits["train"] for u in utterances_of([c])]
            tok = BPETokenizer.train_from_texts(texts, cfg.data.bpe_vocab_size,
                                                save_path=tok_path)
            tok.save(tok_path)
        except ImportError:
            log.warning("`tokenizers` not installed; using char tokenizer")
            tok = CharTokenizer.train_from_texts(
                [u for c in splits["train"] for u in utterances_of([c])])
            tok_path = cache_dir / "tokenizer_char.json"
            tok.save(tok_path)
    else:
        tok = CharTokenizer.train_from_texts(
            [u for c in splits["train"] for u in utterances_of([c])])
        tok.save(tok_path)

    # ------------------------------------------------ 3. processed splits
    for name in ("train", "val", "test"):
        save_messages(splits[name], processed / f"{name}.jsonl")

    # Stage 1: SINGLE-TURN reconstruction only. One pair per utterance
    # (utterance, utterance) + the dedicated paraphrase pairs (A -> B).
    s1 = [(u, u, "ae") for u in utterances_of(splits["train"])]
    s1 += [(a, b, "para") for a, b in PARAPHRASE_PAIRS]
    save_pairs(s1, processed / "stage1_train.jsonl")
    save_pairs([(u, u, "ae") for u in utterances_of(splits["val"])],
               processed / "stage1_val.jsonl")

    # ------------------------------------------------ 4. metadata
    utts = [u for c in splits["train"] for u in utterances_of([c])]
    lengths = [len(tok.encode(u)) for u in utts[:2000]]
    n_turns = [len(c["messages"]) for c in splits["train"]]
    meta = {
        "source": source,
        "format": "messages",
        "offline_used": convos is None,
        "train_conversations": len(splits["train"]),
        "val_conversations": len(splits["val"]),
        "test_conversations": len(splits["test"]),
        "stage1_train_pairs": len(s1),
        "mean_turns_per_conversation": sum(n_turns) / max(1, len(n_turns)),
        "vocab_size": tok.vocab_size,
        "tokenizer": tok_path.as_posix(),
        "mean_len_tokens": sum(lengths) / max(1, len(lengths)),
        "max_len_tokens": max(lengths) if lengths else 0,
    }
    with open(cfg.data.metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    cfg.model.vocab_size = tok.vocab_size
    cfg.save("configs/config.yaml")
    log.info("Data ready: %s", json.dumps(meta, ensure_ascii=False))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare messages-format data (cached)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--offline", action="store_true", help="never access the network")
    args = ap.parse_args()
    setup_logging("logs")
    cfg = load_config(args.config)
    prepare(cfg, offline=args.offline)


if __name__ == "__main__":
    main()