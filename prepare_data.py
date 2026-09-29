"""Data preparation pipeline (v4: STREAMING, memory-bounded).

Every stage processes line-by-line and writes through flush buffers; no
stage materializes the corpus in RAM. Buffer sizes come from the memory
ACTUALLY available (psutil / MemAvailable) x train.mem_safety_fraction.

Output layout (all consumed lazily by the training datasets):
  data/raw/<source>.jsonl                    raw conversations (messages fmt)
  data/processed/{train,val,test}.jsonl      conversations (Stage 2/3)
  data/processed/stage1_{train,val}.{tokbin,tokidx,ptypes.npy}
                                             pre-tokenized Stage-1 pairs,
                                             read back via np.memmap
  data/processed/stage1_{train,val}.jsonl    text pairs (reference / fallback)
  data/cache/tokenizer_*  data/metadata.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import Config, load_config
from src.corpus import PARAPHRASE_PAIRS, build_bundled_corpus
from src.dataset import format_conversation, utterances_of
from src.memutils import stream_flush_rows
from src.tokenization import CharTokenizer
from src.utils.logging_utils import get_logger, setup_logging

log = get_logger("prepare_data")

ROW_BYTES_RAW = 400     # rough bytes/line for flush sizing
ROW_BYTES_TOK = 128     # rough bytes/token-sequence for flush sizing


# ============================================================== HF parsing
# (user's multi-format HF parsing, preserved as pure functions)
_CJK = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_CJK_PUNCT = "，。！？；：、（）【】「」『』《》〈〉“”‘’"

def _clean(x) -> str:
    if x is None:
        return ""

    s = str(x)

    # 统一各种 Unicode 空白
    s = re.sub(r"[\u00A0\u2000-\u200B\u3000\t\r\n]+", " ", s)
    s = s.strip()

    if not s:
        return ""

    # 中文字符之间的错误空格
    s = re.sub(
        rf"([{_CJK}]) +(?=[{_CJK}])",
        r"\1",
        s,
    )

    # 中文字符与中文标点之间的错误空格
    s = re.sub(
        rf"([{_CJK}]) +(?=[{re.escape(_CJK_PUNCT)}])",
        r"\1",
        s,
    )

    s = re.sub(
        rf"([{re.escape(_CJK_PUNCT)}]) +(?=[{_CJK}])",
        r"\1",
        s,
    )

    # 中文标点之间不需要空格
    s = re.sub(
        rf"([{re.escape(_CJK_PUNCT)}]) +(?=[{re.escape(_CJK_PUNCT)}])",
        r"\1",
        s,
    )

    # 英文/数字 + 中文标点
    s = re.sub(
        rf"([A-Za-z0-9]) +(?=[{re.escape(_CJK_PUNCT)}])",
        r"\1",
        s,
    )

    # 中文标点 + 英文/数字
    s = re.sub(
        rf"([{re.escape(_CJK_PUNCT)}]) +(?=[A-Za-z0-9])",
        r"\1",
        s,
    )

    return s

def parse_speaker_dialogue(src: str) -> List[Dict]:
    """Convert <speaker1>/<speaker2> dialogue into messages."""
    src = _clean(src)
    if not src:
        return []
    src = src.replace("<speaker 1>", "<speaker1>")
    src = src.replace("<speaker 2>", "<speaker2>")
    parts = re.split(r"(<speaker[12]>)", src)
    messages, current_role, buffer = [], None, []
    for part in parts:
        if not part:
            continue
        if part in ("<speaker1>", "<speaker2>"):
            if current_role is not None:
                content = _clean("".join(buffer))
                if content:
                    messages.append({"role": current_role, "content": content})
            current_role = "user" if part == "<speaker1>" else "assistant"
            buffer = []
        else:
            buffer.append(part)
    if current_role is not None:
        content = _clean("".join(buffer))
        if content:
            messages.append({"role": current_role, "content": content})
    return messages


def parse_dialog_list(dialog) -> Optional[Dict]:
    """Convert a list of utterances into alternating messages."""
    if not isinstance(dialog, list):
        return None
    utterances = [_clean(x) for x in dialog if _clean(x)]
    if len(utterances) < 2:
        return None
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": u}
                for i, u in enumerate(utterances)]
    return {"messages": messages}


def parse_t5_text(text) -> Optional[Dict]:
    """Parse JSON stored inside the dataset's `text` field."""
    text = _clean(text)
    if not text:
        return None
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    src, tgt = obj.get("src"), obj.get("tgt")
    if src is None or tgt is None:
        return None
    src, tgt = _clean(src), _clean(tgt)
    if not src or not tgt:
        return None
    if "<speaker1>" in src or "<speaker2>" in src:
        messages = parse_speaker_dialogue(src)
        if not messages:
            return None
        messages.append({"role": "assistant", "content": tgt})
        return {"messages": messages}
    return {"messages": [{"role": "user", "content": src},
                         {"role": "assistant", "content": tgt}]}


def parse_example(ex) -> Optional[Dict]:
    """One HF example -> messages conversation. Same dispatch as the user's
    loop: T5-text > src/tgt > prompt/target > dialog list > plain text."""
    if not isinstance(ex, dict):
        return None
    if ex.get("text") is not None:
        parsed = parse_t5_text(ex["text"])
        if parsed is not None:
            return parsed
    if ex.get("src") is not None and ex.get("tgt") is not None:
        src, tgt = _clean(ex["src"]), _clean(ex["tgt"])
        if src and tgt:
            if "<speaker1>" in src or "<speaker2>" in src:
                messages = parse_speaker_dialogue(src)
                if messages:
                    messages.append({"role": "assistant", "content": tgt})
                    return {"messages": messages}
            else:
                return {"messages": [{"role": "user", "content": src},
                                     {"role": "assistant", "content": tgt}]}
        return None
    p = ex.get("prompt") or ex.get("question") or ex.get("input")
    t = ex.get("target") or ex.get("answer") or ex.get("response")
    if p is not None and t is not None:
        if isinstance(p, list):
            messages = parse_dialog_list(p)
            if messages is not None:
                target = _clean(t)
                if target:
                    messages["messages"].append({"role": "assistant", "content": target})
                return messages
            return None
        p, t = _clean(p), _clean(t)
        if p and t:
            return {"messages": [{"role": "user", "content": p},
                                 {"role": "assistant", "content": t}]}
        return None
    dialog = ex.get("dialog") or ex.get("dialogue") or ex.get("dialogues")
    if isinstance(dialog, list):
        return parse_dialog_list(dialog)
    if ex.get("text") is not None:
        text = _clean(ex["text"])
        if text:
            return {"messages": [{"role": "user", "content": text}]}
    return None


# ============================================================== streaming IO
class JsonlWriter:
    """Flush-buffered line writer. The buffer is sized from AVAILABLE RAM
    (safety fraction x share), never from corpus size."""

    def __init__(self, path: Path, row_bytes: int, safety: float, share: float = 0.05) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._f = open(path, "w", encoding="utf-8")
        self._buf: List[str] = []
        self._flush_rows = stream_flush_rows(row_bytes, safety, share)
        self.count = 0

    def write(self, obj) -> None:
        self._buf.append(json.dumps(obj, ensure_ascii=False) + "\n")
        self.count += 1
        if len(self._buf) >= self._flush_rows:
            self._flush()

    def _flush(self) -> None:
        if self._buf:
            self._f.writelines(self._buf)
            self._buf.clear()

    def close(self) -> None:
        self._flush()
        self._f.close()


class TokenBinWriter:
    """Append-only pre-tokenized store: one uint16/uint32 .tokbin (flat token
    stream) + int64 .tokidx offsets (N+1) + optional uint8 .ptypes.npy.
    Read back with np.memmap -- the trainer never re-tokenizes and never
    holds the corpus in RAM."""

    def __init__(self, base: Path, vocab_size: int, safety: float) -> None:
        self.bin_path = base.with_suffix(".tokbin")
        self.idx_path = base.with_suffix(".tokidx")
        self.ptypes_path = base.with_suffix(".ptypes.npy")
        self.dtype = np.uint16 if vocab_size < 65536 else np.uint32
        self._b = open(self.bin_path, "wb")
        self._offsets = [0]
        self._buf: List[np.ndarray] = []
        self._ptypes: List[int] = []
        self._flush_rows = stream_flush_rows(ROW_BYTES_TOK, safety, 0.05)
        self.count = 0  # number of stored sequences

    def append(self, seq: List[int], ptype: int = 0) -> None:
        self._buf.append(np.asarray(seq, dtype=self.dtype))
        self._offsets.append(self._offsets[-1] + len(seq))
        self._ptypes.append(ptype)
        self.count += 1
        if len(self._buf) >= self._flush_rows:
            self._flush()

    def _flush(self) -> None:
        if self._buf:
            self._b.write(np.concatenate(self._buf).tobytes())
            self._buf.clear()

    def close(self) -> None:
        self._flush()
        self._b.close()
        np.save(self.idx_path, np.asarray(self._offsets, dtype=np.int64))
        np.save(self.ptypes_path, np.asarray(self._ptypes, dtype=np.uint8))


def stream_hf_conversations(cfg: Config, offline: bool) -> Optional[Path]:
    """HF -> raw jsonl, STREAMING (never materializes the dataset).
    `hf_dataset: "file:<path>"` streams a LOCAL jsonl dump instead (no
    network) -- useful to re-split/re-tokenize large external files."""
    if offline or not cfg.data.hf_dataset:
        return None
    if cfg.data.hf_dataset.startswith("file:"):
        p = Path(cfg.data.hf_dataset[5:])
        if not p.exists():
            log.warning("file: dump not found: %s", p)
            return None
        return p
    try:
        from datasets import load_dataset
    except ImportError:
        log.warning("`datasets` not installed; using bundled corpus")
        return None
    try:
        try:
            ds = load_dataset(cfg.data.hf_dataset, streaming=True,
                              cache_dir=cfg.data.cache_dir)
        except Exception as e:  # some sets lack streaming configs
            log.warning("streaming unavailable (%s); map-style with cap %s",
                        e, cfg.data.max_conversations)
            ds = load_dataset(cfg.data.hf_dataset, cache_dir=cfg.data.cache_dir)
        raw = JsonlWriter(Path(cfg.data.raw_dir) / "hf_stream.jsonl",
                          ROW_BYTES_RAW, cfg.train.mem_safety_fraction)
        cap = cfg.data.max_conversations
        for split in ("train", "validation", "test"):
            if split not in ds:
                continue
            for ex in ds[split]:
                conv = parse_example(ex)
                if conv is not None and len(conv["messages"]) >= 1:
                    raw.write(conv)
                    if cap is not None and raw.count >= cap:
                        break
            if cap is not None and raw.count >= cap:
                break
        raw.close()
        if raw.count == 0:
            return None
        log.info("HF streaming done: %d conversations -> %s", raw.count, raw.path)
        return raw.path
    except Exception as e:
        log.warning("HF dataset unavailable (%s); using bundled corpus", e)
        return None


def iter_jsonl(path: Path) -> Iterator[dict]:
    """One line in RAM at a time."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ================================================================== prepare
def prepare(cfg: Config, offline: bool = False,
            save_to: Optional[str] = None) -> dict:
    safety = cfg.train.mem_safety_fraction
    raw_dir = Path(cfg.data.raw_dir)
    processed = Path(cfg.data.processed_dir)
    cache_dir = Path(cfg.data.cache_dir)
    for d in (raw_dir, processed, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ---- 1. raw conversations: streamed to disk, only the flush buffer in RAM
    hf_raw = stream_hf_conversations(cfg, offline)
    if hf_raw is not None:
        source = f"hf:{cfg.data.hf_dataset}"
        conv_iter = iter_jsonl(hf_raw)
        with open(hf_raw, "r", encoding="utf-8") as f:  # disk-based count pass
            n_total = sum(1 for _ in f)
    else:
        source = "bundled_synthetic_messages"
        bundled = build_bundled_corpus(seed=cfg.data.seed)
        n_total = sum(len(v) for v in bundled.values())
        conv_iter = iter([c for k in ("train", "val", "test") for c in bundled[k]])

    n_val = max(1, int(n_total * cfg.data.val_split))
    n_test = max(1, int(n_total * cfg.data.test_split))
    n_train = max(0, n_total - n_val - n_test)
    writers = {s: JsonlWriter(processed / f"{s}.jsonl", ROW_BYTES_RAW, safety)
               for s in ("train", "val", "test")}
    for i, conv in enumerate(conv_iter):
        if "messages" not in conv:  # normalize legacy {"prompt","target"} rows
            conv = {"messages": [{"role": "user", "content": str(conv.get("prompt", ""))},
                                 {"role": "assistant", "content": str(conv.get("target", ""))}]}
        split = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
        writers[split].write(conv)
    for w in writers.values():
        w.close()

    # ---- 2. tokenizer over a bounded SAMPLE of train utterances
    tok_path = cache_dir / ("tokenizer_bpe.json" if cfg.data.tokenizer_type == "bpe"
                            else "tokenizer_char.json")
    if tok_path.exists():
        log.info("Tokenizer cache hit: %s", tok_path)
        from src.tokenization import load_tokenizer
        tok = load_tokenizer(tok_path)
    else:
        texts: List[str] = []
        for conv in iter_jsonl(processed / "train.jsonl"):
            texts.extend(utterances_of([conv]))
            if len(texts) >= cfg.data.tokenizer_sample_texts:
                break
        log.info("training tokenizer over %d sampled texts", len(texts))
        if cfg.data.tokenizer_type == "bpe":
            try:
                from src.tokenization import BPETokenizer
                tok = BPETokenizer.train_from_texts(texts, cfg.data.bpe_vocab_size,
                                                    save_path=tok_path)
                tok.save(tok_path)
            except ImportError:
                log.warning("`tokenizers` not installed; char tokenizer instead")
                cfg.data.tokenizer_type = "char"
                tok = CharTokenizer.train_from_texts(texts)
                tok_path = cache_dir / "tokenizer_char.json"
                tok.save(tok_path)
        else:
            tok = CharTokenizer.train_from_texts(texts)
            tok.save(tok_path)

    # ---- 3. streaming tokenization: Stage-1 pairs -> memmap-able tokbin
    def encode(s: str) -> List[int]:
        return [tok.bos_id] + tok.encode(s, max_len=cfg.data.max_seq_len - 2) + [tok.eos_id]

    s1w = JsonlWriter(processed / "stage1_train.jsonl", ROW_BYTES_RAW, safety)
    s1t = TokenBinWriter(processed / "stage1_train", tok.vocab_size, safety)
    s1vw = JsonlWriter(processed / "stage1_val.jsonl", ROW_BYTES_RAW, safety)
    s1vt = TokenBinWriter(processed / "stage1_val", tok.vocab_size, safety)

    for split in ("train", "val"):
        writer, tbin = (s1w, s1t) if split == "train" else (s1vw, s1vt)
        for conv in iter_jsonl(processed / f"{split}.jsonl"):
            # Stage 1 = SINGLE-TURN reconstruction only: one (u, u) pair per utterance
            for u in utterances_of([conv]):
                writer.write({"prompt": u, "target": u, "ptype": "ae"})
                tbin.append(encode(u), ptype=0)
                tbin.append(encode(u), ptype=0)
    # dedicated paraphrase pairs (bounded list, appended before close)
    for a, b in PARAPHRASE_PAIRS:
        s1w.write({"prompt": a, "target": b, "ptype": "para"})
        s1t.append(encode(a), ptype=1)
        s1t.append(encode(b), ptype=1)
    s1w.close(); s1t.close(); s1vw.close(); s1vt.close()

    # ---- 4. metadata (single streaming pass, bounded token probes)
    n_turns, lens = [], []
    for conv in iter_jsonl(processed / "train.jsonl"):
        n_turns.append(len(conv["messages"]))
        if len(lens) < 2000:
            lens.append(len(tok.encode(utterances_of([conv])[0])))
    meta = {
        "source": source,
        "format": "messages",
        "offline_used": hf_raw is None,
        "train_conversations": n_train,
        "val_conversations": n_val,
        "test_conversations": n_test,
        "stage1_train_pairs": s1t.count // 2,
        "mean_turns_per_conversation": sum(n_turns) / max(1, len(n_turns)),
        "vocab_size": tok.vocab_size,
        "tokenizer": tok_path.as_posix(),
        "mean_len_tokens": sum(lens) / max(1, len(lens)),
        "max_len_tokens": max(lens) if lens else 0,
        "flush_rows_per_writer": stream_flush_rows(ROW_BYTES_RAW, safety),
        "tokenizer_sample_texts": cfg.data.tokenizer_sample_texts,
    }
    with open(cfg.data.metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    cfg.model.vocab_size = tok.vocab_size
    cfg.save(save_to or "configs/config.yaml")
    log.info("Data ready (streaming): %s", json.dumps(meta, ensure_ascii=False))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare messages-format data (streaming)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--offline", action="store_true", help="never access the network")
    args = ap.parse_args()
    setup_logging("logs")
    cfg = load_config(args.config)
    prepare(cfg, offline=args.offline, save_to=args.config)


if __name__ == "__main__":
    main()
