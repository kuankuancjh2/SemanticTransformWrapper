"""Tokenizer layer: char-level by default (zero dependencies, fully offline),
optional BPE via `tokenizers` when installed. All special-token bookkeeping
(PAD/BOS/EOS/UNK) lives here so every downstream module stays consistent.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]


class CharTokenizer:
    """Character-level tokenizer. Deterministic, offline, lossless for CJK."""

    def __init__(self, vocab: Optional[Dict[str, int]] = None) -> None:
        if vocab is None:
            vocab = {tok: i for i, tok in enumerate(SPECIALS)}
        self.vocab: Dict[str, int] = vocab
        self.inverse: Dict[int, str] = {i: s for s, i in vocab.items()}

    # ------------------------------------------------------------------ info
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.vocab[PAD]

    @property
    def bos_id(self) -> int:
        return self.vocab[BOS]

    @property
    def eos_id(self) -> int:
        return self.vocab[EOS]

    @property
    def unk_id(self) -> int:
        return self.vocab[UNK]

    # ------------------------------------------------------------------ build
    @classmethod
    def train_from_texts(cls, texts: List[str], min_freq: int = 1, **kw) -> "CharTokenizer":
        counter: Dict[str, int] = {}
        for t in texts:
            for ch in t:
                counter[ch] = counter.get(ch, 0) + 1
        vocab = {tok: i for i, tok in enumerate(SPECIALS)}
        for ch, freq in sorted(counter.items(), key=lambda x: -x[1]):
            if freq >= min_freq and ch not in vocab:
                vocab[ch] = len(vocab)
        return cls(vocab)

    # ------------------------------------------------------------------ encode
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False,
               max_len: Optional[int] = None) -> List[int]:
        ids = [self.vocab.get(ch, self.unk_id) for ch in text]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        specials = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        out = []
        for i in ids:
            if skip_special and i in specials:
                continue
            out.append(self.inverse.get(int(i), UNK))
        return "".join(out)

    def clean_text(self, text: str) -> str:
        return " ".join(text.split())

    # ------------------------------------------------------------------ IO
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "char", "vocab": self.vocab}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if d.get("type") != "char":
            raise ValueError(f"Expected char tokenizer, got {d.get('type')}")
        return cls(d["vocab"])


class BPETokenizer:
    """Thin wrapper over HuggingFace `tokenizers` BPE (optional dependency)."""

    def __init__(self, tokenizer_path: str | Path) -> None:
        from tokenizers import Tokenizer  # lazy import

        self.tk: "Tokenizer" = Tokenizer.from_file(str(tokenizer_path))
        self._pad = self.tk.token_to_id(PAD)
        self._bos = self.tk.token_to_id(BOS)
        self._eos = self.tk.token_to_id(EOS)
        self._unk = self.tk.token_to_id(UNK)

    @property
    def vocab_size(self) -> int:
        return self.tk.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self._pad

    @property
    def bos_id(self) -> int:
        return self._bos

    @property
    def eos_id(self) -> int:
        return self._eos

    @property
    def unk_id(self) -> int:
        return self._unk

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False,
               max_len: Optional[int] = None) -> List[int]:
        ids = self.tk.encode(text, add_special_tokens=False).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        specials = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        if skip_special:
            ids = [i for i in ids if i not in specials]
        return self.tk.decode(ids, skip_special_tokens=skip_special)

    def clean_text(self, text: str) -> str:
        return " ".join(text.split())

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.tk.save(str(path))

    @classmethod
    def train_from_texts(cls, texts: List[str] | Iterable[str], vocab_size: int = 4000, **kw) -> "BPETokenizer":
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

        tk = Tokenizer(models.BPE(unk_token=UNK))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
        
        # 核心修复点：special_tokens 必须是字符串列表 [str, ...]
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIALS,  # SPECIALS 即为 ["<pad>", "<bos>", "<eos>", "<unk>"]
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        
        tk.train_from_iterator(texts, trainer=trainer)
        
        # 保存并初始化
        tmp = Path(kw.get("save_path", "data/cache/tokenizer_bpe.json"))
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tk.save(str(tmp))
        return cls(tmp)

def load_tokenizer(path: str | Path):
    """Load whichever tokenizer flavor was saved at `path`."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    
    # 判断方式：只有 CharTokenizer 会在根节点保存 "type": "char"
    if d.get("type") == "char":
        return CharTokenizer(d["vocab"])
    else:
        # HuggingFace Tokenizer 保存的 JSON 格式
        return BPETokenizer(path)