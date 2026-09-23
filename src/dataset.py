"""Dataset + collate: sentence-level samples, correct PAD/BOS/EOS handling,
dynamic padding, padding-aware masks. Pair types (cont/para/qa/open) are
carried so paraphrase-consistency loss only applies to true paraphrases.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

Triple = Tuple[str, str, str]  # (prompt, target, ptype)


class TextPairDataset(Dataset):
    def __init__(self, triples: Sequence[Triple], tokenizer, max_seq_len: int) -> None:
        self.triples: List[Triple] = list(triples)
        self.tok = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> dict:
        prompt, target, ptype = self.triples[idx]
        limit = self.max_seq_len - 1
        p_ids = self.tok.encode(prompt, max_len=limit - 1)
        t_ids = self.tok.encode(target, max_len=limit - 1)
        p = [self.tok.bos_id] + p_ids + [self.tok.eos_id]
        t = [self.tok.bos_id] + t_ids + [self.tok.eos_id]
        return {
            "prompt_ids": torch.tensor(p, dtype=torch.long),
            "target_ids": torch.tensor(t, dtype=torch.long),
            "ptype": ptype,
        }


PTYPE_MAP = {"cont": 0, "para": 1, "qa": 2, "open": 3}


def collate_pairs(batch: List[dict], pad_id: int) -> dict:
    out: dict = {}
    for key, pkey, mkey in (("prompt_ids", "prompt", "prompt_mask"),
                            ("target_ids", "target", "target_mask")):
        seqs = [b[key] for b in batch]
        max_len = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)  # True = real
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = s
            mask[i, : len(s)] = True
        out[pkey] = ids
        out[mkey] = mask
    out["ptype"] = torch.tensor([PTYPE_MAP.get(b["ptype"], 0) for b in batch],
                                dtype=torch.long)
    return out


def save_split(triples: Sequence[Triple], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p, t, ptype in triples:
            f.write(json.dumps({"prompt": p, "target": t, "ptype": ptype},
                               ensure_ascii=False) + "\n")


def load_split(path: str | Path) -> List[Triple]:
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            triples.append((d["prompt"], d["target"], d.get("ptype", "cont")))
    return triples
