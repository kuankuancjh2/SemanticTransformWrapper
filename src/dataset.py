"""Datasets (Zip-B).

Stage 1: TextPairDataset -- single-turn reconstruction pairs (utterance,
utterance). No multi-turn anywhere in Stage 1.

Stage 2/3: MessagesDataset over the STANDARD MESSAGES FORMAT
  {"messages": [{"role": ..., "content": ...}, ...]}
Each item exposes:
  - prompt_full : the whole conversation formatted as text (concat cores
                  encode this as the prompt)
  - prompt_user : ONLY the final user turn (the stimulus for latent-memory
                  cores; its semantic tokens are transformed with the state)
  - hist        : every turn BEFORE the final user turn (each encoded
                  separately; latent-memory cores ingest them turn-by-turn to
                  build the persistent state = "equivalent context")
  - target      : the final assistant turn (what the decoder must produce)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

Triple = Tuple[str, str, str]  # (prompt, target, ptype) -- stage 1


# ------------------------------------------------------------------ formatting
def format_turn(role: str, content: str) -> str:
    return f"User: {content}\n" if role == "user" else f"Assistant: {content}\n"


def format_conversation(messages: Sequence[Dict[str, str]]) -> str:
    return "".join(format_turn(m["role"], m["content"]) for m in messages)


# ------------------------------------------------------------------ stage 1
class TextPairDataset(Dataset):
    def __init__(self, triples: Sequence[Triple], tokenizer, max_seq_len: int) -> None:
        self.triples: List[Triple] = list(triples)
        self.tok = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt, target, ptype = self.triples[idx]
        limit = self.max_seq_len - 1
        p_ids = self.tok.encode(prompt, max_len=limit - 1)
        t_ids = self.tok.encode(target, max_len=limit - 1)
        p = [self.tok.bos_id] + p_ids + [self.tok.eos_id]
        t = [self.tok.bos_id] + t_ids + [self.tok.eos_id]
        return {"prompt_ids": torch.tensor(p, dtype=torch.long),
                "target_ids": torch.tensor(t, dtype=torch.long),
                "ptype": ptype}


PTYPE_MAP = {"cont": 0, "para": 1, "qa": 2, "open": 3, "ae": 0}


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


# ------------------------------------------------------------------ stage 2/3
class MessagesDataset(Dataset):
    def __init__(self, conversations: Sequence[Dict], tokenizer, max_seq_len: int) -> None:
        self.convos = list(conversations)
        self.tok = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.convos)

    def __getitem__(self, idx: int) -> Dict:
        msgs = self.convos[idx]["messages"]
        # target = final assistant turn; prompt_user = final user turn;
        # history = everything before the final user turn
        a_idx = max(i for i, m in enumerate(msgs) if m["role"] == "assistant")
        target = msgs[a_idx]["content"]
        u_idx = max(i for i in range(a_idx) if msgs[i]["role"] == "user")
        prompt_user = msgs[u_idx]["content"]
        history = [m for m in msgs[:u_idx]]
        limit = self.max_seq_len - 2
        enc = lambda s: [self.tok.bos_id] + self.tok.encode(s, max_len=limit) \
            + [self.tok.eos_id]
        return {
            "prompt_full": torch.tensor(enc(format_conversation(msgs)), dtype=torch.long),
            "prompt_user": torch.tensor(enc(prompt_user), dtype=torch.long),
            "target": torch.tensor(enc(target), dtype=torch.long),
            "hist": [torch.tensor(enc(format_turn(m["role"], m["content"])),
                                  dtype=torch.long) for m in history],
        }


def _pad(seqs: List[torch.Tensor], pad_id: int):
    max_len = max(s.size(0) for s in seqs)
    ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    for i, s in enumerate(seqs):
        ids[i, : s.size(0)] = s
        mask[i, : s.size(0)] = True
    return ids, mask


def collate_messages(batch: List[dict], pad_id: int) -> dict:
    out: dict = {}
    for key in ("prompt_full", "prompt_user", "target"):
        ids, mask = _pad([b[key] for b in batch], pad_id)
        pkey = "prompt" if key == "prompt_full" else key
        out[pkey] = ids
        out[pkey + "_mask"] = mask
    flat, owner = [], []
    for i, b in enumerate(batch):
        for t in b["hist"]:
            flat.append(t)
            owner.append(i)
    if flat:
        ids, mask = _pad(flat, pad_id)
    else:  # single-turn conversations: no history
        ids = torch.zeros(0, 1, dtype=torch.long)
        mask = torch.zeros(0, 1, dtype=torch.bool)
    out["hist_ids"], out["hist_mask"] = ids, mask
    out["hist_counts"] = torch.tensor([len(b["hist"]) for b in batch],
                                      dtype=torch.long)
    return out


# ----------------------------------------------------------------------- IO
def save_pairs(triples: Sequence[Triple], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p, t, ptype in triples:
            f.write(json.dumps({"prompt": p, "target": t, "ptype": ptype},
                               ensure_ascii=False) + "\n")


def load_pairs(path: str | Path) -> List[Triple]:
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            triples.append((d["prompt"], d["target"], d.get("ptype", "cont")))
    return triples


def save_messages(convos: Sequence[Dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in convos:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def load_messages(path: str | Path) -> List[Dict]:
    convos = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            convos.append(json.loads(line))
    return convos


def utterances_of(convos: Sequence[Dict]) -> List[str]:
    """Flatten conversations into individual utterances (stage-1-style text)."""
    return [m["content"] for c in convos for m in c["messages"]]