"""Centralized dataclass configuration. Every hyperparameter lives here (mirrored in YAML).

All numeric hyperparameters (512 / 128 / 16 / lr ...) are managed ONLY through
these dataclasses and their YAML counterparts -- never hard-coded in modules.
"""
from __future__ import annotations

import dataclasses
import dataclasses as dc
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _dump_dc(obj: Any) -> Any:
    if dc.is_dataclass(obj):
        return {k: _dump_dc(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_dump_dc(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _dump_dc(v) for k, v in obj.items()}
    return obj


@dataclass
class ModelConfig:
    vocab_size: int = 0  # filled by tokenizer
    max_seq_len: int = 128
    hidden_dim: int = 512
    num_heads: int = 8
    encoder_layers: int = 3
    decoder_layers: int = 5
    ffn_dim: int = 2048
    dropout: float = 0.1
    num_semantic_tokens: int = 16
    # Layer-Split experiment: apply bottleneck after this many encoder layers
    # (None = after the full encoder stack).
    encoder_layer_split: Optional[int] = None
    # Ablation: if True, decoder gets an extra cross-attention over prompt
    # encoder states. Default must remain False (main experiment).
    decoder_sees_prompt: bool = False
    tie_embeddings: bool = True


@dataclass
class BottleneckConfig:
    # Gaussian noise on semantic tokens, training only (Stage 1).
    noise_std: float = 0.1
    use_noise_train: bool = True
    # Learned latent queries + cross attention only (no stride slicing).
    use_queries: bool = True


@dataclass
class SemanticCoreConfig:
    # mlp | transformer | bihopfield | identity | random
    type: str = "transformer"
    num_layers: int = 2
    num_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1
    mlp_hidden: int = 2048
    hopfield_beta: float = 1.0
    hopfield_steps: int = 3
    seed: int = 1234  # for the random core


@dataclass
class LossConfig:
    # Stage 1
    paraphrase_weight: float = 0.5
    variance_weight: float = 1.0
    covariance_weight: float = 1.0
    variance_target: float = 1.0
    # Stage 2: L = latent_weight * L_latent + decode_weight * L_decode
    latent_weight: float = 1.0
    latent_cosine_weight: float = 0.5
    decode_weight: float = 1.0


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    cache_dir: str = "data/cache"
    metadata_path: str = "data/metadata.json"
    max_seq_len: int = 128
    val_split: float = 0.1
    test_split: float = 0.1
    seed: int = 42
    # Optional HuggingFace dataset name (only used when online). Bundled
    # synthetic corpus is the zero-network fallback.
    hf_dataset: Optional[str] = None
    tokenizer_type: str = "char"  # char | bpe (bpe requires `tokenizers`)
    bpe_vocab_size: int = 4000


@dataclass
class TrainConfig:
    device: str = "auto"  # auto | cpu | cuda
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 20
    max_steps: Optional[int] = None  # overrides epochs when set
    warmup_steps: int = 500
    grad_accum: int = 1
    grad_clip: float = 1.0
    val_interval: int = 500  # optimizer steps between validations
    patience: Optional[int] = None  # early stopping (in # validations)
    amp: str = "auto"  # auto | bf16 | fp16 | off
    num_workers: int = 2
    seed: int = 3407
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    samples_dir: str = "samples"
    log_interval: int = 50
    # Fixed preview prompts used after every validation.
    preview_prompts: list = field(default_factory=lambda: [
        "今天天气很好，我决定",
        "小明拿起苹果，然后",
        "人工智能最重要的问题之一是",
        "如果一个物体从桌子上掉下来，那么",
        "我喜欢音乐，因为",
    ])


@dataclass
class GenerationConfig:
    max_gen_len: int = 48
    temperature: float = 0.7
    top_p: float = 0.9
    greedy: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    bottleneck: BottleneckConfig = field(default_factory=BottleneckConfig)
    core: SemanticCoreConfig = field(default_factory=SemanticCoreConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    gen: GenerationConfig = field(default_factory=GenerationConfig)

    # ------------------------------------------------------------------ IO
    def to_dict(self) -> Dict[str, Any]:
        return _dump_dc(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        cfg = cls()
        for sect in ("model", "bottleneck", "core", "loss", "data", "train", "gen"):
            if sect in d and d[sect]:
                sub = getattr(cfg, sect)
                valid = {f.name for f in dataclasses.fields(sub)}
                for k, v in d[sect].items():
                    if k not in valid:
                        raise KeyError(f"Unknown config key '{sect}.{k}'")
                    setattr(sub, k, v)
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        return cls.from_dict(d)


def load_config(path: Optional[str | Path] = None) -> Config:
    if path is None:
        default = Path("configs/config.yaml")
        if default.exists():
            return Config.from_yaml(default)
        return Config()
    return Config.from_yaml(path)


def resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
