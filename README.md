# SemanticTransformWrapper

研究 **语义接口可插拔** 的语言建模工程：
`Language Encoder → Semantic Bottleneck → Semantic Core → AR Decoder`。
Semantic Core 是唯一可替换组件（9 种实现），Stage 2 冻结语言部件只训 core，
Stage 3 端到端整体训练。

## 安装

```bash
pip install -r requirements.txt   # torch numpy pyyaml matplotlib tqdm tensorboard
                                  # 可选: tokenizers(BPE) datasets(HF数据) scikit-learn(t-SNE)
```

## 快速开始

```bash
python prepare_data.py --offline          # 内置合成语料 → tokenizer → split → cache
python train_stage1.py                    # 语言自编码器（AE/VAE 由 vae.enabled 决定）
python train_stage2.py --stage1-checkpoint checkpoints/stage1/best.pt --core transformer
python generate.py --checkpoint checkpoints/stage2/transformer/best.pt --prompt "Hello"
python evaluate.py   --checkpoint checkpoints/stage2/transformer/best.pt --eval-preset english
python demo.py --once
```

## 用外部数据训练（--data）

自己写一个小数据文件，直接训练并观察模型是否学会它，**不需要跑 prepare_data**：

```bash
python train_stage1.py --data data/example_toy.jsonl --max-steps 500
python train_stage2.py --stage1-checkpoint checkpoints/stage1/best.pt \
    --core bihopfield --data data/example_toy.jsonl
```

文件格式 `.jsonl`（每行一个对象）或 `.json`（数组），支持四种行格式：

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"prompt": "...", "target": "..."}
{"text": "..."}
"plain string"
```

小数据会自动留 20% 做 validation（不足 8 条时全用作训练，val 复用同一份，
方便观察记忆效果）。也可在 YAML 里持久配置：`train.data_path`。

## 多卡训练（torchrun / DDP）

```bash
# 单机 2 卡（stage 2/3 同理，替换脚本名即可）
torchrun --standalone --nproc_per_node=2 train_stage1.py
# 指定 4 卡
torchrun --standalone --nproc_per_node=4 train_stage2.py \
    --stage1-checkpoint checkpoints/stage1/best.pt --core bihopfield
```

- Stage 1 走标准 DDP；Stage 2/3 前向经过模块方法（core 状态路由/记忆），
  梯度用手工 all-reduce 同步，等效平均。
- 每 rank 一个 GPU（`cuda:{local_rank}`），DistributedSampler 切分数据、
  每个 epoch `set_epoch`。
- **只有 rank 0** 写 checkpoint / TensorBoard / preview / run.log。
- 单卡与 CPU 完全不受影响（不设 torchrun 环境变量即为普通单进程）。

## Checkpoint 布局与压缩策略

不同的 core **分目录存放，永不互相覆盖**：

```
checkpoints/
├── stage1/                      # 语言部件（所有 core 共享的初始化来源）
│   ├── best.pt                  #   完整权重, 不压缩（下游 stage 初始化锚点）
│   ├── latest.pt.gz             #   完整状态含 optimizer, gzip（可精确续训）
│   └── epoch_N.pt.gz            #   仅权重 fp16 快照（无 optimizer）, gzip
├── stage2/
│   ├── transformer/{best.pt, latest.pt.gz, epoch_N.pt.gz}
│   ├── bihopfield/{...}         # 每种 core 一个子目录
│   └── global_mlp/{...}
└── stage3/
    └── transformer/{...}
```

- 空间策略（`train.compress_checkpoints: true` 默认开）：
  `best.pt` 保持原样不压缩；`latest` 与 `epoch_N` 走 gzip，且 epoch 快照
  丢弃 optimizer/scheduler/RNG、权重转 fp16（体积约降为 1/8~1/10）。
- 所有加载入口（`--resume`、`--stage1-checkpoint`、`generate/evaluate/
  demo/inspect_latent --checkpoint`）对 `.pt` 与 `.pt.gz` 透明。
- 恢复示例：`python train_stage2.py --core bihopfield --resume checkpoints/stage2/bihopfield/latest.pt.gz --stage1-checkpoint checkpoints/stage1/best.pt`

## 三个训练阶段

| 阶段 | 命令 | 训练内容 | 数据 |
|---|---|---|---|
| Stage 1 | `train_stage1.py` | Encoder + Bottleneck + Decoder（单对单重构；可选 VAE：μ/logvar + 重参数化 + KL 退火 + free bits） | 单轮句对 |
| Stage 2 | `train_stage2.py --stage1-checkpoint ... --core X` | 只训 Semantic Core（语言部件冻结；decode 梯度穿过冻结 decoder 回传） | messages 对话 |
| Stage 3 | `train_stage3.py --stage1-checkpoint ... --core X` | 端到端全部解冻（低 lr `stage3.lr=1e-4`、专属 warmup、grad clip + norm 监控、保留 encoder 辅助损失与 latent 对齐） | messages 对话 |

## Semantic Core 一览（`--core` / `core.type`）

| core | 特点 | 主要旋钮 |
|---|---|---|
| `transformer` | 双向 Transformer，默认基线 | num_layers / num_heads / ffn_dim |
| `mlp` | 逐 token MLP | mlp_depth / mlp_hidden / mlp_activation |
| `global_mlp` | 全 token 全连接（展平 K·D 过深层 MLP，~(K·D)² 参数） | mlp_depth / mlp_hidden |
| `bihopfield` | **持久状态**神经动力学：状态 `[B, depth, K, D]` 跨调用保持 = 等效上下文；`hopfield_steps` 个 tick 内部演化；token/depth 双轴全连接混合；tick/depth embedding；gated-delta 更新；collapse 诊断（`core_state/*`） | num_layers(=状态深度) / hopfield_steps / bi_detach_state |
| `conv` | 1D 卷积 | conv_kernel / conv_dilation / num_layers |
| `mamba` | 选择性 SSM（纯 PyTorch，免 CUDA 扩展） | ssm_d_state / ssm_d_conv / ssm_expand / num_layers |
| `diffusion` | 迭代去噪精炼（确定性 z→z）；`diffusion_conditioned: true` 时以**整段上下文的 encoder 输出为条件**（潜空间记忆模式之二） | diffusion_steps / diffusion_schedule |
| `identity` / `random` | 消融对照（恒等 / 冻结随机投影） | seed |

### 多轮上下文路由（Stage 2/3 自动选择，无需开关）

| core | 模式 | 上下文去向 |
|---|---|---|
| `bihopfield` | 潜空间记忆 | 逐轮编码 → 逐轮 ingest 进持久状态；刺激 = 最后一条 user |
| `diffusion`(conditioned) | 条件扩散 | 条件 = 对话 encoder 输出，对 latent 去噪 |
| 其余全部 | 拼接 | 整段对话文本送入 encoder |

## 数据

- `prepare_data.py [--offline]`：内置合成语料或 HF 数据集（你的
  `maybe_fetch_hf` 支持 prompt/target、dialog 列表、`<speaker1/2>`、T5 text
  内嵌 JSON 等格式）→ 标准 messages 格式 → tokenizer → split → cache。
- Stage 1 只用单轮句对（`stage1_train.jsonl`），多轮对话只进 Stage 2/3。
- `eval.data_preset`：语义测试句集 `english`（默认）/ `chinese` / `custom`
  （`--eval-preset` + `--custom-tests file.json`，JSON 含 tests /
  intervention_a,b / probe_subjects / probe_objects）。

## 实验与分析工具

```bash
python ablation_semantic_core.py --stage1-checkpoint checkpoints/stage1/best.pt \
    --cores identity,random,mlp,global_mlp,transformer,bihopfield,conv,mamba,diffusion
python inspect_latent.py --checkpoint ...   # PCA / t-SNE / covariance / cosine / norm
python evaluate.py --checkpoint ...         # 语义测试 + 干预 + probing + latent 依赖
tensorboard --logdir logs
```

关键监控：`latent/{std,norm,cos_sim,eff_rank}`、`vae/active_units`（=0 即塌缩）、
`latent_dependency`（zero-latent 对照）、`gradient_norm`、`core_state/*`。

## 配置

全部超参集中在 `configs/config.yaml`（另有 `config.lowmem.yaml` 低内存预设、
`config.smoke.yaml` CPU 冒烟预设）。新键：

```yaml
train:
  data_path: null             # 等效 --data；非 null 时跳过 processed splits
  compress_checkpoints: true  # 除 best.pt 外全部 gzip
```
