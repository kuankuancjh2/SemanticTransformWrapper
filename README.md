# Semantic Core Research Codebase
#
# Language Encoder -> Semantic Bottleneck -> Semantic Core -> Semantic Bottleneck
#                                                        -> AR Language Decoder
#
# 研究假设：把「语言理解/表达」与「语义转换」彻底分离，让中间的 Semantic Core
# 成为一个独立、可替换的信息接口 [B, K, D]。未来可以换成 Transformer / MLP /
# BiHopfield / 其他神经动力学系统，而无需重新训练 Encoder/Decoder。

## 三阶段总览

```
Stage 1  Language Autoencoder（训练 Encoder + Bottleneck + Decoder）
  sentence -> Encoder -> [B,K,D] -> AR Decoder -> sentence
  （Decoder 只见 semantic tokens + 已生成 token，绝不见原始 prompt/encoder states）

Stage 2  Semantic Core（冻结 Stage-1 全部，只训 Core）
  prompt -> frozen Encoder -> z_A -> CORE -> ẑ -> frozen Decoder(AR) -> target
  L = L_latent(pooled MSE + cosine) + L_decode(CE through frozen decoder)

Inference
  prompt -> Encoder(1次) -> Core(1次) -> semantic tokens -> AR Decoder(N步)
```

## 快速开始

```bash
pip install -r requirements.txt
bash scripts/run_all.sh          # 一键小规模全流程冒烟
```

## 完整 CLI

| 命令 | 作用 |
|---|---|
| `python prepare_data.py [--offline]` | 下载/生成语料 -> tokenizer -> split -> 本地 cache + metadata |
| `python train_stage1.py [--resume checkpoints/stage1/latest.pt] [--noise-std 0.1]` | 语言自编码器训练 |
| `python train_stage2.py --stage1-checkpoint checkpoints/stage1/best.pt [--core transformer]` | 冻结 Stage-1，训练 Semantic Core |
| `python generate.py --checkpoint ... --prompt "小明今天去了学校"` | 完整推理（Encoder/Core 各一次，AR 解码） |
| `python evaluate.py --checkpoint ...` | 语义测试 + latent 统计 + 干预实验 + probing |
| `python inspect_latent.py --checkpoint ...` | PCA / t-SNE / covariance / cosine / norm 图 |
| `python demo.py [--once]` | 交互式 demo，打印 latent shape / norm / 生成 |
| `python ablation_semantic_core.py --stage1-checkpoint ... --cores identity,random,mlp,transformer,bihopfield` | Core 消融 |

## 设计要点

1. **真正的瓶颈**：16 个 learned latent queries 对 encoder states 做 cross
   attention，输出固定 `[B, 16, 512]`，与文本长度无关；不使用 stride slicing。
2. **Decoder 信息隔离**：默认 `decoder_sees_prompt: false`，cross attention 只
   指向 semantic tokens；ablation 设为 true 后追加 prompt cross-attention 以对比。
3. **Stage-1 losses**：reconstruction + paraphrase consistency（cosine）+
   VICReg variance hinge + off-diagonal covariance penalty + 训练期 latent
   Gaussian noise（`--noise-std` 可调，验证/推理关闭）。
4. **Stage-2 losses**：set-level pooled latent MSE + cosine（不做 position-wise
   配对，支持 prompt/target 长度不同）+ frozen-decoder decode CE（梯度穿过冻结
   Decoder 回传到 Core）。
5. **防 collapse 监控**：latent std/norm/成对 cosine/effective rank/协方差谱 +
   `zero-latent` 依赖分数（真 latent 与零 latent 生成差异）。
6. **Padding 正确性**：encoder `key_padding_mask`、decoder causal + padding、
   CE `ignore_index=pad`、latent 池化全部忽略 padding。
7. **Checkpoint**：model/optimizer/scheduler/epoch/step/best_val_loss/config/
   tokenizer/random_state，支持 resume；Stage-2 自动冻结 Stage-1 参数。
8. **可插拔 Core**：统一接口 `forward([B,K,D]) -> [B,K,D]`，注册表切换
   `mlp / transformer / bihopfield / identity / random`，训练代码不依赖具体实现。

## 实验指南

- **Prompt visibility ablation**：把 `configs/config.yaml` 的
  `model.decoder_sees_prompt` 改为 `true` 重训两个 stage，对比 val loss /
  perplexity / token accuracy / semantic tests / latent dependency。
- **Layer split**：设 `model.encoder_layer_split: 1..3`（在本 3 层 encoder 中
  即第 N 层后接 bottleneck），比较 reconstruction、semantic tests、probing。
- **Semantic tests**：`evaluate.py` 输出 paraphrase vs different-meaning 的
  cosine 对比（contrast margin 越大越好）、主/宾替换差异、表面改写稳定性。
- **Interventions**：`evaluate.py` 输出 latent 插值 / 噪声 / token 掩码 /
  token 交换 / 维度掩码的生成结果，以及 zero-latent 依赖检查。
- **Probing**：semantic probe（主语/宾语识别）应高，surface probe
  （末字符/长度桶）应尽量低——语义保留、表面信息被瓶颈滤除。

## 目录结构

```
project/
├── data/{raw,processed,cache,metadata.json}
├── checkpoints/{stage1,stage2}
├── logs/  samples/  configs/
├── src/
│   ├── model/      (encoder, decoder, bottleneck, cores, model)
│   ├── data/       (dataset, corpus)
│   ├── training/   (common, stage1, stage2)
│   ├── evaluation/ (semantic_eval)
│   └── utils/      (seed, logging)
├── scripts/run_all.sh
├── prepare_data.py  train_stage1.py  train_stage2.py
├── generate.py  evaluate.py  inspect_latent.py  demo.py
├── ablation_semantic_core.py
├── configs/config.yaml  requirements.txt  README.md
```

## Semantic VAE（Stage 1 可选，ablation 用）

默认仍是 deterministic AE；`vae.enabled: true`（或 `--vae`）切换为 Semantic VAE：

```
sentence -> Encoder -> 32x512 -> [mu_head, logvar_head]
   z = mu + exp(0.5*logvar) * eps   (训练，重参数化采样)
   z = mu                            (eval / 推理 / sample=False 确定性锚点)
```

- Loss：`L = L_recon + beta_eff*KL(q(z|x)||N(0,I)) + paraphrase/variance/covariance`
- KL 退火：`beta_eff = beta * min(1, step / kl_warmup_steps)`（TensorBoard: `vae/beta_eff`）
- 防 posterior collapse：free bits（每维 KL 下限 `free_bits`，下限内梯度为零）、
  `logvar_clamp`、active-units 监控（`vae/active_units`，KL(nats) 超过
  `active_unit_threshold` 的维度数；`vae/kl` 持续≈0 且 au 掉到 0 = 塌缩）
- Stage 2 / 全部 8 种 core 零改动：bottleneck 对外仍是 `[B,K,D]` 接口，
  Stage 2 拿到的是确定性的 mu
- 老 checkpoint / 旧 YAML 完全兼容（vae/eval 段缺省 = AE + english preset）

```bash
python train_stage1.py --vae --beta 0.01        # VAE
python train_stage1.py --no-vae                 # AE（对照）
python evaluate.py --checkpoint checkpoints/stage1/best.pt --eval-preset english
```

## Semantic Eval 数据集配置化

`src/evaluation/semantic_eval.py` 原来硬编码中文测试句，现在全部走 config：

```yaml
eval:
  data_preset: english   # english（默认）| chinese | custom
  custom_tests_file: null  # preset=custom 时的 JSON：
                           # {"tests": {"name": ["A", "B"]},
                           #  "intervention_a": "...", "intervention_b": "...",
                           #  "probe_subjects": [...], "probe_objects": [...]}
  intervention_prompt_a: null  # 显式字段 > preset（可只覆盖其中一项）
  probe_subjects: null
  probe_objects: null
```

CLI 覆盖：`--eval-preset english|chinese|custom`、`--custom-tests file.json`。
注意 probe 词表要与语料语言匹配，否则 subject/object probe 为 0。
