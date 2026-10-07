# GraphTokenizer 训练

`graph_tokenizer_trainer.py` 是唯一的正式训练入口。它使用 GammaGL 的
Feuler 序列化器、Graph BPE、tokenizer、数据集以及原生 TensorLayerX
`GraphBERT` / `GraphGTE`；训练前向路径不使用 Hugging Face 模型。

正式路径默认以种子 `42 43 44 45 46` 完成五次彼此独立的完整训练：仅在训练集拟合
BPE／标签归一化，在完整训练集上进行 MLM，再微调、验证、恢复最佳微调 checkpoint 并最终
测试。最终指标为五次最终测试的 `mean ± std`（样本标准差，`ddof=1`）。测试集只会在恢复
最佳 checkpoint 后评估。

长度超过 `max_length` 的序列化图会被拒绝而非截断，因为截断会丢失图结构。CLI 参数会覆盖
`PAPER_CONFIGS` 中对应 preset 的值。

补充数据集的论文报告值与当前复现结果如下：

| Dataset | Metric | BERT (Paper) | BERT (Ours) | GTE (Paper) | GTE (Ours) |
|:--|:--:|--:|--:|--:|--:|
| **Mutagenicity** | Accuracy ↑ | 0.875 ± 0.009 | 0.7673 ± 0.0175 | 0.901 ± 0.007 | 0.6811 ± 0.0588 |
| **DBLP** | Accuracy ↑ | 0.932 ± 0.001 | 0.9225 ± 0.0128 | 0.936 ± 0.001 | 0.9214 ± 0.0045 |
| **AQSOL** | MAE ↓ | 0.648 ± 0.008 | 0.7395 ± 0.0072 | 0.609 ± 0.016 | 0.7077 ± 0.0069 |

论文结果与 GammaGL reproduction 是两个独立栏目：后者只能填写本入口完成的五 seed 正式实验
汇总，不能复制论文数值。

论文结果表中可能将 Mutagenicity 简写为 `mutag` 或 `muta`；这里它始终指论文使用的
**Mutagenicity** 数据集，**不是**经典的 188-graph `MUTAG`。Trainer 也接受 `mutag` 和
`muta` aliases。`dblp` 指 GraphTokenizer 论文的 **DBLP_v1 graph-classification** benchmark，
不是 GammaGL 已有的 heterogeneous `DBLP` dataset；其独立数据集类为
`GraphTokenizerDBLP`。

## 支持的图范围

当前 `FrequencyGuidedEulerianSerializer` 面向简单无向图，覆盖这三个 GraphTokenizer
benchmark 的图结构。它支持非连通图、孤立节点及标量节点/边标签；空节点图会保留为零节点
图。显式声明有向语义的图对象会被拒绝；单向 COO 仅可表示一条无向边的存储形式。

有向图语义、自环和并行边／多重图均不支持，会 fail fast 而不会静默转换。

## 固定的六组实验 preset

`PAPER_CONFIGS` 基于固定的
[GraphTokenizer 官方源码 revision
`98343a6b025a48fbb6859cd812a12b81ec3ac3cc`](https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers/tree/98343a6b025a48fbb6859cd812a12b81ec3ac3cc)
整理，并适配到 GammaGL streamlined trainer 的 benchmark presets，不是对 upstream orchestration
configuration 的逐字段复制。表中的值与 trainer 当前代码一致；GammaGL 保持统一的 serializer/
tokenization pipeline，默认使用 Python BPE，`cpp` 是可选 backend。

所有 preset 均使用 `max_length=8096`、BPE `2000` merges / 最小频率 `2`、100 次序列化、
MLM mask 概率 `0.09`、warm-up `0.12 / 0.025`、梯度裁剪 `2.0 / 0.5`、默认 seeds `42–46`，
并使用 `gradient_accumulation_steps=1`。Graph BPE 仅在训练集拟合，故词表大小为运行时派生值。

| 实验 | 预训练 / 微调 batch | MLM / 微调 LR | MLM / 微调 epoch | 预训练 / 微调 weight decay | 耐心值 | 位置容量 | 架构 / checkpoint |
| :-- | --: | :-- | :-- | :-- | --: | --: | :-- |
| Mutagenicity + BERT | 32 / 32 | `1e-4 / 1e-5` | 200 / 200 | `0.1 / 0.1` | 20 | 8096 | 512 隐藏维、4 层、4 头、2048 FFN |
| Mutagenicity + GTE | 32 / 64 | `1e-4 / 5e-5` | 200 / 100 | `0.1 / 0.01` | 10 | 8192 | 768 隐藏维、12 层、12 头、3072 FFN、RoPE |
| DBLP_v1 + BERT | 32 / 32 | `1e-4 / 1e-5` | 200 / 200 | `0.1 / 0.1` | 20 | 8096 | 512 隐藏维、4 层、4 头、2048 FFN |
| DBLP_v1 + GTE | 32 / 64 | `1e-4 / 5e-5` | 200 / 100 | `0.1 / 0.01` | 10 | 8192 | 768 隐藏维、12 层、12 头、3072 FFN、RoPE |
| AQSOL + BERT | 32 / 32 | `1e-4 / 1e-5` | 200 / 200 | `0.1 / 0.1` | 20 | 8096 | 512 隐藏维、4 层、4 头、2048 FFN |
| AQSOL + GTE | 32 / 32 | `1e-4 / 1e-5` | 200 / 200 | `0.1 / 0.1` | 20 | 8192 | 768 隐藏维、12 层、12 头、3072 FFN、RoPE |

`--gradient-accumulation-steps` 仍是通用 CLI 功能；每个完整或最后的 partial accumulation
window 都使用其实际 window size 作为相同 divisor，optimizer 与 scheduler 的 step 数保持一致。

### 序列长度与位置容量

`max_length` 是包含 special token 的最终 GraphTokenizer 输入上限；它不同于模型
`max_position_embeddings` 容量。BERT 的两个值均为 8096，来自固定 revision 的
[`config/default_config.yml`](https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers/blob/98343a6b025a48fbb6859cd812a12b81ec3ac3cc/config/default_config.yml#L77-L83)。
GammaGL 不采用上游将 BERT 运行时上限改为 768 的路径，因为这会违反完整图序列不得截断的
协议；collate 只会 pad 到当前 batch 中最长的序列。

GTE 的最终输入上限为 8096、位置容量为 8192；后者来自固定源码的
[`gte_model/config.json`](https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers/blob/98343a6b025a48fbb6859cd812a12b81ec3ac3cc/gte_model/config.json#L32)。

### GTE checkpoint 与 BPE

正式 `--encoder gte` 运行构建原生 TLX GraphGTE，并通过 GammaGL HF-to-TLX converter 加载
`Alibaba-NLP/gte-multilingual-base` revision
`9bbca17d9273fd0d03d5725c7a4b0f6b45142062`。它验证 checkpoint SHA-256
`f5a35a10faa54da7717870af1517c9b41e9bd8e3880bc5a8e9363d4c3c63e9b0`、官方架构、135/135
parameter coverage 和 tensor shape；任一校验失败都会停止运行。`--allow-random-gte-init`
仅用于开发，会标记 `reproduction: false`，不是论文复现实验命令。

`--bpe-backend python` 是默认值；`auto` 在可用时使用可选的 `graph_bpe_cpp` 原生扩展，否则
回退 Python；`cpp` 则要求该扩展并在缺失时 fail fast。无论 backend，BPE 只在训练 split 拟合，
验证与测试从不参与词表或回归标签统计。

## 安装

在运行正式 preset 前，安装论文训练运行时：

```bash
pip install -e ".[graph-tokenizer-paper]"
```

该 extra 包含原生 PyTorch 运行时及 `huggingface-hub`、`safetensors`；正式 GTE 初始化需要它们
来获取并转换固定 checkpoint。训练前请按 CUDA 环境选择适当的 PyTorch 版本。

## 数据准备

Mutagenicity、DBLP_v1 和 AQSOL 共用官方 release bundle。启动任意训练 worker 前，先运行一次
单进程准备步骤：

```bash
python examples/graph_tokenizer/graph_tokenizer_trainer.py \
  --prepare-data \
  --data-root data
```

它会在同一 `--data-root` 下下载、SHA-256 校验、缓存并 materialize bundle；重复运行复用已校验
缓存。Dataset 构造和训练 worker 不会隐式下载它，因此请勿让多个 worker 并发执行准备步骤。
设置 `GAMMAGL_GRAPH_TOKENIZER_DATA_BUNDLE=/path/to/local/bundle` 可使用本地已验证 bundle。
训练时已拟合的 tokenizer 与编码后的 train/val/test records 会缓存到
`DATA_ROOT/.graph_tokenizer_preprocessing_cache/`；同一配置的后续 seed 可复用该缓存，
`--preprocessing-cache-dir PATH` 可改变位置。

## 训练与评估

```bash
export TL_BACKEND=torch

python examples/graph_tokenizer/graph_tokenizer_trainer.py --dataset mutagenicity --encoder bert --data-root data --device cuda
python examples/graph_tokenizer/graph_tokenizer_trainer.py --dataset mutagenicity --encoder gte --data-root data --device cuda
python examples/graph_tokenizer/graph_tokenizer_trainer.py --dataset dblp --encoder bert --data-root data --device cuda
python examples/graph_tokenizer/graph_tokenizer_trainer.py --dataset dblp --encoder gte --data-root data --device cuda
python examples/graph_tokenizer/graph_tokenizer_trainer.py --dataset aqsol --encoder bert --data-root data --device cuda
python examples/graph_tokenizer/graph_tokenizer_trainer.py --dataset aqsol --encoder gte --data-root data --device cuda
```

默认每个命令运行五个 seeds；可显式指定：

```bash
python examples/graph_tokenizer/graph_tokenizer_trainer.py \
  --dataset mutagenicity --encoder bert --data-root data --device cuda \
  --seeds 42 43 44 45 46
```

Mutagenicity 和 DBLP_v1 使用两个 logits 的 CrossEntropy，并在每个图的多次 serialization
logits 平均后取 `argmax` 计算 Accuracy。AQSOL 使用 MSE 训练；其标签归一化只用 train split
统计，评估将预测和标签恢复到原始尺度后计算 MAE。

常用覆盖参数包括 `--pretrain-batch-size`、`--finetune-batch-size`、
`--pretrain-learning-rate`、`--learning-rate`、
`--pretrain-epochs`、`--finetune-epochs`、`--max-length`、`--bpe-merges`、
`--bpe-backend`、`--num-serializations`、`--early-stopping-patience`、
`--pretrain-weight-decay`、`--finetune-weight-decay`、`--gradient-accumulation-steps` 与 `--seeds`。
使用 `--gte-checkpoint-cache-dir PATH` 指定 GTE checkpoint 的 Hugging Face 缓存。

每次运行将 checkpoint 与 summary 写入 `run_01_seed_42/`、`run_02_seed_43/` 等目录；父级
`summary.json` 保存聚合结果。early stopping 的方向由 dataset metadata 驱动：Accuracy 取最大值，
MAE 取最小值。

## 极小 smoke 测试

```bash
export TL_BACKEND=torch
python examples/graph_tokenizer/graph_tokenizer_trainer.py \
  --smoke --dataset aqsol --encoder bert --device cpu \
  --seeds 42 \
  --output-dir /tmp/graph-tokenizer-smoke
```

BERT 的 `--smoke` 使用 tiny architecture。GTE 的 tiny smoke 必须显式加入
`--allow-random-gte-init`；该随机初始化模式仅用于开发／pipeline 检查，不能作为正式
reproduced result。
