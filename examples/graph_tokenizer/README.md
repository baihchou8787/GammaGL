# GraphTokenizer 训练

`graph_tokenizer_trainer.py` 是唯一的正式训练入口。它使用 GammaGL 的
Feuler 序列化器、Graph BPE、tokenizer、数据集以及原生 TensorLayerX
`GraphBERT` / `GraphGTE`；训练前向路径不使用 Hugging Face 模型。

正式路径默认以种子 `42 43 44 45 46` 完成五次彼此独立的完整训练。每次训练依次执行：

```text
仅在训练集拟合 BPE / 标签归一化
→ 在完整训练集上按配置的预训练轮数进行 MLM 适配
→ 微调训练 → 验证 → 恢复最佳微调 checkpoint → 最终测试
```

最终汇总将五次最终测试指标报告为 `mean ± std`（样本标准差，`ddof=1`），以匹配
GraphTokenizer 的重复运行统计口径。五次运行不等于五个 epoch：每次运行仍使用 preset
中的 `pretrain_epochs`、`finetune_epochs` 与早停设置（默认 preset 例如为 200 个预训练
epoch 和 200 个微调 epoch）。

仅在恢复最佳微调 checkpoint 后才评估测试集。CLI 参数会覆盖 `PAPER_CONFIGS` 中相应的
preset 值。长度超过 `max_length` 的序列化图会被拒绝而非截断，因为截断会丢失图结构。

## 支持的图范围

当前 `FrequencyGuidedEulerianSerializer` 是面向简单无向图的 GraphTokenizer 序列化器，
并不保证可通用于所有 GammaGL 图类型。这覆盖了随附分子 GraphTokenizer benchmark 所用的
图结构。

支持非连通图、孤立节点以及标量节点和边标签。显式声明有向语义的图对象会被拒绝；单向 COO
存储仅可表示一条无向边的存储形式。

目前不支持有向图语义、自环和并行边／多重图；它们会被拒绝，不会被静默转换。

## 固定的六组实验 preset

`PAPER_CONFIGS` 是基于固定的
[GraphTokenizer 官方源码 revision
`98343a6b025a48fbb6859cd812a12b81ec3ac3cc`](https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers/tree/98343a6b025a48fbb6859cd812a12b81ec3ac3cc)
整理并适配到 streamlined GammaGL 训练入口的 preset。它们保留当前实现所需的核心模型和训练设置，但并非对
原仓库默认配置、调优配置或 benchmark 编排进行逐字节、逐字段复刻。因此，下表中的任一行均不应
被理解为其每个字段都从同一个上游配置文件原样继承。

固定源码为 BERT 架构、200 / 200 的预训练／微调 epoch、0.09 的 mask 概率、BPE（2,000 次
merge、最小频率 2）、100 次序列化、0.1 的权重衰减、0.12 / 0.025 的 warm-up 比例以及 MLM／
微调的 2.0 / 0.5 梯度裁剪提供参考默认值。GammaGL 将这些值组合进统一的六组 preset 入口，并在
下文明确记录实现侧的选择。`vocab size` 并非固定超参数：Graph BPE 只在训练集拟合，模型在运行时
接收由此得到的实际词表大小，因此实际 codebook 可能少于 2,000 次 merge。

| 实验 | Batch | MLM / 微调 LR | MLM / 微调 epoch | 最终最大长度 | BPE / 词表 | 序列化次数 | 微调耐心值 | 权重衰减 | 种子 | 架构 | Checkpoint revision |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| QM9 + BERT | 32 | 1e-4 / 1e-5 | 200 / 200 | 8096 | 2000 / 训练集派生 | 100 | 20 | 0.1 | 42–46（默认 CLI） | 512 隐藏维、4 层、4 头、2048 FFN、位置容量 8096 | 不适用 |
| QM9 + GTE | 32 | 5e-5 / 1e-5 | 200 / 200 | 8096 | 2000 / 训练集派生 | 100 | 20 | 0.1 | 42–46（默认 CLI） | 768 隐藏维、12 层、12 头、3072 FFN、RoPE、位置容量 8192 | `Alibaba-NLP/gte-multilingual-base` `9bbca17d9273fd0d03d5725c7a4b0f6b45142062`* |
| MolHIV + BERT | 32 | 1e-4 / 5e-5 | 200 / 200 | 8096 | 2000 / 训练集派生 | 100 | 20 | 0.1 | 42–46（默认 CLI） | 512 隐藏维、4 层、4 头、2048 FFN、位置容量 8096 | 不适用 |
| MolHIV + GTE | 32 | 5e-5 / 5e-5 | 200 / 200 | 8096 | 2000 / 训练集派生 | 100 | 20 | 0.1 | 42–46（默认 CLI） | 768 隐藏维、12 层、12 头、3072 FFN、RoPE、位置容量 8192 | `Alibaba-NLP/gte-multilingual-base` `9bbca17d9273fd0d03d5725c7a4b0f6b45142062`* |
| Peptides-struct + BERT | 16 | 1e-4 / 1e-5 | 200 / 200 | 8096 | 2000 / 训练集派生 | 100 | 20 | 0.1 | 42–46（默认 CLI） | 512 隐藏维、4 层、4 头、2048 FFN、位置容量 8096 | 不适用 |
| Peptides-struct + GTE | 16 | 1e-4 / 1e-5 | 200 / 200 | 8096 | 2000 / 训练集派生 | 100 | 20 | 0.1 | 42–46（默认 CLI） | 768 隐藏维、12 层、12 头、3072 FFN、RoPE、位置容量 8192 | `Alibaba-NLP/gte-multilingual-base` `9bbca17d9273fd0d03d5725c7a4b0f6b45142062`* |

### 序列长度与位置容量的来源

`max_length` 是最终 GraphTokenizer 输入的最大长度，包含 tokenization 添加的 special token。
它与 Transformer 的位置容量 `max_position_embeddings` 是两个不同的设置。

对于 BERT，正式 preset 的 `max_length=8096` 和
`max_position_embeddings=8096` 均来自固定 revision 的
[`config/default_config.yml`](https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers/blob/98343a6b025a48fbb6859cd812a12b81ec3ac3cc/config/default_config.yml#L77-L83)
中 BERT 的 `max_seq_length=8096` 与 `max_position_embeddings=8096`。上游
[`src/models/bert/data.py`](https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers/blob/98343a6b025a48fbb6859cd812a12b81ec3ac3cc/src/models/bert/data.py#L76)
还含有将 BERT 运行时上限覆盖为 768 的运行时路径；GammaGL 不采用该路径，因为它与当前“完整图
序列不得截断”的严格协议冲突。GammaGL 对长度超过 `max_length` 的序列化图抛出 `ValueError`，
绝不会静默截断；trainer 的 collate 仅 pad 到当前 batch 中最长的序列，而不会固定 pad 到 8096。

对于 GTE，GammaGL 使用 `max_length=8096` 和 `max_position_embeddings=8192`。后者是固定
源码随附 GTE 的位置容量：
[`gte_model/config.json`](https://github.com/BUPT-GAMMA/Graph-Tokenization-for-Bridging-Graphs-and-Transformers/blob/98343a6b025a48fbb6859cd812a12b81ec3ac3cc/gte_model/config.json#L32)；
前者则是当前入口的最终输入限制，仍与该位置容量相互独立。

### 与原始 benchmark 工作流的差异

- **Warm-up 与梯度裁剪：** GammaGL 保留源码默认的 warm-up 比例（0.12 / 0.025）和分阶段
  梯度裁剪（2.0 / 0.5），并由本 trainer 的按 step warm-up/cosine scheduler 实现。
- **长度处理：** BERT 的最终输入上限与 `max_position_embeddings` 均为 8096，二者仍是
  语义不同的设置。GammaGL 对超长序列化图直接失败，而不静默截断，并且按 batch 动态 padding。
- **MLM 与早停：** GammaGL 在完整下游训练集上按配置的预训练 epoch 进行 MLM，只在微调阶段
  保留验证选择和早停；不保留上游的 MLM 验证／早停路径。
- **Benchmark 编排：** 上游仓库有独立的数据准备、预训练、微调、批量启动和聚合程序。GammaGL
  有意使用此单一 trainer，未保留该 benchmark orchestrator。
- **五次运行聚合：** GammaGL 默认保留五次完整独立运行（`42`–`46`），并报告最终测试指标的均值
  加样本标准差。固定上游默认配置本身从一次运行开始，因此 GammaGL 的默认值是明确的入口策略。
- **数据准备：** GammaGL 通过单独且显式的 `--prepare-data` 步骤准备共享的论文 release bundle；
  Dataset 构造和训练 worker 不会隐式下载它。
- **结果状态：** 历史 benchmark 输出尚未由此 streamlined trainer 重新运行；下方的 pending
  状态是有意保留的。

\* 正式 `--encoder gte` 运行会构建原生 TLX GraphGTE，并通过 GammaGL 的 HF-to-TLX converter
加载这个固定 checkpoint。它会验证 revision、SHA-256
`f5a35a10faa54da7717870af1517c9b41e9bd8e3880bc5a8e9363d4c3c63e9b0`、官方架构、converter
覆盖率和 tensor shape；任一校验失败都会停止运行。`--allow-random-gte-init` 是仅用于开发的
显式逃生选项，并会记录 `reproduction: false`；它不是论文复现实验命令。

## 安装

在运行任何正式 preset 前，安装 GraphTokenizer 论文运行时：

```bash
pip install -e ".[graph-tokenizer-paper]"
```

该 extra 包含原生 PyTorch 运行时及 `huggingface-hub`、`safetensors`；正式 GTE 初始化需要
它们来获取并转换固定的官方 encoder checkpoint。训练前请根据 CUDA 环境选择适当的 PyTorch 版本。

## 数据准备

QM9、OGBG-MolHIV 和 Peptides-struct 共用一个官方 release bundle。启动任意训练 worker 前，
先运行一次这个单进程准备步骤：

```bash
python examples/graph_tokenizer/graph_tokenizer_trainer.py \
  --prepare-data \
  --data-root data
```

它会在训练使用的同一 `--data-root` 下下载、校验、缓存并 materialize 共享 bundle。重复运行会复用
已校验的缓存；不要让多个训练 worker 并发执行此步骤。设置
`GAMMAGL_GRAPH_TOKENIZER_DATA_BUNDLE=/path/to/local/bundle` 可改用本地 bundle。

## 训练

设置一次 backend 后，可运行任一组六组 preset。命令会选择相应的 `PAPER_CONFIGS` 条目；默认每次
调用运行种子 `42 43 44 45 46`，并报告最终测试的 `mean ± std`。

```bash
export TL_BACKEND=torch

python examples/graph_tokenizer/graph_tokenizer_trainer.py --dataset qm9 --encoder bert --data-root data --device cuda
python examples/graph_tokenizer/graph_tokenizer_trainer.py --dataset qm9 --encoder gte --data-root data --device cuda
python examples/graph_tokenizer/graph_tokenizer_trainer.py --dataset molhiv --encoder bert --data-root data --device cuda
python examples/graph_tokenizer/graph_tokenizer_trainer.py --dataset molhiv --encoder gte --data-root data --device cuda
python examples/graph_tokenizer/graph_tokenizer_trainer.py --dataset peptides-struct --encoder bert --data-root data --device cuda
python examples/graph_tokenizer/graph_tokenizer_trainer.py --dataset peptides-struct --encoder gte --data-root data --device cuda
```

默认五次运行的命令也可以显式指定种子：

```bash
python examples/graph_tokenizer/graph_tokenizer_trainer.py \
  --dataset qm9 --encoder bert --data-root data --device cuda \
  --seeds 42 43 44 45 46
```

对于 CI、smoke test 或调试，可只运行一次完整实验：

```bash
python examples/graph_tokenizer/graph_tokenizer_trainer.py \
  --dataset qm9 --encoder bert --data-root data --device cuda --seeds 42
```

每次运行将 checkpoint 和 summary 写入 `run_01_seed_42/`、`run_02_seed_43/` 等目录；父级
`summary.json` 保存聚合结果。

常用覆盖参数包括 `--batch-size`、`--pretrain-learning-rate`、`--learning-rate`、
`--pretrain-epochs`、`--finetune-epochs`、`--max-length`、`--bpe-merges`、
`--num-serializations`、`--early-stopping-patience`、`--weight-decay` 和 `--seeds`。
使用 `--gte-checkpoint-cache-dir PATH` 为正式 GTE 运行指定 Hugging Face 缓存。

## 评估语义与结果

- QM9 只训练／评估 HOMO 目标；评估时对预测值做逆变换，并报告原始标签空间的 HOMO MAE。
- MolHIV 将两个 logits 转为正类概率，对每个图的序列化概率取平均，然后报告 ROC-AUC。
- Peptides-struct 对全部 11 个目标做逆变换，并报告各目标 MAE 的均值。

GammaGL 结果：**pending / 尚未重新运行**。此次 streamlined trainer 重构后，历史 benchmark
输出仅保存在本地，不能声称是由此入口产生的结果。

## 极小 smoke 测试

```bash
export TL_BACKEND=torch
python examples/graph_tokenizer/graph_tokenizer_trainer.py \
  --smoke --dataset qm9 --encoder bert --device cpu \
  --seeds 42 \
  --output-dir /tmp/graph-tokenizer-smoke
```
