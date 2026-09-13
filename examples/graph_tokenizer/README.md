# GraphTokenizer training

`graph_tokenizer_trainer.py` is the sole formal training entrypoint. It uses
GammaGL's Feuler serializer, Graph BPE, tokenizer, datasets, and native
TensorLayerX `GraphBERT` / `GraphGTE`; no Hugging Face model participates in
the training forward path.

The formal path defaults to five independent complete runs with seeds `42 43
44 45 46`. Each run performs:

```text
train-only BPE / train-only label normalization
→ MLM train → MLM validation → restore best MLM checkpoint
→ finetune train → validation → restore best finetune checkpoint → final test
```

The final summary reports the five final-test metrics as `mean ± std`
(population standard deviation, `ddof=0`).
Five runs do not mean five epochs: every run retains the preset's
`pretrain_epochs`, `finetune_epochs`, and early-stopping settings (for example,
200 pretraining and 200 finetuning epochs in the default presets).

The test split is evaluated only after the best finetune checkpoint is
restored. CLI values override every `PAPER_CONFIGS` preset value.

## Supported graph scope

The current `FrequencyGuidedEulerianSerializer` is a GraphTokenizer serializer
for simple undirected graphs, not a general serialization guarantee for every
GammaGL graph type. This covers the graph structure used by the included
molecular GraphTokenizer benchmarks.

Supported graph structures include disconnected graphs, isolated nodes, and
scalar node and edge labels. Graph objects that explicitly declare directed
semantics are rejected; a single COO orientation is accepted only as the
storage form of one undirected edge.

Directed graph semantics, self-loops, and parallel edges / multigraphs are not
currently supported. They are rejected rather than silently converted.

## Pinned six-experiment presets

The following values are transcribed from the previous pinned paper protocol
(`source revision 98343a6b025a48fbb6859cd812a12b81ec3ac3cc`). They are
defaults in `PAPER_CONFIGS`, except where noted below. `vocab size` is not a
fixed hyperparameter: Graph BPE learns it from the training split only and
the model receives the resulting required vocabulary size at runtime. BPE is
requested to perform 2,000 merges with minimum frequency 2, so the realised
codebook may be smaller.

| Experiment | Batch | MLM / finetune LR | MLM / finetune epochs | Max length | BPE / vocab | Serializations | Patience | Weight decay | Seed(s) | Architecture | Checkpoint revision |
| --- | ---: | --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| QM9 + BERT | 32 | 1e-4 / 1e-5 | 200 / 200 | 768 | 2000 / train-derived | 100 | 20 / 20 | 0.1 | 42–46 (default CLI) | 512 hidden, 4 layers, 4 heads, 2048 FFN, positional cap 8096 | N/A |
| QM9 + GTE | 32 | 5e-5 / 1e-5 | 200 / 200 | 8096 | 2000 / train-derived | 100 | 20 / 20 | 0.1 | 42–46 (default CLI) | 768 hidden, 12 layers, 12 heads, 3072 FFN, RoPE, positional cap 8192 | `Alibaba-NLP/gte-multilingual-base` `9bbca17d9273fd0d03d5725c7a4b0f6b45142062`* |
| MolHIV + BERT | 32 | 1e-4 / 5e-5 | 200 / 200 | 768 | 2000 / train-derived | 100 | 20 / 20 | 0.1 | 42–46 (default CLI) | 512 hidden, 4 layers, 4 heads, 2048 FFN, positional cap 8096 | N/A |
| MolHIV + GTE | 32 | 5e-5 / 5e-5 | 200 / 200 | 8096 | 2000 / train-derived | 100 | 20 / 20 | 0.1 | 42–46 (default CLI) | 768 hidden, 12 layers, 12 heads, 3072 FFN, RoPE, positional cap 8192 | `Alibaba-NLP/gte-multilingual-base` `9bbca17d9273fd0d03d5725c7a4b0f6b45142062`* |
| Peptides-struct + BERT | 16 | 1e-4 / 1e-5 | 200 / 200 | 768 | 2000 / train-derived | 100 | 20 / 20 | 0.1 | 42–46 (default CLI) | 512 hidden, 4 layers, 4 heads, 2048 FFN, positional cap 8096 | N/A |
| Peptides-struct + GTE | 16 | 1e-4 / 1e-5 | 200 / 200 | 8096 | 2000 / train-derived | 100 | 20 / 20 | 0.1 | 42–46 (default CLI) | 768 hidden, 12 layers, 12 heads, 3072 FFN, RoPE, positional cap 8192 | `Alibaba-NLP/gte-multilingual-base` `9bbca17d9273fd0d03d5725c7a4b0f6b45142062`* |

The preceding protocol also specified mask probability 0.09, mean pooling,
Feuler serialization, and warm-up ratios 0.12 / 0.025. The streamlined
trainer exposes the training parameters it implements; it intentionally does
not preserve benchmark-only warm-up.

\* Formal `--encoder gte` runs construct native TLX GraphGTE and load this
pinned checkpoint through GammaGL's HF-to-TLX converter. It validates the
revision, SHA-256
`f5a35a10faa54da7717870af1517c9b41e9bd8e3880bc5a8e9363d4c3c63e9b0`, official
architecture, converter coverage, and tensor shapes; any failure stops the
run. `--allow-random-gte-init` is an explicit development-only escape hatch
and records `reproduction: false`; it is not a paper reproduction command.

## Installation

Install the GraphTokenizer paper runtime before running any formal preset:

```bash
pip install -e ".[graph-tokenizer-paper]"
```

This extra includes the native PyTorch runtime plus `huggingface-hub` and
`safetensors`, which formal GTE initialization needs to fetch and convert the
pinned official encoder checkpoint. Select the PyTorch build appropriate for
your CUDA environment before training.

## Data preparation

QM9, OGBG-MolHIV, and Peptides-struct share one official release bundle. Run
this single-process preparation step once before starting any training workers:

```bash
python examples/graph_tokenizer/graph_tokenizer_trainer.py \
  --prepare-data \
  --data-root data
```

It downloads, verifies, caches, and materializes the shared bundle under the
same `--data-root` used for training. Re-running it reuses the verified cache;
do not have multiple training workers perform this step concurrently. Set
`GAMMAGL_GRAPH_TOKENIZER_DATA_BUNDLE=/path/to/local/bundle` to use a local
bundle instead.

## Training

Set the backend once, then run any of the six preset combinations. The command
chooses the matching `PAPER_CONFIGS` entry; by default, each invocation runs
the five seeds `42 43 44 45 46` and reports final-test `mean ± std`.

```bash
export TL_BACKEND=torch

python examples/graph_tokenizer/graph_tokenizer_trainer.py --dataset qm9 --encoder bert --data-root data --device cuda
python examples/graph_tokenizer/graph_tokenizer_trainer.py --dataset qm9 --encoder gte --data-root data --device cuda
python examples/graph_tokenizer/graph_tokenizer_trainer.py --dataset molhiv --encoder bert --data-root data --device cuda
python examples/graph_tokenizer/graph_tokenizer_trainer.py --dataset molhiv --encoder gte --data-root data --device cuda
python examples/graph_tokenizer/graph_tokenizer_trainer.py --dataset peptides-struct --encoder bert --data-root data --device cuda
python examples/graph_tokenizer/graph_tokenizer_trainer.py --dataset peptides-struct --encoder gte --data-root data --device cuda
```

The default five-run command can also state its seeds explicitly:

```bash
python examples/graph_tokenizer/graph_tokenizer_trainer.py \
  --dataset qm9 --encoder bert --data-root data --device cuda \
  --seeds 42 43 44 45 46
```

For CI, smoke tests, or debugging, run one complete experiment instead:

```bash
python examples/graph_tokenizer/graph_tokenizer_trainer.py \
  --dataset qm9 --encoder bert --data-root data --device cuda --seeds 42
```

Each run writes its checkpoints and summary under
`run_01_seed_42/`, `run_02_seed_43/`, and so on; the parent `summary.json`
contains the aggregate result.

Useful overrides include `--batch-size`, `--pretrain-learning-rate`,
`--learning-rate`, `--pretrain-epochs`, `--finetune-epochs`, `--max-length`,
`--bpe-merges`, `--num-serializations`, `--early-stopping-patience`,
`--pretrain-early-stopping-patience`, `--weight-decay`, and `--seeds`.
Use `--gte-checkpoint-cache-dir PATH` to select the Hugging Face cache for a
formal GTE run.

## Evaluation semantics and results

- QM9 trains/evaluates the HOMO target only; evaluation inversely transforms
  predictions and reports raw-label-space HOMO MAE.
- MolHIV converts two logits to the positive-class probability, averages
  serialization probabilities per graph, then reports ROC-AUC.
- Peptides-struct inversely transforms all 11 targets and reports the mean of
  their target-wise MAEs.

GammaGL result: **pending / not rerun** after this streamlined trainer
refactor. Historical benchmark outputs are local-only and are not claimed as
results from this entrypoint.

## Tiny smoke

```bash
export TL_BACKEND=torch
python examples/graph_tokenizer/graph_tokenizer_trainer.py \
  --smoke --dataset qm9 --encoder bert --device cpu \
  --seeds 42 \
  --output-dir /tmp/graph-tokenizer-smoke
```
