# GraphTokenizer training

`graph_tokenizer_trainer.py` is the sole formal training entrypoint. It uses
GammaGL's Feuler serializer, Graph BPE, tokenizer, datasets, and native
TensorLayerX `GraphBERT` / `GraphGTE`; no Hugging Face model participates in
the training forward path.

The formal path is one run, not a benchmark orchestrator:

```text
train-only BPE / train-only label normalization
→ MLM train → MLM validation → restore best MLM checkpoint
→ finetune train → validation → restore best finetune checkpoint → final test
```

The test split is evaluated only after the best finetune checkpoint is
restored. CLI values override every `PAPER_CONFIGS` preset value.

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
| QM9 + BERT | 32 | 1e-4 / 1e-5 | 200 / 200 | 768 | 2000 / train-derived | 100 | 20 / 20 | 0.1 | 42 (CLI; historical benchmark 42–46) | 512 hidden, 4 layers, 4 heads, 2048 FFN, positional cap 8096 | N/A |
| QM9 + GTE | 32 | 5e-5 / 1e-5 | 200 / 200 | 8096 | 2000 / train-derived | 100 | 20 / 20 | 0.1 | 42 (CLI; historical benchmark 42–46) | 768 hidden, 12 layers, 12 heads, 3072 FFN, RoPE, positional cap 8192 | `Alibaba-NLP/gte-multilingual-base` `9bbca17d9273fd0d03d5725c7a4b0f6b45142062`* |
| MolHIV + BERT | 32 | 1e-4 / 5e-5 | 200 / 200 | 768 | 2000 / train-derived | 100 | 20 / 20 | 0.1 | 42 (CLI; historical benchmark 42–46) | 512 hidden, 4 layers, 4 heads, 2048 FFN, positional cap 8096 | N/A |
| MolHIV + GTE | 32 | 5e-5 / 5e-5 | 200 / 200 | 8096 | 2000 / train-derived | 100 | 20 / 20 | 0.1 | 42 (CLI; historical benchmark 42–46) | 768 hidden, 12 layers, 12 heads, 3072 FFN, RoPE, positional cap 8192 | `Alibaba-NLP/gte-multilingual-base` `9bbca17d9273fd0d03d5725c7a4b0f6b45142062`* |
| Peptides-struct + BERT | 16 | 1e-4 / 1e-5 | 200 / 200 | 768 | 2000 / train-derived | 100 | 20 / 20 | 0.1 | 42 (CLI; historical benchmark 42–46) | 512 hidden, 4 layers, 4 heads, 2048 FFN, positional cap 8096 | N/A |
| Peptides-struct + GTE | 16 | 1e-4 / 1e-5 | 200 / 200 | 8096 | 2000 / train-derived | 100 | 20 / 20 | 0.1 | 42 (CLI; historical benchmark 42–46) | 768 hidden, 12 layers, 12 heads, 3072 FFN, RoPE, positional cap 8192 | `Alibaba-NLP/gte-multilingual-base` `9bbca17d9273fd0d03d5725c7a4b0f6b45142062`* |

The preceding protocol also specified mask probability 0.09, mean pooling,
Feuler serialization, and warm-up ratios 0.12 / 0.025. The streamlined
trainer exposes the training parameters it implements; it intentionally does
not preserve benchmark-only warm-up, five-seed orchestration, or report
aggregation.

\* Formal `--encoder gte` runs construct native TLX GraphGTE and load this
pinned checkpoint through GammaGL's HF-to-TLX converter. It validates the
revision, SHA-256
`f5a35a10faa54da7717870af1517c9b41e9bd8e3880bc5a8e9363d4c3c63e9b0`, official
architecture, converter coverage, and tensor shapes; any failure stops the
run. `--allow-random-gte-init` is an explicit development-only escape hatch
and records `reproduction: false`; it is not a paper reproduction command.

## Commands

Set the backend once, then run any of the six single-run presets directly.
The command chooses the matching `PAPER_CONFIGS` entry; append any listed CLI
option to override it.

```bash
export TL_BACKEND=torch

python examples/graph_tokenizer/graph_tokenizer_trainer.py --dataset qm9 --encoder bert --data-root data --device cuda
python examples/graph_tokenizer/graph_tokenizer_trainer.py --dataset qm9 --encoder gte --data-root data --device cuda
python examples/graph_tokenizer/graph_tokenizer_trainer.py --dataset molhiv --encoder bert --data-root data --device cuda
python examples/graph_tokenizer/graph_tokenizer_trainer.py --dataset molhiv --encoder gte --data-root data --device cuda
python examples/graph_tokenizer/graph_tokenizer_trainer.py --dataset peptides-struct --encoder bert --data-root data --device cuda
python examples/graph_tokenizer/graph_tokenizer_trainer.py --dataset peptides-struct --encoder gte --data-root data --device cuda
```

Useful overrides include `--batch-size`, `--pretrain-learning-rate`,
`--learning-rate`, `--pretrain-epochs`, `--finetune-epochs`, `--max-length`,
`--bpe-merges`, `--num-serializations`, `--early-stopping-patience`,
`--pretrain-early-stopping-patience`, `--weight-decay`, and `--seed`.
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
  --output-dir /tmp/graph-tokenizer-smoke
```
