# GraphTokenizer Example

This directory hosts the GraphTokenizer preprocessing, MLM pretraining, fine-tuning, and benchmark entrypoint.

## Paper protocol on a Linux server

The `QM9`, `OGBGMolHIV`, and `PeptidesStruct` classes now download the official
GraphTokenizer release bundle on first use. The archive is cached once under
`<data-root>/.graph_tokenizer_release`, and each dataset is copied into the
normal GammaGL `raw/` directory before processing. To reuse an existing bundle
or an extracted release directory, set
`GAMMAGL_GRAPH_TOKENIZER_DATA_BUNDLE=/absolute/path/to/bundle-or-directory`.

The strict GT+GTE runtime used for the three benchmarks is installed under
`/local/wrq/gammagl-graphtokenizer-paper-cu121`. It contains PyTorch 2.1.2
with CUDA 12.1, DGL 2.4.0 with CUDA 12.1, and PyTorch Geometric 2.4.0.
GraphBERT and GraphGTE are implemented directly with public TensorLayerX layers
and operations; Hugging Face Transformers is not used by the paper models. All
caches, temporary files, datasets, checkpoints, and results must remain under
`/local/wrq`.

Build the required native BPE backend and validate the data/runtime first:

```bash
cd /home/wrq/GammaGL
export TL_BACKEND=torch DGLBACKEND=pytorch
export TORCH_HOME=/local/wrq/cache/graphtokenizer-paper/torch
export XDG_CACHE_HOME=/local/wrq/cache/graphtokenizer-paper/xdg
export TMPDIR=/local/wrq/tmp/graphtokenizer-paper
PYTHON=/local/wrq/gammagl-graphtokenizer-paper-cu121/bin/python

$PYTHON third_party/graph_bpe_cpp/setup.py build_ext --inplace

$PYTHON examples/graph_tokenizer/graph_tokenizer_trainer.py \
  --protocol paper --preflight --dataset qm9 --model gte \
  --data-root /local/wrq/graph-tokenizer-data --bpe-backend cpp
$PYTHON examples/graph_tokenizer/graph_tokenizer_trainer.py \
  --protocol paper --preflight --dataset OGBG-molhiv --model gte \
  --data-root /local/wrq/graph-tokenizer-data --bpe-backend cpp
$PYTHON examples/graph_tokenizer/graph_tokenizer_trainer.py \
  --protocol paper --preflight --dataset Peptides-struct --model gte \
  --data-root /local/wrq/graph-tokenizer-data --bpe-backend cpp
```

Run the five-seed paper protocol:

```bash
$PYTHON examples/graph_tokenizer/graph_tokenizer_trainer.py \
  --protocol paper --dataset qm9 --model gte \
  --paper-config examples/graph_tokenizer/configs/qm9_paper.json \
  --data-root /local/wrq/graph-tokenizer-data --bpe-backend cpp \
  --paper-cache-root /local/wrq/cache/graphtokenizer-paper/artifacts \
  --num-merges 2000 --runs 5 --seed 42 \
  --output-dir /local/wrq/graph-tokenizer-paper-formal/qm9

$PYTHON examples/graph_tokenizer/graph_tokenizer_trainer.py \
  --protocol paper --dataset OGBG-molhiv --model gte \
  --paper-config examples/graph_tokenizer/configs/molhiv_paper.json \
  --data-root /local/wrq/graph-tokenizer-data --bpe-backend cpp \
  --paper-cache-root /local/wrq/cache/graphtokenizer-paper/artifacts \
  --num-merges 2000 --runs 5 --seed 42 \
  --output-dir /local/wrq/graph-tokenizer-paper-formal/molhiv

$PYTHON examples/graph_tokenizer/graph_tokenizer_trainer.py \
  --protocol paper --dataset Peptides-struct --model gte \
  --paper-config examples/graph_tokenizer/configs/peptides_struct_paper.json \
  --data-root /local/wrq/graph-tokenizer-data --bpe-backend cpp \
  --paper-cache-root /local/wrq/cache/graphtokenizer-paper/artifacts \
  --num-merges 2000 --runs 5 --seed 42 \
  --output-dir /local/wrq/graph-tokenizer-paper-formal/peptides_struct
```

Paper mode requires the TensorLayerX PyTorch backend, CUDA, DGL, PyTorch
Geometric, NumPy, and OGB. It jointly predicts all 16 QM9 targets, uses Feuler
for all three datasets, fits BPE only on the training split, restores the best
validation checkpoint, evaluates the test split once, and reports
mean/population-standard-deviation over seeds 42-46. The paper does not publish
all five seeds, the full scheduler, or an overlength policy; this implementation
records the explicit choices of consecutive seeds, linear warmup/decay, and
failing rather than truncating above 8192 tokens.

Paper mode stores the train-only tokenizer and encoded ragged splits under
`--paper-cache-root` (or `<data-root>/.graph_tokenizer_cache` when omitted).
Use an absolute `/local/...` path on servers with a small home partition.
Sequences are padded only to the longest item in each batch and training
batches are length-bucketed. Before model construction, the runner estimates
dense-attention memory and reduces the micro-batch when necessary while using
gradient accumulation to preserve the configured effective paper batch size.
If even a one-example micro-batch is unsafe, the run fails before allocating
the model.

The reproducibility default remains FP32 with TF32 disabled and the original
dataset losses. Optional performance/ablation flags are explicit:

```bash
# Faster CUDA execution; record these deviations when comparing with FP32.
--paper-amp bf16 --paper-tf32

# Optional regression or classification-loss studies.
--paper-loss huber
--paper-loss focal --molhiv-pos-weight
```

`--resume` restores the latest phase, optimizer, scheduler, RNG, and AMP scaler
state from `last_state.pt`; `best.pt` remains a lightweight best-model artifact.
Each epoch emits JSON containing loss/validation metric, elapsed seconds,
examples per second, and CUDA peak memory. Final JSON includes per-target MAE
for QM9 and Peptides-struct plus five-run mean and population standard
deviation for the primary metric.

The current implementation provides an end-to-end pipeline for:

- graph serialization with Feuler
- Graph BPE training and encoding
- `GraphTokenizer.build_mlm_batch()` for masked-token pretraining batches
- BERT/GTE model construction
- MLM masking
- MLM pretraining with `--pretrain-epoch`
- supervised fine-tuning with train/val/test evaluation
- best-checkpoint saving and multi-run result aggregation

Supported paper dataset names at the command layer:

- `QM9` / `qm9`
- `OGBG-molhiv` / `molhiv`
- `Peptides-func` / `p-func`
- `Peptides-struct` / `p-struct`

Graph BPE backends:

- `python`: pure Python reference implementation
- `auto`: use the native C++ extension when importable, otherwise fall back to Python
- `cpp`: require the native C++ extension and fail fast if it is not installed

Smoke commands:

```bash
python examples/graph_tokenizer/graph_tokenizer_trainer.py --smoke --dataset qm9 --model bert
python examples/graph_tokenizer/graph_tokenizer_trainer.py --smoke --dataset molhiv --model gte
python examples/graph_tokenizer/graph_tokenizer_trainer.py --smoke --dataset p-func --model bert
python examples/graph_tokenizer/graph_tokenizer_trainer.py --smoke --dataset p-struct --model bert
```

Small train/val/test test run:

```bash
python examples/graph_tokenizer/graph_tokenizer_trainer.py \
  --dataset qm9 \
  --model bert \
  --data-root data \
  --runs 1 \
  --pretrain-epoch 1 \
  --n-epoch 1 \
  --bpe-backend python \
  --num-merges 20 \
  --vocab-size 512 \
  --max-length 64 \
  --hidden-size 24 \
  --num-hidden-layers 1 \
  --num-attention-heads 4 \
  --intermediate-size 48 \
  --batch-size 16 \
  --output-dir logs/graph_tokenizer
```

Paper-like long-run shape:

```bash
python examples/graph_tokenizer/graph_tokenizer_trainer.py \
  --dataset qm9 \
  --model bert \
  --runs 5 \
  --pretrain-epoch 200 \
  --n-epoch 200 \
  --patience 20 \
  --mask-prob 0.15 \
  --output-dir logs/graph_tokenizer
```

Experiment matrix command:

```bash
python examples/graph_tokenizer/graph_tokenizer_trainer.py \
  --datasets qm9,molhiv,p-func,p-struct \
  --models bert,gte \
  --data-root data \
  --runs 3 \
  --pretrain-epoch 5 \
  --n-epoch 20 \
  --output-dir logs/graph_tokenizer \
  --resume \
  --keep-going
```

For matrix runs, the trainer also writes `graph_tokenizer_all_summary.json`, `graph_tokenizer_all_runs.csv`, `graph_tokenizer_all_paper_table.md`, `graph_tokenizer_all_paper_table.tex`, and `graph_tokenizer_all_manifest.json`. Use `--resume` to skip completed dataset/model summaries and `--keep-going` to record failed entries while continuing the matrix.


Reusable JSON config example:

```json
{
  "datasets": "qm9,molhiv,p-func,p-struct",
  "models": "bert,gte",
  "data_root": "data",
  "runs": 3,
  "pretrain_epoch": 5,
  "n_epoch": 20,
  "patience": 5,
  "bpe_backend": "python",
  "num_merges": 200,
  "vocab_size": 2048,
  "max_length": 128,
  "batch_size": 32,
  "output_dir": "logs/graph_tokenizer",
  "resume": true,
  "keep_going": true
}
```


Preflight before a long run:

```bash
python examples/graph_tokenizer/graph_tokenizer_trainer.py \
  --config configs/graph_tokenizer_matrix.json \
  --preflight
```

`--preflight` validates dataset directories, `data.pkl`/`data.pkl.gz`, `train_index.json`, `val_index.json`, `test_index.json`, model names, and the selected BPE backend without starting training.

Run it with:

```bash
python examples/graph_tokenizer/graph_tokenizer_trainer.py \
  --config configs/graph_tokenizer_matrix.json \
  --write-config logs/graph_tokenizer/resolved_config.json
```

Values in the JSON file become parser defaults; explicit CLI flags override them. `--write-config` saves the resolved arguments actually used by the run.

The trainer writes:

- `<output-dir>/<dataset>_<model>_summary.json`
- `<output-dir>/<dataset>_<model>_runs.csv`
- `<output-dir>/<dataset>_<model>_paper_table.md`
- `<output-dir>/<dataset>_<model>_paper_table.tex`
- `<output-dir>/<dataset>_<model>_manifest.json`
- `<output-dir>/checkpoints/<dataset>/<model>/run_<id>_best.npz`

Use `--no-save-checkpoint` to skip checkpoint files during quick tests. Set `--pretrain-epoch 0` to skip MLM pretraining.

Each summary JSON contains a `manifest` block, and the same block is also written as a standalone manifest file. It records the command arguments, Python executable/version/platform, and key package versions. Git metadata collection is disabled for these local-only runs, so the trainer does not execute Git commands.

Server tmux example:

```bash
tmux new -s gt_qm9
conda activate ggl_graphtokenizer_py310
cd ~/GmmaGL_newest/GammaGL
mkdir -p logs/graph_tokenizer

python examples/graph_tokenizer/graph_tokenizer_trainer.py \
  --dataset qm9 \
  --model bert \
  --data-root data \
  --runs 3 \
  --pretrain-epoch 5 \
  --n-epoch 20 \
  --patience 5 \
  --bpe-backend python \
  --num-merges 200 \
  --vocab-size 2048 \
  --max-length 128 \
  --batch-size 32 \
  --output-dir logs/graph_tokenizer \
  2>&1 | tee logs/graph_tokenizer/qm9_bert_train.log
```

Current preprocessed-data adapter

The trainer can now read a lightweight preprocessed directory without importing the full GammaGL loader stack:

```text
data/
  qm9/
    data.pkl
    train_index.json
    val_index.json
    test_index.json
```

`data.pkl` may contain graph dictionaries or `(graph, label)` tuples. Peptides-style `data.pkl.gz` is also accepted. The split files must contain integer graph indices. The primary metrics are aligned with the paper protocol:

- QM9: MAE
- OGBG-molhiv: ROC-AUC
- Peptides-func: AP
- Peptides-struct: Average MAE
- Classification datasets with `metric="acc"` or `metric="accuracy"`: ACC
