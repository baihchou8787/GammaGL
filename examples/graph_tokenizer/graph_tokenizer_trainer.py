"""Single-run GraphTokenizer training entrypoint.

The trainer deliberately owns only the train/validation/test pipeline.  It
uses GammaGL's serializers, BPE implementation, tokenizer, datasets, and
native TensorLayerX GraphBERT/GraphGTE models; it does not orchestrate paper
benchmarks. Formal GTE runs load the pinned official encoder through GammaGL's
native HF-to-TLX converter.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    aliases: tuple[str, ...]
    task_type: str
    label_width: int


DATASETS = (
    DatasetSpec("qm9", ("qm9",), "regression", 16),
    DatasetSpec("molhiv", ("molhiv", "ogbg-molhiv", "ogbg_molhiv"),
                "binary_classification", 1),
    DatasetSpec("peptides-struct", ("peptides-struct", "peptides_struct", "p-struct"),
                "multi_target_regression", 11),
)


# One small source of defaults.  CLI values override every item below.
_BASE_CONFIG = {
    "batch_size": 32,
    "learning_rate": 1e-5,
    "pretrain_learning_rate": 1e-4,
    "pretrain_epochs": 200,
    "finetune_epochs": 200,
    "max_length": 768,
    "max_position_embeddings": 8096,
    "bpe_merges": 2000,
    "bpe_min_frequency": 2,
    "num_serializations": 100,
    "early_stopping_patience": 20,
    "pretrain_early_stopping_patience": 20,
    "mask_probability": 0.09,
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,
    "mlm_validation_fraction": 0.1,
    "pooling": "mean",
}
PAPER_CONFIGS = {
    ("qm9", "bert"): {**_BASE_CONFIG},
    ("qm9", "gte"): {**_BASE_CONFIG, "pretrain_learning_rate": 5e-5,
                        "max_length": 8096, "max_position_embeddings": 8192},
    ("molhiv", "bert"): {**_BASE_CONFIG, "learning_rate": 5e-5},
    ("molhiv", "gte"): {**_BASE_CONFIG, "pretrain_learning_rate": 5e-5,
                           "learning_rate": 5e-5, "max_length": 8096,
                           "max_position_embeddings": 8192},
    ("peptides-struct", "bert"): {**_BASE_CONFIG, "batch_size": 16},
    ("peptides-struct", "gte"): {**_BASE_CONFIG, "batch_size": 16,
                                    "max_length": 8096, "max_position_embeddings": 8192},
}


class GraphRecord:
    def __init__(self, edge_index, x, edge_attr, y):
        self.edge_index, self.x, self.edge_attr, self.y = edge_index, x, edge_attr, y
        self.num_nodes = len(x)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_repo_on_path() -> None:
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def resolve_dataset(name: str) -> DatasetSpec:
    normalized = str(name).lower().replace("_", "-")
    for spec in DATASETS:
        if normalized in {item.replace("_", "-") for item in spec.aliases}:
            return spec
    raise ValueError("dataset must be one of: qm9, molhiv, peptides-struct")


def ensure_torch_backend():
    if os.environ.get("TL_BACKEND", "torch").lower() != "torch":
        raise RuntimeError("GraphTokenizer training requires TL_BACKEND=torch.")
    import tensorlayerx as tlx
    import torch

    if str(tlx.BACKEND).lower() != "torch":
        raise RuntimeError("GraphTokenizer training requires TensorLayerX's torch backend.")
    return torch


def set_seed(torch, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_list(value):
    for method in ("detach", "cpu"):
        if hasattr(value, method):
            value = getattr(value, method)()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return value.tolist() if hasattr(value, "tolist") else value


def is_sequence(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict))


def flatten_feature_ids(values) -> List[int]:
    if values is None:
        return []
    result = []
    for value in to_list(values):
        if is_sequence(value):
            if len(value) != 1 or is_sequence(value[0]):
                raise ValueError("Multi-dimensional features need a dataset-specific token adapter.")
            value = value[0]
        result.append(int(value))
    return result


def graph_label(value, width: int, allow_nan: bool) -> List[float]:
    values = to_list(value)
    while is_sequence(values) and len(values) == 1 and is_sequence(values[0]):
        values = values[0]
    if not is_sequence(values):
        values = [values]
    if len(values) != width:
        raise ValueError(f"Expected {width} labels, received {len(values)}.")
    result = [float(item) for item in values]
    if not allow_nan and not all(math.isfinite(item) for item in result):
        raise ValueError("Labels must be finite for this dataset.")
    return result


def graph_from_gammagl(graph, spec: DatasetSpec) -> GraphRecord:
    edge_index = to_list(getattr(graph, "edge_index", [[], []]))
    if len(edge_index) != 2:
        edge_index = [[pair[0] for pair in edge_index], [pair[1] for pair in edge_index]]
    return GraphRecord(
        [[int(item) for item in edge_index[0]], [int(item) for item in edge_index[1]]],
        flatten_feature_ids(getattr(graph, "x", None)),
        flatten_feature_ids(getattr(graph, "edge_attr", None)),
        graph_label(getattr(graph, "y", None), spec.label_width,
                    allow_nan=spec.name == "peptides-struct"),
    )


def validate_splits(size: int, indices: Dict[str, Sequence[int]]) -> None:
    if set(indices) != {"train", "val", "test"}:
        raise ValueError("Dataset must define exactly train, val, and test splits.")
    seen = set()
    for split in ("train", "val", "test"):
        for index in indices[split]:
            if not 0 <= int(index) < size or int(index) in seen:
                raise ValueError("Dataset splits must be in range and disjoint.")
            seen.add(int(index))


def load_dataset_splits(data_root: str, spec: DatasetSpec) -> Dict[str, List[GraphRecord]]:
    ensure_repo_on_path()
    from gammagl import datasets

    classes = {"qm9": "QM9", "molhiv": "OGBGMolHIV", "peptides-struct": "PeptidesStruct"}
    dataset = getattr(datasets, classes[spec.name])(root=str(data_root))
    indices = {name: [int(item) for item in to_list(values)]
               for name, values in dataset.get_idx_split().items()}
    validate_splits(len(dataset), indices)
    graphs = [graph_from_gammagl(dataset[index], spec) for index in range(len(dataset))]
    return {name: [graphs[index] for index in indices[name]] for name in indices}


def synthetic_splits(spec: DatasetSpec) -> Dict[str, List[GraphRecord]]:
    labels = []
    for index in range(5):
        if spec.name == "molhiv":
            labels.append([float(index % 2)])
        elif spec.name == "peptides-struct":
            labels.append([float(index + column) / 10 for column in range(11)])
        else:
            labels.append([float(index + column) / 10 for column in range(16)])
    graphs = [
        GraphRecord([[0, 1], [1, 2]], [1, 2, 3], [1, 2], labels[0]),
        GraphRecord([[0, 1], [1, 2]], [2, 3, 4], [1, 3], labels[1]),
        GraphRecord([[0, 1], [1, 2]], [3, 1, 2], [2, 1], labels[2]),
        GraphRecord([[0, 1], [1, 2]], [4, 2, 1], [2, 3], labels[3]),
        GraphRecord([[0, 1], [1, 2]], [1, 4, 2], [3, 1], labels[4]),
    ]
    return {"train": graphs[:3], "val": graphs[3:4], "test": graphs[4:]}


def make_tokenizer(args, train_graphs: Sequence[GraphRecord]):
    ensure_repo_on_path()
    from gammagl.transforms.graph_bpe import GraphBPE
    from gammagl.transforms.graph_serializer import FrequencyGuidedEulerianSerializer
    from gammagl.transforms.graph_tokenizer import GraphTokenizer

    tokenizer = GraphTokenizer(
        serializer=FrequencyGuidedEulerianSerializer(),
        bpe=GraphBPE(num_merges=args.bpe_merges, min_frequency=args.bpe_min_frequency,
                     backend=args.bpe_backend),
    )
    # The only tokenizer fit in this program receives the downstream train split.
    tokenizer.fit(train_graphs, graph_ids=list(range(len(train_graphs))),
                  num_realizations=args.num_serializations)
    return tokenizer


def truncate(sequence: Sequence[int], max_length: int, sep_token: int) -> List[int]:
    if len(sequence) <= max_length:
        return list(sequence)
    if max_length < 2:
        raise ValueError("max_length must be at least two tokens.")
    return [*sequence[:max_length - 1], int(sep_token)]


def encode_splits(tokenizer, splits, args) -> Dict[str, List[Dict[str, Any]]]:
    encoded = {}
    for split_name, graphs in splits.items():
        records = []
        for graph_id, graph in enumerate(graphs):
            for start in tokenizer.realization_start_nodes(graph, args.num_serializations):
                result = tokenizer.encode_graph(graph, start_node=start)
                records.append({
                    "input_ids": truncate(result.input_ids, args.max_length,
                                          tokenizer.special_tokens.sep_token_id),
                    "labels": list(graph.y), "graph_id": graph_id,
                })
        if not records:
            raise ValueError(f"{split_name} split is empty.")
        encoded[split_name] = records
    return encoded


def normalize_labels(encoded, spec: DatasetSpec):
    if spec.name == "molhiv":
        return None, 2
    if spec.name == "qm9":
        target = 2  # HOMO in QM9's canonical target order.
        train_values = [record["labels"][target] for record in encoded["train"]]
        names = ["homo"]
    else:
        target = None
        train_values = [[record["labels"][column] for record in encoded["train"]
                         if math.isfinite(record["labels"][column])]
                        for column in range(11)]
        names = [f"target_{column}" for column in range(11)]
    if spec.name == "qm9":
        train_values = [train_values]
    means, stds = [], []
    for values in train_values:
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("Normalization must be fitted from finite training labels only.")
        mean = sum(values) / len(values)
        means.append(mean)
        stds.append(max(math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)), 1e-12))
    for records in encoded.values():
        for record in records:
            values = [record["labels"][target]] if target is not None else record["labels"]
            record["labels"] = [
                ((value - means[index]) / stds[index] if math.isfinite(value) else float("nan"))
                for index, value in enumerate(values)
            ]
    return {"mean": means, "std": stds, "targets": names}, len(means)


def group_records(records: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record["graph_id"]), []).append(record)
    return list(grouped.values())


def make_loader(torch, records, batch_size, pad_token_id, shuffle, choose_variant):
    data = group_records(records) if choose_variant else list(records)

    def collate(items):
        if choose_variant:
            items = [random.choice(group) for group in items]
        length = max(len(item["input_ids"]) for item in items)
        input_ids = torch.full((len(items), length), int(pad_token_id), dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        for row, item in enumerate(items):
            tokens = torch.as_tensor(item["input_ids"], dtype=torch.long)
            input_ids[row, :len(tokens)] = tokens
            attention[row, :len(tokens)] = 1
        return (input_ids, attention,
                torch.as_tensor([item["labels"] for item in items], dtype=torch.float32),
                torch.as_tensor([item["graph_id"] for item in items], dtype=torch.long))

    return torch.utils.data.DataLoader(data, batch_size=max(1, int(batch_size)),
                                      shuffle=bool(shuffle), collate_fn=collate)


def split_mlm_records(records, fraction: float, seed: int):
    groups = group_records(records)
    if len(groups) < 2:
        raise ValueError("MLM validation needs at least two training graphs.")
    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    validation_count = min(len(groups) - 1, max(1, round(len(groups) * float(fraction))))
    validation = {order[index] for index in range(validation_count)}
    train = [record for index, group in enumerate(groups) for record in group if index not in validation]
    val = [record for index, group in enumerate(groups) for record in group if index in validation]
    return train, val


def set_model_mode(model, training: bool) -> None:
    method = getattr(model, "set_train" if training else "set_eval", None)
    if callable(method):
        method()
    else:
        (model.train if training else model.eval)()


def make_model(args, vocab_size: int, pad_token_id: int, output_dim: int):
    ensure_repo_on_path()
    from gammagl.models.graph_bert import GraphBERT
    from gammagl.models.graph_gte import GraphGTE

    kwargs = {"vocab_size": vocab_size, "output_dim": output_dim,
              "pad_token_id": pad_token_id, "task_type": resolve_dataset(args.dataset).task_type,
              "pooling": args.pooling,
              "max_position_embeddings": args.max_position_embeddings}
    for name in ("hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size"):
        value = getattr(args, name)
        if value is not None:
            kwargs[name] = value
    if args.encoder == "gte" and not args.allow_random_gte_init:
        model = GraphGTE.from_pretrained(cache_dir=args.gte_checkpoint_cache_dir, **kwargs)
        gte_initialization_summary(model, args)
        return model
    return (GraphBERT if args.encoder == "bert" else GraphGTE)(**kwargs)


def gte_initialization_summary(model, args):
    if args.encoder != "gte":
        return None
    manifest = getattr(model, "pretrained_manifest", None)
    if manifest is None:
        if args.allow_random_gte_init:
            return {"pretrained": False, "reproduction": False,
                    "mode": "random_development"}
        raise RuntimeError("Formal GTE training requires the pinned official checkpoint.")
    valid = (manifest.get("pretrained") is True and manifest.get("reproduction") is True
             and manifest.get("encoder_coverage") == 1.0
             and not manifest.get("missing") and not manifest.get("unexpected")
             and not manifest.get("shape_mismatches"))
    if not valid:
        raise RuntimeError("Formal GTE training requires complete official checkpoint conversion.")
    return manifest


def mask_for_mlm(torch, input_ids, attention, tokenizer, probability, generator):
    labels = input_ids.clone()
    blocked = ~attention.bool()
    for token in (tokenizer.special_tokens.pad_token_id, tokenizer.special_tokens.cls_token_id,
                  tokenizer.special_tokens.sep_token_id, tokenizer.special_tokens.component_sep_token_id):
        blocked |= input_ids.eq(int(token))
    selected = torch.rand(input_ids.shape, generator=generator, device=input_ids.device) < probability
    selected &= ~blocked
    for row in range(len(input_ids)):
        if not selected[row].any():
            available = torch.where(~blocked[row])[0]
            if len(available):
                selected[row, available[0]] = True
    labels[~selected] = -100
    masked = input_ids.clone()
    masked[selected] = int(tokenizer.special_tokens.mask_token_id)
    return masked, labels


def mlm_loss(torch, model, input_ids, attention, tokenizer, probability, generator):
    masked, labels = mask_for_mlm(torch, input_ids, attention, tokenizer, probability, generator)
    logits = model(masked, attention, task="mlm")
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                                             ignore_index=-100)


def supervised_loss(torch, logits, labels, spec: DatasetSpec):
    if spec.name == "molhiv":
        valid = torch.isfinite(labels.reshape(-1))
        return torch.nn.functional.cross_entropy(logits[valid], labels.reshape(-1)[valid].long())
    valid = torch.isfinite(labels)
    if not valid.any():
        raise ValueError("Regression batch has no finite labels.")
    return (torch.nn.functional.l1_loss if spec.name == "peptides-struct" else torch.nn.functional.mse_loss)(
        logits[valid], labels[valid])


def save_checkpoint(torch, path: Path, model, epoch: int, score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save({"model": model.state_dict(), "epoch": int(epoch), "score": float(score)}, temporary)
    os.replace(temporary, path)


def restore_checkpoint(torch, path: Path, model, device):
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    return state


def train_mlm_epoch(torch, model, loader, optimizer, tokenizer, args, device, seed):
    set_model_mode(model, True)
    generator = torch.Generator(device=device).manual_seed(seed)
    total, count = 0.0, 0
    for input_ids, attention, _, _ in loader:
        input_ids, attention = input_ids.to(device), attention.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = mlm_loss(torch, model, input_ids, attention, tokenizer,
                        args.mask_probability, generator)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        total += float(loss.detach()) * len(input_ids)
        count += len(input_ids)
    return total / max(count, 1)


def evaluate_mlm(torch, model, loader, tokenizer, args, device, seed):
    set_model_mode(model, False)
    generator = torch.Generator(device=device).manual_seed(seed)
    total, count = 0.0, 0
    with torch.no_grad():
        for input_ids, attention, _, _ in loader:
            input_ids, attention = input_ids.to(device), attention.to(device)
            loss = mlm_loss(torch, model, input_ids, attention, tokenizer,
                            args.mask_probability, generator)
            total += float(loss) * len(input_ids)
            count += len(input_ids)
    return total / max(count, 1)


def run_mlm(torch, model, train_loader, val_loader, tokenizer, args, device, checkpoint: Path):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.pretrain_learning_rate,
                                  weight_decay=args.weight_decay)
    best, stale, history = None, 0, []
    for epoch in range(1, args.pretrain_epochs + 1):
        train_loss = train_mlm_epoch(torch, model, train_loader, optimizer, tokenizer, args, device,
                                     args.seed + epoch)
        val_loss = evaluate_mlm(torch, model, val_loader, tokenizer, args, device, args.seed)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if best is None or val_loss < best:
            best, stale = val_loss, 0
            save_checkpoint(torch, checkpoint, model, epoch, val_loss)
        else:
            stale += 1
            if stale >= args.pretrain_early_stopping_patience:
                break
    state = restore_checkpoint(torch, checkpoint, model, device)
    return {"best_epoch": state["epoch"], "best_val_loss": state["score"], "history": history}


def roc_auc(labels: Sequence[float], scores: Sequence[float]) -> float:
    positives = sum(label == 1 for label in labels)
    negatives = sum(label == 0 for label in labels)
    if not positives or not negatives:
        return float("nan")
    ranked = sorted(enumerate(scores), key=lambda item: item[1])
    rank_sum, index = 0.0, 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][1] == ranked[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2
        rank_sum += average_rank * sum(labels[ranked[item][0]] == 1 for item in range(index, end))
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def denormalize(torch, values, normalizer):
    if normalizer is None:
        return values
    mean = torch.as_tensor(normalizer["mean"], dtype=values.dtype, device=values.device)
    std = torch.as_tensor(normalizer["std"], dtype=values.dtype, device=values.device)
    return values * std + mean


def evaluate_downstream(torch, model, loader, spec, normalizer, device):
    set_model_mode(model, False)
    total_loss, count, grouped = 0.0, 0, {}
    with torch.no_grad():
        for input_ids, attention, labels, graph_ids in loader:
            input_ids, attention, labels = input_ids.to(device), attention.to(device), labels.to(device)
            logits = model(input_ids, attention, task="supervised")
            loss = supervised_loss(torch, logits, labels, spec)
            total_loss += float(loss) * len(input_ids)
            count += len(input_ids)
            predictions = torch.softmax(logits, -1)[:, 1:2] if spec.name == "molhiv" else logits
            for graph_id, label, prediction in zip(graph_ids.tolist(), labels.cpu(), predictions.cpu()):
                grouped.setdefault(graph_id, {"label": label, "predictions": []})["predictions"].append(prediction)
    labels = torch.stack([item["label"] for item in grouped.values()])
    predictions = torch.stack([torch.stack(item["predictions"]).mean(0) for item in grouped.values()])
    raw_labels, raw_predictions = denormalize(torch, labels, normalizer), denormalize(torch, predictions, normalizer)
    if spec.name == "molhiv":
        metric = roc_auc(labels.reshape(-1).tolist(), predictions.reshape(-1).tolist())
        detail = {"rocauc": metric, "metric_space": "positive_class_probability"}
    elif spec.name == "qm9":
        metric = float(torch.abs(raw_labels - raw_predictions).mean())
        detail = {"homo_mae": metric, "metric_space": "raw_label"}
    else:
        per_target = torch.nanmean(torch.abs(raw_labels - raw_predictions), dim=0)
        metric = float(per_target.mean())
        detail = {"average_mae": metric, "per_target_mae": per_target.tolist(), "metric_space": "raw_label"}
    return {"loss": total_loss / max(count, 1), "metric": metric,
            "num_graphs": len(grouped), **detail}


def train_downstream_epoch(torch, model, loader, optimizer, spec, args, device):
    set_model_mode(model, True)
    total, count = 0.0, 0
    for input_ids, attention, labels, _ in loader:
        input_ids, attention, labels = input_ids.to(device), attention.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = supervised_loss(torch, model(input_ids, attention, task="supervised"), labels, spec)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        total += float(loss.detach()) * len(input_ids)
        count += len(input_ids)
    return total / max(count, 1)


def run_finetuning(torch, model, train_loader, val_loader, test_loader, spec, normalizer, args, device, checkpoint):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best, stale, history = None, 0, []
    higher_is_better = spec.name == "molhiv"
    for epoch in range(1, args.finetune_epochs + 1):
        train_loss = train_downstream_epoch(torch, model, train_loader, optimizer, spec, args, device)
        validation = evaluate_downstream(torch, model, val_loader, spec, normalizer, device)
        score = validation["metric"]
        history.append({"epoch": epoch, "train_loss": train_loss, "validation": validation})
        improved = best is None or (score > best if higher_is_better else score < best)
        if improved:
            best, stale = score, 0
            save_checkpoint(torch, checkpoint, model, epoch, score)
        else:
            stale += 1
            if stale >= args.early_stopping_patience:
                break
    state = restore_checkpoint(torch, checkpoint, model, device)
    # This is the only downstream test evaluation.
    test = evaluate_downstream(torch, model, test_loader, spec, normalizer, device)
    return {"best_epoch": state["epoch"], "best_validation_metric": state["score"],
            "history": history, "test": test}


def apply_preset(args):
    config = PAPER_CONFIGS[(resolve_dataset(args.dataset).name, args.encoder)]
    for name, value in config.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


def run_training(args):
    args = apply_preset(args)
    torch = ensure_torch_backend()
    set_seed(torch, args.seed)
    spec = resolve_dataset(args.dataset)
    splits = synthetic_splits(spec) if args.smoke else load_dataset_splits(args.data_root, spec)
    tokenizer = make_tokenizer(args, splits["train"])
    encoded = encode_splits(tokenizer, splits, args)
    normalizer, output_dim = normalize_labels(encoded, spec)
    tokenizer.validate_model_vocab(tokenizer.max_token_id + 1)
    device = torch.device(args.device)
    model = make_model(args, tokenizer.max_token_id + 1, tokenizer.special_tokens.pad_token_id, output_dim).to(device)
    mlm_train, mlm_val = split_mlm_records(encoded["train"], args.mlm_validation_fraction, args.seed)
    loader_args = {"torch": torch, "batch_size": args.batch_size,
                   "pad_token_id": tokenizer.special_tokens.pad_token_id}
    mlm_train_loader = make_loader(**loader_args, records=mlm_train,
                                   shuffle=True, choose_variant=True)
    mlm_val_loader = make_loader(**loader_args, records=mlm_val,
                                 shuffle=False, choose_variant=False)
    train_loader = make_loader(**loader_args, records=encoded["train"],
                               shuffle=True, choose_variant=True)
    val_loader = make_loader(**loader_args, records=encoded["val"],
                             shuffle=False, choose_variant=False)
    test_loader = make_loader(**loader_args, records=encoded["test"],
                              shuffle=False, choose_variant=False)
    directory = Path(args.output_dir) / spec.name / args.encoder
    mlm = run_mlm(torch, model, mlm_train_loader, mlm_val_loader, tokenizer, args, device,
                  directory / "best_mlm.pt")
    finetune = run_finetuning(torch, model, train_loader, val_loader, test_loader, spec, normalizer,
                              args, device, directory / "best_finetune.pt")
    summary = {"dataset": spec.name, "encoder": args.encoder, "seed": args.seed,
            "tokenizer_fit_split": "train", "normalization_fit_split": "train",
            "mlm": mlm, "finetune": finetune,
            "checkpoints": {"mlm": str(directory / "best_mlm.pt"),
                            "finetune": str(directory / "best_finetune.pt")}}
    if args.encoder == "gte":
        summary["gte_initialization"] = gte_initialization_summary(model, args)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="Train GraphTokenizer once with validation-selected checkpoints.")
    parser.add_argument("--dataset", default="qm9")
    parser.add_argument("--encoder", "--model", dest="encoder", choices=("bert", "gte"), default="bert")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="runs/graph_tokenizer")
    parser.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float)
    parser.add_argument("--pretrain-learning-rate", "--pretrain-lr", dest="pretrain_learning_rate", type=float)
    parser.add_argument("--pretrain-epochs", "--pretrain-epoch", dest="pretrain_epochs", type=int)
    parser.add_argument("--finetune-epochs", "--n-epoch", dest="finetune_epochs", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--max-position-embeddings", type=int)
    parser.add_argument("--bpe-merges", "--num-merges", dest="bpe_merges", type=int)
    parser.add_argument("--bpe-min-frequency", "--min-frequency", dest="bpe_min_frequency", type=int)
    parser.add_argument("--bpe-backend", choices=("python", "auto", "cpp"), default="python")
    parser.add_argument("--num-serializations", "--num-realizations", dest="num_serializations", type=int)
    parser.add_argument("--early-stopping-patience", "--patience", dest="early_stopping_patience", type=int)
    parser.add_argument("--pretrain-early-stopping-patience", type=int)
    parser.add_argument("--mlm-validation-fraction", type=float)
    parser.add_argument("--mask-probability", "--mask-prob", dest="mask_probability", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--pooling", choices=("mean", "cls"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gte-checkpoint-cache-dir")
    parser.add_argument("--allow-random-gte-init", action="store_true",
                        help="Development only; this is not a paper reproduction run.")
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--num-hidden-layers", type=int)
    parser.add_argument("--num-attention-heads", type=int)
    parser.add_argument("--intermediate-size", type=int)
    parser.add_argument("--smoke", action="store_true", help="Use five tiny synthetic graphs and small model defaults.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.smoke:
        smoke_overrides = {"batch_size": 2, "pretrain_epochs": 2, "finetune_epochs": 2,
                           "max_length": 32, "max_position_embeddings": 32,
                           "bpe_merges": 8, "num_serializations": 1,
                           "early_stopping_patience": 1, "pretrain_early_stopping_patience": 1,
                           "hidden_size": 16, "num_hidden_layers": 1,
                           "num_attention_heads": 4, "intermediate_size": 32}
        for name, value in smoke_overrides.items():
            if getattr(args, name) is None:
                setattr(args, name, value)
    summary = run_training(args)
    print(json.dumps(summary, indent=2, allow_nan=True))
    return summary


if __name__ == "__main__":
    main()
