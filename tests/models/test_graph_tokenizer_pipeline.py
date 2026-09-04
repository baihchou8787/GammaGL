import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def trainer():
    spec = importlib.util.spec_from_file_location(
        "graph_tokenizer_pipeline_trainer", ROOT / "examples" / "graph_tokenizer" / "graph_tokenizer_trainer.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _qm9_graphs(trainer, count, offset=0):
    return [trainer.GraphRecord(
        [[0, 1], [1, 2]], [index % 4 + 1, 2, 3], [1, 2],
        [float(index + column) for column in range(16)])
        for index in range(offset, offset + count)]


def _record(graph_id, labels):
    return {"input_ids": [3, 8, 4], "labels": labels, "graph_id": graph_id}


class FixedLogits(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.as_tensor(logits, dtype=torch.float32))

    def forward(self, input_ids, attention_mask=None, task=None):
        return self.logits[:len(input_ids)]


def test_tiny_pipeline_runs_pretrain_restore_finetune_and_final_test(trainer, monkeypatch, tmp_path):
    splits = {
        "train": _qm9_graphs(trainer, 8),
        "val": _qm9_graphs(trainer, 4, 20),
        "test": _qm9_graphs(trainer, 4, 40),
    }
    monkeypatch.setattr(trainer, "synthetic_splits", lambda _spec: splits)

    summary = trainer.main([
        "--smoke", "--dataset", "qm9", "--encoder", "bert", "--device", "cpu",
        "--output-dir", str(tmp_path),
    ])

    assert Path(summary["checkpoints"]["mlm"]).is_file()
    assert Path(summary["checkpoints"]["finetune"]).is_file()
    assert all(math.isfinite(row["train_loss"]) and math.isfinite(row["val_loss"])
               for row in summary["mlm"]["history"])
    assert all(math.isfinite(row["train_loss"]) and math.isfinite(row["validation"]["metric"])
               for row in summary["finetune"]["history"])
    assert math.isfinite(summary["finetune"]["test"]["metric"])


def test_formal_gte_preset_uses_the_official_loader(trainer, monkeypatch):
    from gammagl.models.graph_gte import GraphGTE

    args = trainer.build_parser().parse_args(["--dataset", "qm9", "--encoder", "gte"])
    args = trainer.apply_preset(args)
    calls, loaded = {}, SimpleNamespace(pretrained_manifest={
        "pretrained": True, "reproduction": True, "encoder_coverage": 1.0,
        "missing": [], "unexpected": [], "shape_mismatches": [],
    })

    def from_pretrained(_cls, **kwargs):
        calls.update(kwargs)
        return loaded

    monkeypatch.setattr(GraphGTE, "from_pretrained", classmethod(from_pretrained))

    assert trainer.make_model(args, vocab_size=31, pad_token_id=0, output_dim=1) is loaded
    assert calls["vocab_size"] == 31
    assert calls["max_position_embeddings"] == 8192
    assert trainer.gte_initialization_summary(loaded, args)["reproduction"] is True


def test_formal_gte_loader_failure_is_not_replaced_with_random_initialization(trainer, monkeypatch):
    from gammagl.models.graph_gte import GraphGTE

    args = trainer.apply_preset(trainer.build_parser().parse_args([
        "--dataset", "qm9", "--encoder", "gte",
    ]))

    def rejected_loader(_cls, **_kwargs):
        raise RuntimeError("GTE checkpoint SHA-256 mismatch")

    monkeypatch.setattr(GraphGTE, "from_pretrained", classmethod(rejected_loader))

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        trainer.make_model(args, vocab_size=31, pad_token_id=0, output_dim=1)


def test_explicit_random_gte_development_mode_is_not_a_reproduction(trainer, monkeypatch):
    from gammagl.models.graph_gte import GraphGTE

    args = trainer.apply_preset(trainer.build_parser().parse_args([
        "--dataset", "qm9", "--encoder", "gte", "--allow-random-gte-init",
        "--hidden-size", "16", "--num-hidden-layers", "1",
        "--num-attention-heads", "4", "--intermediate-size", "32",
    ]))
    monkeypatch.setattr(GraphGTE, "from_pretrained", classmethod(
        lambda *_args, **_kwargs: pytest.fail("random development mode must not load a checkpoint")))

    model = trainer.make_model(args, vocab_size=31, pad_token_id=0, output_dim=1)

    assert isinstance(model, GraphGTE)
    assert trainer.gte_initialization_summary(model, args) == {
        "pretrained": False, "reproduction": False, "mode": "random_development",
    }


def test_train_only_bpe_and_normalization_do_not_use_validation_or_test(trainer):
    train = _qm9_graphs(trainer, 2)
    held_out = trainer.GraphRecord([[0], [1]], [101, 102], [5], [1000.0] * 16)
    args = SimpleNamespace(bpe_merges=2, bpe_min_frequency=2, bpe_backend="python",
                           num_serializations=1)
    tokenizer = trainer.make_tokenizer(args, train)
    before = (dict(tokenizer.vocabulary), list(tokenizer.bpe.codebook.merge_rules))

    held_out_encoding = tokenizer.encode_graph(held_out)
    encoded = {
        "train": [_record(0, [0.0, 0.0, 1.0] + [0.0] * 13),
                  _record(1, [0.0, 0.0, 3.0] + [0.0] * 13)],
        "val": [_record(0, [0.0, 0.0, 1000.0] + [0.0] * 13)],
        "test": [_record(0, [0.0, 0.0, -1000.0] + [0.0] * 13)],
    }
    normalizer, output_dim = trainer.normalize_labels(encoded, trainer.resolve_dataset("qm9"))

    assert tokenizer.special_tokens.unk_token_id in held_out_encoding.input_ids
    assert (tokenizer.vocabulary, tokenizer.bpe.codebook.merge_rules) == before
    assert output_dim == 1
    assert normalizer["mean"] == [2.0]
    assert normalizer["std"] == [1.0]


def test_qm9_metric_uses_homo_inverse_normalization_and_raw_mae(trainer):
    normalizer = {"mean": [10.0], "std": [2.0], "targets": ["homo"]}
    loader = trainer.make_loader(torch, [_record(0, [0.0])], 1, 0, False, False)

    result = trainer.evaluate_downstream(
        torch, FixedLogits([[1.0]]), loader, trainer.resolve_dataset("qm9"), normalizer,
        torch.device("cpu"))

    assert result["metric_space"] == "raw_label"
    assert result["homo_mae"] == pytest.approx(2.0)


def test_molhiv_metric_averages_serializations_before_roc_auc(trainer):
    probabilities = [0.1, 0.3, 0.7, 0.9]
    logits = [[math.log(1 - probability), math.log(probability)] for probability in probabilities]
    records = [_record(0, [0.0]), _record(0, [0.0]), _record(1, [1.0]), _record(1, [1.0])]
    loader = trainer.make_loader(torch, records, 4, 0, False, False)

    result = trainer.evaluate_downstream(
        torch, FixedLogits(logits), loader, trainer.resolve_dataset("molhiv"), None,
        torch.device("cpu"))

    assert result["num_graphs"] == 2
    assert result["metric_space"] == "positive_class_probability"
    assert result["rocauc"] == pytest.approx(1.0)


def test_peptides_metric_inverse_transforms_each_target_before_averaging(trainer):
    normalizer = {"mean": [10.0] * 11, "std": [2.0] * 11, "targets": ["x"] * 11}
    loader = trainer.make_loader(torch, [_record(0, [0.0] * 11)], 1, 0, False, False)

    result = trainer.evaluate_downstream(
        torch, FixedLogits([[0.5] * 11]), loader,
        trainer.resolve_dataset("peptides-struct"), normalizer, torch.device("cpu"))

    assert result["average_mae"] == pytest.approx(1.0)
    assert result["per_target_mae"] == pytest.approx([1.0] * 11)


def test_mlm_early_stopping_restores_the_best_checkpoint(trainer, monkeypatch, tmp_path):
    model = torch.nn.Linear(1, 1, bias=False)
    train_epochs, validation_losses = [], iter((0.5, 0.3, 0.4))
    args = SimpleNamespace(pretrain_learning_rate=1e-3, weight_decay=0.0,
                           pretrain_epochs=5, pretrain_early_stopping_patience=1, seed=7)

    def train(*_args):
        train_epochs.append(len(train_epochs) + 1)
        model.weight.data.fill_(train_epochs[-1])
        return 0.1

    monkeypatch.setattr(trainer, "train_mlm_epoch", train)
    monkeypatch.setattr(trainer, "evaluate_mlm", lambda *_args: next(validation_losses))
    result = trainer.run_mlm(torch, model, None, None, None, args, torch.device("cpu"), tmp_path / "mlm.pt")

    assert result["best_epoch"] == 2
    assert len(result["history"]) == 3
    assert float(model.weight.detach()) == 2.0


def test_finetune_early_stopping_restores_best_before_test(trainer, monkeypatch, tmp_path):
    model = torch.nn.Linear(1, 1, bias=False)
    epochs, validation_metrics, validation_calls, test_weights = [], iter((2.0, 1.0, 3.0)), [], []
    args = SimpleNamespace(learning_rate=1e-3, weight_decay=0.0, finetune_epochs=5,
                           early_stopping_patience=1)

    def train(*_args):
        epochs.append(len(epochs) + 1)
        model.weight.data.fill_(epochs[-1])
        return 0.1

    def evaluate(*_args):
        if len(validation_calls) < 3:
            validation_calls.append(True)
            return {"loss": 0.1, "metric": next(validation_metrics)}
        test_weights.append(float(model.weight.detach()))
        return {"loss": 0.1, "metric": test_weights[-1]}

    monkeypatch.setattr(trainer, "train_downstream_epoch", train)
    monkeypatch.setattr(trainer, "evaluate_downstream", evaluate)
    result = trainer.run_finetuning(
        torch, model, None, None, None, trainer.resolve_dataset("qm9"), None, args,
        torch.device("cpu"), tmp_path / "finetune.pt")

    assert result["best_epoch"] == 2
    assert len(result["history"]) == 3
    assert result["test"]["metric"] == 2.0
    assert test_weights == [2.0]
