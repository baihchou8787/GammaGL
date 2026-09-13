import importlib.util
import io
import json
import math
import pickle
import sys
import tarfile
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


class FixedLengthTokenizer:
    def __init__(self, input_ids):
        from gammagl.transforms.graph_tokenizer import GraphTokenizer

        self.input_ids = input_ids
        self.special_tokens = SimpleNamespace(sep_token_id=4)
        self.strict_tokenizer = GraphTokenizer()

    @staticmethod
    def realization_start_nodes(_graph, _num_serializations):
        return [None]

    def encode_graph(self, _graph, start_node=None):
        return SimpleNamespace(input_ids=list(self.input_ids))

    def validate_token_sequences(self, token_sequences, max_length):
        return self.strict_tokenizer.validate_token_sequences(token_sequences, max_length)


def _write_graph_tokenizer_bundle(path):
    qm9_properties = {name: float(index) for index, name in enumerate(
        ("mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0",
         "u298", "h298", "g298", "cv", "u0_atom", "u298_atom",
         "h298_atom", "g298_atom"))}
    datasets = {
        "qm9": [{"edge_index": [[0], [1]], "x": [6, 8], "edge_attr": [1],
                 "properties": qm9_properties}],
        "ogbg-molhiv": [({"edges": [[0], [1]], "node_type_ids": [6, 8],
                           "edge_type_ids": [1]}, [1])],
        "peptides-struct": [({"edges": [[0], [1]], "node_token_ids": [[5], [9]],
                               "edge_token_ids": [[2]]}, [float(index) for index in range(11)])],
    }
    with tarfile.open(path, "w:gz") as archive:
        for dataset_name, samples in datasets.items():
            files = {
                f"release/{dataset_name}/data.pkl": pickle.dumps(samples),
                f"release/{dataset_name}/train_index.json": b"[0]",
                f"release/{dataset_name}/val_index.json": b"[]",
                f"release/{dataset_name}/test_index.json": b"[]",
            }
            for name, content in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))


class FixedLogits(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.as_tensor(logits, dtype=torch.float32))

    def forward(self, input_ids, attention_mask=None, task=None):
        return self.logits[:len(input_ids)]


def test_prepare_data_mode_materializes_shared_bundle_and_training_reuses_it(
        trainer, tmp_path, monkeypatch, capsys):
    from gammagl.datasets import _graph_tokenizer_download as download

    bundle = tmp_path / "bundle.tar.gz"
    _write_graph_tokenizer_bundle(bundle)
    monkeypatch.setattr(download, "PAPER_DATA_BUNDLE_SHA256", download.sha256_file(bundle))
    monkeypatch.setenv(download.DATA_BUNDLE_ENV, str(bundle))
    monkeypatch.setattr("gammagl.data.download.download_google_url",
                        lambda *_args, **_kwargs: pytest.fail("prepare mode must reuse its local bundle"))
    root = tmp_path / "data"

    assert trainer.main(["--prepare-data", "--data-root", str(root)]) is None
    first_bytes = (root / "qm9" / "raw" / "data.pkl").read_bytes()
    assert "QM9: READY" in capsys.readouterr().out

    assert trainer.main(["--prepare-data", "--data-root", str(root)]) is None
    assert (root / "qm9" / "raw" / "data.pkl").read_bytes() == first_bytes
    assert trainer.load_dataset_splits(str(root), trainer.resolve_dataset("qm9"))["train"]


def test_training_requires_prepared_data_without_network(trainer, tmp_path, monkeypatch):
    monkeypatch.delenv("GAMMAGL_GRAPH_TOKENIZER_DATA_BUNDLE", raising=False)
    monkeypatch.setattr("gammagl.data.download.download_google_url",
                        lambda *_args, **_kwargs: pytest.fail("training must not download the shared bundle"))

    with pytest.raises(FileNotFoundError, match="GraphTokenizer dataset is not prepared"):
        trainer.load_dataset_splits(str(tmp_path / "empty"), trainer.resolve_dataset("qm9"))


@pytest.mark.parametrize("split_name", ("train", "val", "test"))
def test_encode_splits_rejects_overlong_sequences_for_every_split(trainer, split_name):
    args = SimpleNamespace(dataset="qm9", num_serializations=1, max_length=8)
    tokenizer = FixedLengthTokenizer([3, 8, 9, 10, 11, 12, 13, 14, 4])
    graph = SimpleNamespace(y=[0.0] * 16)

    with pytest.raises(ValueError) as error:
        trainer.encode_splits(tokenizer, {split_name: [graph]}, args)

    message = str(error.value)
    assert "dataset=qm9" in message
    assert f"split={split_name}" in message
    assert "sample=0" in message
    assert "length=9" in message
    assert "max_length=8" in message


def test_encode_splits_allows_sequence_at_max_length(trainer):
    args = SimpleNamespace(dataset="qm9", num_serializations=1, max_length=8)
    tokenizer = FixedLengthTokenizer([3, 8, 9, 10, 11, 12, 13, 4])
    graph = SimpleNamespace(y=[0.0] * 16)

    encoded = trainer.encode_splits(tokenizer, {"train": [graph]}, args)

    assert encoded["train"][0]["input_ids"] == [3, 8, 9, 10, 11, 12, 13, 4]


def _run_result(seed, run_index, metric):
    return {
        "dataset": "qm9",
        "encoder": "bert",
        "seed": seed,
        "run_index": run_index,
        "metric_name": "MAE",
        "metric": metric,
    }


def test_default_seeds_are_five_explicit_runs(trainer):
    assert trainer.build_parser().parse_args([]).seeds == [42, 43, 44, 45, 46]


def test_main_runs_requested_single_experiment(trainer, monkeypatch, tmp_path):
    calls = []

    def run_single(args, config, seed, run_index):
        calls.append((args, config, seed, run_index))
        return _run_result(seed, run_index, 1.0)

    monkeypatch.setattr(trainer, "run_single_experiment", run_single)

    summary = trainer.main(["--smoke", "--dataset", "qm9", "--encoder", "bert",
                            "--seeds", "42", "--output-dir", str(tmp_path)])
    assert len(calls) == 1
    assert calls[0][1] == trainer.PAPER_CONFIGS[("qm9", "bert")]
    assert calls[0][2] == 42
    assert calls[0][3] == 0
    assert summary["seeds"] == [42]
    assert summary["run_metrics"] == [1.0]
    assert summary["complete"] is True


def test_main_orchestrates_five_runs_and_isolates_summaries(trainer, monkeypatch, tmp_path):
    calls = []

    def run_single(_args, _config, seed, run_index):
        calls.append((seed, run_index))
        return _run_result(seed, run_index, float(run_index + 1))

    monkeypatch.setattr(trainer, "run_single_experiment", run_single)

    summary = trainer.main(["--dataset", "qm9", "--encoder", "bert",
                            "--output-dir", str(tmp_path)])

    assert calls == [(42, 0), (43, 1), (44, 2), (45, 3), (46, 4)]
    assert summary["run_metrics"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    for seed, run_index in calls:
        assert (tmp_path / "qm9" / "bert" /
                f"run_{run_index + 1:02d}_seed_{seed}" / "summary.json").is_file()


def test_aggregate_run_results_reports_population_standard_deviation(trainer):
    metrics = [0.10, 0.11, 0.09, 0.12, 0.08]
    summary = trainer.aggregate_run_results([
        _run_result(42, 0, metrics[0]), _run_result(43, 1, metrics[1]),
        _run_result(44, 2, metrics[2]), _run_result(45, 3, metrics[3]),
        _run_result(46, 4, metrics[4]),
    ])

    assert summary["run_metrics"] == metrics
    assert summary["mean"] == pytest.approx(0.10)
    assert summary["std"] == pytest.approx(math.sqrt(0.0002))
    assert summary["std_ddof"] == 0
    assert summary["display"] == f"{summary['mean']} ± {summary['std']}"
    assert summary["complete"] is True


def test_main_does_not_write_complete_summary_when_a_run_fails(trainer, monkeypatch, tmp_path):
    def run_single(_args, _config, seed, run_index):
        if seed == 44:
            raise RuntimeError("run failed")
        return _run_result(seed, run_index, float(run_index + 1))

    monkeypatch.setattr(trainer, "run_single_experiment", run_single)

    with pytest.raises(RuntimeError, match="run failed"):
        trainer.main(["--dataset", "qm9", "--encoder", "bert", "--seeds", "42", "43", "44",
                      "--output-dir", str(tmp_path)])

    partial = json.loads((tmp_path / "qm9" / "bert" / "summary.json").read_text())
    assert partial["complete"] is False
    assert partial["run_metrics"] == [1.0, 2.0]
    assert "mean" not in partial
    assert "std" not in partial
    assert "display" not in partial


def test_tiny_pipeline_runs_pretrain_restore_finetune_and_final_test(trainer, monkeypatch, tmp_path):
    splits = {
        "train": _qm9_graphs(trainer, 8),
        "val": _qm9_graphs(trainer, 4, 20),
        "test": _qm9_graphs(trainer, 4, 40),
    }
    monkeypatch.setattr(trainer, "synthetic_splits", lambda _spec: splits)

    summary = trainer.main([
        "--smoke", "--dataset", "qm9", "--encoder", "bert", "--device", "cpu",
        "--seeds", "42",
        "--output-dir", str(tmp_path),
    ])

    run_directory = tmp_path / "qm9" / "bert" / "run_01_seed_42"
    run_summary = json.loads((run_directory / "summary.json").read_text())
    assert Path(run_summary["checkpoints"]["mlm"]).is_file()
    assert Path(run_summary["checkpoints"]["finetune"]).is_file()
    assert all(math.isfinite(row["train_loss"]) and math.isfinite(row["val_loss"])
               for row in run_summary["mlm"]["history"])
    assert all(math.isfinite(row["train_loss"]) and math.isfinite(row["validation"]["metric"])
               for row in run_summary["finetune"]["history"])
    assert math.isfinite(run_summary["finetune"]["test"]["metric"])
    assert summary["run_metrics"] == [run_summary["metric"]]


def test_tiny_pipeline_writes_summary_file(trainer, tmp_path):
    summary = trainer.main([
        "--smoke", "--dataset", "qm9", "--encoder", "bert", "--device", "cpu",
        "--seeds", "42",
        "--output-dir", str(tmp_path),
    ])

    saved = tmp_path / "qm9" / "bert" / "summary.json"
    assert saved.is_file()
    assert saved.read_text() == json.dumps(summary, indent=2, allow_nan=True)


def test_tiny_pipeline_emits_preprocessing_and_training_progress(trainer, tmp_path, capsys):
    trainer.main([
        "--smoke", "--dataset", "qm9", "--encoder", "bert", "--device", "cpu",
        "--seeds", "42",
        "--output-dir", str(tmp_path),
    ])

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()
              if line.startswith('{"event"')]

    assert {"event": "stage", "stage": "tokenizer_fit", "status": "started"} in events
    assert any(event.get("phase") == "pretrain" for event in events)
    assert any(event.get("phase") == "finetune" for event in events)


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


def test_formal_gte_tokenizer_uses_the_official_padding_id(trainer):
    args = trainer.apply_preset(trainer.build_parser().parse_args([
        "--dataset", "qm9", "--encoder", "gte",
    ]))

    tokenizer = trainer.make_tokenizer(args, _qm9_graphs(trainer, 2))

    assert tokenizer.special_tokens.pad_token_id == 1
    assert tokenizer.special_tokens.unk_token_id == 0


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
