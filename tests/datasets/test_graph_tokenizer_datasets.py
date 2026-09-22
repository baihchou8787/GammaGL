import hashlib
import importlib.util
import io
import json
import pickle
import shutil
import tarfile
from pathlib import Path

import pytest
import tensorlayerx as tlx

import gammagl.datasets._graph_tokenizer_download as graph_tokenizer_download
from gammagl.datasets import AQSOL, GraphTokenizerDBLP, Mutagenicity


class FakeDGLGraph:
    """Minimal pickleable DGL-shaped graph for released bundle fixtures."""

    def __init__(self, src, dst, ndata, edata):
        self._src = src
        self._dst = dst
        self.ndata = ndata
        self.edata = edata

    def edges(self):
        return self._src, self._dst


def _graph(ndata, edata):
    return FakeDGLGraph([0], [1], ndata, edata)


def _write_raw(root, name, samples, splits):
    raw_dir = root / name / "raw"
    raw_dir.mkdir(parents=True)
    with (raw_dir / "data.pkl").open("wb") as handle:
        pickle.dump(samples, handle)
    for split, indices in splits.items():
        (raw_dir / f"{split}_index.json").write_text(
            json.dumps(indices), encoding="utf-8")


def _download_module():
    path = Path(__file__).resolve().parents[2] / "gammagl" / "datasets" / "_graph_tokenizer_download.py"
    spec = importlib.util.spec_from_file_location("graph_tokenizer_download_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_bundle(path, datasets):
    with tarfile.open(path, "w:gz") as archive:
        for name, samples in datasets.items():
            files = {
                f"release/{name}/data.pkl": pickle.dumps(samples),
                f"release/{name}/train_index.json": b"[0]",
                f"release/{name}/val_index.json": b"[]",
                f"release/{name}/test_index.json": b"[]",
            }
            for member, content in files.items():
                info = tarfile.TarInfo(member)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))


def test_mutagenicity_loads_official_split_and_default_edge_token(tmp_path):
    _write_raw(tmp_path, "mutagenicity", [
        (_graph({"node_token_ids": [[5], [9]]}, {}), 1),
    ], {"train": [0], "val": [], "test": []})

    dataset = Mutagenicity(root=str(tmp_path))
    graph = dataset[0]

    assert dataset.get_idx_split() == {"train": [0], "val": [], "test": []}
    assert dataset.num_classes == 2
    assert tlx.convert_to_numpy(graph.x).tolist() == [5, 9]
    assert tlx.convert_to_numpy(graph.edge_attr).tolist() == [0]
    assert tlx.convert_to_numpy(graph.y).shape == (1, 1)
    assert tlx.convert_to_numpy(graph.y).dtype.kind == "f"


def test_graph_tokenizer_dblp_loads_official_split_and_default_edge_token(tmp_path):
    _write_raw(tmp_path, "dblp", [
        (_graph({"node_token_ids": [[7], [11]]}, {}), 0),
    ], {"train": [0], "val": [], "test": []})

    dataset = GraphTokenizerDBLP(root=str(tmp_path))
    graph = dataset[0]

    assert dataset.get_idx_split() == {"train": [0], "val": [], "test": []}
    assert dataset.num_classes == 2
    assert tlx.convert_to_numpy(graph.x).tolist() == [7, 11]
    assert tlx.convert_to_numpy(graph.edge_attr).tolist() == [0]
    assert float(tlx.convert_to_numpy(graph.y)[0, 0]) == 0.0


@pytest.mark.parametrize("dataset_class,dataset_dir", [
    (Mutagenicity, "mutagenicity"),
    (GraphTokenizerDBLP, "dblp"),
])
@pytest.mark.parametrize("label", [0, 1, 0.0, 1.0])
def test_binary_classification_labels_accept_only_valid_classes(
        tmp_path, dataset_class, dataset_dir, label):
    _write_raw(tmp_path, dataset_dir, [
        (_graph({"node_token_ids": [[5], [9]]}, {}), label),
    ], {"train": [0], "val": [], "test": []})

    graph = dataset_class(root=str(tmp_path))[0]
    assert float(tlx.convert_to_numpy(graph.y)[0, 0]) == label


@pytest.mark.parametrize("dataset_class,dataset_dir", [
    (Mutagenicity, "mutagenicity"),
    (GraphTokenizerDBLP, "dblp"),
])
@pytest.mark.parametrize("label", [1.5, -1, 2, float("nan"), float("inf")])
def test_binary_classification_labels_fail_fast_when_not_a_valid_class(
        tmp_path, dataset_class, dataset_dir, label):
    _write_raw(tmp_path, dataset_dir, [
        (_graph({"node_token_ids": [[5], [9]]}, {}), label),
    ], {"train": [0], "val": [], "test": []})

    with pytest.raises(ValueError) as error:
        dataset_class(root=str(tmp_path))

    message = str(error.value)
    assert dataset_class.display_name in message
    assert "label" in message


@pytest.mark.parametrize("label", [0.37, -1.42])
def test_aqsol_preserves_finite_float_labels(tmp_path, label):
    _write_raw(tmp_path, "aqsol", [
        (_graph({"feat": [6, 8]}, {"feat": [1]}), label),
    ], {"train": [0], "val": [], "test": []})

    graph = AQSOL(root=str(tmp_path))[0]
    assert float(tlx.convert_to_numpy(graph.y)[0, 0]) == pytest.approx(label)


def test_aqsol_converts_feat_tokens_and_solubility_label(tmp_path):
    _write_raw(tmp_path, "aqsol", [
        (_graph({"feat": [6, 8]}, {"feat": [1]}), -2.5),
    ], {"train": [0], "val": [], "test": []})

    dataset = AQSOL(root=str(tmp_path))
    graph = dataset[0]

    assert dataset.get_idx_split() == {"train": [0], "val": [], "test": []}
    assert dataset.num_tasks == 1
    assert tlx.convert_to_numpy(graph.x).tolist() == [13, 17]
    assert tlx.convert_to_numpy(graph.edge_attr).tolist() == [2]
    assert float(tlx.convert_to_numpy(graph.y)[0, 0]) == -2.5


def test_aqsol_fails_fast_without_real_bond_features(tmp_path):
    _write_raw(tmp_path, "aqsol", [
        (_graph({"feat": [6, 8]}, {}), -2.5),
    ], {"train": [0], "val": [], "test": []})

    with pytest.raises(ValueError, match="edge features"):
        AQSOL(root=str(tmp_path))


def test_dataset_rejects_overlapping_official_splits(tmp_path):
    samples = [
        (_graph({"node_token_ids": [[5], [9]]}, {"edge_token_ids": [[2]]}), 0),
        (_graph({"node_token_ids": [[6], [8]]}, {"edge_token_ids": [[3]]}), 1),
    ]
    _write_raw(tmp_path, "mutagenicity", samples,
               {"train": [0], "val": [0], "test": [1]})

    with pytest.raises(ValueError, match="overlaps"):
        Mutagenicity(root=str(tmp_path))


@pytest.mark.parametrize("dataset_class,dataset_dir,sample", [
    (Mutagenicity, "mutagenicity", (_graph({"node_token_ids": [[5], [9]]}, {}), 1)),
    (GraphTokenizerDBLP, "dblp", (_graph({"node_token_ids": [[7], [11]]}, {}), 0)),
    (AQSOL, "aqsol", (_graph({"feat": [6, 8]}, {"feat": [1]}), -2.5)),
])
def test_verified_bundle_materializes_real_release_directory_names(
        tmp_path, monkeypatch, dataset_class, dataset_dir, sample):
    module = _download_module()
    bundle = tmp_path / "bundle.tar.gz"
    _write_bundle(bundle, {dataset_dir: [sample]})
    monkeypatch.setattr(module, "PAPER_DATA_BUNDLE_SHA256", module.sha256_file(bundle))
    monkeypatch.setenv(module.DATA_BUNDLE_ENV, str(bundle))
    raw_dir = tmp_path / "data" / dataset_class.name / "raw"

    module.materialize_paper_dataset(
        dataset_class.name, dataset_class.aliases, raw_dir, tmp_path / "data",
        allow_download=False)

    assert (raw_dir / "data.pkl").is_file()
    assert (raw_dir / "train_index.json").read_text(encoding="utf-8") == "[0]"


@pytest.mark.parametrize("dataset_class,dataset_dir,sample", [
    (Mutagenicity, "mutagenicity", (_graph({"node_token_ids": [[5], [9]]}, {}), 1)),
    (GraphTokenizerDBLP, "dblp", (_graph({"node_token_ids": [[7], [11]]}, {}), 0)),
    (AQSOL, "aqsol", (_graph({"feat": [6, 8]}, {"feat": [1]}), -2.5)),
])
def test_dataset_initialization_reuses_prepared_bundle_without_network(
        tmp_path, monkeypatch, dataset_class, dataset_dir, sample):
    bundle = tmp_path / "bundle.tar.gz"
    _write_bundle(bundle, {dataset_dir: [sample]})
    monkeypatch.setattr(
        graph_tokenizer_download,
        "PAPER_DATA_BUNDLE_SHA256",
        graph_tokenizer_download.sha256_file(bundle),
    )
    monkeypatch.setenv(graph_tokenizer_download.DATA_BUNDLE_ENV, str(bundle))
    monkeypatch.setattr(
        "gammagl.data.download.download_google_url",
        lambda *_args, **_kwargs: pytest.fail("prepared bundle must not download"),
    )
    root = tmp_path / "data"
    graph_tokenizer_download.materialize_paper_dataset(
        dataset_class.name, dataset_class.aliases,
        root / dataset_class.name / "raw", root, allow_download=False)

    dataset = dataset_class(root=str(root))

    assert len(dataset) == 1


def test_missing_or_corrupted_bundle_fails_without_network(tmp_path, monkeypatch):
    module = _download_module()
    monkeypatch.delenv(module.DATA_BUNDLE_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match="prepared"):
        module.materialize_paper_dataset("aqsol", (), tmp_path / "raw", tmp_path)

    archive = tmp_path / "corrupted.tar.gz"
    archive.write_bytes(b"corrupted")
    monkeypatch.setattr(module, "PAPER_DATA_BUNDLE_SHA256", hashlib.sha256(b"expected").hexdigest())
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        module._verify_official_bundle(archive, remove_on_failure=False)
