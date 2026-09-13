import gzip
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
from gammagl.datasets import OGBGMolHIV, PeptidesStruct, QM9


def _write_raw(root, name, samples, splits, compressed=False):
    raw_dir = root / name / "raw"
    raw_dir.mkdir(parents=True)
    opener = gzip.open if compressed else open
    with opener(raw_dir / ("data.pkl.gz" if compressed else "data.pkl"), "wb") as handle:
        pickle.dump(samples, handle)
    for split, indices in splits.items():
        (raw_dir / f"{split}_index.json").write_text(json.dumps(indices), encoding="utf-8")


def _download_module():
    path = Path(__file__).resolve().parents[2] / "gammagl" / "datasets" / "_graph_tokenizer_download.py"
    spec = importlib.util.spec_from_file_location("graph_tokenizer_download_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_bundle_archive(path, dataset_name="qm9"):
    members = {
        f"release/{dataset_name}/data.pkl": pickle.dumps([dataset_name]),
        f"release/{dataset_name}/train_index.json": b"[]",
        f"release/{dataset_name}/val_index.json": b"[]",
        f"release/{dataset_name}/test_index.json": b"[]",
    }
    with tarfile.open(path, "w:gz") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def _materialize_bundle(module, tmp_path, allow_download):
    raw_dir = tmp_path / "raw"
    module.materialize_paper_dataset(
        "qm9", (), raw_dir, tmp_path / "cache", allow_download=allow_download)
    return raw_dir


def _write_dataset_lifecycle_bundle(path):
    qm9_properties = {name: float(index) for index, name in enumerate(QM9.label_keys)}
    datasets = {
        "qm9": [{
            "edge_index": [[0], [1]], "x": [6, 8], "edge_attr": [1],
            "properties": qm9_properties,
        }],
        "ogbg-molhiv": [(
            {"edges": [[0], [1]], "node_type_ids": [6, 8], "edge_type_ids": [1]}, [1],
        )],
        "peptides-struct": [(
            {"edges": [[0], [1]], "node_token_ids": [[5], [9]], "edge_token_ids": [[2]]},
            [float(index) for index in range(11)],
        )],
    }
    with tarfile.open(path, "w:gz") as archive:
        for dataset_name, samples in datasets.items():
            members = {
                f"release/{dataset_name}/data.pkl": pickle.dumps(samples),
                f"release/{dataset_name}/train_index.json": b"[0]",
                f"release/{dataset_name}/val_index.json": b"[]",
                f"release/{dataset_name}/test_index.json": b"[]",
            }
            for name, content in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))


def test_qm9_constructs_with_official_split_and_homo_label(tmp_path):
    properties = {name: float(index) for index, name in enumerate(QM9.label_keys)}
    _write_raw(tmp_path, "qm9", [{
        "edge_index": [[0], [1]], "x": [6, 8], "edge_attr": [1], "properties": properties,
    }], {"train": [0], "val": [], "test": []})

    dataset = QM9(root=str(tmp_path))
    graph = dataset[0]

    assert dataset.get_idx_split() == {"train": [0], "val": [], "test": []}
    assert dataset.num_tasks == 16
    assert tlx.convert_to_numpy(graph.edge_index).shape == (2, 1)
    assert tlx.convert_to_numpy(graph.x).dtype.kind in "iu"
    assert tlx.convert_to_numpy(graph.y).shape == (1, 16)
    assert float(tlx.convert_to_numpy(graph.y)[0, QM9.label_keys.index("homo")]) == 2.0


def test_molhiv_constructs_with_official_split_and_binary_label(tmp_path):
    _write_raw(tmp_path, "ogbg-molhiv", [
        ({"edges": [[0], [1]], "node_type_ids": [6, 8], "edge_type_ids": [1]}, [1]),
    ], {"train": [0], "val": [], "test": []})

    dataset = OGBGMolHIV(root=str(tmp_path))
    graph = dataset[0]

    assert dataset.get_idx_split() == {"train": [0], "val": [], "test": []}
    assert dataset.metric == "rocauc"
    assert tlx.convert_to_numpy(graph.edge_index).shape == (2, 1)
    assert tlx.convert_to_numpy(graph.y).shape == (1, 1)
    assert float(tlx.convert_to_numpy(graph.y)[0, 0]) == 1.0


def test_peptides_struct_constructs_with_eleven_targets(tmp_path):
    _write_raw(tmp_path, "peptides-struct", [
        ({"edges": [[0], [1]], "node_token_ids": [[5], [9]], "edge_token_ids": [[2]]},
         [float(index) for index in range(11)]),
    ], {"train": [0], "val": [], "test": []}, compressed=True)

    dataset = PeptidesStruct(root=str(tmp_path))
    graph = dataset[0]

    assert dataset.get_idx_split() == {"train": [0], "val": [], "test": []}
    assert dataset.metric == "average_mae"
    assert tlx.convert_to_numpy(graph.edge_index).shape == (2, 1)
    assert tlx.convert_to_numpy(graph.y).shape == (1, 11)
    assert tlx.convert_to_numpy(graph.y).dtype.kind == "f"


def test_dataset_rejects_overlapping_official_splits(tmp_path):
    _write_raw(tmp_path, "ogbg-molhiv", [
        ({"edges": [[0], [1]], "node_type_ids": [6, 8], "edge_type_ids": [1]}, [0]),
        ({"edges": [[0], [1]], "node_type_ids": [6, 6], "edge_type_ids": [1]}, [1]),
    ], {"train": [0], "val": [0], "test": [1]})

    with pytest.raises(ValueError, match="overlaps"):
        OGBGMolHIV(root=str(tmp_path))


@pytest.mark.parametrize("name,data_file", [
    ("qm9", "data.pkl"), ("ogbg-molhiv", "data.pkl"), ("peptides-struct", "data.pkl.gz"),
])
def test_verified_bundle_materializes_dataset_cache(tmp_path, monkeypatch, name, data_file):
    module = _download_module()
    source = tmp_path / "release" / "data" / name
    source.mkdir(parents=True)
    (source / data_file).write_bytes(pickle.dumps([name]))
    for split in ("train", "val", "test"):
        (source / f"{split}_index.json").write_text("[]", encoding="utf-8")
    monkeypatch.setenv(module.DATA_BUNDLE_ENV, str(tmp_path / "release"))
    raw_dir = tmp_path / "cache" / name / "raw"

    module.materialize_paper_dataset(name, (), raw_dir, tmp_path / "cache", allow_download=True)

    assert (raw_dir / data_file).read_bytes() == (source / data_file).read_bytes()
    assert (raw_dir / "train_index.json").is_file()


def test_missing_or_corrupted_artifact_fails_without_download(tmp_path, monkeypatch):
    module = _download_module()
    monkeypatch.delenv(module.DATA_BUNDLE_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match="prepared"):
        module.materialize_paper_dataset("qm9", (), tmp_path / "raw", tmp_path)

    archive = tmp_path / "corrupted.tar.gz"
    archive.write_bytes(b"corrupted")
    monkeypatch.setattr(module, "PAPER_DATA_BUNDLE_SHA256", hashlib.sha256(b"expected").hexdigest())
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        module._verify_official_bundle(archive, remove_on_failure=False)


@pytest.mark.parametrize("allow_download", [False, True])
def test_local_bundle_materializes_without_network(tmp_path, monkeypatch, allow_download):
    module = _download_module()
    bundle = tmp_path / "bundle.tar.gz"
    _write_bundle_archive(bundle)
    monkeypatch.setattr(module, "PAPER_DATA_BUNDLE_SHA256", module.sha256_file(bundle))
    monkeypatch.setenv(module.DATA_BUNDLE_ENV, str(bundle))
    download_calls = 0

    def download(*args, **kwargs):
        nonlocal download_calls
        download_calls += 1

    monkeypatch.setattr("gammagl.data.download.download_google_url", download)

    raw_dir = _materialize_bundle(module, tmp_path, allow_download=allow_download)

    assert (raw_dir / "data.pkl").is_file()
    assert download_calls == 0


@pytest.mark.parametrize("allow_download", [False, True])
def test_cached_bundle_materializes_without_network(tmp_path, monkeypatch, allow_download):
    module = _download_module()
    cache_dir = tmp_path / "cache" / ".graph_tokenizer_release"
    cache_dir.mkdir(parents=True)
    bundle = cache_dir / module.PAPER_DATA_BUNDLE_FILENAME
    _write_bundle_archive(bundle)
    monkeypatch.setattr(module, "PAPER_DATA_BUNDLE_SHA256", module.sha256_file(bundle))
    monkeypatch.delenv(module.DATA_BUNDLE_ENV, raising=False)
    download_calls = 0

    def download(*args, **kwargs):
        nonlocal download_calls
        download_calls += 1

    monkeypatch.setattr("gammagl.data.download.download_google_url", download)

    raw_dir = _materialize_bundle(module, tmp_path, allow_download=allow_download)

    assert (raw_dir / "data.pkl").is_file()
    assert download_calls == 0


def test_missing_bundle_fails_when_download_is_disabled(tmp_path, monkeypatch):
    module = _download_module()
    monkeypatch.delenv(module.DATA_BUNDLE_ENV, raising=False)

    with pytest.raises(FileNotFoundError, match="prepared"):
        _materialize_bundle(module, tmp_path, allow_download=False)


def test_missing_configured_local_bundle_fails_without_network(tmp_path, monkeypatch):
    module = _download_module()
    cache_dir = tmp_path / "cache" / ".graph_tokenizer_release"
    cache_dir.mkdir(parents=True)
    _write_bundle_archive(cache_dir / module.PAPER_DATA_BUNDLE_FILENAME)
    monkeypatch.setenv(module.DATA_BUNDLE_ENV, str(tmp_path / "missing.tar.gz"))
    download_calls = 0

    def download(*args, **kwargs):
        nonlocal download_calls
        download_calls += 1

    monkeypatch.setattr("gammagl.data.download.download_google_url", download)

    with pytest.raises(FileNotFoundError, match="points to a missing path"):
        _materialize_bundle(module, tmp_path, allow_download=True)

    assert download_calls == 0


def test_missing_bundle_downloads_once_and_materializes(tmp_path, monkeypatch):
    module = _download_module()
    source_bundle = tmp_path / "source.tar.gz"
    _write_bundle_archive(source_bundle)
    monkeypatch.setattr(module, "PAPER_DATA_BUNDLE_SHA256", module.sha256_file(source_bundle))
    monkeypatch.delenv(module.DATA_BUNDLE_ENV, raising=False)
    download_calls = 0

    def download(file_id, folder, filename):
        nonlocal download_calls
        download_calls += 1
        shutil.copy2(source_bundle, Path(folder) / filename)

    monkeypatch.setattr("gammagl.data.download.download_google_url", download)

    raw_dir = _materialize_bundle(module, tmp_path, allow_download=True)
    cache_bundle = tmp_path / "cache" / ".graph_tokenizer_release" / module.PAPER_DATA_BUNDLE_FILENAME

    assert (raw_dir / "data.pkl").is_file()
    assert download_calls == 1
    assert cache_bundle.is_file()
    assert module.sha256_file(cache_bundle) == module.PAPER_DATA_BUNDLE_SHA256


@pytest.mark.parametrize("dataset_class", [QM9, OGBGMolHIV, PeptidesStruct])
def test_dataset_initialization_processes_prepared_bundle(
        tmp_path, monkeypatch, dataset_class):
    source_bundle = tmp_path / "source.tar.gz"
    _write_dataset_lifecycle_bundle(source_bundle)
    monkeypatch.setattr(
        graph_tokenizer_download,
        "PAPER_DATA_BUNDLE_SHA256",
        graph_tokenizer_download.sha256_file(source_bundle),
    )
    monkeypatch.delenv(graph_tokenizer_download.DATA_BUNDLE_ENV, raising=False)
    download_calls = 0

    def download(file_id, folder, filename):
        nonlocal download_calls
        download_calls += 1
        shutil.copy2(source_bundle, Path(folder) / filename)

    monkeypatch.setattr("gammagl.data.download.download_google_url", download)

    data_root = tmp_path / "data"
    graph_tokenizer_download.materialize_paper_dataset(
        dataset_class.name,
        dataset_class.aliases,
        data_root / dataset_class.name / "raw",
        data_root,
        allow_download=True,
    )
    dataset = dataset_class(root=str(data_root))

    assert len(dataset) == 1
    assert (tmp_path / "data" / dataset.name / "raw" / "data.pkl").is_file()
    assert download_calls == 1


def test_dataset_initialization_reuses_shared_bundle_cache(tmp_path, monkeypatch):
    source_bundle = tmp_path / "source.tar.gz"
    _write_dataset_lifecycle_bundle(source_bundle)
    monkeypatch.setattr(
        graph_tokenizer_download,
        "PAPER_DATA_BUNDLE_SHA256",
        graph_tokenizer_download.sha256_file(source_bundle),
    )
    monkeypatch.delenv(graph_tokenizer_download.DATA_BUNDLE_ENV, raising=False)
    download_calls = 0

    def download(file_id, folder, filename):
        nonlocal download_calls
        download_calls += 1
        shutil.copy2(source_bundle, Path(folder) / filename)

    monkeypatch.setattr("gammagl.data.download.download_google_url", download)

    data_root = tmp_path / "data"
    graph_tokenizer_download.materialize_paper_dataset(
        QM9.name,
        QM9.aliases,
        data_root / QM9.name / "raw",
        data_root,
        allow_download=True,
    )
    first = QM9(root=str(data_root))
    second = OGBGMolHIV(root=str(data_root))

    assert len(first) == len(second) == 1
    assert download_calls == 1
