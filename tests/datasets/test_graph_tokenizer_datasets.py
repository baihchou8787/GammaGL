import gzip
import hashlib
import importlib.util
import json
import pickle
from pathlib import Path

import pytest
import tensorlayerx as tlx

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
