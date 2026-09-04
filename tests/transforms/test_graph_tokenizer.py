import random

import pytest

from gammagl.transforms.graph_bpe import GraphBPE
from gammagl.transforms.graph_serializer import FrequencyGuidedEulerianSerializer
from gammagl.transforms.graph_tokenizer import GraphTokenizer


class SimpleGraph:
    def __init__(self, edge_index, x, edge_attr=None, num_nodes=None):
        self.edge_index = edge_index
        self.x = x
        self.edge_attr = edge_attr
        self.num_nodes = len(x) if num_nodes is None else num_nodes


def _edges(edge_index, edge_attr):
    labels = edge_attr if edge_attr is not None else [0] * len(edge_index[0])
    return sorted((min(source, target), max(source, target), label)
                  for source, target, label in zip(edge_index[0], edge_index[1], labels))


def _round_trip(graph):
    serializer = FrequencyGuidedEulerianSerializer().fit([graph])
    restored = serializer.deserialize(serializer.serialize(graph))
    assert restored["num_nodes"] == graph.num_nodes
    assert restored["x"] == graph.x
    assert _edges(restored["edge_index"], restored["edge_attr"]) == _edges(
        graph.edge_index, graph.edge_attr)


def _relabel(graph, old_to_new):
    new_to_old = {new: old for old, new in old_to_new.items()}
    return SimpleGraph(
        [[old_to_new[index] for index in graph.edge_index[0]],
         [old_to_new[index] for index in graph.edge_index[1]]],
        [graph.x[new_to_old[index]] for index in range(graph.num_nodes)],
        list(graph.edge_attr), graph.num_nodes)


def _structure_tokens(token_ids):
    return [token for token in token_ids if token == -1 or token % 3 != 2]


@pytest.mark.parametrize(
    "edge_index,edge_attr",
    [([[0], [1]], [3]), ([[0, 1], [1, 0]], [3, 3])],
    ids=("single_coo", "symmetric_coo"),
)
def test_serializer_accepts_equivalent_undirected_coo(edge_index, edge_attr):
    graph = SimpleGraph(edge_index, [7, 8], edge_attr)
    serializer = FrequencyGuidedEulerianSerializer().fit([graph])

    result = serializer.serialize(graph)

    assert result.metadata["method"] == "feuler"
    assert result.metadata["num_edges_traversed"] == 2
    assert result.token_ids


def test_serializer_normalizes_single_and_symmetric_coo_identically():
    single = SimpleGraph([[0], [1]], [7, 8], [3])
    symmetric = SimpleGraph([[0, 1], [1, 0]], [7, 8], [3, 3])

    assert FrequencyGuidedEulerianSerializer().fit([single]).serialize(single).token_ids == (
        FrequencyGuidedEulerianSerializer().fit([symmetric]).serialize(symmetric).token_ids)


@pytest.mark.parametrize(
    "graph",
    [
        SimpleGraph([[0, 1], [1, 2]], [4, 9, 4], [7, 8]),
        SimpleGraph([[0, 1, 2], [1, 2, 0]], [1, 1, 1], [2, 3, 4]),
        SimpleGraph([[0, 2], [1, 3]], [1, 2, 3, 4], [5, 6]),
        SimpleGraph([[0], [1]], [1, 2, 3], [5], num_nodes=3),
    ],
    ids=("path", "cycle", "disconnected", "isolated_node"),
)
def test_serializer_round_trip(graph):
    _round_trip(graph)


def test_serializer_permutation_regression():
    graph = SimpleGraph(
        [[0, 1, 1, 3], [1, 2, 3, 4]], [9, 1, 4, 7, 4], [3, 2, 5, 6])
    serializer = FrequencyGuidedEulerianSerializer().fit([graph])
    expected = _structure_tokens(serializer.serialize(graph).token_ids)
    permutation = list(range(graph.num_nodes))
    random.Random(7).shuffle(permutation)
    relabeled = _relabel(graph, dict(enumerate(permutation)))

    assert _structure_tokens(serializer.serialize(relabeled).token_ids) == expected
    _round_trip(relabeled)


@pytest.mark.parametrize(
    "graph,operation,error",
    [
        (SimpleGraph([[0], [0]], [4], [1]), "serialize", "self-loops"),
        (SimpleGraph([[0, 0], [1, 1]], [4, 5], [1, 1]), "fit", "Parallel edges"),
        (SimpleGraph([[0], [1]], [4], [1], num_nodes=2), "fit", "x must contain exactly"),
    ],
    ids=("self_loop", "parallel_edge", "bad_node_features"),
)
def test_serializer_rejects_invalid_graphs(graph, operation, error):
    serializer = FrequencyGuidedEulerianSerializer()

    with pytest.raises(ValueError, match=error):
        getattr(serializer, operation)([graph] if operation == "fit" else graph)


def test_serializer_rejects_explicit_directed_graphs():
    graph = {"edge_index": [[0], [1]], "x": [4, 5], "edge_attr": [1], "directed": True}

    with pytest.raises(ValueError, match="directed graph semantics"):
        FrequencyGuidedEulerianSerializer().serialize(graph)


def test_bpe_fits_encodes_and_builds_a_deterministic_codebook(tmp_path):
    sequences = [[1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 3]]
    first = GraphBPE(num_merges=2, min_frequency=2).fit(sequences)
    second = GraphBPE(num_merges=2, min_frequency=2).fit(list(reversed(sequences)))

    assert first.codebook.merge_rules == [(1, 2, 4), (4, 4, 5)]
    assert first.codebook == second.codebook
    assert first.encode([1, 2, 1, 2, 3]) == [5, 3]
    path = tmp_path / "codebook.json"
    first.save_codebook(path)
    assert GraphBPE.load_codebook(path).encode([1, 2, 1, 2]) == [5]


def test_bpe_keeps_protected_tokens_out_of_merges():
    bpe = GraphBPE(num_merges=2, min_frequency=2, protected_token_ids={7})
    bpe.fit([[1, 2, 7, 1, 2], [1, 2, 7, 1, 2]])

    assert all(7 not in rule[:2] for rule in bpe.codebook.merge_rules)
    assert 7 in bpe.encode([1, 2, 7, 1, 2])


def test_cpp_bpe_matches_python_when_extension_is_available():
    from third_party import graph_bpe_cpp

    if not graph_bpe_cpp.is_available():
        pytest.skip("graph_bpe_cpp native extension is unavailable")
    sequences = [[1, 2, 7, 1, 2], [1, 2, 7, 1, 2], [1, 2]]
    python_bpe = GraphBPE(2, 2, backend="python", protected_token_ids={7}).fit(sequences)
    cpp_bpe = GraphBPE(2, 2, backend="cpp", protected_token_ids={7}).fit(sequences)

    assert cpp_bpe.codebook.merge_rules == python_bpe.codebook.merge_rules
    assert cpp_bpe.encode(sequences[0]) == python_bpe.encode(sequences[0])


def test_tokenizer_encodes_special_tokens_and_pads():
    graph = SimpleGraph([[0], [1]], [1, 2], [5])
    tokenizer = GraphTokenizer(bpe=GraphBPE(num_merges=1, min_frequency=2)).fit([graph, graph])

    encoded = tokenizer.encode_graph(graph)
    padded, attention = tokenizer.pad_token_sequences([encoded.input_ids], max_length=12)

    assert encoded.input_ids[0] == tokenizer.special_tokens.cls_token_id
    assert encoded.input_ids[-1] == tokenizer.special_tokens.sep_token_id
    assert attention[0][:len(encoded.input_ids)] == [1] * len(encoded.input_ids)
    assert padded[0][-1] == tokenizer.special_tokens.pad_token_id


def test_tokenizer_does_not_expand_train_vocabulary_for_held_out_tokens():
    train = SimpleGraph([[0], [1]], [1, 2], [5])
    held_out = SimpleGraph([[0], [1]], [101, 102], [5])
    tokenizer = GraphTokenizer(bpe=GraphBPE(num_merges=1, min_frequency=2)).fit([train, train])
    vocabulary = dict(tokenizer.vocabulary)

    encoded = tokenizer.encode_graph(held_out)

    assert tokenizer.vocabulary == vocabulary
    assert tokenizer.special_tokens.unk_token_id in encoded.input_ids


def test_tokenizer_rejects_unfitted_and_overlong_sequences():
    graph = SimpleGraph([[0], [1]], [1, 2], [5])
    tokenizer = GraphTokenizer()

    with pytest.raises(RuntimeError, match="must be fit"):
        tokenizer.encode_graph(graph)
    with pytest.raises(ValueError, match="exceeds max_length"):
        tokenizer.pad_token_sequences([[3, 8, 9, 4]], max_length=3)
