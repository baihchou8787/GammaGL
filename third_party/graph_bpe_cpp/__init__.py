from typing import Dict, Iterable, List, Sequence, Tuple


MergeRule = Tuple[int, int, int]


def _load_native():
    try:
        from . import _graph_bpe
    except ImportError:
        return None
    return _graph_bpe


def is_available() -> bool:
    """Return whether the optional native BPE extension is importable."""

    return _load_native() is not None


def backend_name() -> str:
    return "cpp" if is_available() else "python"


def train_bpe(token_sequences, num_merges: int, min_frequency: int):
    native = _load_native()
    if native is not None:
        result = native.train_bpe(token_sequences, int(num_merges), int(min_frequency))
        return _normalize_result(result, "cpp")
    return _train_bpe_python(token_sequences, int(num_merges), int(min_frequency))


def encode(token_sequence, merge_rules: Sequence[Sequence[int]]) -> List[int]:
    native = _load_native()
    if native is not None:
        return [int(token) for token in native.encode(token_sequence, merge_rules)]
    encoded = _as_int_list(token_sequence)
    for rule in merge_rules:
        encoded = _apply_merge(encoded, _as_merge_rule(rule))
    return encoded


def batch_encode(token_sequences, merge_rules: Sequence[Sequence[int]]) -> List[List[int]]:
    native = _load_native()
    if native is not None and hasattr(native, "batch_encode"):
        return [
            [int(token) for token in sequence]
            for sequence in native.batch_encode(token_sequences, merge_rules)
        ]
    return [encode(sequence, merge_rules) for sequence in token_sequences]


def _normalize_result(result, backend: str):
    merge_rules = [_as_merge_rule(rule) for rule in result.get("merge_rules", [])]
    return {
        "merge_rules": merge_rules,
        "vocab_size": int(result.get("vocab_size", 0)),
        "metadata": {
            **{key: int(value) for key, value in result.get("metadata", {}).items() if isinstance(value, int)},
            "backend": backend,
        },
    }


def _train_bpe_python(token_sequences, num_merges: int, min_frequency: int):
    sequences = [_as_int_list(sequence) for sequence in token_sequences]
    merge_rules: List[MergeRule] = []
    next_token_id = _next_token_id(sequences)

    for _ in range(num_merges):
        pair_counts = _count_pairs(sequences)
        if not pair_counts:
            break
        best_pair, best_count = _select_best_pair(pair_counts)
        if best_count < min_frequency:
            break
        rule = (best_pair[0], best_pair[1], next_token_id)
        merge_rules.append(rule)
        sequences = [_apply_merge(sequence, rule) for sequence in sequences]
        next_token_id += 1

    return {
        "merge_rules": merge_rules,
        "vocab_size": next_token_id,
        "metadata": {
            "backend": "python",
            "num_merges_requested": num_merges,
            "num_merges_performed": len(merge_rules),
            "min_frequency": min_frequency,
        },
    }


def _as_int_list(sequence) -> List[int]:
    if hasattr(sequence, "tolist"):
        sequence = sequence.tolist()
    return [int(token) for token in sequence]


def _as_merge_rule(rule) -> MergeRule:
    return tuple(int(value) for value in rule)


def _next_token_id(sequences: Iterable[List[int]]) -> int:
    max_token = -1
    for sequence in sequences:
        if sequence:
            max_token = max(max_token, max(sequence))
    return max_token + 1


def _count_pairs(sequences: Iterable[List[int]]) -> Dict[Tuple[int, int], int]:
    counts: Dict[Tuple[int, int], int] = {}
    for sequence in sequences:
        for left, right in zip(sequence, sequence[1:]):
            pair = (left, right)
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def _select_best_pair(pair_counts: Dict[Tuple[int, int], int]) -> Tuple[Tuple[int, int], int]:
    best_pair = min(pair_counts.keys(), key=lambda pair: (-pair_counts[pair], pair[0], pair[1]))
    return best_pair, pair_counts[best_pair]


def _apply_merge(sequence: List[int], merge_rule: MergeRule) -> List[int]:
    left, right, new_id = merge_rule
    merged: List[int] = []
    index = 0
    while index < len(sequence):
        if index + 1 < len(sequence) and sequence[index] == left and sequence[index + 1] == right:
            merged.append(new_id)
            index += 2
        else:
            merged.append(sequence[index])
            index += 1
    return merged
