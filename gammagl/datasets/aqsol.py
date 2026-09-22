from ._graph_tokenizer_benchmark import PreprocessedGraphBenchmark


class AQSOL(PreprocessedGraphBenchmark):
    r"""GraphTokenizer's AQSOL solubility-regression benchmark."""

    name = 'aqsol'
    display_name = 'AQSOL'
    aliases = ('aqsol',)
    task_type = 'regression'
    num_tasks = 1
    metric = 'mae'
    label_keys = ('solubility',)
