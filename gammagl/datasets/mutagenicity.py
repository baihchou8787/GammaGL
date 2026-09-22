from ._graph_tokenizer_benchmark import PreprocessedGraphBenchmark


class Mutagenicity(PreprocessedGraphBenchmark):
    r"""GraphTokenizer's TU Mutagenicity graph-classification benchmark."""

    name = 'mutagenicity'
    display_name = 'Mutagenicity'
    aliases = ('mutagenicity', 'mutag', 'muta')
    task_type = 'binary_classification'
    num_tasks = 1
    num_classes = 2
    metric = 'accuracy'
    label_keys = ('label',)
    default_edge_token = 0
