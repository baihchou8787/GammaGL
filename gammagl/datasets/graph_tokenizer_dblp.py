from ._graph_tokenizer_benchmark import PreprocessedGraphBenchmark


class GraphTokenizerDBLP(PreprocessedGraphBenchmark):
    r"""GraphTokenizer's TU DBLP_v1 graph-classification benchmark."""

    name = 'dblp'
    display_name = 'DBLP_v1'
    aliases = ('dblp', 'dblp-v1', 'dblp_v1')
    task_type = 'binary_classification'
    num_tasks = 1
    num_classes = 2
    metric = 'accuracy'
    label_keys = ('label',)
    default_edge_token = 0
