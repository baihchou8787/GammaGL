from ._molecular_benchmark import PreprocessedMolecularBenchmark


class QM9(PreprocessedMolecularBenchmark):
    r"""The QM9 molecular property prediction benchmark.

    The GraphTokenizer paper data must be explicitly prepared before
    constructing this dataset. It is materialized from the shared paper
    release bundle; dataset construction does not download that bundle
    implicitly. The GraphTokenizer example training entrypoint provides the
    explicit preparation step.
    """

    name = 'qm9'
    display_name = 'QM9'
    aliases = ('qm9',)
    task_type = 'regression'
    num_tasks = 16
    metric = 'mae'
    label_keys = (
        'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0',
        'u298', 'h298', 'g298', 'cv', 'u0_atom', 'u298_atom',
        'h298_atom', 'g298_atom',
    )
    node_feature_columns = {'attr': 5, 'x': 0}
