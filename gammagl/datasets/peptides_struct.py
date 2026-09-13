from ._molecular_benchmark import PreprocessedMolecularBenchmark


class PeptidesStruct(PreprocessedMolecularBenchmark):
    r"""The Peptides-struct molecular graph regression benchmark.

    The GraphTokenizer paper data must be explicitly prepared before
    constructing this dataset. It is materialized from the shared paper
    release bundle; dataset construction does not download that bundle
    implicitly. The GraphTokenizer example training entrypoint provides the
    explicit preparation step.
    """

    name = 'peptides-struct'
    display_name = 'Peptides-struct'
    aliases = ('peptides-struct', 'peptides_struct', 'p-struct', 'p_struct')
    task_type = 'multi_target_regression'
    num_tasks = 11
    metric = 'average_mae'
    label_keys = ('labels',)
    allow_nan_labels = True
