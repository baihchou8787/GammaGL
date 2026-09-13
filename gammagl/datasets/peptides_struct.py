from ._molecular_benchmark import PreprocessedMolecularBenchmark


class PeptidesStruct(PreprocessedMolecularBenchmark):
    r"""The Peptides-struct molecular graph regression benchmark.

    The GraphTokenizer paper bundle is downloaded and cached automatically
    when the raw files are missing.
    """

    name = 'peptides-struct'
    display_name = 'Peptides-struct'
    aliases = ('peptides-struct', 'peptides_struct', 'p-struct', 'p_struct')
    task_type = 'multi_target_regression'
    num_tasks = 11
    metric = 'average_mae'
    label_keys = ('labels',)
    allow_nan_labels = True
