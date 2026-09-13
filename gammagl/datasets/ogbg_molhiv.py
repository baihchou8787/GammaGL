from ._molecular_benchmark import PreprocessedMolecularBenchmark


class OGBGMolHIV(PreprocessedMolecularBenchmark):
    r"""The OGBG-molhiv molecular property prediction benchmark.

    The GraphTokenizer paper bundle is downloaded and cached automatically
    when the raw files are missing.
    """

    name = 'ogbg-molhiv'
    display_name = 'OGBG-molhiv'
    aliases = ('molhiv', 'ogbg-molhiv', 'ogbg_molhiv')
    task_type = 'binary_classification'
    num_tasks = 1
    metric = 'rocauc'
    label_keys = ('label',)
