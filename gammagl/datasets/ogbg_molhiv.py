from ._molecular_benchmark import PreprocessedMolecularBenchmark


class OGBGMolHIV(PreprocessedMolecularBenchmark):
    r"""The OGBG-molhiv molecular property prediction benchmark.

    The GraphTokenizer paper data must be explicitly prepared before
    constructing this dataset. It is materialized from the shared paper
    release bundle; dataset construction does not download that bundle
    implicitly. The GraphTokenizer example training entrypoint provides the
    explicit preparation step.
    """

    name = 'ogbg-molhiv'
    display_name = 'OGBG-molhiv'
    aliases = ('molhiv', 'ogbg-molhiv', 'ogbg_molhiv')
    task_type = 'binary_classification'
    num_tasks = 1
    metric = 'rocauc'
    label_keys = ('label',)
