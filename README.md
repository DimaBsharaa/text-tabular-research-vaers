# Text-tabular research

Track 2 NLP final-project research for choosing a new text-tabular dataset.

Open the [executed EDA notebook](research/notebooks/01_multabench_dataset_analysis.ipynb) or its [standalone HTML view](research/notebooks/01_multabench_dataset_analysis.html) to see the tables and 13 visualizations.

For the next runtime check, open the [tennis smoke-test notebook in Colab](https://colab.research.google.com/github/DimaBsharaa/text-tabular-research-2/blob/main/research/notebooks/02_tennis_official_benchmark_colab.ipynb), select **A100 GPU**, and run all cells. It calls the official MulTaBench evaluator unchanged on a bounded Kaggle CSV and displays the four condition scores and gains. Default: one learner, one holdout, two TAR epochs. See [execution details](evaluation/README.md). The CSV has temporary generated histories, not the final research text; model cells have not yet been executed.

See [research/README.md](research/README.md) for findings, data inventories, sources, limitations, and offline reproduction commands. No new dataset has been selected and no benchmark training has run.
