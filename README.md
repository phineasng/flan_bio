# FLANs

Reference code for Feature-wise Latent Additive Networks.

## Setup environment to run code

To run the code, it is advised to create an environment:

```console
conda create --name flan python=3.6
conda activate flan
```

And install the repo (in editable mode if you wish to play around with the code)

```console
pip install -e .
```

_Troubleshooting_: If you incur in some issues with the requirements, try to comment out the requirements throwing the error. Some of the requirements included are not necessary.

## Main scripts

The main scripts are:

1. `bin/run_experiment.py` for running the training of a FLANetwork. The flag `--interactive` can be used to run in verbose mode (i.e. showing the training updates). If the config file points to a folder containing a checkpoint, this will be loaded and the training will continue from there. In particular if the model already reached the epoch indicated, then only the test accuracy will be printed. To specify the config file, use the flag `--config_file` followed by the path to the config file.
2. `bin/benchmark_tabular.py` to run the benchmarking code for tabular experiments.
3. `bin/results_analysis.py` to analyze the results (especially creta the figures for the cub results). The `--help` flag should report the relevant flags and arguments to use to generate results figures. In particular, the `--print_test_accuracy` flag can be used to print the test accuracy.

_Troubleshooting_: If you incur in some issues with missing datasets, a relevant error mentioning where to download the dataset should be raised.

## Included configs and ckpt

We include the config files and checkpoints of the best performing models. Note that the config files _should_ be modified to point to the correct folders.