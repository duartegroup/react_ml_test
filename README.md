# react_ml_test_public

Tools and scripts for training / evaluating machine-learned interatomic potentials (MLIPs) for chemical reactions in vacuum and solution, as part of the pre-print: "Data-efficient fine-tuning for reaction modelling in explicit solvent" (Holownia, J.; Juraskova, V.; Deringer, V. L., Duarte, F., 2026): https://chemrxiv.org/doi/abs/10.26434/chemrxiv.15007217/v1.

This repository is a pip-installable Python package, installed as the distribution `react_ml_test_public`.

## Setup

1. Create a conda environment with the required MLIP stack. For example, clone `mlptrain` separately and run its `./install_mace.sh` script - this provides the conda environment named `mlptrain-mace`:
   ```bash
   conda activate mlptrain-mace
   pip install hdf5plugin
   pip install seaborn
   ```
2. Install this package into the environment. **This install is required** — it provides the
   `rmlt` command used to run the workflow scripts (see [Running the scripts](#running-the-scripts)),
   rather than loose files in a checkout. An editable install is recommended while developing:
   ```bash
   pip install -e .
   ```
   A regular install (`pip install .`) also works, but will ignore future code edits.

Notes:
- Run all scripts from the repository root directory. The scripts read `data/`, `models/`, `inputs/` and write to `out/` relative to the current working directory, so they must be launched from the repo root (or a directory containing these folders).
- Only lightweight, PyPI-only dependencies are listed in `pyproject.toml`. The MLIP stack (MACE, mlptrain, autode, torchani, graph-pes, load-atoms, etc.) is installed from via the mlp-train environment.
- For some features a specific fork of MACE is required (e.g. train loss logging + patience warmup, please let me know if you need these features!)
- To run the ani-1nxr model, you must have the neurchem project in the parent directory of this repo. You must also then ensure you have the `libcublas.so.9.2` file (from CUDA 9.2) added to the `LD_LIBRARY_PATH`.

## Running scripts

Installing the package (see [Setup](#setup)) puts a single `rmlt` command on your `PATH`.
Workflow scripts live in the `react_ml_test_public.scripts` package, and `rmlt` runs one of
them by name from the repository root (the scripts read `data/`, `models/`, `inputs/` and
write to `out/` relative to the current working directory, so this is required):

```bash
rmlt run-test-set-comparison
```

Dashes and underscores are interchangeable, so `rmlt run_test_set_comparison` is the same
command. List all available scripts (auto-discovered from `react_ml_test_public/scripts/`)
with:

```bash
rmlt list
```

Anything after the script name is passed through to the script as its arguments. The scripts
remain ordinary modules, so the same one can also be run directly if you prefer:
`python -m react_ml_test_public.scripts.run_test_set_comparison`.

## Examples

Note:

### 1. Energy + force errors of zero-shot and fine-tuned models for CPD + MVK DA system.

Run ```rmlt run-test-set-comparison-yaml``` (edit the input parameters / YAML path at the bottom of the script, or pass in `inputs/eg1.yaml` as shown in the script header).

### 2. 

### 3.

### 4.

### 5.

## Key Workflows

### 1. Dataset preparation (existing labelled xyz files)

#### 1. `rmlt save-npz-as-xyz`

#### 2. `scripts/ref_energy_ref_force_label.sh <dir>`

    - DESCRIPTION: Bash script to relabel energy and force labels to REF_energy and REF_force, compatible with MACE.
    - USAGE:

#### 3. `rmlt add-xyz-metadata`

    - DESCRIPTION: Adds some xyz metadata to an .xyz file (= i.e. config type labels / box dimensions).
    - USAGE: Add relevant parameters in the python script and then run.

#### 4. `rmlt filter-datasets`

    - DESCRIPTION: Script to filter and / or combine .xyz file datasets.
    - USAGE: Add relevant parameters in the python script and then run, use this as in the other examples to first do force filtering, then filter by config type to create subsets.

#### 5. `rmlt save-train-test-split`

    - DESCRIPTION: Script to perform a train / valid / test split of an input dataset (or group of datasets) and save these as separate files.
    - USAGE: Combine all dataset subset sources (i.e. AL vac / solv + additional vacuum / solv configs) and split them into for e.g. a 70:10:20 train/valid/test split.
             Then use script in 4. to create train / valid / test vacuum / solvated subsets.

#### 6. `rmlt run-ds-analysis`

    - DESCRIPTION: analyse the spread of your dataset about a reaction coordinate, with SOAP descriptors, energy / force distributions etc.

### 2. Analysis

- `rmlt run-test-set-comparison`
  - DESCRIPTION: Runs the main energy / force comparisons of a selection of specified models. Precalculated results will be loaded from cached npz files.
  - USAGE: Add relevant parameters in the python script and then run.

## Package layout

```
react_ml_test_public/     # the installable Python package
├── helpers/              # library code (import react_ml_test_public.helpers.*)
└── scripts/              # workflow modules (run via `rmlt <name>`)
data/                     # runtime data (read relative to the working directory)
models/
inputs/
out/
testing/
```

All Python code uses absolute imports under the `react_ml_test_public` namespace, so it can be
run as an installed package. The package must be installed with `pip install -e .` (or
`pip install .`) for the scripts to be runnable.
