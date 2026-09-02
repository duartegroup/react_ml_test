# react_ml_test_public

Tools and scripts for building and testing reactive machine-learned interatomic potentials (MLIPs) for chemical reactions in vacuum and solution.

This repository is a pip-installable Python package, installed as the distribution `react_ml_test_public` and imported as `react_ml_test_public.*`. The import namespace is intentionally distinct from `react_ml_test` so that this public copy can coexist in the same environment as the private `react_ml_test` repository. The heavy MLIP stack (MACE, mlptrain, autode, ...) is **not** installed by pip - it is expected to be provided by your (conda) environment (see [Setup](#setup)).

## Setup

1. Create a conda environment with the required MLIP stack. For example, clone `mlptrain` separately and run its `./install_mace.sh` script - this provides the conda environment named `mlptrain-mace`:
   ```bash
   conda activate mlptrain-mace
   pip install hdf5plugin
   pip install seaborn
   ```
2. Install this package (editable is recommended while developing):
   ```bash
   pip install -e .
   ```
   A regular install also works:
   ```bash
   pip install .
   ```

Notes:
- Run all scripts from the repository root directory. The scripts read `data/`, `models/`, `inputs/` and write to `out/` relative to the current working directory, so they must be launched from the repo root (or a directory containing these folders).
- Only lightweight, PyPI-only dependencies are listed in `pyproject.toml`. The MLIP stack (MACE, mlptrain, autode, torchani, graph-pes, load-atoms, etc.) is typically installed from conda / custom forks and must be present in the environment.
- For some features a specific fork of MACE is required (e.g. train loss logging + patience warmup, please let me know if you need these features!)
- To run the ani-1nxr model, you must have the neurchem project in the parent directory of this repo. You must also then ensure you have the `libcublas.so.9.2` file (from CUDA 9.2) added to the `LD_LIBRARY_PATH`.

## Running the scripts

Each workflow script can be run in its historical form from the repository root:

```bash
python scripts/run_test_set_comparison.py
```

or, equivalently, as a package module:

```bash
python -m react_ml_test_public.scripts.run_test_set_comparison
```

The `scripts/` folder at the repository root contains thin launchers that hand off to
`react_ml_test_public.scripts.*` (so the two invocations are identical), and they also work from a
plain checkout without installing the package.

## Examples

Note:

### 1. Energy + force errors of zero-shot and fine-tuned models for CPD + MVK DA system.

Run ```python scripts/run_test_set_comparison_yaml.py``` (edit the input parameters / YAML path at the bottom of the script, or pass in `inputs/eg1.yaml` as shown in the script header).

### 2.

### 3.

### 4.

### 5.

## Key Workflows

### 1. Dataset preparation (existing labelled xyz files)

#### 1. `python scripts/save_npz_as_xyz.py`

#### 2. `scripts/ref_energy_ref_force_label.sh <dir>`

    - DESCRIPTION: Bash script to relabel energy and force labels to REF_energy and REF_force, compatible with MACE.
    - USAGE:

#### 3. `python scripts/add_xyz_metadata.py`

    - DESCRIPTION: Adds some xyz metadata to an .xyz file (= i.e. config type labels / box dimensions).
    - USAGE: Add relevant parameters in the python script and then run.

#### 4. `python scripts/filter_datasets.py`

    - DESCRIPTION: Script to filter and / or combine .xyz file datasets.
    - USAGE: Add relevant parameters in the python script and then run, use this as in the other examples to first do force filtering, then filter by config type to create subsets.

#### 5. `python scripts/save_train_test_split.py`

    - DESCRIPTION: Script to perform a train / valid / test split of an input dataset (or group of datasets) and save these as separate files.
    - USAGE: Combine all dataset subset sources (i.e. AL vac / solv + additional vacuum / solv configs) and split them into for e.g. a 70:10:20 train/valid/test split.
             Then use script in 4. to create train / valid / test vacuum / solvated subsets.

#### 6. `python scripts/run_ds_analysis.py`

    - DESCRIPTION: analyse the spread of your dataset about a reaction coordinate, with SOAP descriptors, energy / force distributions etc.

### 2. Analysis

- `python scripts/run_test_set_comparison.py`
  - DESCRIPTION: Runs the main energy / force comparisons of a selection of specified models. Precalculated results will be loaded from cached npz files.
  - USAGE: Add relevant parameters in the python script and then run.

## Package layout

```
react_ml_test_public/     # the installable Python package
├── helpers/              # library code (import react_ml_test_public.helpers.*)
└── scripts/              # workflow modules (import react_ml_test_public.scripts.<name>)
scripts/                  # repo-root launchers: `python scripts/<name>.py`
data/                     # runtime data (read relative to the working directory)
models/
inputs/
out/
testing/
```

All Python code uses absolute imports under the `react_ml_test_public` namespace, so it can be
used both as an installed package and run directly from a checkout.
