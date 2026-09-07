# react_ml_test_public

Tools and scripts for training / evaluating machine-learned interatomic potentials (MLIPs) for chemical reactions in vacuum and solution, as part of the pre-print: "Data-efficient fine-tuning for reaction modelling in explicit solvent" (Holownia, J.; Juraskova, V.; Deringer, V. L., Duarte, F., 2026): https://chemrxiv.org/doi/abs/10.26434/chemrxiv.15007217/v1.

## Setup

1. Create a conda environment with the required MLIP stack. For example, clone `mlptrain` separately and run its `./install_mace.sh` script - this provides the conda environment named `mlptrain-mace`:
   ```bash
   conda activate mlptrain-mace
   pip install hdf5plugin
   ```
2. Install this package into the environment. **This install is required** — it provides the
   `rmlt` command used to run the workflow scripts (see [Running the scripts](#running-the-scripts)). An editable install is recommended while developing:
   ```bash
   pip install -e .
   ```
   A regular install (`pip install .`) also works, but will ignore future code edits.

Notes:
- Run all scripts from the repository root directory. The scripts read `data/`, `models/`, `inputs/` and write to `out/` relative to the current working directory, so they must be launched from the repo root (or a directory containing these folders).
- Only lightweight, PyPI-only dependencies are listed in `pyproject.toml`. The MLIP stack (MACE, mlptrain, autode, torchani, graph-pes, load-atoms, etc.) is installed via the `mlptrain-mace` environment.
- For some features a specific fork of MACE is required (e.g. train loss logging + patience warmup, please let me know if you need these features!)
- To run the ani-1nxr model, you must have the neurchem project in the parent directory of this repo. You must also then ensure you have the `libcublas.so.9.2` file (from CUDA 9.2) added to the `LD_LIBRARY_PATH`.

## Running scripts

Installing the package (see [Setup](#setup)) adds the `rmlt` command to your `PATH`.
Workflow scripts live in the `react_ml_test_public.scripts` package, and `rmlt` runs one of them by name from the repository root (the scripts read `data/`, `models/`, `inputs/` and write to `out/` relative to the current working directory), for e.g.:

```bash
rmlt run-test-set-comparison <args>
```

List all available scripts with:

```bash
rmlt list
```

Scripts can also be run as ordinary modules, if you prefer:

```
python -m react_ml_test_public.scripts.run_test_set_comparison
```

## Examples

Note: these examples will load packaged precalculated .npz or .csv file results to produce figures, as running the calculations requires a GPU and the respective models. If you wish to run these examples and fully re-calculate results, remove the relevant .npz cache files and add the required models in react_ml_test_public/models, which can be found in our ORA repository: <>. 

### 1. Test set energy + force errors:

For zero-shot and fine-tuned models for CPD + MVK DA system:

Run: ```rmlt run-test-set-comparison inputs/test/CPD_MVK_DA_FT42_test.yaml```.

Output figures will be saved in `out/plt/bar_plot/`, and `out/csv/`, along with relevant values / statistics printed to the terminal.

This script reproduces the violin plot in Figure 3b\) in the paper, along with relevant supplementary bar plots.

To run this for the other two systems, use `inputs/test/anthracene_DA_FT46_test.yaml` and `inputs/test/guajadial_DA_FT50_test.yaml`.

### 2. IRC energy barrier:

For zero-shot and fine-tuned models for CPD + MVK DA system:

Run: ```rmlt run-reaction-barrier-comparison inputs/CPD_MVK_DA_FT42_IRC.yaml```.

Output figures will be saved in `out/plt/bar_plot/`, and `out/csv/`, along with relevant values / statistics printed to the terminal.

This script reproduces Figure 3c\) in the paper, along with relevant supplementary plots.

To run this for the other systems, use `inputs/test/anthracene_DA_FT46_IRC.yaml`, `inputs/test/anthracene_1_4_DA_FT46_IRC.yaml`, and `inputs/test/guajadial_DA_FT50_IRC.yaml`.

### 3. TS Optimisation + dynamics:

1. TS Optimisation

For fine-tuned models for CPD + MVK DA system:

Run: ```rmlt run-reaction-barrier-comparison inputs/CPD_MVK_DA_FT42_IRC.yaml```.

2. Downhill dynamics



### 4. Pretraining bias analysis:

For CPD + MVK DA system:


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
models/                   # store MLIP models here
inputs/                   # input YAML files for scripts
out/                      # runtime outputs stored here
testing/                  # some additional testing
```
