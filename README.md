# react_ml_test


## Setup:

1. clone mlptrain separately, and run the ./install_mace.sh script - this will provide the conda environment with the name mlptrain-mace 
2. conda activate mlptrain-mace
3. pip install hdf5plugin
4. pip install seaborn

Notes:
- Run all scripts from the parent directory, i.e. python scripts/<script_name>.py, or if on a HPC cluster: sbatch scripts/<script_name>.sh
- For some features a specific fork of MACE is required (e.g. train loss logging + patience warmup, please let me know if you need these features!)
- To run the ani-1nxr model, you must have the neurchem project in parent directory of react_ml_test. You must also then ensure you have the libcublas.so.9.2 file (from CUDA 9.2) added to the LD_LIBRARY_PATH.

## Examples

Note: 

### 1. Energy + force errors of zero-shot and fine-tuned models for CPD + MVK DA system.

Run ```python scripts/run_test_set_comparison.py inputs/eg1.yaml```. Out

### 2.

### 3.

### 4.

### 5. 

## Key Workflows

### 1. Dataset preparation (existing labelled xyz files)

#### 1. scripts/save_npz_as_xyz.py 

#### 2. scripts/ref_energy_ref_force_label.sh <dir>

    - DESCRIPTION: Bash script to relabel energy and force labels to REF_energy and REF_force, compatible with MACE.
    - USAGE: 

#### 3. scripts/add_xyz_metadata.py

    - DESCRIPTION: Adds some xyz metadata to an .xyz file.= (i.e. config type labels / box dimensions).
    - USAGE: Add relevant parameters in python script and then run 

#### 4. scripts/filter_datasets.py 

    - DESCRIPTION: Script to filter and / or combine .xyz file datasets.
    - USAGE: Add relevant parameters in python script and then run, use this as in the other examples to first do force filtering, then filter by config type to create subsets. 

#### 5. scripts/get_train_test_split.py 

    - DESCRIPTION: Script to perform a train / valid / test split of an input dataset (or group of datasets) and save these as separate files.
    - USAGE: Combine all dataset subset sources (i.e. AL vac / solv + additional vacuum / solv configs) and split them into for e.g. a 70:10:20 train/valid/test split.
             Then use script in 4. to create train / valid / test vacuum / solvated subsets

#### 6. scripts/run_ds_analysis.py 

    - DESCRIPTION: analyse the spread of your dataset about a reaction coordinate, with SOAP descriptors, energy / force distributions etc.

###  2. Analysis

- scripts/run_test_set_comparison.py
  - DESCRIPTION: Runs the main energy / force comparisons of a selection of specified models. Precalculated reuslts will be loaded from cached npz files.
  - USAGE: Add relevant parameters in python script and then run.