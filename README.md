# react_ml_test

Setup:

1. clone mlptrain separately, and run the ./install_mace.sh script - this will provide the conda environment with the name mlptrain-mace 
2. conda activate mlptrain-mace
3. pip install hdf5plugin
4. pip install seaborn

Dependencies:
- To run the ani-1nxr model, you must have the neurchem project in parent directory of react_ml_test. You must also then ensure you have the libcublas.so.9.2 file (from CUDA 9.2) added to the LD_LIBRARY_PATH.

Notes:
- Run all scripts from the parent directory, i.e. python scripts/<script_name>.py, or if on a HPC cluster: sbatch scripts/<script_name>.sh
- For some features a specific fork of MACE is required (e.g. train loss logging + patience warmup, please let me know if you want these features!)

Scripts:

When running scripts, run all from the top level parent directory, i.e. python scripts/filter_datasets.py.

Key Workflows

# 1. Dataset preparation (existing labelled xyz files)

0. scripts/save_npz_as_xyz.py 

1. scripts/ref_energy_ref_force_label.sh <dir>

    - DESCRIPTION: Bash script to relabel energy and force labels to REF_energy and REF_force, compatible with MACE.
    - USAGE: 

2. scripts/add_xyz_metadata.py

    - DESCRIPTION: Adds some xyz metadata to an .xyz file.= (i.e. config type labels / box dimensions).
    - USAGE: Add relevant parameters in python script and then run 

3. scripts/filter_datasets.py 

    - DESCRIPTION: Script to filter and / or combine .xyz file datasets.
    - USAGE: Add relevant parameters in python script and then run, use this as in the other examples to first do force filtering, then filter by config type to create subsets. 

4. scripts/get_train_test_split.py 

    - DESCRIPTION: Script to perform a train / valid / test split of an input dataset (or group of datasets) and save these as separate files.
    - USAGE: Combine all dataset subset sources (i.e. AL vac / solv + additional vacuum / solv configs) and split them into for e.g. a 70:10:20 train/valid/test split.
             Then use script in 4. to create train / valid / test vacuum / solvated subsets

5. scripts/run_ds_analysis.ipynb 

    - DESCRIPTION: analyse the spread of your dataset about a reaction coordinate, with SOAP descriptors, energy / force distributions etc.
    - 

scripts/save_xyz_sample.py 


# 2. Train

# 2. ARC / HTC / any slurm HPC scripts

scripts/arc_cpu_script.sh

scripts/arc_descriptors_script.sh

scripts/arc_dimer_opt_script.sh 

scripts/arc_inference_script.sh 

scripts/arc_mlptrain_cpu_script.sh

scripts/arc_mlptrain_gpu_script.sh

scripts/arc_orca_script.sh 

scripts/arc_train_script.sh 


# 3. Running ORCA DFT Calculations

1. scripts/save_orca_inputs_from_xyz.py

2. scripts/multi_orca_queue.sh + scripts/arc_multi_orca_script.sh 

3. scripts/check_orca_output.py 

4. scripts/orca_out_to_xyz.py 


# 4. Run Training


# 5. Analysis

scripts/plot_fine_tuning_line_plots.py 

scripts/plot_from_csv.py 


# 6. ...

scripts/compress_out_cache.py 

scripts/conda_build_script.sh 

scripts/create_conda_environment_htc.sh 

scripts/desc_analysis.ipynb 

scripts/download_datasets.py 


scripts/get_mace_descriptors.py 

scripts/get_similarities.py 

scripts/maps_soap.ipynb 

scripts/model_inspector.ipynb 

scripts/omega_script.sh 


scripts/pca_plot.ipynb 

scripts/plot_2D_pes.ipynb 

scripts/plot_react_solv_grid.ipynb 

scripts/plot_react_solv_samples.ipynb 

scripts/print_hdf5_cache.py 

scripts/prop_opp_analysis.ipynb 

scripts/react_coord_solvent_sample.py 

scripts/ref_energy_ref_force_label.sh 

scripts/run_chemiscope.ipynb 

scripts/run_constrained_sampling.py 

scripts/run_dim_red.py 

scripts/run_geom_opt.py 

scripts/run_graph_pes_test_set_comp.py 

scripts/run_greedy_iter_train.py 

scripts/run_influence_analysis.ipynb 

scripts/run_influence_analysis.py 

scripts/run_iter_train.py 

scripts/run_mace_fine_tuning.py 

scripts/run_mlip_md.py 

scripts/run_mlip_ts_opt.py

scripts/run_mlp_analysis.py 

scripts/run_model_pred_analysis.ipynb 

scripts/run_reaction_barrier_comparison.ipynb 

scripts/run_soap_analysis.py

scripts/run_test_set_comparison.py 

scripts/run_trajectory_analysis.ipynb 

scripts/run_ts_comparison.ipynb 

scripts/run_vib_analysis.py 

scripts/save_dataset_batches.py 

scripts/save_iter_train_ds.py 

scripts/sim_comp.ipynb 

scripts/solvate_config.py 

scripts/sort_and_save_ds.py


# 7. UNUSED (under development)

scripts/get_loss_hessian.ipynb 

scripts/get_loss_hessian.py 