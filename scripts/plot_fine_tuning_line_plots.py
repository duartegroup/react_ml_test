"""
Script to plot csv data from fine tuning runs.
"""
import csv
import sys

import numpy as np
import pandas as pd

sys.path.append('../')

from react_ml_test.helpers.plot import line_plot
from react_ml_test.helpers.settings import (
    CSV_DIR,
    MACE_MODELS,
    PAPER_MODEL_COLOURS,
    PLOT_DIR,
    PlotSettings,
)
from react_ml_test.helpers.utils import get_formatted_model_name

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

if __name__ == "__main__":

    # defaults
    include_error_range = True
    include_scratch = True
    reference_indices = set()
    energy_full_ft_points = None
    force_full_ft_points = None

    # ===== SPECIFY FILE LIST AND LABELS FOR COMPARISONS HERE =====

    # === CHOOSE MODEL ===
    # model_name, from_scratch_name = "GO-MACE-23", "GO-MACE-23", 'GO-MACE-23'
    # model_name, from_scratch_name = "MACE-OFF23_medium", "MACE-OFF-23", "MACE-OFF23"
    # model_name, from_scratch_name = "MACE-MP0-128-L1", "MACE-MP0", "MACE_MP0"

    # --- paper models ---
    # format: (ft_name, from_scratch_name)
    models = [
              ("GO-MACE-23", 'GO-MACE-23'), 
              ("MACE-OFF23_medium", "MACE-OFF23_medium"), 
              ("MACE-MP-0b3_medium", "MACE_MP0")
             ]
    models_name = "paper_models"

    # --- test set comp ---
    # test_set_name, include_forces = "endo_DA_test", True
    # csv_col_stub, col_name = "Endo DA Vac Traj", "vac_traj"
    # csv_col_stub, col_name = "Endo DA Water Traj", "water_traj"
    # csv_col_stub, col_name = "Endo DA Valid", "valid"
    # csv_col_stub, col_name = "Endo DA NEB Gas", "neb_gas"

    # --- CPD + MVK DA test set comp ---
    # test_set_name, include_forces = "CPD_MVK_DA_test", True
    # csv_col_stub, col_name = "All", "all"
    # energies_ylim, use_energy_log_scale = (1.0, 30.0), False
    # forces_ylim, use_force_log_scale = (30.0, 1000.0), True
    # run_names = ['CPD_MVK_DA']
    # dataset_labels = [f"{model_name} FT" for model_name, _ in models] + [f"{model_name} From Scratch" for model_name, _ in models]
    # x_data = [0, 2, 4, 6, 8, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50]
    # n_full_ft = 314
    # energy_full_ft_points = [(n_full_ft, 3.8), (n_full_ft, 0.885), (n_full_ft, 0.824), (n_full_ft, 1.780), (n_full_ft, 1.250), (n_full_ft, 0.955)] #
    # force_full_ft_points = [(n_full_ft, 59.9), (n_full_ft, 51.7), (n_full_ft, 51.2), (n_full_ft, 67.8), (n_full_ft, 56.14), (n_full_ft, 55.47)] # 
    # include_scratch = True
    # include_error_range = True
    # # sample_str = ''
    # sample_str = 'ts_gauss_'

    # --- anthracene DA test ---
    # test_set_name, include_forces = "anthracene_DA_test", True
    # csv_col_stub, col_name = "All", "all"
    # # csv_col_stub, col_name = "Vacuum", "vacuum"
    # # csv_col_stub, col_name = "Solvated Reactants", "solvated_reactants"
    # # csv_col_stub, col_name = "Solvent", "solvent"
    # energies_ylim, use_energy_log_scale = (0.09, 30.0), False
    # forces_ylim, use_force_log_scale = (20, 1000.0), True
    # # forces_ylim, use_force_log_scale = (20, 600.0), False
    # run_names = ['anthracene_DA']
    # dataset_labels = [f"{model_name} FT" for model_name, _ in models] + [f"{model_name} From Scratch" for model_name, _ in models]
    # x_data = [0, 2, 4, 6, 8, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50]
    # n_full_ft = 455
    # energy_full_ft_points = [(n_full_ft, 2.19), (n_full_ft, 0.610), (n_full_ft, 0.423), (n_full_ft, 2.24), (n_full_ft, 0.782), (n_full_ft, 0.691)] #
    # force_full_ft_points = [(n_full_ft, 27.2), (n_full_ft, 35.2), (n_full_ft, 25.7), (n_full_ft, 42.5), (n_full_ft, 40.1), (n_full_ft, 40.4)] # 
    # include_scratch = True
    # include_error_range = True
    # sample_str = ''

    # --- guajadial DA test ---
    test_set_name, include_forces = "guajadial_DA_test", True
    csv_col_stub, col_name = "All", "all"
    # csv_col_stub, col_name = "Vacuum", "vacuum"
    # csv_col_stub, col_name = "Solvated Reactants", "solvated_reactants"
    # csv_col_stub, col_name = "Solvent", "solvent"
    energies_ylim, use_energy_log_scale = (0.09, 15.0), False
    forces_ylim, use_force_log_scale = (9.9, 1001.0), True
    # forces_ylim, use_force_log_scale = (20, 600.0), False
    run_names = ['guajadial_DA']
    dataset_labels = [f"{model_name} FT" for model_name, _ in models] + [f"{model_name} From Scratch" for model_name, _ in models]
    x_data = [0, 2, 4, 6, 8, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50]
    n_full_ft = 587
    energy_full_ft_points = [(n_full_ft, 0.584), (n_full_ft, 0.427), (n_full_ft,0.5), (n_full_ft, 0.726), (n_full_ft, 0.654), (n_full_ft, 0.670)]  # Put full ft converged energy + force Points here (n_samples, error)), corresponding to model names TODO
    force_full_ft_points = [(n_full_ft, 49.7), (n_full_ft, 45.6), (n_full_ft, 42.5), (n_full_ft, 67.3), (n_full_ft, 57.5), (n_full_ft, 54.1)]  # Put full ft converged energy + force Points here
    include_scratch = True
    include_error_range = True
    sample_str = ''

    # --- I Cl SN2 test set comp ---
    # test_set_name, include_forces = "I_Cl_SN2_test", True
    # csv_col_stub, col_name = "All", "all"
    # csv_col_stub, col_name = "Vacuum", "vacuum"
    # csv_col_stub, col_name = "Solvated Reactants", "solvated_reactants"
    # csv_col_stub, col_name = "Solvent", "solvent"
    # energies_ylim=(0.09, 6.0)
    # forces_ylim=(20, 500.0)
    # run_names = ['I_Cl_SN2']
    # dataset_labels = [f"{model_name} FT" for model_name, _ in models] + [f"{model_name} From Scratch" for model_name, _ in models]
    # x_data = [0, 2, 4, 6, 8, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50]

    # --- valid set comp ---
    # test_set_name, include_forces = "endo_DA_valid", True
    # csv_col_stub = "Endo DA Valid"
    # energies_ylim=(0.01, 180.0)
    # forces_ylim=(50, 1400.0)

    # run_names = ['endo_DA_train_random', 'endo_DA_train', 'endo_DA_train_cv_sorted', 'endo_DA_train_energy_sorted_reversed',
    #              f'endo_DA_train_soap_sim_{pretrain_ds_name}', f'endo_DA_train_soap_sim_{pretrain_ds_name}_reversed', 
    #              f'endo_DA_train_size_sorted_{model_name}_overall_loss_influence_sorted']  # full comparison#
    # dataset_labels = [f"Random", f"AL Order", f"Collective Variable", f"High Energy", "SOAP Sim.", "SOAP Sim. Reverse", "Influence Sorted", "Random From Scratch"]

    # full comparison of orderings
    # run_names = ['endo_DA_train_random', 'endo_DA_train', f'endo_DA_train_soap_sim_{pretrain_ds_name}', f'endo_DA_train_soap_sim_{pretrain_ds_name}_reversed', 
    #              f'endo_DA_train_size_sorted_{model_name}_overall_loss_influence_sorted']  # full comparison
    # dataset_labels = [f"Random", f"AL Order", "SOAP Sim.", "SOAP Sim. Reverse", "Influence Sorted", "Random From Scratch"]

    # random order only
    # run_names = ['endo_DA_train_random']
    # dataset_labels = [f"{model_name} FT" for model_name, _ in models] + [f"{model_name} From Scratch" for model_name, _ in models]
    # x_data = [0, 1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    # === PLOT SETTINGS === 
    grid_lines = False
    show_legend = False

    # ================================================

    fine_tuned_csv_fpaths = [f"{CSV_DIR}/test_set_comparison_{test_set_name}_{model_name}_iter_train_{sample_str}fine_tuned_{run_name}.csv"
                            for model_name, _ in models 
                            for run_name in run_names]
    
    energy_col_name = f'{csv_col_stub} Energies (meV / atom)'
    force_col_name = f'{csv_col_stub} Forces (meV / A)'
    save_name = f"{models_name}_ordered_fine_tune_{test_set_name}_{col_name}"

    if include_scratch:
        from_scratch_csv_fpaths = [f"{CSV_DIR}/test_set_comparison_{test_set_name}_{from_scratch_name}_iter_train_{sample_str}from_scratch_{run_name}.csv"
                                for _, from_scratch_name in models
                                for run_name in run_names]  # from scratch run
        reference_indices = [len(dataset_labels) - i for i in range(1, (len(dataset_labels) // 2) + 1)]
        cust_legend_linestyles = ['-', 'dotted'] * len(models)
        colour_list = [colour for label in dataset_labels for model_name, colour in PAPER_MODEL_COLOURS.items() if model_name in label]
        cust_legend_colours = [list(PAPER_MODEL_COLOURS.values())[i] for i in [k for j in range(len(models)) for k in [j, j]]]  # for three models with from scratch training
        cust_legend_labels = [label for model_name, _ in models for label in [f"{get_formatted_model_name(model_name)} FT", f"{get_formatted_model_name(model_name)} Scratch"]]
        x_datasets = [x_data for _ in range(0, len(fine_tuned_csv_fpaths))] + [x_data[1:] for _ in range(0, len(from_scratch_csv_fpaths))]   # don't include pretrained model in from scratch
    
    else:
        from_scratch_csv_fpaths = []
        cust_legend_linestyles = ['-'] * len(models)
        colour_list = [colour for label in dataset_labels for model_name, colour in PAPER_MODEL_COLOURS.items() if model_name in label]
        cust_legend_colours = [PAPER_MODEL_COLOURS.get(model_name) for model_name, _ in models]  # for three models without from scratch training 
        cust_legend_labels = [f"{get_formatted_model_name(model_name)} FT" for model_name, _ in models]
        x_datasets = [x_data for _ in range(0, len(fine_tuned_csv_fpaths))]
    
    csv_fpaths = fine_tuned_csv_fpaths + from_scratch_csv_fpaths

    print('Input CSV Fpaths: ', csv_fpaths)
    print('Dataset Labels: ', dataset_labels)

    print('reference_indices: ', reference_indices)
    print('Cust legend linestyles: ', cust_legend_linestyles)
    print('Colours: ', colour_list)
    print('Cust legend colours: ', cust_legend_colours)
    
    # === 3. ENDO DA FINE-TUNING NEB ACTIVATION ENERGY BARRIERS ===
    # model_name, pretrain_ds_name, from_scratch_name = "GO-MACE-23", "GO-MACE-23", 'GO-MACE-23'
    # model_name, pretrain_ds_name, from_scratch_name = "MACE-OFF23_medium", "MACE-OFF-23", "MACE-OFF23"
    # model_name, pretrain_ds_name, from_scratch_name = "MACE-MP0-128-L1", "MACE-MP0", 'MACE_MP_0'

    # test_set_name, include_forces = "endo_DA_neb_gas_wB97M", True
    # test_set_name, include_forces = "endo_DA_neb_water_wB97M", False
    # csv_col_stub = 'Ref Peak'
    # energies_ylim=(0.09, 110.0)
    # forces_ylim=(50, 1400.0)

    # run_names = ['endo_DA_train_random', 'endo_DA_train', 'endo_DA_train_cv_sorted', 'endo_DA_train_energy_sorted_reversed',
    #              f'endo_DA_train_soap_sim_{pretrain_ds_name}', f'endo_DA_train_soap_sim_{pretrain_ds_name}_reversed']
    # csv_fpaths = [f"{CSV_DIR}/{test_set_name}_{model_name}_{run_name}_fine_tuned.csv" for run_name in run_names]
    # csv_fpaths += [f"{CSV_DIR}/{test_set_name}_{from_scratch_name}_from_scratch_endo_DA_train_random.csv"]  # from scratch run

    # dataset_labels = [f"Random", f"AL Order", f"Collective Variable", f"High Energy", "SOAP Sim.", "SOAP Sim. Reverse", "From Scratch"]
    # x_data = [0, 1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # energy_col_name = f'{csv_col_stub} Energy Error (meV / atom)'
    # force_col_name = f'{csv_col_stub} Force Error (meV / A)'
    # save_name = f"{test_set_name}_{model_name}_ordered_fine_tune_activation_barriers"
    # reference_indices = [0]
    # colour_dict = {'From Scratch': 'grey'}
    # x_datasets = [x_data for _ in range(0, len(csv_fpaths))]
    # x_datasets.append(x_data[1:]) # for from scratch dataset don't use 0 value
    
    # =============================================================
    
    # add y datasets, with means and standard deviations
    energy_mean_rmse_series, energy_sd_rmse_series, force_mean_rmse_series, force_sd_rmse_series = [], [], [], []
    for csv_fpath in csv_fpaths:
        df = pd.read_csv(csv_fpath)

        # do not include base model in from scratch runs
        if 'from_scratch' in csv_fpath:
            df = df.iloc[1:]

        df['Model Key'] = df['Model'].str.replace(r'_rep_\d+', '', regex=True)
        grouped_df = df.groupby('Model Key', sort=False)

        # get mean and std of each group of replicates:
        energy_means, energy_sds, force_means, force_sds = [], [], [], [] 
        for key, group_data in grouped_df:
            energy_means.append(np.mean(group_data[energy_col_name]))
            energy_sds.append(np.nan_to_num(np.std(group_data[energy_col_name], ddof=1), nan=0.0))
            if include_forces:
                force_means.append(np.mean(group_data[force_col_name]))
                force_sds.append(np.nan_to_num(np.std(group_data[force_col_name], ddof=1), nan=0.0))

        energy_mean_rmse_series.append(np.array(energy_means))
        energy_sd_rmse_series.append(np.array(energy_sds))
        force_mean_rmse_series.append(np.array(force_means))
        force_sd_rmse_series.append(np.array(force_sds))

    # print('Energy mean rmses: ', energy_mean_rmse_series)
    # print('Energy std rmses: ', energy_sd_rmse_series)
    # print('Force mean rmses: ', force_mean_rmse_series)
    # print('Force std rmses: ', force_sd_rmse_series)

    # plot energies line plot
    plot_settings = PlotSettings(title=None,
                                 x_label="Samples", 
                                 y_label="Energy RMSE (meV / atom)", 
                                 ylim=energies_ylim,
                                 use_log_y_scale=use_energy_log_scale,
                                 custom_colour_list=colour_list,
                                 cust_legend_colours=cust_legend_colours,
                                 cust_legend_labels=cust_legend_labels,
                                 cust_legend_linestyles=cust_legend_linestyles,
                                 legend_loc="upper right",
                                 reference_indices=reference_indices,
                                 grid_lines=grid_lines,
                                 show_legend=show_legend,
                                 line_width=2.0)
    save_fp = f"{PLOT_DIR}/line_plots/{save_name}_energies_line_plot.png"
    line_plot(x_datasets,
              energy_mean_rmse_series, 
              dataset_labels, 
              save_fp,
              y_errors=energy_sd_rmse_series if include_error_range else None,
              second_ax_points=energy_full_ft_points,
              plot_settings=plot_settings)

    # plot forces line plot
    if include_forces:
        plot_settings = PlotSettings(title=None,
                                     x_label="Samples", 
                                     y_label=r"Force Norm RMSE (meV / $\AA$)", 
                                     ylim=forces_ylim,
                                     use_log_x_scale=False,
                                     use_log_y_scale=use_force_log_scale,
                                     custom_colour_list=colour_list,
                                     cust_legend_colours=cust_legend_colours,
                                     cust_legend_labels=cust_legend_labels,
                                     cust_legend_linestyles=cust_legend_linestyles,
                                     legend_loc="upper right",
                                     reference_indices=reference_indices,
                                     grid_lines=grid_lines,
                                     show_legend=show_legend
                                     )
        save_fp = f"{PLOT_DIR}/line_plots/{save_name}_forces_line_plot.png"
        line_plot(x_datasets, 
                  force_mean_rmse_series, 
                  dataset_labels,
                  save_fp,
                  y_errors=force_sd_rmse_series if include_error_range else None,
                  second_ax_points=force_full_ft_points,
                  plot_settings=plot_settings)
        
    # save means and SDs of each model energies and forces to another CSV
    mean_csv_fpaths = [f"{CSV_DIR}/test_set_comparison_{test_set_name}_{model_name}_{run_name}_fine_tuned_means.csv" 
                for model_name, _ in models 
                for run_name in run_names]
    mean_csv_fpaths += [f"{CSV_DIR}/test_set_comparison_{test_set_name}_{from_scratch_name}_from_scratch_{run_name}_means.csv"
                            for _, from_scratch_name in models
                            for run_name in run_names]  # from scratch run

    for mean_csv_fpath, dataset_label, x_dataset, energy_means, energy_sds, force_means, force_sds in zip(mean_csv_fpaths, dataset_labels, x_datasets, energy_mean_rmse_series, energy_sd_rmse_series, force_mean_rmse_series, force_sd_rmse_series):
        
        csv_tuples = [('N Configs', 'Energy RMSE Mean (meV / atom)', 'Energy RMSE SD (meV / atom)', 
                      'Forces RMSE Mean (meV / A)', 'Forces RMSE SD (meV / A)')]
        csv_tuples += [(n_configs, energy_mean, energy_sd, force_mean, force_sd) 
                      for n_configs, energy_mean, energy_sd, force_mean, force_sd 
                      in zip(x_dataset, energy_means, energy_sds, force_means, force_sds)]

        with open(mean_csv_fpath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_tuples)
 