import argparse
import csv
import glob
import os
import re

import numpy as np
import torch
from matplotlib import pyplot as plt

from react_ml_test_public.helpers import inputs as rmlt_inputs
from react_ml_test_public.helpers.io import (
    flexible_ase_read,
    get_name_from_path,
)
from react_ml_test_public.helpers.logging import logger
from react_ml_test_public.helpers.model_eval import get_test_and_pred_data
from react_ml_test_public.helpers.plot import index_plot
from react_ml_test_public.helpers.settings import (
    ANTHRACENE_DA_DATA_DIR,
    CPD_MVK_DA_DATA_DIR,
    CSV_DIR,
    DATASETS_DIR,
    GUAJADIAL_DA_DATA_DIR,
    KCAL_PER_MOL,
    KJ_PER_MOL,
    MEV,
    MEV_PER_ATOM,
    MODEL_DIR,
    MODELS,
    PAPER_MODEL_COLOURS,
    PAPER_MODELS,
    PLOT_DIR,
    PlotSettings,
)
from react_ml_test_public.helpers.stats import calculate_rmse_on_errors
from react_ml_test_public.helpers.utils import (
    get_formatted_model_name,
    natural_sort_key,
)

# set CUDA device
CUDA_DEVICE_NUM = 0
CUDA_DEVICE = f"cuda:{CUDA_DEVICE_NUM}"

if torch.cuda.is_available():
    torch.cuda.set_device(CUDA_DEVICE_NUM)
    torch.set_default_dtype(torch.float32)
    print("CUDA Version: ", torch.version.cuda)
else:
    print("No CUDA device available, using CPU...")

if __name__ == "__main__":

    """Run an MLIP reaction-barrier comparison along an IRC trajectory.

    The path to the input YAML file is given as a single command-line argument:

        rmlt run-reaction-barrier-comparison inputs/irc/<test_name>.yaml

    (or equivalently: python -m react_ml_test_public.scripts.run_reaction_barrier_comparison
    inputs/irc/<test_name>.yaml)

    Input YAML format (see inputs/IRC/guajadial_DA_IRC_barrier.yaml for a full example):

    Fields:
        n:                      # training set size.
        train_name:             # fine-tuning training set name.
        sample_str:             # sampling-strategy suffix used in fine-tuned names (e.g. "ts_gauss_").
        energy_str:             # energy label string used in fine-tuned names (e.g. "energy_").
        rep_num:                # fine-tuned model repetition number.
        omol_rep_num:           # OMOL model repetition number.
        omol_energy_str:        # OMOL energy suffix/prefix string.
        model_names:            # list of model names to compare (supports ${var} templating).
        save_name:              # output name.
        run_names:              # run names to process (defaults to [save_name]).
        num_base_models:        # number of base models.
        num_base_model_comps:   # number of comparisons per base model.
        alpha_list:             # alpha values used for plotting.
        linestyle_list:         # line style per base-model comparison.

        test_trj_fpath:         # path to the IRC trajectory .xyz to compare against.
        custom_xyz_parser:      # "orca_props_parser" / "autode_props_parser" / empty.
        include_forces:         # whether to also compare force norms.
        ref_prod_index:         # index of the product reference config (e.g. -1).
        xlabel:                 # x-axis label, e.g. "IRC Image".
        energy_units:           # energy unit for plots / CSV, e.g. "kcal / mol".
        force_units:            # force unit for force-error plots, e.g. "meV / A".
        energies_ylim:          # [lo, hi] y-limits for the reaction-profile plot.
        forces_ylim:            # [lo, hi] y-limits for force-error plots.
        grid_lines:             # whether to draw grid lines.
        show_legend:            # whether to show the legend.
        cust_legend_colours:    # legend colours for the reference (DFT) line.
        cust_legend_labels:     # legend labels for the reference (DFT) line.
        cust_legend_linestyles: # legend line styles for the reference (DFT) line.
        ref_point_labels:       # labels for the RS / TS / PS reference points (or empty).

    Any omitted field is set to None.
    """


    # input YAML config path from a single command-line argument
    parser = argparse.ArgumentParser(
        description="Run an MLIP reaction-barrier comparison driven by an input YAML "
                    "config file (format described above).",
        epilog="Example: rmlt run-reaction-barrier-comparison inputs/IRC/guajadial_DA_IRC_barrier.yaml",
    )
    parser.add_argument(
        "input_yaml",
        help="Path to the input YAML config file "
             "(e.g. inputs/IRC/guajadial_DA_IRC_barrier.yaml).",
    )
    args = parser.parse_args()

    # load + process the input YAML (see react_ml_test_public/helpers/inputs.py)
    resolved = rmlt_inputs.resolve_reaction_barrier_comparison_yaml(args.input_yaml, parser)

    # ===== model / fine-tuning selection =====
    n = resolved["n"]
    train_name = resolved["train_name"]
    sample_str = resolved["sample_str"]
    energy_str = resolved["energy_str"]
    rep_num = resolved["rep_num"]
    omol_rep_num = resolved["omol_rep_num"]
    omol_energy_str = resolved["omol_energy_str"]
    model_names = resolved["model_names"]
    save_name = resolved["save_name"]
    run_names = resolved["run_names"]
    num_base_models = resolved["num_base_models"]
    num_base_model_comps = resolved["num_base_model_comps"]
    alpha_list = resolved["alpha_list"]
    linestyle_list = resolved["linestyle_list"]

    # ===== IRC trajectory comparison =====
    test_trj_fpath = resolved["test_trj_fpath"]
    custom_xyz_parser = resolved["custom_xyz_parser"]
    include_forces = resolved["include_forces"]
    ref_prod_index = resolved["ref_prod_index"]
    xlabel = resolved["xlabel"]
    force_units = resolved["force_units"]
    grid_lines = resolved["grid_lines"]
    show_legend = resolved["show_legend"]
    cust_legend_colours = resolved["cust_legend_colours"]
    cust_legend_labels = resolved["cust_legend_labels"]
    cust_legend_linestyles = resolved["cust_legend_linestyles"]
    ref_point_labels = resolved["ref_point_labels"]
    energy_units = resolved["energy_units"]
    energies_ylim = resolved["energies_ylim"]
    forces_ylim = resolved["forces_ylim"]
    del resolved

    # ===== derived plot lists =====
    labels = [get_formatted_model_name(model_name) for model_name in model_names]
    custom_linestyle_list = linestyle_list * num_base_models
    custom_alpha_list = alpha_list * num_base_models
    custom_zorder_list = [len(model_names) + 2 - i for i in range(len(model_names))]
    colours = [x for colour in PAPER_MODEL_COLOURS.values() for x in [colour] * num_base_model_comps]


    # ===== 1. SELECT BASE MODEL HERE =====

    # --- full zero-shot model comparison ---
    # n, ft_system_name, ft_train_name, rep_num = 42, "CPD_MVK_DA", "CPD_MVK_DA_train", 1
    # # n, ft_system_name, ft_train_name, rep_num = 46, "ant
    # # hracene_DA", "anthracene_DA_train", 2
    # # n, ft_system_name, ft_train_name, rep_num = 50, "guajadial_DA", "guajadial_DA_train", 1
    # model_names = [
    #         # 'GO-MACE-23', 
    #         # 'MACE-OFF23_medium', 
    #         # 'MACE-MP-0b3_medium',
    #         # 'EGRET_1T', 
    #         # 'MACE-NEB',
    #         # 'ani-1xnr',
    #         # 'MACE-ANI-CC',
    #         'MACE-omol-0-extra-large-1024',
    #         # f"MACE-OFF23_medium_{ft_train_name}_{n}_config_strat_fine_tuned_energy_rep_{rep_num}",
    #         f"MACE-OFF23_medium_{ft_train_name}_{n}_config_strat_ts_gauss_fine_tuned_energy_rep_2",
    # ]
    # save_name = "zero_shot_paper_models"
    # run_names = ['']

    # # just models (no reference)
    # colours = list(PAPER_MODEL_COLOURS.values())
    # custom_linestyle_list=['-'] * 3
    # custom_alpha_list = None

    # fully fine-tuned (positive control) paper models
    # model_names = [
    #               "GO-MACE-23_CPD_MVK_DA_final_train_clean_vac_solv_only_force_filtered_fine_tuned_rep_1",
    #               "MACE-OFF23_medium_CPD_MVK_DA_final_train_clean_vac_solv_only_force_filtered_fine_tuned_rep_1",
    #               "MACE-MP0-128-L1_CPD_MVK_DA_final_train_clean_vac_solv_only_force_filtered_fine_tuned_rep_1",
    #               ]
    # save_name = "paper_positive_control_ft_models"
    # labels = [f'{get_formatted_model_name(model_name)} Full FT' for model_name in PAPER_MODELS]
    # run_names = ['']

    # --- specific fine-tuned model comparison ---
    # n, ft_system_name, ft_train_name, sample_str, rep_num, omol_rep_num, omol_energy_str = 42, "CPD_MVK_DA", "CPD_MVK_DA_train", 'ts_gauss_', 1, 1, 'energy_'
    # n, ft_system_name, ft_train_name, sample_str, rep_num, omol_rep_num, omol_energy_str = 46, "anthracene_DA", "anthracene_DA_train", '', 2, 3, 'energy_'

    # regular fine-tuned models
    # energy_str = ''
    # energy_str = 'energy_'
                # f"GO-MACE-23_{ft_train_name}_{n}_config_strat_fine_tuned_rep_{rep_num}",
                # f"anthracene_DA_GO46_{ft_train_name}_{n}_config_strat_w_1_4_vac_sample_fine_tuned_rep_{rep_num}",
                # f"anthracene_DA_GO46_{ft_train_name}_{n}_config_strat_w_1_4_vac_sample_fine_tuned_energy_rep_{rep_num}",
                # f"GO-MACE-23_{ft_train_name}_fine_tuned_energy_rep_1",
                # f"MACE-OFF23_medium_{ft_train_name}_{n}_config_strat_fine_tuned_rep_{rep_num}", 
                # f"MACE-OFF23_medium_{ft_train_name}_{n}_config_strat_ts_gauss_fine_tuned_energy_rep_{rep_num}",
                # f"anthracene_DA_OFF46_{ft_train_name}_{n}_config_strat_w_1_4_vac_sample_fine_tuned_rep_{rep_num}",
                # f"anthracene_DA_OFF46_{ft_train_name}_{n}_config_strat_w_1_4_vac_sample_fine_tuned_energy_rep_{rep_num}",
                # f"MACE-OFF23_medium_{ft_train_name}_fine_tuned_energy_rep_1",
                # f"MACE-MP-0b3_medium_{ft_train_name}_{n}_config_strat_fine_tuned_rep_{rep_num}",
                # f"anthracene_DA_MP46_{ft_train_name}_{n}_config_strat_w_1_4_vac_sample_fine_tuned_rep_{rep_num}",
                # f"anthracene_DA_MP46_{ft_train_name}_{n}_config_strat_w_1_4_vac_sample_fine_tuned_energy_rep_{rep_num}",
                # f"MACE-MP-0b3_medium_{ft_train_name}_fine_tuned_energy_rep_1",
                # f"MACE-omol-0-extra-large-1024_{ft_train_name}_{n}_config_strat_ts_gauss_fine_tuned_energy_rep_{rep_num}"

    # save_name = f"paper_ft_models_n={n}_w_extra_sample"

    # just models (no reference)
    # colours = list(PAPER_MODEL_COLOURS.values())
    # custom_linestyle_list=['-'] * 3
    # custom_alpha_list = None

    # paper models with a reference each
    # num_base_model_comps, alpha_list = 3, [0.2, 0.6, 0.9]  # 2 by default
    # num_base_model_comps, alpha_list, linestyle_list = 1, [0.9], ['-', '-']  # 2 by default


    # ===== 3. SPECIFY TRAJECTORY FOR COMPARISON HERE =====

    # --- CPD MVK irc in vac (wB97M) ---
    # test_trj_fpath = f"{CPD_MVK_DA_DATA_DIR}/valid/CPD_MVK_wB97M_IRC.xyz"
    # custom_xyz_parser = orca_props_parser
    # include_forces = False
    # ref_prod_index = -1
    # xlabel = 'IRC Image'
    # energies_ylim=(-30.0, 50.0)
    # forces_ylim=(0.0, 700.0)

    # --- anthracene IRC ---
    # # test_trj_fpath, energies_ylim = f"{ANTHRACENE_DA_DATA_DIR}/valid/anthracene_DA_wB97M_IRC.xyz", (-30.0, 35.0)
    # test_trj_fpath, energies_ylim = f"{ANTHRACENE_DA_DATA_DIR}/valid/anthracene_1_4_DA_wB97M_IRC.xyz", (-30.0, 35.0)
    # custom_xyz_parser = orca_props_parser
    # include_forces = False
    # ref_prod_index = -1
    # xlabel = 'IRC Image'

    # --- guajadial IRC ---
    # test_trj_fpath, energies_ylim  = f"{GUAJADIAL_DA_DATA_DIR}/valid/guajadial_wB97M_IRC_trunc.xyz", (-45.0, 45.0)

    # === UNITS ===
    # energy_units = MEV
    # energy_units = KJ_PER_MOL
    # energy_units = MEV_PER_ATOM

    # === PLOT SETTINGS ===
    # ref_point_labels =['RS', 'TS', 'PS']


    # Choose Test System

    for run_name in run_names:

        # === 2a. FINE-TUNING RUNS ===
        # base_model_name = model_names[0]
        # save_name = f"{base_model_name}_{run_name}_fine_tuned"
        # model_name_stub = f"{model_names[0]}_{run_name}_*_config_fine_tuned*"
        # model_fpaths = sorted(glob.glob(os.path.join(MODEL_DIR, model_name_stub)), key=natural_sort_key)
        # assert len(model_fpaths) > 0, f"No files detected in directory {MODEL_DIR} with name stub {model_name_stub}..."
        # model_names = [base_model_name] + [get_name_from_path(model_fpath) for model_fpath in model_fpaths]

        # === OR 2b. FROM SCRATCH RUNS ===
        # save_name = f"{from_scratch_name}_from_scratch_{run_name}"
        # model_name_stub = f"{from_scratch_name}_from_scratch_{run_name}_*_config*"
        # model_fpaths = sorted(glob.glob(os.path.join(MODEL_DIR, model_name_stub)), key=natural_sort_key)
        # assert len(model_fpaths) > 0, f"No files detected in directory {MODEL_DIR} with name stub {model_name_stub}..."
        # model_names += [get_name_from_path(model_fpath) for model_fpath in model_fpaths]

        print("Testing models: ", model_names)

        # run comparison
        # get test set ground truth  and forces
        test_set_name = get_name_from_path(test_trj_fpath)

        # load energies / forces
        # npz_fpath = f"{}"
        # cache_data = load_data_cache(npz_fpath)
        # assert cache_data is not None, f'{npz_fpath} not found...'
        # test_energies, config_lens = cache_data['mace_desc'], cache_data['config_lens']
        all_test_energies, all_test_forces, all_pred_energies_dict, all_pred_forces_dict = get_test_and_pred_data(model_names,
                                                                                                                [test_trj_fpath],
                                                                                                                include_forces=include_forces,
                                                                                                                apply_energy_offsets=True,  # True
                                                                                                                custom_xyz_parsers=[custom_xyz_parser],
                                                                                                                energy_units=energy_units,
                                                                                                                force_units=force_units,
                                                                                                                cuda_device=CUDA_DEVICE)
                                                                                                                
        combined_datasets = all_test_energies
        combined_datasets.extend([pred_energies 
                                for model_pred_energies in all_pred_energies_dict.values() 
                                for pred_energies in model_pred_energies])
        formatted_model_names = [get_formatted_model_name(model_name) for model_name in model_names]
        dataset_names = ["Reference"] + formatted_model_names
        logger.info(f"Dataset Names: {dataset_names}")

        energy_errors = [np.abs(combined_datasets[0] - pred_energies) for pred_energies in combined_datasets[1:]]
        # energy_errors = [energy_errors[i] for i in indices_filter] if indices_filter is not None else energy_errors # TODO: apply index filter!!!

        for i, model_name in enumerate(formatted_model_names):
            energy_rmse = calculate_rmse_on_errors(energy_errors[i])
            logger.info(f'{model_name} Absolute Energy RMSE: {energy_rmse}')

        # plot index plot of absolute energy errors
        # save_fp = f"{PLOT_DIR}/index/{save_name}_{test_set_names[i]}_energy_errors.png"
        plot_settings = PlotSettings(x_label='Index', y_label=f"Energy RMSE ({energy_units})", custom_colour_list=colours, 
                                    show_legend=False,
                                    custom_alpha_list=custom_alpha_list,
                                    custom_linestyle_list=custom_linestyle_list)
        save_fp = f"{PLOT_DIR}/irc/{test_set_name}_{save_name}_energy_errors.png"
        index_plot(energy_errors,                   formatted_model_names,
                   save_fp,
                   plot_as_lines=[i for i in range(len(formatted_model_names))],  # or, only 0
                   plot_settings=plot_settings
                  )
        logger.info(f"Saved energy error index plot to {save_fp}...")

        # set all energies relative to reference config
        for dataset in combined_datasets:
            dataset -= dataset[0]  # all reference themselves
            # dataset -= combined_datasets[0][0]    # reference difference to ground truth

        # get key reference values
        ref_peak_energy = np.max(combined_datasets[0])
        ref_peak_pos = np.argmax(combined_datasets[0])
        ref_prod_energy = combined_datasets[0][ref_prod_index]

        # produce plot of step vs. energy
        save_fp = f"{PLOT_DIR}/irc/{test_set_name}_{save_name}_reaction_plot.png"
        step_plot_settings = PlotSettings(x_label=xlabel, 
                                    y_label=f"Potential Energy ({energy_units})",
                                    reference_indices=[0], 
                                    ylim=energies_ylim, 
                                    grid_lines=grid_lines, 
                                    show_legend=show_legend,
                                    cust_legend_colours=cust_legend_colours,
                                    cust_legend_labels=cust_legend_labels,
                                    cust_legend_linestyles=cust_legend_linestyles,
                                    custom_alpha_list=custom_alpha_list,
                                    custom_zorder_list=custom_zorder_list,
                                    legend_fontsize=8,
                                    custom_colour_list=colours,
                                    custom_linestyle_list=[(0, (1, 1))] + custom_linestyle_list,
                                    line_width=2.0,
                                    )

        index_plot(combined_datasets,
                dataset_names,
                save_fp,
                plot_as_lines=[i for i in range(len(combined_datasets))],  # or, only 0
                plot_settings=step_plot_settings,
                ref_points=[(0, 0.0), (ref_peak_pos, ref_peak_energy), (len(combined_datasets[0]) + ref_prod_index, ref_prod_energy)],
                ref_point_labels=ref_point_labels,
                ref_label_offsets=[5.0, -10.0, 20.0]
                )
        logger.info(f"Saved reaction plot to {save_fp}...")

        # if forces are included
        model_force_norm_errors = []
        if include_forces:
            configs = flexible_ase_read(test_trj_fpath, custom_xyz_parser)
            test_force_norms = np.linalg.norm(all_test_forces[0], axis=1)
            for model_name in model_names:
                pred_force_norms = np.linalg.norm(all_pred_forces_dict[model_name][0], axis=1)
                force_norm_errors = np.abs(test_force_norms - pred_force_norms)

                # average over configs
                config_ave_force_norm_errors = []
                start = 0
                for config in configs:
                    config_force_errors = np.array(force_norm_errors[start: start + len(config)])
                    config_mean_force_error = np.mean(config_force_errors)
                    config_ave_force_norm_errors.append(config_mean_force_error)
                    start += len(config)
                model_force_norm_errors.append(np.array(config_ave_force_norm_errors))

            # index plot of force errors
            save_fp = f"{PLOT_DIR}/irc/{test_set_name}_{save_name}_force_errors.png"
            plot_settings = plot_settings._replace(y_label=f"|DFT - Pred Force Norm| ({force_units})", 
                                                custom_linestyle_list=custom_linestyle_list, ylim=forces_ylim)
            index_plot(model_force_norm_errors,
                    formatted_model_names,
                    save_fp,
                    plot_as_lines=[i for i in range(len(model_force_norm_errors))],  # or, only 0
                    plot_settings=plot_settings
                    )
            # indices = np.arange(0, len(model_force_norm_errors[0]))
            # for model_name in formatted_model_names:
            #     plt.plot(indices, dataset, label=model_name, linestyle=line_style, c=colour, zorder=z_order, linewidth=plot_settings.line_width)
            logger.info(f"Saved force error index plot to {save_fp}...")

        csv_tuples = []
        for i, model_name in enumerate(model_names):
            pred_energies = all_pred_energies_dict[model_name][0]
            reactant_energy = pred_energies[0]
            # print(pred_energies)
            peak_energy = np.max(pred_energies)
            peak_pos = np.argmax(pred_energies)

            abs_peak_energy_error = peak_energy - ref_peak_energy 
            ref_peak_energy_error = peak_energy - ref_peak_energy 
            ref_peak_force_error = model_force_norm_errors[i][peak_pos] if include_forces else 'N/A'
            ref_peak_pos_error = peak_pos - ref_peak_pos
            product_energy = pred_energies[ref_prod_index]
            csv_tuples.append((model_name, reactant_energy, product_energy, peak_energy, peak_pos, ref_prod_energy, ref_peak_energy, ref_peak_pos, ref_peak_energy_error, ref_peak_force_error, ref_peak_pos_error))

        # save csv 
        csv_save_fp = f"{CSV_DIR}/{test_set_name}_{save_name}.csv"
        with open(csv_save_fp, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            header_row = ['Model', f'Reactant Energy {energy_units}', f'Product Energy {energy_units}', f'Peak Energy {energy_units}', 'Peak Position (Step)',
                            f'Ref Prod Energy {energy_units}', f'Ref Peak Energy {energy_units}', 'Ref Peak Position', 
                            f'Ref Peak Energy Error {energy_units}', f'Ref Peak Force Error {energy_units}', 'Ref Peak Position Error (Step)']
            writer.writerow(header_row)
            print(header_row)
            writer.writerows(csv_tuples)
            for csv_tuple in csv_tuples:
                print(csv_tuple)

        logger.info(f"Saved csv to: {csv_save_fp}...")
