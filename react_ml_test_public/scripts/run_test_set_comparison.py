"""
Script to compare MLIPs running inference on specified test sets.
"""
import glob
import os
import re
from typing import Callable, List, Tuple

import ase
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from react_ml_test_public.helpers import inputs as rmlt_inputs
from react_ml_test_public.helpers.io import (
    autode_props_parser,
    flexible_ase_read,
    get_name_from_path,
    orca_props_parser,
    resolve_cuda_device,
)
from react_ml_test_public.helpers.logging import logger
from react_ml_test_public.helpers.model_eval import get_test_and_pred_data
from react_ml_test_public.helpers.plot import (
    bar_plot,
    correlation_scatter_plot,
    index_plot,
    violin_plot,
)
from react_ml_test_public.helpers.settings import (
    ANTHRACENE_DA_DATA_DIR,
    CPD_MVK_DA_DATA_DIR,
    CSV_DIR,
    DATASETS_DIR,
    DPI,
    GRAPH_PES_MODEL_LOADER,
    GUAJADIAL_DA_DATA_DIR,
    ISOLATED_ATOMS_DIR,
    MACE_MODEL_LOADER,
    MODELS,
    PAPER_MODEL_COLOURS,
    PAPER_MODELS,
    PLOT_DIR,
    SINGLE_COLUMN_FIG_SIZE,
    I_Cl_SN2_DATA_DIR,
    PlotSettings,
)
from react_ml_test_public.helpers.stats import calculate_rmse_on_errors, moving_rmse
from react_ml_test_public.helpers.utils import (
    generate_error_table,
    get_formatted_model_name,
    natural_sort_key,
    unflatten_array_by_lengths,
)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def run_model_comparison_on_test_set(model_names: List[str],
                                     test_set_fpaths: List[str],
                                     offset_ref_fpath: str = None,
                                     save_name: str = None,
                                     test_name: str = None,
                                     test_set_names: List[str] = None,
                                     pbc_box: List[float] = None,
                                     model_loaders: List[str] = None,
                                     custom_xyz_parsers: List[Callable] = None,
                                     csv_category_names: List[str] = None,
                                     plot: bool = True,
                                     glob_plot_set: PlotSettings = PlotSettings(),
                                     corr_plot_set: PlotSettings = PlotSettings(),
                                     config_type_filter: str = None,
                                     test_set_force_atom_ids: List[List[int]] = None,
                                     react_coords: List[Tuple[int]] = None,
                                     ref_ts_config: ase.Atoms = None,
                                     enable_cueq: bool = False,
                                     cuda_device: str = "cuda:0"
                                    ):

    """
    Compare a list of MLIP models on a list of specified test sets. Prints a table of results comparing RMSE and MAE
    of energies and forces for each model on the test set.

    Units are meV / atom for energies and meV / A for forces.

    :param model_fpaths: a list of file paths to .model objects to be evaluated
    :param test_set_fpaths: a list of file paths to .xyz files (which should include reference energies and forces)
    :param test_set_force_atom_ids: a list of atom ids per test set (so a list of lists) to only be considered for force errors (i.e. reactants in solvent)
    :param cuda_device: device for the MLIP calculators ('cuda:0' default, 'cpu' to force CPU)
    :return:
    """

    test_set_names = [get_name_from_path(test_set_fpath) for test_set_fpath in test_set_fpaths] if test_set_names is None else test_set_names

    # get test set energies / forces and run inference on test set with models 
    all_test_energies, all_test_forces, all_pred_energies_dict, all_pred_forces_dict = get_test_and_pred_data(model_names,
                                                                                                              test_set_fpaths,
                                                                                                              offset_ref_fpath=offset_ref_fpath,
                                                                                                              model_loaders=model_loaders,
                                                                                                              custom_xyz_parsers=custom_xyz_parsers,
                                                                                                              pbc_box=pbc_box,
                                                                                                              config_type_filter=config_type_filter,
                                                                                                              cuda_device=cuda_device,
                                                                                                              enable_cueq=enable_cueq,
                                                                                                              model_dir=model_dir)

    # filter only specific atom ids for force comparisons
    if test_set_force_atom_ids is not None: 
        
        for i, test_set_fpath in enumerate(test_set_fpaths):

            force_atom_ids = test_set_force_atom_ids[i]
            if force_atom_ids is None:
                continue

            # update test forces for atom ids only
            # logger.info(f'Test Set: {test_set_fpath}')
            config_lens = [len(config) for config in flexible_ase_read(test_set_fpath, custom_xyz_parsers[i] if custom_xyz_parsers is not None else None)]
            config_test_forces = unflatten_array_by_lengths(all_test_forces[i], config_lens)
            # logger.info(f"Num config test force arrays: {len(config_test_forces)}")
            atom_id_test_forces = np.concatenate([config_forces[force_atom_ids] for config_forces in config_test_forces])
            # logger.info(f"Finalised atom_id_test_forces shape: {atom_id_test_forces.shape}")
            all_test_forces[i] = atom_id_test_forces

            # update pred forces for atom ids only
            for model_name in all_pred_forces_dict.keys():
                config_pred_forces = unflatten_array_by_lengths(all_pred_forces_dict[model_name][i], config_lens)
                # logger.info(f"Num config pred force arrays: {len(config_test_forces)}")
                atom_id_pred_forces = np.concatenate([config_forces[force_atom_ids] for config_forces in config_pred_forces])
                all_pred_forces_dict[model_name][i] = atom_id_pred_forces
                # logger.info(f"Finalised atom_id_pred_forces shape: {all_pred_forces_dict[model_name][i].shape}")
    
        logger.info(f'Filtered test and pred forces for specified atom ids, now with shape for e.g.: {all_test_forces[0].shape}')

    # create combined correlation plot of energies and forces
    all_rs_force_errors, all_ts_force_errors, all_ps_force_errors = [], [], []
    if plot:
        for i, (test_energies, test_forces) in enumerate(zip(all_test_energies, all_test_forces)):

            # energies
            formatted_model_names, all_pred_energies = [], []
            for model_name, model_pred_energies, in all_pred_energies_dict.items():
                formatted_model_names.append(get_formatted_model_name(model_name))
                all_pred_energies.append(model_pred_energies[i])

            corr_plot_set = corr_plot_set._replace(x_label=f"DFT Norm. E meV / atom", y_label=f"Pred. Norm. E meV / atom")
            correlation_scatter_plot(test_energies,
                                     all_pred_energies,
                                     formatted_model_names,
                                     save_fp=f"{PLOT_DIR}/correlation/{save_name}_{test_set_names[i]}_energies_correlation_scatter.png",
                                     plot_settings=corr_plot_set
                                    )

            # forces correlation
            if test_forces is not None:
                test_force_norms = np.linalg.norm(test_forces, axis=1)
                formatted_model_names, all_pred_force_norms = [], []
                for model_name, model_pred_forces, in all_pred_forces_dict.items():
                    formatted_model_names.append(get_formatted_model_name(model_name))
                    pred_force_norms = np.linalg.norm(model_pred_forces[i], axis=1)
                    all_pred_force_norms.append(pred_force_norms)

                corr_plot_set = corr_plot_set._replace(x_label=r"DFT Force Norm meV / $\AA$", y_label=r"Pred Force Norm meV / $\AA$")
                correlation_scatter_plot(test_force_norms,
                                         all_pred_force_norms,
                                         formatted_model_names,
                                         save_fp=f"{PLOT_DIR}/correlation/{save_name}_{test_set_names[i]}_force_norms_correlation_scatter.png",
                                         plot_settings=corr_plot_set
                                        )

            # sort data by reaction coordinate for index plots
            ts_region = None
            configs = flexible_ase_read(test_set_fpaths[i], custom_xyz_parsers[i] if custom_xyz_parsers is not None else None)
            sorted_indices = [i for i, _ in enumerate(configs)]
            rs_ids, ts_ids, ps_ids = [i for i, _ in enumerate(configs)], [i for i, _ in enumerate(configs)], [i for i, _ in enumerate(configs)] # default
            # ts_dist_threshold = 1.4
            ts_dist_low_threshold = 1.0
            ts_dist_up_threshold = 1.0
            if react_coords is not None:
                config_index_tuples = [(config, config_index, sum([config.get_distance(i, j) for i, j in react_coords])) 
                                       for config_index, config in enumerate(configs)]
                sorted_config_index_tuples = sorted(config_index_tuples, key=lambda config_tuple: config_tuple[2], reverse=True)
                configs = [config for config, _, _ in sorted_config_index_tuples]
                sorted_indices = [config_id for _, config_id, _ in sorted_config_index_tuples]
                sorted_dists = [dist for _, _, dist in sorted_config_index_tuples]

                # get shaded region about TS
                if ref_ts_config is not None:
                    ref_ts_dist = sum([ref_ts_config.get_distance(i, j) for i, j in react_coords])
                    print(f"Ref TS react coords: (r1: {ref_ts_config.get_distance(*react_coords[0])}, r2: {ref_ts_config.get_distance(*react_coords[1])})")
                    print("Ref TS Dist: ", ref_ts_dist)
                    # half_threshold = ts_dist_threshold / 2
                    ts_region_indices = [i for i in range(len(sorted_indices)) if -ts_dist_low_threshold <= sorted_dists[i] - ref_ts_dist <= ts_dist_up_threshold]
                    print("TS Region indices: ", ts_region_indices)
                    ts_region = ts_region_indices[0], ts_region_indices[-1]
                    print("TS Region: ", ts_region)
                    # save rs, ts, ps configs
                    rs_ids = [i for i in sorted_indices if sorted_dists[i] > ref_ts_dist + ts_dist_up_threshold]
                    ts_ids = [i for i in sorted_indices if -ts_dist_low_threshold <= sorted_dists[i] - ref_ts_dist <= ts_dist_up_threshold]
                    ps_ids = [i for i in sorted_indices if sorted_dists[i] < ref_ts_dist - ts_dist_low_threshold]
                    print('Total Num configs: ', len(sorted_indices))
                    print('Num RS Configs: ', len(rs_ids))
                    # print([f'(r1: {configs[i].get_distance(*react_coords[0])}, r2: {configs[i].get_distance(*react_coords[1])})' for i in rs_ids])
                    print('Num TS Configs: ', len(ts_ids))
                    # print([f'(r1: {configs[i].get_distance(*react_coords[0])}, r2: {configs[i].get_distance(*react_coords[1])})' for i in ts_ids])
                    print('Num PS Configs: ', len(ps_ids))
                    # print([f'(r1: {configs[i].get_distance(*react_coords[0])}, r2: {configs[i].get_distance(*react_coords[1])})' for i in ps_ids])


            test_energies = test_energies[sorted_indices]
            rs_test_energies = test_energies[rs_ids]
            ts_test_energies = test_energies[ts_ids]
            ps_test_energies = test_energies[ps_ids]

            # get errors for each data point
            formatted_model_names, model_energy_errors = [], []
            model_rs_energy_errors, model_ts_energy_errors, model_ps_energy_errors = [], [], []
            for model_name in model_names:
                model_pred_energies = all_pred_energies_dict[model_name][i]
                model_pred_energies = model_pred_energies[sorted_indices]
                formatted_model_names.append(get_formatted_model_name(model_name))
                model_energy_errors.append(np.abs(test_energies - model_pred_energies))

                # get rs / ts/ ps specific energy errors
                rs_pred_energies, ts_pred_energies, ps_pred_energies = model_pred_energies[rs_ids], model_pred_energies[ts_ids], model_pred_energies[ps_ids]
                model_rs_energy_errors.append(np.abs(rs_test_energies - rs_pred_energies))
                model_ts_energy_errors.append(np.abs(ts_test_energies - ts_pred_energies))
                model_ps_energy_errors.append(np.abs(ps_test_energies - ps_pred_energies))

            # create index plot of energy errors
            save_fp = f"{PLOT_DIR}/index/{save_name}_{test_set_names[i]}_energy_errors.png"
            plot_settings = glob_plot_set._replace(x_label='Index', y_label="Energy RMSE (meV / atom)", 
                                                   ylim=global_plot_settings.energies_ylim, 
                                                   use_log_y_scale=global_plot_settings.use_energy_log_scale)
            index_plot(model_energy_errors,
                       formatted_model_names,
                       save_fp,
                       plot_as_lines=[i for i in range(len(model_energy_errors))],  # or, only 0
                       plot_settings=plot_settings,
                       shade_region=ts_region
                       )

            logger.info(f"Saved energy error index plot to {save_fp}...")

            # create violin plot of energy errors

            plot_settings = glob_plot_set._replace(y_label="Energy Errors (meV/atom)")
            violin_plot(model_datasets = [model_rs_energy_errors, model_ts_energy_errors, model_ps_energy_errors],
                        model_names = model_names,
                        dataset_names = ["RS", "TS", "PS"],
                        save_fp=f"{PLOT_DIR}/bar_plots/{save_name}_{test_set_names[i]}_energy_errors_violin.png",
                        plot_settings=plot_settings
                        )

            # create index plot of force errors
            if test_forces is not None:

                # correct for filtered config lengths
                if test_set_force_atom_ids is not None and test_set_force_atom_ids[i] is not None:
                    config_lens = [len(test_set_force_atom_ids[i]) for config in configs]
                else:
                    config_lens = [len(config) for config in configs]

                # get test force norms per config
                test_force_norms = np.linalg.norm(test_forces, axis=1)
                config_test_force_norms = unflatten_array_by_lengths(test_force_norms, config_lens)
                config_test_force_norms = [config_test_force_norms[j] for j in sorted_indices]
                rs_config_test_force_norms = [config_test_force_norms[j] for j in rs_ids]
                ts_config_test_force_norms = [config_test_force_norms[j] for j in ts_ids]
                ps_config_test_force_norms = [config_test_force_norms[j] for j in ps_ids]

                model_config_force_errors = []
                model_rs_config_force_errors, model_ts_config_force_errors, model_ps_config_force_errors = [], [], []
                for model_name in model_names:

                    model_pred_forces = all_pred_forces_dict[model_name][i]

                    # get pred force norms per config
                    pred_force_norms = np.linalg.norm(model_pred_forces, axis=1)
                    config_pred_force_norms = unflatten_array_by_lengths(pred_force_norms, config_lens)
                    config_pred_force_norms = [config_pred_force_norms[j] for j in sorted_indices]

                    config_ave_force_norm_errors = [np.mean(np.abs(test_force_norms - pred_force_norms)) for test_force_norms, pred_force_norms in zip(config_test_force_norms, config_pred_force_norms)]
                    model_config_force_errors.append(np.array(config_ave_force_norm_errors))

                    # get rs / ts / ps specific errors
                    rs_config_pred_force_norms = [config_pred_force_norms[j] for j in rs_ids]
                    ts_config_pred_force_norms = [config_pred_force_norms[j] for j in ts_ids]
                    ps_config_pred_force_norms = [config_pred_force_norms[j] for j in ps_ids]
                    model_rs_config_force_errors.append([np.mean(np.abs(test_force_norms - pred_force_norms)) for test_force_norms, pred_force_norms in zip(rs_config_test_force_norms, rs_config_pred_force_norms)])
                    model_ts_config_force_errors.append([np.mean(np.abs(test_force_norms - pred_force_norms)) for test_force_norms, pred_force_norms in zip(ts_config_test_force_norms, ts_config_pred_force_norms)])
                    model_ps_config_force_errors.append([np.mean(np.abs(test_force_norms - pred_force_norms)) for test_force_norms, pred_force_norms in zip(ps_config_test_force_norms, ps_config_pred_force_norms)])

                plot_settings = plot_settings._replace(y_label=r"Force Norm Error (meV / $\AA$)",
                                                       grid_lines=False,
                                                       ylim=global_plot_settings.forces_ylim,
                                                       use_log_y_scale=global_plot_settings.use_forces_log_scale)
                
                # config averaged force norm errors
                save_fp = f"{PLOT_DIR}/index/{save_name}_{test_set_names[i]}_config_force_errors.png"
                index_plot(model_config_force_errors,
                           formatted_model_names,
                           save_fp,
                           plot_as_lines=[i for i in range(len(model_config_force_errors))],  # or, only 0
                           plot_settings=plot_settings,
                           shade_region=ts_region
                          )
                logger.info(f"Saved config force error index plot to {save_fp}...")

                # create violin plot of errors
                violin_plot(model_datasets = [model_rs_config_force_errors, model_ts_config_force_errors, model_ps_config_force_errors],
                            model_names = model_names,
                            dataset_names = ["RS", "TS", "PS"],
                            save_fp=f"{PLOT_DIR}/bar_plots/{save_name}_{test_set_names[i]}_config_force_errors_violin.png",
                            plot_settings=plot_settings
                            )

                all_rs_force_errors.append(model_rs_config_force_errors)
                all_ts_force_errors.append(model_ts_config_force_errors)
                all_ps_force_errors.append(model_ps_config_force_errors)

    # create rsme table of energies and forces for each model
    energy_rmse_df = generate_error_table(all_pred_energies_dict,
                                          all_test_energies,
                                          test_set_names,
                                          'rmse',
                                          title="Energies (Shifted) RMSE (meV / atom)")
    forces_rmse_df = generate_error_table(all_pred_forces_dict,
                                          all_test_forces,
                                          test_set_names,
                                          'rmse_comp',
                                          title="Forces Norm RMSE (meV / A)")

    # calculate R squared
    energy_r2_df = generate_error_table(all_pred_energies_dict, all_test_energies, test_set_names, 'R2', title="Energies R2 (meV / atom)")
    forces_r2_df = generate_error_table(all_pred_forces_dict, all_test_forces, test_set_names, 'R2', title="Forces R2 (meV / A)")

    # save combined df as csv
    if save_name is not None:

        out_csv_fpath = f"{CSV_DIR}/test_set_comparison_{test_name}_{save_name}.csv"

        # rename columns
        energy_rmse_df.columns = [f"{csv_category_names[i]} Energies (meV / atom)" for i, _ in enumerate(energy_rmse_df.columns)]
        forces_rmse_df.columns = [f"{csv_category_names[i]} Forces (meV / A)" for i, _ in enumerate(forces_rmse_df.columns)]
        energy_r2_df.columns = [f"{csv_category_names[i]} Energy R2 " for i, _ in enumerate(energy_rmse_df.columns)]
        forces_r2_df.columns = [f"{csv_category_names[i]} Force R2 " for i, _ in enumerate(forces_rmse_df.columns)]

        merged_df = pd.concat([energy_rmse_df, energy_r2_df, forces_rmse_df, forces_r2_df], axis=1)
        merged_df = merged_df.rename_axis(index="Model")
        merged_df.to_csv(out_csv_fpath, index=True)
        logger.info(f"Saved combined RMSE table to {out_csv_fpath}...")

        # create combined bar plot of energy rmses
        plot_settings = PlotSettings(x_label=None, 
                                     y_label='Energy RMSE (meV / atom)', 
                                     ylim=global_plot_settings.energies_ylim,
                                     use_log_y_scale=global_plot_settings.use_energy_log_scale,
                                     show_legend=glob_plot_set.show_legend, 
                                     custom_colour_list=glob_plot_set.custom_colour_list,
                                     custom_alpha_list=glob_plot_set.custom_alpha_list,
                                     legend_loc=glob_plot_set.legend_loc
                                     )
        # bar_plot([energy_rmse_df[col].tolist() for col in energy_rmse_df.columns],
        #          energy_rmse_df.index.tolist(),
        #          csv_category_names,
        #          save_fp=f"{PLOT_DIR}/bar_plots/{save_name}_{test_name}_energies_bar_plot.png",
        #          mode=plot_settings.bar_plot_mode,
        #          plot_settings=plot_settings
        #         )

        bar_plot(energy_rmse_df.values.tolist(),
            csv_category_names,
            energy_rmse_df.index.tolist(),
            save_fp=f"{PLOT_DIR}/bar_plots/{save_name}_{test_name}_energies_bar_plot.png",
            mode=plot_settings.bar_plot_mode,
            plot_settings=plot_settings,
            show_x_tick_labels=False
        )

        # force rmses
        plot_settings = plot_settings._replace(y_label='Force Norm RMSE (meV / A)',
                                               ylim=global_plot_settings.forces_ylim,
                                               use_log_y_scale=global_plot_settings.use_forces_log_scale,)

        # bar_plot([forces_rmse_df[col].tolist() for col in forces_rmse_df.columns],
        #          forces_rmse_df.index.tolist(),
        #          csv_category_names,
        #          save_fp=f"{PLOT_DIR}/bar_plots/{save_name}_{test_name}_forces_bar_plot.png",
        #          plot_settings=plot_settings)

        bar_plot(forces_rmse_df.values.tolist(),
            csv_category_names,
            forces_rmse_df.index.tolist(),
            save_fp=f"{PLOT_DIR}/bar_plots/{save_name}_{test_name}_forces_bar_plot.png",
            mode=plot_settings.bar_plot_mode,
            plot_settings=plot_settings,
            show_x_tick_labels=False
        )

        # create combined violin plot with desired columns
        if plot:
            plot_settings = plot_settings._replace(num_base_models=global_plot_settings.num_base_models,
                                                   num_base_model_comps=global_plot_settings.num_base_model_comps,
                                                   use_log_y_scale=glob_plot_set.use_forces_log_scale,
                                                   y_label="Force Norm Errors (meV/Å)")
            # --- select datasets - HERE SELECT SUMMARY DATASET! ---
            config_type_index, combined_name = len(test_set_fpaths) - 2, 'all'  # -2 = solute for vac and solv
            # solv_react_index, combined_name = len(test_set_fpaths) - 3, 'solvated'  # -3 = solvated solute only
            # solv_react_index, combined_name = 1, 'vac'                         # = vac only
            solv_index = len(test_set_fpaths) - 1
            rs_errors, ts_errors, ps_errors = all_rs_force_errors[config_type_index], all_ts_force_errors[config_type_index], all_ps_force_errors[config_type_index]  # solvated reactants
            solvent_errors = [np.concatenate([np.array(all_rs_force_errors[solv_index][j]), np.array(all_ts_force_errors[solv_index][j]), np.array(all_ps_force_errors[solv_index][j])]) 
                                      for j in range(len(model_names))] # solvent only force errors
            dataset_names = ["RS", "TS", "PS", "Solvent"]
            violin_plot(model_datasets = [rs_errors, ts_errors, ps_errors, solvent_errors],
                        model_names = model_names,
                        dataset_names = dataset_names,
                        save_fp=f"{PLOT_DIR}/bar_plots/{save_name}_{test_name}_combined_{combined_name}_config_force_errors_violin_w_solv.png",
                        plot_settings=plot_settings
                        )
            for i, model_name in enumerate(model_names):
                print(f"Model: {model_name}")
                for ds_name, region_errors in zip(dataset_names, [rs_errors[i], ts_errors[i], ps_errors[i], solvent_errors[i]]):
                    print(f"{ds_name} RMSE: {calculate_rmse_on_errors(region_errors)}")


if __name__ == "__main__":

    """Run an MLIP test-set comparison.

    The path to the input YAML file is given as a single command-line argument:

        rmlt run-test-set-comparison inputs/eg1.yaml

    (or equivalently: python -m react_ml_test_public.scripts.run_test_set_comparison
    inputs/eg1.yaml)

    Input YAML format (see inputs/test/CPD_MVK_DA_FT42_test.yaml for a full example):

    Fields (path fields may reference environment variables, e.g. model_dir: "$DATA/models"):
        model_names:            # list of model names to compare.
        model_dir:              # directory those models are resolved in, when they live
                                # outside the default MODEL_DIR (defaults to MODEL_DIR).
        n:                      # training set size.
        ft_train_name:          # fine-tuning training set name.
        test_name:              # test set name.
        rep_num:                # fine-tuned model repetition number.
        omol_rep_num:           # OMOL specific model replicate number.
        omol_energy_str:        # OMOL specific energy suffix/prefix string.
        test_set_ids:           # test-set indices to calculate.
        save_name:              # output name.
        run_names:              # run names to process.
        plot:                   # whether to generate plots.
        num_base_models:        # number of base models.
        num_base_model_comps:   # number of comparisons per base model.
        alpha_list:             # alpha values used for plotting.
        cuda_device:            # GPU device for the MLIP calculators (default "cuda:0",
                                # falls back to "cpu" if that device is not found).

    Any omitted field is set to None.
    """

    import argparse

    model_loaders = None
    global_plot_settings = PlotSettings()
    corr_plot_set = PlotSettings()
    custom_xyz_parsers = None
    test_set_force_atom_ids = None
    test_set_names = None
    react_coords = None
    ref_ts_config = None
    enable_cueq = False
    offset_ref_fpath = None
    sample_str = ""

    # input YAML config path from command-line argument.
    parser = argparse.ArgumentParser(
        description="Run an MLIP test-set comparison driven by an input YAML config "
                    "file (format described above).",
        epilog="Example: rmlt run-test-set-comparison inputs/eg1.yaml",
    )
    parser.add_argument(
        "input_yaml",
        help="Path to the input YAML config file (e.g. inputs/eg1.yaml).",
    )
    args = parser.parse_args()

    # load + process the input YAML (see react_ml_test_public/helpers/inputs.py)
    resolved = rmlt_inputs.resolve_test_set_comparison_yaml(args.input_yaml, parser)

    # input vars
    model_dir = resolved["model_dir"]
    model_names = resolved["model_names"]
    n = resolved["n"]
    ft_train_name = resolved["ft_train_name"]
    test_name = resolved["test_name"]
    rep_num = resolved["rep_num"]
    omol_rep_num = resolved["omol_rep_num"]
    omol_energy_str = resolved["omol_energy_str"]
    test_set_ids = resolved["test_set_ids"]
    save_name = resolved["save_name"]
    run_names = resolved["run_names"]
    plot = resolved["plot"]
    num_base_models = resolved["num_base_models"]
    num_base_model_comps = resolved["num_base_model_comps"]
    alpha_list = resolved["alpha_list"]
    cuda_device = resolved["cuda_device"]
    del resolved

    # resolve the requested device (i.e. fall back to 'cpu' if the GPU is not available)
    cuda_device = resolve_cuda_device(cuda_device)
    if cuda_device != "cpu":
        torch.set_default_dtype(torch.float32)

    energies_ylim, forces_ylim = (-0.1, 120), (7.0, 1001.0)

    # ==============================================

    labels = [get_formatted_model_name(model_name) for model_name in model_names]

    # just one set of models
    # global_plot_settings = PlotSettings(show_legend=False, legend_loc="upper right", 
    #                                     custom_colour_list=list(PAPER_MODEL_COLOURS.values()),
    #                                     energies_ylim=(0.0, 20.0), forces_ylim=(0.0, 200.0))

    # comparison to reference set too
    global_plot_settings = global_plot_settings._replace(custom_colour_list = [x for colour in PAPER_MODEL_COLOURS.values() for x in [colour] * num_base_model_comps], 
                                                         custom_linestyle_list=['-'] * len(model_names), 
                                                         custom_alpha_list=alpha_list * num_base_models, # (len(model_names) // 2),
                                                         cust_legend_colours = [x for colour in PAPER_MODEL_COLOURS.values() for x in [colour] * num_base_model_comps],
                                                         show_legend=False,
                                                         #  cust_legend_labels = [f"{get_formatted_model_name(model_name)}" for model_name in PAPER_MODELS] + [f"{get_formatted_model_name(model_name)} TS FT"  for model_name in PAPER_MODELS],
                                                         cust_legend_labels = [label for model_name in model_names for label in [f"{get_formatted_model_name(model_name)}", f"{get_formatted_model_name(model_name)} FT"]],
                                                         cust_legend_linestyles = [(0, (1, 1)), '-'] * (len(model_names) // 2),
                                                         legend_ncols=len(model_names) // 2,
                                                         energies_ylim=energies_ylim,
                                                         forces_ylim=forces_ylim,
                                                         use_energy_log_scale=True,
                                                         use_forces_log_scale=True,
                                                         num_base_models=num_base_models,
                                                         num_base_model_comps=num_base_model_comps
                                                        )

    # # --- fully fine-tuned (positive control) paper models ---
    # # ft_system_name, ft_train_name = "anthracene_DA", "anthracene_DA_train"
    # # model_names = [
    # #               f"GO-MACE-23_{ft_train_name}_fine_tuned_rep_1",
    # #               f"MACE-OFF23_medium_{ft_train_name}_fine_tuned_rep_1",
    # #               f"MACE-MP-0b3_medium_{ft_train_name}_fine_tuned_rep_1",
    # #               ]
    # # save_name = f"{ft_system_name}_paper_full_ft_models"
    # # labels = [f'{get_formatted_model_name(model_name)} Full FT' for model_name in PAPER_MODELS]
    # # run_names = ['']

    # labels = [get_formatted_model_name(model_name) for model_name in model_names]
    # run_names = ['']
    # plot = False  # True

    # ==============================================

    # --- grid based reactivity + solvent runs ---
    # n, system_name, test_name = 8, "endo_DA", "CPD_MVK_DA_test"
    # n, system_name, test_name = 8, "anthracene_DA", "anthracene_DA_test"
    # n, system_name, test_name = 8, "guajadial_DA", "guajadial_DA_test"

    # # model_names = ["GO-MACE-23"]
    # # model_names = ["MACE-OFF23_medium"]
    # model_names = ["MACE-MP-0b3_medium"]
    # save_name = f"{model_names[0]}_{system_name}_grid_n={n}"
    # plot = False
    # rep_num = None
    # run_names = [f"{system_name}_n={n}_r_dist"]

    # --- for iterative fine-tuning ---
    # model_names = ["GO-MACE-23"]
    # from_scratch_name = "GO-MACE-23"

    # # model_names = ["MACE-OFF23_medium"]
    # # from_scratch_name = "MACE-OFF23_medium"
    
    # # model_names = ["MACE-MP-0b3_medium"]
    # # from_scratch_name = "MACE_MP0"

    # # # run_names, test_name, sample_str = ['CPD_MVK_DA'], "CPD_MVK_DA_test", ""
    # # # run_names, test_name, sample_str = ['CPD_MVK_DA'], "CPD_MVK_DA_test", "ts_gauss_"
    # # # run_names, test_name = ['anthracene_DA'], "anthracene_DA_test"
    # run_names, test_name = ['guajadial_DA'], "guajadial_DA_test"
    # # # # run_names = ['I_Cl_SN2']

    # save_name = model_names[0] + f"_iter_train"
    # plot = False
    # rep_num = None  # set None to process all!

    # --- single TS run names ---
    # global_plot_settings = PlotSettings(show_legend=False, 
    #                                     legend_loc="upper left")
    # plot = True
    # run_names = ['endo_DA_vac_TS_wB97M']
    # rep_num = 1

    # all models
    # model_names = PAPER_MODELS
    # save_name = f"paper_models_single_shot_TS_fine_tuned"
    # global_plot_settings = global_plot_settings._replace(custom_colour_list = [x for colour in PAPER_MODEL_COLOURS.values() for x in [colour, colour]], 
    #                                                      custom_linestyle_list=[(0, (1, 1)), '-'] * 3, 
    #                                                      custom_alpha_list=[0.5, 1.0] * 3,
    #                                                      cust_legend_colours = [x for colour in PAPER_MODEL_COLOURS.values() for x in [colour, colour]],
    #                                                     #  cust_legend_labels = [f"{get_formatted_model_name(model_name)}" for model_name in PAPER_MODELS] + [f"{get_formatted_model_name(model_name)} TS FT"  for model_name in PAPER_MODELS],
    #                                                      cust_legend_labels = [label for model_name in PAPER_MODELS for label in [f"{get_formatted_model_name(model_name)}", f"{get_formatted_model_name(model_name)} TS FT"]],
    #                                                      cust_legend_linestyles = [(0, (1, 1)), '-'] * 3,
    #                                                      legend_ncols=3)
    # model_names = ['GO-MACE-23', 'GO-MACE-23_endo_DA_vac_TS_wB97M_fine_tuned_rep_1', 
    #                'MACE-OFF23_medium', 'MACE-OFF23_medium_endo_DA_vac_TS_wB97M_fine_tuned_rep_1',
    #                'MACE-MP0-128-L1', 'MACE-MP0-128-L1_endo_DA_vac_TS_wB97M_fine_tuned_rep_1']

    # one specific model
    # model_names = ['MACE-MP0-128-L1']
    # model_names = ['MACE-OFF23_medium']
    # model_names = ['GO-MACE-23']
    # save_name = f"{model_names[0]}_single_shot_TS_fine_tuned"
    # corr_plot_set = PlotSettings(show_legend=True,
    #                               custom_colour_list=[PAPER_MODEL_COLOURS[model_names[0]]] * 2,#, '#801850'],
    #                               custom_alpha_list=[0.2, 1.0],
    #                               cust_marker_linewidth=[0.0, 0.1],
    #                               cust_size_list=[4, 2],
    #                               custom_markers=['o', 'o'],
    #                               cust_legend_labels=[f"Zero-Shot", f"TS Fine-tuned"],
    #                               )

    # ===============================================
    rep_str = f'_rep_{rep_num}' if rep_num is not None else ''
    print('Run Names: ', run_names)
    for run_name in run_names:

        # === FINE-TUNED RUNS ===
        # !!! UNCOMMENT THIS TO DETECT MULTIPLE FINE-TUNED RUNS!!!
        # base_model_names = [model_name for model_name in model_names]
        # save_name = f"{save_name}_{sample_str}fine_tuned_{run_name}"
        # for base_model_name in base_model_names:
        #     model_name_stub = f"{base_model_name}_{run_name}*{sample_str}fine_tuned{rep_str}*" # fine_tuned
        #     model_fpaths = sorted(glob.glob(os.path.join(MODEL_DIR, model_name_stub)), key=natural_sort_key)
        #     assert len(model_fpaths) > 0, f"No files detected in directory {MODEL_DIR} with name stub {model_name_stub}..."
        #     model_names += [get_name_from_path(model_fpath) for model_fpath in model_fpaths]

        # === OR FROM SCRATCH RUNS ===
        # save_name = f"{from_scratch_name}_from_scratch_{run_name}"
        # model_name_stub = f"{save_name}_*_config*{sample_str}*"
        # model_fpaths = sorted(glob.glob(os.path.join(MODEL_DIR, model_name_stub)), key=natural_sort_key)
        # # model_fpaths = [fpath for fpath in model_fpaths if re.match(rf".*{from_scratch_name}_from_scratch_endo_DA_train_random_\d+_config.*", fpath)]
        # assert len(model_fpaths) > 0, f"No files detected in directory {MODEL_DIR} with name stub {model_name_stub}..."
        # model_names += [get_name_from_path(model_fpath) for model_fpath in model_fpaths]
        # plot = False

        print("Testing models: ", model_names)
        
        # ===== SPECIFY TEST SETS FOR COMPARISON HERE =====

        # --- endo DA trajectories / NEB ---
        if test_name == "CPD_MVK_DA_traj":
            test_set_fpaths = [
                f"{CPD_MVK_DA_DATA_DIR}/CPD_MVK_wB97M_IRC.xyz",
                # f"{CPD_MVK_DA_DATA_DIR}/endo_ace_wB97M_nonpca_orca.xyz",
                f"{CPD_MVK_DA_DATA_DIR}/endo_in_water_ace_wB97M_short_orca_non_periodic.xyz",
                ]
            pbc_box = None
            csv_category_names = ["Vac IRC", "Water Traj"]
            # csv_category_names = ["Endo DA Vacuum NEB", "Endo DA Vac Traj", "Endo DA Water Traj"]
            index_xlabels = ["IRC Step", "MD Step", "MD Step"]
            custom_xyz_parsers = [orca_props_parser, None, None]
            react_coords = [(1, 12), (6, 11)]
            ref_ts_config = flexible_ase_read(f'{CPD_MVK_DA_DATA_DIR}/TS/endo_DA_vac_TS_wB97M.xyz')[0]
            offset_ref_fpath = f"{CPD_MVK_DA_DATA_DIR}/test/{test_name}.xyz"

        # --- CPD + MVK DA final paper combined test set ---
        elif test_name == "CPD_MVK_DA_test":
            test_set_fpaths = [
                f"{CPD_MVK_DA_DATA_DIR}/test/{test_name}.xyz",
                f"{CPD_MVK_DA_DATA_DIR}/test/{test_name}_vac_only.xyz",
                f"{CPD_MVK_DA_DATA_DIR}/test/{test_name}_solv_only.xyz",
                f"{CPD_MVK_DA_DATA_DIR}/test/{test_name}_solv_only.xyz",
                f"{CPD_MVK_DA_DATA_DIR}/test/{test_name}.xyz",
                f"{CPD_MVK_DA_DATA_DIR}/test/{test_name}_solv_only.xyz"
                ]
            test_set_names = [
                        "CPD_MVK_DA_test",
                        "CPD_MVK_DA_test_vac_only",
                        "CPD_MVK_DA_test_solv_only",
                        "CPD_MVK_DA_test_solv_only_reactants",
                        "CPD_MVK_DA_test_solute",
                        "CPD_MVK_DA_test_solv_only_solvent",
                        ]
            # csv_category_names = ["All"]
            # csv_category_names = ["All", "Vacuum", "Solvated"]
            # csv_category_names = ["All", "Vacuum", "Solvated Reactants", 'Solvent']
            csv_category_names = ["All", "Vacuum", "Solvated", "Solvated Reactants", 'All Solute', 'Solvent']
            test_set_force_atom_ids = [
                                    None,
                                    None, 
                                    None, 
                                    [i for i in range(22)],      # reactants in solvent only
                                    [i for i in range(22)],      # solute atoms
                                    [i for i in range(22, 109)]  # solvent molecules only (NOTE: THE NUM OF SOLVENT MOLECULES IS VARIABLE, WE ARE JUT TAKING THE MIN NUMBER )
                                    ]
            pbc_box = None
            react_coords = [(1, 12), (6, 11)]
            ref_ts_config = flexible_ase_read(f'{CPD_MVK_DA_DATA_DIR}/TS/CPD_MVK_DA_vac_TS_wB97M.xyz')[0]
            offset_ref_fpath = f"{CPD_MVK_DA_DATA_DIR}/test/{test_name}.xyz"

        # --- anthracene test set ---
        elif test_name == "anthracene_DA_test":

            test_set_fpaths = [
                                f"{ANTHRACENE_DA_DATA_DIR}/test/{test_name}.xyz",
                                f"{ANTHRACENE_DA_DATA_DIR}/test/{test_name}_vac_only.xyz",
                                f"{ANTHRACENE_DA_DATA_DIR}/test/{test_name}_solv_only.xyz", 
                                f"{ANTHRACENE_DA_DATA_DIR}/test/{test_name}_solv_only.xyz",
                                f"{ANTHRACENE_DA_DATA_DIR}/test/{test_name}.xyz",
                                f"{ANTHRACENE_DA_DATA_DIR}/test/{test_name}_solv_only.xyz", 
                            ]
            test_set_names = [
                                "anthracene_DA_test",
                                "anthracene_DA_test_vac_only",
                                "anthracene_DA_test_solv_only",
                                "anthracene_DA_test_solv_only_reactants",
                                "anthracene_DA_test_solute",
                                "anthracene_DA_test_solv_only_solvent",
                            ]
            pbc_box = None
            csv_category_names = ["All", "Vacuum", "Solvated", "Solvated Reactants", 'Solute', 'Solvent']
            test_set_force_atom_ids = [
                                    None, 
                                    None, 
                                    None, 
                                    [i for i in range(41)],      # reactants in solvent only
                                    [i for i in range(41)],      # reactants in solvent only
                                    [i for i in range(41, 356)]  # solvent molecules only
                                    ]

            react_coords = [(2, 20), (5, 22)]
            ref_ts_config = flexible_ase_read(f'{ANTHRACENE_DA_DATA_DIR}/TS/anthracene_DA_vac_TS_wB97M.xyz')[0]
            offset_ref_fpath = f"{ANTHRACENE_DA_DATA_DIR}/test/{test_name}.xyz"

        elif test_name == "anthracene_DA_traj":

            test_set_fpaths = [
                                f"{ANTHRACENE_DA_DATA_DIR}/valid/anthracene_DA_wB97M_IRC.xyz",
                                f"{ANTHRACENE_DA_DATA_DIR}/valid/anthracene_1_4_DA_wB97M_IRC.xyz",
                             ]
            test_set_names = [
                                "anthracene_DA_wB97M_IRC",
                                "anthracene_1_4_DA_wB97M_IRC",
                             ]
            csv_category_names = ["anthracene_DA_wB97M_IRC", "anthracene_DA_wB97M_IRC"]
            pbc_box = None
            custom_xyz_parsers = [orca_props_parser, orca_props_parser]

        # --- guajadial test set ---
        elif test_name == 'guajadial_DA_test':

            test_set_fpaths = [
                                f"{GUAJADIAL_DA_DATA_DIR}/test/{test_name}.xyz",
                                f"{GUAJADIAL_DA_DATA_DIR}/test/{test_name}_vac_only.xyz",
                                f"{GUAJADIAL_DA_DATA_DIR}/test/{test_name}_solv_only.xyz", 
                                f"{GUAJADIAL_DA_DATA_DIR}/test/{test_name}_solv_only.xyz", 
                                f"{GUAJADIAL_DA_DATA_DIR}/test/{test_name}.xyz",
                                f"{GUAJADIAL_DA_DATA_DIR}/test/{test_name}_solv_only.xyz", 
                            ]
            test_set_names = [
                                "guajadial_DA_test",
                                "guajadial_DA_test_vac_only",
                                "guajadial_DA_test_solv_only",
                                "guajadial_DA_test_solv_only_reactants",
                                "guajadial_DA_test_solute",
                                "guajadial_DA_test_solv_only_solvent",
                            ]
            pbc_box = None
            # csv_category_names = ["All"]
            # csv_category_names = ["Vacuum", "Solvated Reactants"]
            # csv_category_names = ["All", "Vacuum", "Solvated Reactants", 'Solvent']
            csv_category_names = ["All", "Vacuum", "Solvated", "Solvated Reactants", 'Solute', 'Solvent']
            test_set_force_atom_ids = [
                                    None, 
                                    None, 
                                    None, 
                                    [i for i in range(69)],      # reactants in solvent only
                                    [i for i in range(69)],      # reactants in solvent only
                                    [i for i in range(69, 429)]  # solvent molecules only
                                    ]
            react_coords = [(13, 31), (6, 32)]
            ref_ts_config = flexible_ase_read(f'{GUAJADIAL_DA_DATA_DIR}/TS/guajadial_DA_vac_TS_wB97M.xyz')[0]
            offset_ref_fpath = f"{GUAJADIAL_DA_DATA_DIR}/test/{test_name}.xyz"
        
        # --- spice test set ---
        # test_name = "spice"
        # test_configs_fp = [f"{DATASETS_DIR}/datasets/spice/test_large_neut_all.xyz"]

        # --- mptrj test set ---
        # test_name = "mptrj"
        # test_configs_fp = [f"{DATASETS_DIR}/datasets/mptrj/"]

        logger.info(f"Testing models on {test_set_fpaths}...")
        # logger.info(f"Using atom ids: {test_set_force_atom_ids} for forces...")

        # filter test set info based on provided ids
        test_set_fpaths = [test_set_fpaths[i] for i in test_set_ids]
        test_set_names = [test_set_names[i] for i in test_set_ids]
        csv_category_names = [csv_category_names[i] for i in test_set_ids]
        test_set_force_atom_ids = [test_set_force_atom_ids[i] for i in test_set_ids]

        # run comparison
        run_model_comparison_on_test_set(model_names,
                                         test_set_fpaths,
                                         save_name=save_name,
                                         test_name=test_name,
                                         test_set_names=test_set_names,
                                         offset_ref_fpath=offset_ref_fpath,
                                         pbc_box=pbc_box,
                                         model_loaders=model_loaders,
                                         custom_xyz_parsers=custom_xyz_parsers,
                                         csv_category_names=csv_category_names,
                                         plot=plot,
                                         glob_plot_set=global_plot_settings,
                                         corr_plot_set=corr_plot_set,
                                         test_set_force_atom_ids=test_set_force_atom_ids,
                                         react_coords=react_coords,
                                         ref_ts_config=ref_ts_config,
                                         enable_cueq=enable_cueq,
                                         cuda_device=cuda_device
                                         )
