"""
Helpers for eval of models.
"""
import csv
import glob
import os
from typing import Callable, Literal, Tuple, List
import numpy as np
from react_ml_test_public.helpers.io import flexible_ase_read, get_name_from_path, load_isolated_atom_energies, load_data_cache, load_model_calculator, save_npz_data_cache
from react_ml_test_public.helpers.plot import index_plot
from react_ml_test_public.helpers.settings import CSV_DIR, EV, EV_PER_AMSTRONG, KCAL_PER_MOL, MEV, MEV_PER_AMSTRONG, MEV_PER_ATOM, MODEL_DIR, NPZ_DIR, PAPER_MODEL_COLOURS, PLOT_DIR, PlotSettings
from react_ml_test_public.helpers.stats import calculate_rmse_on_errors
from react_ml_test_public.helpers.utils import convert_units, get_ase_props
from react_ml_test_public.helpers.utils import calc_energy_shifts, guess_per_element_mean_energies
from react_ml_test_public.helpers.utils import get_formatted_model_name, natural_sort_key
from react_ml_test_public.helpers.logging import logger


def get_model_predictions(model_name: str,
                          test_set_fpaths: List[str],
                          include_forces: bool = True,
                          apply_energy_offsets: bool = True,
                          offset_ref_fpath: str = None,
                          pred_isolated_atom_fpath: str = None,
                          custom_xyz_parsers: List[Callable] = None,
                          cuda_device: str = 'cuda:0',
                          pbc_box: List[float] = None,
                          energy_units: Literal[EV, MEV, MEV_PER_ATOM] = MEV_PER_ATOM,
                          force_units: Literal[EV_PER_AMSTRONG, MEV_PER_AMSTRONG] = MEV_PER_AMSTRONG,
                          model_loader: str = None,
                          enable_cueq: bool = False) -> list[np.ndarray]:
    """
    Generates predictions from one or more trained models on a given test dataset.

    Arguments:
        model_name (List[str]): The model name used for prediction.
        test_set_fpaths (List[str]): A list of file paths to the test set XYZ files.
        include_forces (bool): Whether to include force predictions. Defaults to True.
        apply_energy_offsets (bool): Whether to apply energy offsets for energy predictions. Defaults to True.
                                     (set to False for relative energy comparisons).
        offset_ref_fpath (str): Path to reference offset file for energy adjustment. Optional.
        pred_isolated_atom_fpath (str): Path to isolated atom predictions for reference. Optional.
        custom_xyz_parsers (List[Callable]): Custom parser functions for reading XYZ files. Optional.
        cuda_device (str): CUDA device identifier, e.g., 'cuda:0'. Defaults to 'cuda:0',
                           falling back to 'cpu' if that device is not available.
        pbc_box (List[float]): Dimensions of the periodic boundary condition box, if applicable. Optional.
        energy_units (Literal['meV', 'meV / atom']): Desired output units for energy. Defaults to 'meV / atom'.
        force_units (Literal['meV', 'meV / A']): Desired output units for forces. Defaults to 'meV / A'.
        model_loader (str): Custom model loader (i.e. MACE, ACE, ANI...). Optional.
        enable_cueq (bool): Whether to enable cuequivariance acceleration. Defaults to False.

    Returns:
        list[np.ndarray]: a list of predictions by the model for each specified test_set file
    """
    
    input_energy_units = EV
    input_force_units = EV_PER_AMSTRONG

    calculator = load_model_calculator(model_name, 
                                       device=cuda_device, 
                                       model_loader=model_loader, 
                                       enable_cueq=enable_cueq)

    model_pred_energies, model_pred_forces = [], []
    for j, test_set_fpath in enumerate(test_set_fpaths):

        custom_xyz_parser = custom_xyz_parsers[j] if custom_xyz_parsers is not None else None
        configs = flexible_ase_read(test_set_fpath, custom_xyz_parser)
        # test_set_atom_nums = list(set([int(z) for config in configs for z in config.numbers]))

        # get model offsets
        pred_energy_offsets = None
        if apply_energy_offsets:
            if pred_isolated_atom_fpath is not None:
                logger.info(f"Getting prediction offsets from isolated atoms using: {pred_isolated_atom_fpath}...")
                pred_energy_offsets = load_isolated_atom_energies(pred_isolated_atom_fpath)
            elif offset_ref_fpath is not None:
                logger.info(f"Getting model prediction per-element average offsets from {offset_ref_fpath}...")
                pred_energy_offsets = guess_per_element_mean_energies(offset_ref_fpath,
                                                                    calc=calculator,
                                                                    calc_name=model_name,
                                                                    dtype=np.float32,
                                                                    pbc_box=pbc_box,
                                                                    )
            else:
                logger.info(f"Getting model prediction per-element average offsets based on test set: {test_set_fpath}...")
                pred_energy_offsets = guess_per_element_mean_energies(test_set_fpath,
                                                                    calc=calculator,
                                                                    calc_name=model_name,
                                                                    dtype=np.float32,
                                                                    pbc_box=pbc_box,
                                                                    )
        logger.info(f"Model: {model_name} Per-element means: {pred_energy_offsets}")

        # load cached npz file of energies / forces if it exists
        offset_str = '_no_offset' if not apply_energy_offsets else ''
        npz_cache_path = f"{NPZ_DIR}/{model_name}_{get_name_from_path(test_set_fpath)}_energies_forces{offset_str}.npz"
        cache_data = load_data_cache(npz_cache_path)
        if cache_data is not None:
            pred_energies, pred_forces = cache_data['energies'], cache_data['forces']

            # resave if forces are missing
            if pred_forces is None and include_forces:
                pred_forces = get_ase_props(configs, 'forces', calc=calculator, pbc_cell=pbc_box)
                save_npz_data_cache(npz_cache_path, {'energies': pred_energies, 'forces': pred_forces, 
                                                     'energy_unit': EV, 'force_unit': EV_PER_AMSTRONG})
        else:
            logger.info("Calculating model energies and force predictions...")
            if calculator is not None:
                pred_energies = get_ase_props(configs, 'energy', calc=calculator, pbc_cell=pbc_box)
                pred_forces = get_ase_props(configs, 'forces', calc=calculator, pbc_cell=pbc_box) if include_forces else None       

                # save predictions to npz cache
                save_npz_data_cache(npz_cache_path, {'energies': pred_energies, 'forces': pred_forces, 
                                                    'energy_unit': EV, 'force_unit': EV_PER_AMSTRONG})
            else:
                # logger.warning(f"NO MODEL CALCULATOR, AND NO NPZ CACHE FOUND for {model_name}... Add {model_name} to ./models and try again.")
                raise OSError(f"NO MODEL CALCULATOR, AND NO NPZ CACHE FOUND for {model_name}... Add {model_name} to ./models and try again.")

        # apply offset if specified
        if pred_energy_offsets is not None:
            config_shifts = calc_energy_shifts(configs, pred_energy_offsets)
            pred_energies = pred_energies - config_shifts

        # apply unit conversions      
        pred_energies = convert_units(pred_energies, input_energy_units, energy_units, n_atoms_list=[len(config) for config in configs])

        if include_forces:
            pred_forces = convert_units(pred_forces, input_force_units, force_units)

        model_pred_energies.append(pred_energies)
        model_pred_forces.append(pred_forces)

    return model_pred_energies, model_pred_forces


def get_test_and_pred_data(model_names: List[str],
                           test_set_fpaths: List[str],
                           include_forces: bool = True,
                           apply_energy_offsets: bool = True,
                           offset_ref_fpath: str = None,
                           test_isolated_atom_fpath: str = None,
                           pred_isolated_atom_fpath: str = None,
                           custom_xyz_parsers: List[Callable] = None,
                           cuda_device: str = 'cuda:0',
                           pbc_box: List[float] = None,
                           energy_units: Literal[MEV, MEV_PER_ATOM] = MEV_PER_ATOM,
                           force_units: Literal[EV_PER_AMSTRONG, MEV_PER_AMSTRONG] = MEV_PER_AMSTRONG,
                           model_loaders: List[str] = None,
                           enable_cueq: bool = False,
                           config_type_filter: str = None) -> Tuple[list, list, dict, dict]:
    """
    Get the test and predicted energies / forces for a set of test sets with a set of models.

    Arguments:
        model_names (List[str]): The list of model names to use for prediction.
        test_set_fpaths (List[str]): A list of file paths to the test set XYZ files.
        include_forces (bool): Whether to include force predictions. Defaults to True.
        offset_ref_fpath (str): Path to reference offset file for energy adjustment. Optional.
        pred_isolated_atom_fpath (str): Path to isolated atom predictions for reference. Optional.
        custom_xyz_parsers (List[Callable]): Custom parser functions for reading XYZ files. Optional.
        cuda_device (str): CUDA device identifier, e.g., 'cuda:0'. Defaults to 'cuda:0',
                           falling back to 'cpu' if that device is not available.
        pbc_box (List[float]): Dimensions of the periodic boundary condition box, if applicable. Optional.
        energy_units (Literal['meV', 'meV / atom']): Output units for energy. Defaults to 'meV / atom'.
        force_units (Literal['meV', 'meV / A']): Output units for forces. Defaults to 'meV / A'.
        model_loaders (List[str]): Custom model loaders. Optional.
        enable_cueq (bool): Whether to enable cuequivariance acceleration. Defaults to False.

    Returns:
        Tuple[list, list, dict, dict]: two lists of test energies / forces and two dicts of per-model pred energies / forces.
    """

    all_test_energies = []
    all_test_forces = []
    all_pred_energies_dict = {}
    all_pred_forces_dict = {}

    # get test set ground truth energies and forces
    for i, test_set_fpath in enumerate(test_set_fpaths):
        custom_xyz_parser = custom_xyz_parsers[i] if custom_xyz_parsers is not None else None

        # get offsets for test set
        test_energy_offsets = None
        if apply_energy_offsets:
            if test_isolated_atom_fpath is not None:
                logger.info(f"Getting test offsets from isolated atoms using: {test_isolated_atom_fpath}...")
                test_energy_offsets = load_isolated_atom_energies(test_isolated_atom_fpath)
            elif offset_ref_fpath is not None:
                logger.info(f"Calculating per-element average ground truth offsets from {offset_ref_fpath}...")
                test_energy_offsets = guess_per_element_mean_energies(offset_ref_fpath, dtype=np.float32, pbc_box=pbc_box, custom_xyz_parser=custom_xyz_parser)
            else:
                logger.info(f"Calculating per-element average ground truth offsets based on test set: {test_set_fpath}...")
                test_energy_offsets = guess_per_element_mean_energies(test_set_fpath, dtype=np.float32, pbc_box=pbc_box, custom_xyz_parser=custom_xyz_parser)

        logger.info(f"True Per-element means for {test_set_fpath}: {test_energy_offsets}")

        configs = flexible_ase_read(test_set_fpath, custom_xyz_parser)

        test_energies = get_ase_props(configs, 'energy')
        test_forces = get_ase_props(configs, 'forces') if include_forces else None

        if test_energy_offsets is not None:
            config_shifts = calc_energy_shifts(configs, test_energy_offsets)
            test_energies = test_energies - config_shifts

        test_energies = convert_units(test_energies, EV, energy_units, n_atoms_list=[len(config) for config in configs])

        if include_forces:
            test_forces = convert_units(test_forces, EV_PER_AMSTRONG, force_units)

        all_test_energies.append(test_energies)
        all_test_forces.append(test_forces)

    # run inference on test sets:
    for i, model_name in enumerate(model_names):

        model_loader = model_loaders[i] if model_loaders is not None else None

        model_pred_energies, model_pred_forces = get_model_predictions(model_name,
                                                                       test_set_fpaths,
                                                                       include_forces=include_forces,
                                                                       apply_energy_offsets=apply_energy_offsets,
                                                                       offset_ref_fpath=offset_ref_fpath,
                                                                       pred_isolated_atom_fpath=pred_isolated_atom_fpath,
                                                                       custom_xyz_parsers=custom_xyz_parsers,
                                                                       cuda_device=cuda_device,
                                                                       pbc_box=pbc_box,
                                                                       energy_units=energy_units,
                                                                       force_units=force_units,
                                                                       model_loader=model_loader,
                                                                       enable_cueq=enable_cueq)

        # add predictions to dictionary for each model
        all_pred_energies_dict[model_name] = model_pred_energies
        all_pred_forces_dict[model_name] = model_pred_forces

    return all_test_energies, all_test_forces, all_pred_energies_dict, all_pred_forces_dict


def reaction_barrier_comparison(model_names: List[str],
                                test_trj_fpath: str,
                                run_names: List[str] = None,
                                save_name: str = None,
                                run_type: str = None,
                                from_scratch_name: str = None,
                                num_base_models: int = 1,
                                num_base_model_comps: int = 1,
                                alpha_list: List[float] = None,
                                linestyle_list: List[str] = None,
                                include_forces: bool = False,
                                ref_prod_index: int = -1,
                                custom_xyz_parser: Callable = None,
                                xlabel: str = "Index",
                                energy_units: str = KCAL_PER_MOL,
                                force_units: str = "meV / A",
                                energies_ylim: Tuple[float, float] = None,
                                forces_ylim: Tuple[float, float] = None,
                                grid_lines: bool = False,
                                show_legend: bool = False,
                                cust_legend_colours: List[str] = None,
                                cust_legend_labels: List[str] = None,
                                cust_legend_linestyles: List[str] = None,
                                ref_point_labels: List[str] = None,
                                cuda_device: str = "cuda:0"):
    """
    Compare a selection of MLIP models along an IRC trajectory, and write the resulting
    energy-error / reaction-profile / force-error plots and per-model CSV to disk.

    For each name in ``run_names`` the models are either taken from ``model_names`` as listed
    (the unnamed ``['']`` run) or discovered in ``MODEL_DIR`` by glob, as selected by
    ``run_type``. Everything is saved under ``PLOT_DIR/irc`` and ``CSV_DIR``, named after the
    trajectory and the run's ``save_name``.

    ``cuda_device`` is expected to be already resolved (see ``resolve_cuda_device``), as the
    calculators fall back to 'cpu' internally anyway.

    :param model_names: list of model names to compare, in plotting order
    :param test_trj_fpath: path to the IRC trajectory (an .xyz / .extxyz file) to compare against
    :param run_names: run names to process; defaults to [''], i.e. a single unnamed run using
                      model_names as listed. A named run is instead discovered by glob and needs
                      run_type to be set.
    :param save_name: output name, used in the saved plot / CSV file names. Overwritten per named
                      run as '{base_model}_{run_name}_fine_tuned' / '{from_scratch_name}_from_scratch_{run_name}'.
    :param run_type: 'fine_tuned' / 'from_scratch', i.e. how a named run's models are discovered.
                     None disables discovery.
    :param from_scratch_name: base model name used in from-scratch model names, for
                              run_type == 'from_scratch'
    :param num_base_models: number of base models, i.e. how many times alpha_list / linestyle_list
                            are repeated
    :param num_base_model_comps: number of comparisons per base model, i.e. how many times each
                                 paper model colour is repeated
    :param alpha_list: alpha values used for plotting, repeated num_base_models times.
                       Defaults to [1.0]
    :param linestyle_list: line style per base-model comparison, repeated num_base_models times.
                           Defaults to ["-"]
    :param include_forces: whether to also compare force norms
    :param ref_prod_index: index of the product reference config (e.g. -1)
    :param custom_xyz_parser: custom parser for reading the trajectory's energy / force labels
    :param xlabel: x-axis label for the reaction-profile plot (e.g. "IRC Image")
    :param energy_units: energy unit for the plots / CSV
    :param force_units: force unit for the force-error plot
    :param energies_ylim: [lo, hi] y-limits for the reaction-profile plot (or None)
    :param forces_ylim: [lo, hi] y-limits for the force-error plot (or None)
    :param grid_lines: whether to draw grid lines
    :param show_legend: whether to show the legend
    :param cust_legend_colours: legend colours for the reference (DFT) line
    :param cust_legend_labels: legend labels for the reference (DFT) line
    :param cust_legend_linestyles: legend line styles for the reference (DFT) line
    :param ref_point_labels: labels for the RS / TS / PS reference points (or None)
    :param cuda_device: device for the MLIP calculators
    :return: None. The plots are written to PLOT_DIR/irc and the CSV to CSV_DIR.
    """

    # defaults duplicated from the input-YAML resolver, so the helper also works standalone
    run_names = run_names if run_names else ['']
    alpha_list = alpha_list if alpha_list is not None else [1.0]
    linestyle_list = linestyle_list if linestyle_list is not None else ["-"]

    for run_name in run_names:

        if run_name != '':
            if run_type == 'fine_tuned':
                base_model_name = model_names[0]
                save_name = f"{base_model_name}_{run_name}_fine_tuned"
                model_name_stub = f"{model_names[0]}_{run_name}_*_config_fine_tuned*"
                model_fpaths = sorted(glob.glob(os.path.join(MODEL_DIR, model_name_stub)), key=natural_sort_key)
                assert len(model_fpaths) > 0, f"No files detected in directory {MODEL_DIR} with name stub {model_name_stub}..."
                model_names = [base_model_name] + [get_name_from_path(model_fpath) for model_fpath in model_fpaths]

            elif run_type == 'from_scratch':
                save_name = f"{from_scratch_name}_from_scratch_{run_name}"
                model_name_stub = f"{from_scratch_name}_from_scratch_{run_name}_*_config*"
                model_fpaths = sorted(glob.glob(os.path.join(MODEL_DIR, model_name_stub)), key=natural_sort_key)
                assert len(model_fpaths) > 0, f"No files detected in directory {MODEL_DIR} with name stub {model_name_stub}..."
                model_names = [get_name_from_path(model_fpath) for model_fpath in model_fpaths]

        print("Testing models: ", model_names)

        # ===== derived plot lists =====
        custom_linestyle_list = linestyle_list * num_base_models
        custom_alpha_list = alpha_list * num_base_models
        custom_zorder_list = [len(model_names) + 2 - i for i in range(len(model_names))]
        colours = [x for colour in PAPER_MODEL_COLOURS.values() for x in [colour] * num_base_model_comps]

        # run comparison
        # get test set ground truth  and forces
        test_set_name = get_name_from_path(test_trj_fpath)

        all_test_energies, all_test_forces, all_pred_energies_dict, all_pred_forces_dict = get_test_and_pred_data(model_names,
                                                                                                                [test_trj_fpath],
                                                                                                                include_forces=include_forces,
                                                                                                                apply_energy_offsets=True,  # True
                                                                                                                custom_xyz_parsers=[custom_xyz_parser],
                                                                                                                energy_units=energy_units,
                                                                                                                force_units=force_units,
                                                                                                                cuda_device=cuda_device)

        combined_datasets = all_test_energies
        combined_datasets.extend([pred_energies
                                for model_pred_energies in all_pred_energies_dict.values()
                                for pred_energies in model_pred_energies])
        formatted_model_names = [get_formatted_model_name(model_name) for model_name in model_names]
        dataset_names = ["Reference"] + formatted_model_names
        logger.info(f"Dataset Names: {dataset_names}")

        energy_errors = [np.abs(combined_datasets[0] - pred_energies) for pred_energies in combined_datasets[1:]]

        for i, model_name in enumerate(formatted_model_names):
            energy_rmse = calculate_rmse_on_errors(energy_errors[i])
            logger.info(f'{model_name} Absolute Energy RMSE: {energy_rmse}')

        # plot index plot of absolute energy errors
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
            logger.info(f"Saved force error index plot to {save_fp}...")

        csv_tuples = []
        for i, model_name in enumerate(model_names):
            pred_energies = all_pred_energies_dict[model_name][0]
            reactant_energy = pred_energies[0]
            peak_energy = np.max(pred_energies)
            peak_pos = np.argmax(pred_energies)

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
