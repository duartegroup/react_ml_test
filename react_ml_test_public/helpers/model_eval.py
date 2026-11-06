"""
Helpers for eval of models.
"""
import csv
import os
from typing import Callable, Literal, Tuple, List
import numpy as np
from react_ml_test_public.helpers.io import flexible_ase_read, get_name_from_path, load_isolated_atom_energies, load_data_cache, load_model_calculator, save_npz_data_cache
from react_ml_test_public.helpers.plot import index_plot, react_profile_plot
from react_ml_test_public.helpers.settings import CSV_DIR, EV, EV_PER_AMSTRONG, KCAL_PER_MOL, MEV, MEV_PER_AMSTRONG, MEV_PER_ATOM, NPZ_DIR, PLOT_DIR, PlotSettings, get_model_colour, resolve_model_dir
from react_ml_test_public.helpers.stats import calculate_rmse_on_errors
from react_ml_test_public.helpers.utils import convert_units, get_ase_props
from react_ml_test_public.helpers.utils import calc_energy_shifts, guess_per_element_mean_energies
from react_ml_test_public.helpers.utils import get_formatted_model_name
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
                          enable_cueq: bool = False,
                          model_dir: str = None,
                          fair_chem_task: str = None,
                          fair_chem_setting: str = None) -> list[np.ndarray]:
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
        fair_chem_task (str): FairChem only - the task a fairchem model is run as, e.g. 'omol'
                              for molecules. Optional, ignored by the other model loaders.
        fair_chem_setting (str): FairChem only - the fairchem inference settings, e.g. 'turbo'.
                                 Optional, ignored by the other model loaders.

    Returns:
        list[np.ndarray]: a list of predictions by the model for each specified test_set file
    """
    
    input_energy_units = EV
    input_force_units = EV_PER_AMSTRONG

    calculator = load_model_calculator(model_name,
                                       device=cuda_device,
                                       model_loader=model_loader,
                                       enable_cueq=enable_cueq,
                                       model_dir=model_dir,
                                       fair_chem_task=fair_chem_task,
                                       fair_chem_setting=fair_chem_setting)

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
                           config_type_filter: str = None,
                           model_dir: str = None,
                           fair_chem_task: str = None,
                           fair_chem_setting: str = None) -> Tuple[list, list, dict, dict]:
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
        fair_chem_task (str): FairChem only - the task a fairchem model is run as, e.g. 'omol'
                              for molecules. Optional, ignored by the other model loaders.
        fair_chem_setting (str): FairChem only - the fairchem inference settings, e.g. 'turbo'.
                                 Optional, ignored by the other model loaders.

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
                                                                       model_dir=model_dir,
                                                                       cuda_device=cuda_device,
                                                                       pbc_box=pbc_box,
                                                                       energy_units=energy_units,
                                                                       force_units=force_units,
                                                                       model_loader=model_loader,
                                                                       enable_cueq=enable_cueq,
                                                                       fair_chem_task=fair_chem_task,
                                                                       fair_chem_setting=fair_chem_setting)

        # add predictions to dictionary for each model
        all_pred_energies_dict[model_name] = model_pred_energies
        all_pred_forces_dict[model_name] = model_pred_forces

    return all_test_energies, all_test_forces, all_pred_energies_dict, all_pred_forces_dict


def _infer_base_model_grouping(model_names: List[str]) -> Tuple[int, int]:
    """Infer (num_base_models, num_base_model_comps) from a list of model names.

    A name counts as a base model when no other name in the list is a prefix of it (e.g.
    'GO-MACE-23'); every other name is a comparison of whichever base name is the longest
    matching prefix of it (e.g. a fine-tuned 'GO-MACE-23_<train_name>_<n>_config_fine_tuned'
    name). ``num_base_model_comps`` requires every base model to have the same number of
    comparisons, since alpha_list / linestyle_list apply the same way to each.

    :param model_names: model names to group, e.g. as listed in the input YAML
    :return: (number of unique base models, number of comparisons per base model)
    """
    base_model_names = [name for name in model_names
                        if not any(other != name and name.startswith(other) for other in model_names)]

    comp_counts = {
        base: sum(1 for name in model_names if name == base or name.startswith(base))
        for base in base_model_names
    }
    assert len(set(comp_counts.values())) == 1, (
        f"model_names groups by base model have differing numbers of comparisons: {comp_counts} "
        "- alpha_list / linestyle_list need the same number of comparisons for every base model."
    )

    return len(base_model_names), comp_counts[base_model_names[0]]


def reaction_barrier_comparison(model_names: List[str],
                                test_trj_fpath: str,
                                global_plot_settings: PlotSettings,
                                plot: bool = True,
                                save_name: str = None,
                                alpha_list: List[float] = None,
                                linestyle_list: List[str] = None,
                                include_forces: bool = False,
                                ref_prod_index: int = -1,
                                custom_xyz_parser: Callable = None,
                                energy_units: str = KCAL_PER_MOL,
                                force_units: str = "meV / A",
                                ref_point_labels: List[str] = None,
                                cuda_device: str = "cuda:0",
                                plot_dir: str = PLOT_DIR,
                                csv_dir: str = CSV_DIR,
                                model_dir: str = None,
                                fair_chem_task: str = None,
                                fair_chem_setting: str = None):
    """
    Compare a selection of MLIP models along an IRC trajectory, write the resulting
    energy-error / reaction-profile / force-error plots and per-model CSV to disk, and return
    each model's key error values (see ``:return:`` below) for a caller to aggregate further
    (e.g. ``run_multi_react_barrier``, across several trajectories).

    Everything is saved under ``plot_dir/irc`` and ``csv_dir``, named after the trajectory and
    ``save_name``; both ``plot_dir`` / ``csv_dir`` default to ``PLOT_DIR`` / ``CSV_DIR``.

    ``cuda_device`` is expected to be already resolved (see ``resolve_cuda_device``), as the
    calculators fall back to 'cpu' internally anyway.

    :param model_names: list of model names to compare, in plotting order.
    :param test_trj_fpath: path to the IRC trajectory (an .xyz / .extxyz file) to compare against
    :param global_plot_settings: plot appearance settings, used (and specialised per plot) for
                          the energy-error / reaction-profile / force-error plots.
                          ``use_react_profile_format`` applies only to the reaction-profile plot:
                          when set, it is drawn with ``react_profile_plot`` instead of
                          ``index_plot`` - every series as a traditional organic-chemistry
                          reaction profile (a thick horizontal energy-level marker over each
                          point, joined by thin lines) instead of straight interpolated line
                          segments; ``profile_segment_halfwidth`` sets each marker's half-width.
    :param plot: whether to produce the per-reaction energy-error / reaction-profile /
                force-error plots (the CSV is always written, and the returned error values
                are always computed, regardless of this flag).
    :param save_name: output name, used in the saved plot / CSV file names.
    :param alpha_list: alpha values used for plotting, repeated once per base model.
                       Defaults to [1.0]
    :param linestyle_list: line style per base-model comparison, repeated once per base model.
                           Defaults to ["-"]
    :param include_forces: whether to also compare force norms
    :param ref_prod_index: index of the product reference config (e.g. -1)
    :param custom_xyz_parser: custom parser for reading the trajectory's energy / force labels
    :param energy_units: energy unit for the plots / CSV
    :param force_units: force unit for the force-error plot
    :param ref_point_labels: labels for the RS / TS / PS reference points (or None)
    :param cuda_device: device for the MLIP calculators
    :param plot_dir: directory to write the plots to (default PLOT_DIR)
    :param csv_dir: directory to write the CSV to (default CSV_DIR)
    :param model_dir: directory the listed models are loaded from (default MODEL_DIR)
    :param fair_chem_task: FairChem only - the task a fairchem model is run as, e.g. 'omol'
                           for molecules. Optional, ignored by the other model loaders.
    :param fair_chem_setting: FairChem only - the fairchem inference settings, e.g. 'turbo'.
                              Optional, ignored by the other model loaders.
    :return: a dict mapping each name in ``model_names`` to its key error values, for a caller
            (e.g. ``run_multi_react_barrier``) that aggregates these over several trajectories:
              - "energy_errors": per-config absolute energy errors (np.ndarray)
              - "force_errors": per-config mean absolute force-norm errors (np.ndarray), or
                None when ``include_forces`` is False
              - "barrier_energy_error": predicted - reference barrier (peak) energy (float)
              - "ref_barrier_energy": the DFT reference barrier (peak) energy, relative to its
                own reactant - the same value for every model, since it does not depend on any
                model's predictions (float)
            The plots are written to plot_dir/irc and the CSV to csv_dir regardless.
    """

    # defaults duplicated from the input-YAML resolver, so the helper also works standalone
    alpha_list = alpha_list if alpha_list is not None else [1.0]
    linestyle_list = linestyle_list if linestyle_list is not None else ["-"]
    if plot:
        os.makedirs(f"{plot_dir}/irc", exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # unpacked from plot_settings, and specialised per plot further down
    xlabel = global_plot_settings.x_label
    energies_ylim = global_plot_settings.energies_ylim
    forces_ylim = global_plot_settings.forces_ylim
    grid_lines = global_plot_settings.grid_lines
    show_legend = global_plot_settings.show_legend
    cust_legend_colours = global_plot_settings.cust_legend_colours
    cust_legend_labels = global_plot_settings.cust_legend_labels
    cust_legend_linestyles = global_plot_settings.cust_legend_linestyles
    custom_colour_dict = global_plot_settings.custom_colour_dict
    use_react_profile_format = global_plot_settings.use_react_profile_format
    profile_segment_halfwidth = global_plot_settings.profile_segment_halfwidth
    line_width = global_plot_settings.line_width

    model_dir = resolve_model_dir(model_dir)

    print("Testing models: ", model_names)

    # ===== derived plot lists =====
    num_base_models, num_base_model_comps = _infer_base_model_grouping(model_names)
    custom_linestyle_list = linestyle_list * num_base_models
    custom_alpha_list = alpha_list * num_base_models
    custom_zorder_list = [len(model_names) + 2 - i for i in range(len(model_names))]

    # colour each model by name, so fine-tuned / from-scratch runs inherit the colour of
    # the base model whose name is a stub of theirs
    colours = [get_model_colour(x, custom_colour_dict=custom_colour_dict) for x in model_names]

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
                                                                                                            cuda_device=cuda_device,
                                                                                                            model_dir=model_dir,
                                                                                                            fair_chem_task=fair_chem_task,
                                                                                                            fair_chem_setting=fair_chem_setting)

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

    # plot index plot of absolute energy errors (energy_plot_settings is also the basis for
    # the force-error plot settings further down, so it is built regardless of `plot`)
    energy_plot_settings = PlotSettings(x_label='Index', y_label=f"Energy Error ({energy_units})", custom_colour_list=colours,
                                show_legend=show_legend,
                                custom_alpha_list=custom_alpha_list,
                                custom_linestyle_list=custom_linestyle_list,
                                line_width=global_plot_settings.line_width)
    if plot:
        save_fp = f"{plot_dir}/irc/{test_set_name}_{save_name}_energy_errors.png"
        index_plot(energy_errors,                   formatted_model_names,
                   save_fp,
                   plot_as_lines=[i for i in range(len(formatted_model_names))],  # or, only 0
                   plot_settings=energy_plot_settings
                  )
        logger.info(f"Saved energy error index plot to {save_fp}...")

    # set all energies relative to reference config
    for dataset in combined_datasets:
        dataset -= dataset[0]  # all reference reactant state

    # get key reference values
    ref_peak_energy = np.max(combined_datasets[0])
    ref_peak_pos = np.argmax(combined_datasets[0])
    ref_prod_energy = combined_datasets[0][ref_prod_index]

    # produce plot of step vs. energy
    if plot:
        save_fp = f"{plot_dir}/irc/{test_set_name}_{save_name}_reaction_plot.png"
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
                                    line_width=line_width,
                                    profile_segment_halfwidth=profile_segment_halfwidth,
                                    )


        if use_react_profile_format:
            react_profile_plot(combined_datasets,
                dataset_names,
                save_fp,
                plot_settings=step_plot_settings,
            )
        else:
            index_plot(combined_datasets,
                dataset_names,
                save_fp,
                plot_settings=step_plot_settings,
                ref_points=[(0, 0.0), (ref_peak_pos, ref_peak_energy), (len(combined_datasets[0]) + ref_prod_index, ref_prod_energy)],
                ref_point_labels=ref_point_labels,
                ref_label_offsets=[5.0, -10.0, 20.0],
                plot_as_lines=[i for i in range(len(combined_datasets))]
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
        if plot:
            save_fp = f"{plot_dir}/irc/{test_set_name}_{save_name}_force_errors.png"
            force_plot_settings = energy_plot_settings._replace(y_label=f"|DFT - Pred Force Norm| ({force_units})",
                                                custom_linestyle_list=custom_linestyle_list, ylim=forces_ylim)
            index_plot(model_force_norm_errors,
                    formatted_model_names,
                    save_fp,
                    plot_as_lines=[i for i in range(len(model_force_norm_errors))],  # or, only 0
                    plot_settings=force_plot_settings
                    )
            logger.info(f"Saved force error index plot to {save_fp}...")

    csv_tuples = []
    model_error_values = {}
    for i, model_name in enumerate(model_names):
        pred_energies = all_pred_energies_dict[model_name][0]
        reactant_energy = pred_energies[0]
        peak_energy = np.max(pred_energies)
        peak_pos = np.argmax(pred_energies)

        ref_peak_energy_error = peak_energy - ref_peak_energy
        ref_peak_force_error = model_force_norm_errors[i][peak_pos] if include_forces else 'N/A'
        ref_peak_pos_error = peak_pos - ref_peak_pos
        product_energy = pred_energies[ref_prod_index]
        csv_tuples.append((model_name, float(reactant_energy), float(product_energy), float(peak_energy), int(peak_pos),
                           float(ref_prod_energy), float(ref_peak_energy), int(ref_peak_pos), float(ref_peak_energy_error),
                           float(ref_peak_force_error) if include_forces else 'N/A',
                           int(ref_peak_pos_error)))

        model_error_values[model_name] = {
            "energy_errors": energy_errors[i],
            "force_errors": model_force_norm_errors[i] if include_forces else None,
            "barrier_energy_error": float(ref_peak_energy_error),
            "ref_barrier_energy": float(ref_peak_energy),
        }

    # save csv
    csv_save_fp = f"{csv_dir}/{test_set_name}_{save_name}.csv"
    with open(csv_save_fp, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        header_row = ['Model', f'Reactant Energy {energy_units}', f'Product Energy {energy_units}', f'Peak Energy {energy_units}', 'Peak Position (Step)',
                        f'Ref Prod Energy {energy_units}', f'Ref Peak Energy {energy_units}', 'Ref Peak Position',
                        f'Ref Peak Energy Error {energy_units}', f'Ref Peak Force Error {force_units}', 'Ref Peak Position Error (Step)']
        writer.writerow(header_row)
        print(header_row)
        writer.writerows(csv_tuples)
        for csv_tuple in csv_tuples:
            print(csv_tuple)

    logger.info(f"Saved csv to: {csv_save_fp}...")

    return model_error_values
