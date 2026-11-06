import argparse
import os
from typing import Callable, List, Tuple

import numpy as np
import torch
from ase.io import read as ase_read

from react_ml_test_public.helpers import inputs as rmlt_inputs
from react_ml_test_public.helpers.io import get_name_from_path, resolve_cuda_device
from react_ml_test_public.helpers.logging import logger
from react_ml_test_public.helpers.model_eval import (
    _infer_base_model_grouping,
    reaction_barrier_comparison,
)
from react_ml_test_public.helpers.plot import index_plot, violin_plot
from react_ml_test_public.helpers.settings import (
    CSV_DIR,
    KCAL_PER_MOL,
    PLOT_DIR,
    PlotSettings,
    get_model_colour,
)
from react_ml_test_public.helpers.stats import calculate_rmse_on_errors
from react_ml_test_public.helpers.utils import get_formatted_model_name


def _group_profile_indices_by_atom_count(atom_counts: np.ndarray) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Sorted unique atom counts, and for each one the indices of the profiles that have it - so
    several profiles sharing a system size are combined into a single point in the atom-count
    scaling plots, rather than one point per profile.
    """
    unique_atom_counts = np.array(sorted(set(atom_counts.tolist())))
    groups = [[i for i in range(len(atom_counts)) if atom_counts[i] == atom_count] for atom_count in unique_atom_counts]
    return unique_atom_counts, groups


def _split_errors_by_config_type(combined_errors: dict,
                                 config_types: np.ndarray,
                                 model_names: List[str]) -> Tuple[List[str], List[List[np.ndarray]]]:
    """
    Splits each model's combined (concatenated-over-profiles) per-config error array into one
    group per unique config_type (in order of first appearance in config_types), as the
    dataset_names / model_datasets violin_plot expects - i.e. one x-axis column per config_type.
    """
    unique_config_types = list(dict.fromkeys(config_types.tolist()))
    model_datasets = [[combined_errors[model_name][config_types == config_type] for model_name in model_names]
                      for config_type in unique_config_types]
    return unique_config_types, model_datasets


def _aggregate_by_profile_groups(values_per_profile: List[np.ndarray], groups: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean and std-dev per atom-count group, each computed over every individual value (e.g. every
    per-config error) from every profile in that group combined into one array - not the
    per-profile means averaged together.
    """
    means, stds = [], []
    for group in groups:
        combined = np.concatenate([np.atleast_1d(values_per_profile[i]) for i in group])
        means.append(np.mean(combined))
        stds.append(np.std(combined))
    return np.array(means), np.array(stds)


def run_multi_react_barrier(test_trj_fpaths: List[str],
                            model_names: List[str],
                            plot_settings: PlotSettings,
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
                            fair_chem_setting: str = None,
                            split_violin_by_config_type: bool = False,
                            plot_each_barrier: bool = False):
    """
    Run a reaction barrier comparison for all test_trj_fpath in test_trj_fpaths, report the absolute energy and force RMSEs over all profiles for each model,
    and report the RMSE in predicted barrier height for each model. These RMSE values are output to the terminal as a table, immediately followed by a
    second table summarising the DFT reference's own (relative) barrier height for each profile, and also three plots are produces
    1) a overall violin plot of absolute energy errors for each model, 2) an overall violin plot of absolute force errors for each model,
    and 3) the same but for the reaction barrier energy errors.

    When ``split_violin_by_config_type`` is set, the energy- and force-error violin plots (1 and
    2 above) are split into one column per ``config_type`` found in each trajectory's .extxyz info
    line (e.g. "TS" / "min"), in place of the single combined "All Profiles" column - every
    trajectory must then label every config with a ``config_type``. The barrier-error violin plot
    (3) is left as a single column regardless, since the barrier height is one value per profile,
    not per-config.

    A further three plots are produced of model errors against system size (number of atoms): profiles sharing an
    atom count are combined into a single point per model, 1) the mean absolute energy error (with the std-dev
    over every underlying config as an error bar), 2) the same for absolute force errors, and 3) the same for the
    barrier energy error.

    ``plot_each_barrier`` is passed straight through to ``reaction_barrier_comparison`` as its
    ``plot`` argument: when False (the default) each trajectory's own energy-error /
    reaction-profile / force-error plots are skipped (we draw our own aggregate violin plots
    below instead); when True those per-trajectory plots are produced too.
    """
    # one reaction_barrier_comparison call per trajectory - plot_each_barrier controls whether
    # its per-trajectory energy-error / reaction-profile / force-error plots are produced (we
    # draw our own aggregate violin plots below regardless), but it always writes that
    # trajectory's CSV and hands back each model's per-config errors and barrier-height error,
    # which we collect here and aggregate "over all profiles" below
    per_model_energy_errors = {model_name: [] for model_name in model_names}
    per_model_force_errors = {model_name: [] for model_name in model_names}
    per_model_barrier_errors = {model_name: [] for model_name in model_names}
    ref_barrier_energies = []
    per_profile_config_types = []  # only populated when split_violin_by_config_type is set

    for test_trj_fpath in test_trj_fpaths:
        model_error_values = reaction_barrier_comparison(model_names,
                                                        test_trj_fpath,
                                                        plot_settings,
                                                        plot=plot_each_barrier,
                                                        save_name=save_name,
                                                        alpha_list=alpha_list,
                                                        linestyle_list=linestyle_list,
                                                        include_forces=include_forces,
                                                        ref_prod_index=ref_prod_index,
                                                        custom_xyz_parser=custom_xyz_parser,
                                                        energy_units=energy_units,
                                                        force_units=force_units,
                                                        ref_point_labels=ref_point_labels,
                                                        cuda_device=cuda_device,
                                                        plot_dir=plot_dir,
                                                        csv_dir=csv_dir,
                                                        model_dir=model_dir,
                                                        fair_chem_task=fair_chem_task,
                                                        fair_chem_setting=fair_chem_setting)

        for model_name in model_names:
            per_model_energy_errors[model_name].append(model_error_values[model_name]["energy_errors"])
            if include_forces:
                per_model_force_errors[model_name].append(model_error_values[model_name]["force_errors"])
            per_model_barrier_errors[model_name].append(model_error_values[model_name]["barrier_energy_error"])
        # same for every model within a profile - it is the DFT reference, not a prediction
        ref_barrier_energies.append(model_error_values[model_names[0]]["ref_barrier_energy"])

        if split_violin_by_config_type:
            # config_type as labelled in the trajectory's .extxyz info line (e.g. "TS" / "min"),
            # one per config, in the same order as that profile's energy_errors / force_errors
            config_types = [config.info.get("config_type") for config in ase_read(test_trj_fpath, index=":")]
            assert all(config_type is not None for config_type in config_types), (
                f"split_violin_by_config_type is set but {test_trj_fpath} has configs with no "
                "config_type labelled in their .extxyz info line.")
            per_profile_config_types.append(config_types)

    # combine each model's per-config errors across every trajectory into one array, so the
    # RMSEs / violins below are "over all profiles" rather than per trajectory
    combined_energy_errors = {model_name: np.concatenate(errors)
                              for model_name, errors in per_model_energy_errors.items()}
    combined_force_errors = ({model_name: np.concatenate(errors) for model_name, errors in per_model_force_errors.items()}
                            if include_forces else None)
    combined_barrier_errors = {model_name: np.array(errors) for model_name, errors in per_model_barrier_errors.items()}

    # flat per-config config_type labels, in the same order as the energy_errors / force_errors
    # concatenation above - used to split the energy- / force-error violin plots below
    combined_config_types = (np.concatenate(per_profile_config_types) if split_violin_by_config_type else None)

    energy_rmses = {model_name: calculate_rmse_on_errors(errors) for model_name, errors in combined_energy_errors.items()}
    force_rmses = ({model_name: calculate_rmse_on_errors(errors) for model_name, errors in combined_force_errors.items()}
                  if include_forces else None)
    barrier_rmses = {model_name: calculate_rmse_on_errors(errors) for model_name, errors in combined_barrier_errors.items()}

    # ===== report the RMSEs over all profiles as a table =====
    formatted_model_names = [get_formatted_model_name(model_name) for model_name in model_names]
    header = ["Model", f"Energy RMSE ({energy_units})"]
    if include_forces:
        header.append(f"Force RMSE ({force_units})")
    header.append(f"Barrier Energy RMSE ({energy_units})")

    rows = []
    for model_name, formatted_name in zip(model_names, formatted_model_names):
        row = [formatted_name, f"{energy_rmses[model_name]:.4f}"]
        if include_forces:
            row.append(f"{force_rmses[model_name]:.4f}")
        row.append(f"{barrier_rmses[model_name]:.4f}")
        rows.append(row)

    col_widths = [max(len(header[col]), *(len(row[col]) for row in rows)) for col in range(len(header))]
    logger.info(f"Multi-trajectory reaction-barrier comparison over {len(test_trj_fpaths)} profiles:")
    print(" | ".join(f"{header[col]:<{col_widths[col]}}" for col in range(len(header))))
    print("-+-".join("-" * col_widths[col] for col in range(len(header))))
    for row in rows:
        print(" | ".join(f"{row[col]:<{col_widths[col]}}" for col in range(len(header))))

    # ===== report the DFT reference's relative barrier height for each profile as a table =====
    # (the reference barrier height is relative to that profile's own reactant, and does not
    # depend on any model - so this is purely a summary of how it differs across profiles)
    profile_names = [get_name_from_path(test_trj_fpath) for test_trj_fpath in test_trj_fpaths]
    ref_header = ["Profile", f"Reference Barrier Height ({energy_units})"]
    ref_rows = [[profile_name, f"{ref_barrier_energy:.4f}"]
               for profile_name, ref_barrier_energy in zip(profile_names, ref_barrier_energies)]

    ref_col_widths = [max(len(ref_header[col]), *(len(row[col]) for row in ref_rows)) for col in range(len(ref_header))]
    logger.info(f"DFT reference barrier heights across {len(test_trj_fpaths)} profiles:")
    print(" | ".join(f"{ref_header[col]:<{ref_col_widths[col]}}" for col in range(len(ref_header))))
    print("-+-".join("-" * ref_col_widths[col] for col in range(len(ref_header))))
    for row in ref_rows:
        print(" | ".join(f"{row[col]:<{ref_col_widths[col]}}" for col in range(len(ref_header))))

    # ===== violin plots of the combined per-profile / per-config errors =====
    colours = [get_model_colour(model_name, custom_colour_dict=plot_settings.custom_colour_dict) for model_name in model_names]
    num_base_models, num_base_model_comps = _infer_base_model_grouping(model_names)
    # alpha_list is repeated once per base model, exactly as reaction_barrier_comparison does,
    # so each model's violin gets the same alpha it was plotted with in the per-trajectory plots
    custom_alpha_list = (alpha_list if alpha_list is not None else [1.0]) * num_base_models
    violin_settings = plot_settings._replace(custom_colour_list=colours,
                                            custom_alpha_list=custom_alpha_list,
                                            num_base_models=num_base_models,
                                            num_base_model_comps=num_base_model_comps,
                                            xtick_labels=False)

    plot_dir_irc = f"{plot_dir}/irc"
    os.makedirs(plot_dir_irc, exist_ok=True)
    save_name_prefix = f"{save_name}_" if save_name else ""

    if split_violin_by_config_type:
        energy_dataset_names, energy_model_datasets = _split_errors_by_config_type(combined_energy_errors, combined_config_types, model_names)
    else:
        energy_dataset_names, energy_model_datasets = ["All Profiles"], [[combined_energy_errors[model_name] for model_name in model_names]]

    violin_plot(energy_model_datasets,
                formatted_model_names,
                energy_dataset_names,
                save_fp=f"{plot_dir_irc}/{save_name_prefix}multi_energy_errors_violin.png",
                plot_settings=violin_settings._replace(y_label=f"Energy Error ({energy_units})", xtick_labels=split_violin_by_config_type))

    if include_forces:
        if split_violin_by_config_type:
            force_dataset_names, force_model_datasets = _split_errors_by_config_type(combined_force_errors, combined_config_types, model_names)
        else:
            force_dataset_names, force_model_datasets = ["All Profiles"], [[combined_force_errors[model_name] for model_name in model_names]]

        violin_plot(force_model_datasets,
                    formatted_model_names,
                    force_dataset_names,
                    save_fp=f"{plot_dir_irc}/{save_name_prefix}multi_force_errors_violin.png",
                    plot_settings=violin_settings._replace(y_label=f"Force Error ({force_units})", xtick_labels=split_violin_by_config_type))
    else:
        logger.info("include_forces is False - skipping the force-error violin plot.")

    violin_plot([[combined_barrier_errors[model_name] for model_name in model_names]],
                formatted_model_names,
                ["All Profiles"],
                save_fp=f"{plot_dir_irc}/{save_name_prefix}multi_barrier_energy_errors_violin.png",
                plot_settings=violin_settings._replace(y_label=f"Barrier Energy Error ({energy_units})"))

    # ===== mean/variance vs. system size (number of atoms), combining every profile of a given
    # atom count into a single point =====
    # reading just the first config of each trajectory is enough, since atom count does not
    # change along an IRC
    num_atoms_per_profile = np.array([len(ase_read(test_trj_fpath, index=0)) for test_trj_fpath in test_trj_fpaths])
    unique_atom_counts, atom_count_groups = _group_profile_indices_by_atom_count(num_atoms_per_profile)

    scaling_settings = plot_settings._replace(custom_colour_list=colours,
                                             custom_alpha_list=custom_alpha_list,
                                             x_label="Number of Atoms",
                                             show_legend=False)

    # absolute energies
    energy_error_means, energy_error_stds = [], []
    for model_name in model_names:
        means, stds = _aggregate_by_profile_groups(per_model_energy_errors[model_name], atom_count_groups)
        energy_error_means.append(means)
        energy_error_stds.append(stds)
    save_fp = f"{plot_dir_irc}/{save_name_prefix}multi_energy_errors_vs_natoms.png"
    index_plot(energy_error_means,
              formatted_model_names,
              save_fp=save_fp,
              plot_settings=scaling_settings._replace(y_label=f"Energy Error ({energy_units})"),
              x_values=unique_atom_counts,
              y_errors=energy_error_stds)
    logger.info(f"Saved index plot to {save_fp}")

    if include_forces:
        force_error_means, force_error_stds = [], []
        for model_name in model_names:
            means, stds = _aggregate_by_profile_groups(per_model_force_errors[model_name], atom_count_groups)
            force_error_means.append(means)
            force_error_stds.append(stds)
        save_fp = f"{plot_dir_irc}/{save_name_prefix}multi_force_errors_vs_natoms.png"
        index_plot(force_error_means,
                  formatted_model_names,
                  save_fp=save_fp,
                  plot_settings=scaling_settings._replace(y_label=f"Force Error ({force_units})"),
                  x_values=unique_atom_counts,
                  y_errors=force_error_stds)
        logger.info(f"Saved index plot to {save_fp}")
    else:
        logger.info("include_forces is False - skipping the force-error vs. atom-count plot.")

    # plot barrier error vs num_atoms - absolute, unlike the signed barrier_energy_error used
    # for the RMSE table / violin plot above (squaring there already erases the sign anyway)
    barrier_error_means, barrier_error_stds = [], []
    for model_name in model_names:
        abs_barrier_errors = [np.abs(error) for error in per_model_barrier_errors[model_name]]
        means, stds = _aggregate_by_profile_groups(abs_barrier_errors, atom_count_groups)
        barrier_error_means.append(means)
        barrier_error_stds.append(stds)
    save_fp = f"{plot_dir_irc}/{save_name_prefix}multi_barrier_errors_vs_natoms.png"
    index_plot(barrier_error_means,
              formatted_model_names,
              save_fp=save_fp,
              plot_settings=scaling_settings._replace(y_label=f"Barrier Energy Error ({energy_units})"),
              x_values=unique_atom_counts,
              y_errors=barrier_error_stds)
    logger.info(f"Saved index plot to {save_fp}")

    return {
        "energy_rmses": energy_rmses,
        "force_rmses": force_rmses,
        "barrier_rmses": barrier_rmses,
    }


if __name__ == "__main__":

    """Run an MLIP reaction-barrier comparison along several IRC trajectories at once, and
    aggregate each model's errors over all of them.

    The path to the input YAML file is given as a single command-line argument:

        rmlt run-multi-react-barrier inputs/irc/<test_name>.yaml

    (or equivalently: python -m react_ml_test_public.scripts.eval.run_multi_react_barrier
    inputs/irc/<test_name>.yaml)

    Input YAML format (see inputs/irc/multi_react_barrier_eg.yaml for a full example; this is
    the same format run_reaction_barrier_comparison.py uses, with test_trj_fpaths in place of
    test_trj_fpath):

    Fields (path fields may reference environment variables, e.g. model_dir: "$DATA/models"):
        data_dir:               # data root directory for relative test_trj_fpaths paths
                                # (default DATA_DIR from settings.py).
        out_dir:                # output root directory, i.e. plots go to <out_dir>/plt and
                                # CSVs to <out_dir>/csv (default OUT_DIR from settings.py).
        model_dir:              # directory the listed models are resolved in, when they live
                                # outside the default MODEL_DIR (default MODEL_DIR).
        model_names:            # list of model names to compare (supports ${var} templating),
                                # evaluated against every trajectory in test_trj_fpaths.
                                # The number of base models and the number of comparisons per
                                # base model are inferred from this list: a name counts as a
                                # base model when no other name in the list is a prefix of it,
                                # and every other name is a comparison of whichever base name is
                                # the longest matching prefix of it (e.g. a fine-tuned name).
        save_name:              # output name, used as a prefix for the aggregate violin plots
                                # (each per-trajectory CSV is also named after its own
                                # trajectory, as in run_reaction_barrier_comparison.py; the
                                # per-trajectory plots themselves are skipped unless
                                # plot_each_barrier is set - see plot_each_barrier below).
        alpha_list:             # alpha values used for plotting.
        linestyle_list:         # line style per base-model comparison.

        test_trj_fpaths:        # list of IRC trajectory .xyz paths to compare model_names
                                # against, taken to be relative to data_dir when they do not
                                # resolve from the cwd.
        custom_xyz_parser:      # "orca_props_parser" / "autode_props_parser" /
                                # "hartree_props_parser" (REF_energy=<float> in Hartree) / empty.
        include_forces:         # whether to also compare force norms.
        ref_prod_index:         # index of the product reference config (e.g. -1).
        energy_units:           # energy unit for plots / CSV, e.g. "kcal / mol".
        force_units:            # force unit for force-error plots, e.g. "meV / A".
        show_legend:            # whether to show the legend. Only applies to the per-trajectory
                                # plots, which are skipped unless plot_each_barrier is set.
        cust_legend_colours:    # legend colours for the reference (DFT) line. Only applies to
                                # the per-trajectory plots (see show_legend above).
        cust_legend_labels:     # legend labels for the reference (DFT) line. Only applies to
                                # the per-trajectory plots (see show_legend above).
        cust_legend_linestyles: # legend line styles for the reference (DFT) line. Only applies
                                # to the per-trajectory plots (see show_legend above).
        ref_point_labels:       # labels for the RS / TS / PS reference points (or empty).
        fair_chem_task:         # FairChem only: the task a FAIRCHEM_MODELS entry runs as, e.g.
                                # "omol" for molecules. Ignored by the other model loaders.
        fair_chem_setting:      # FairChem only: e.g. "turbo", fairchem's inference settings.
                                # Ignored by the other model loaders.
        cuda_device:            # GPU device for the MLIP calculators (default "cuda:0",
                                # falls back to "cpu" if that device is not found).
        custom_colour_dict:     # optional model_name : colour mapping to colour the plots with,
                                # in place of settings.py's PAPER_MODEL_COLOURS (see
                                # get_model_colour).
        colour_dict_source:     # optional alternative to custom_colour_dict: a name from
                                # settings.EXTERNAL_COLOUR_DICT_SOURCES whose repo's PAPER_MODEL_COLOURS
                                # is loaded instead (mutually exclusive with custom_colour_dict).
        use_react_profile_format: # whether to draw the reaction-profile plot in the style of a
                                # traditional organic-chemistry reaction profile (default False).
                                # Only applies to the per-trajectory plots, which are skipped
                                # unless plot_each_barrier is set.
        profile_segment_halfwidth: # half-width (in index units) of each horizontal energy-level
                                # marker, when use_react_profile_format is set (default 0.3).
                                # Only applies when plot_each_barrier is set, for the same reason
                                # as use_react_profile_format.
        split_violin_by_config_type: # whether to split the energy- / force-error violin plots
                                # into one column per config_type found in each trajectory's
                                # .extxyz info line (e.g. "TS" / "min"), in place of the single
                                # combined "All Profiles" column (default False). Every trajectory
                                # must then label every config with a config_type.
        plot_each_barrier:      # whether to also produce each trajectory's own energy-error /
                                # reaction-profile / force-error plots, i.e. passed straight
                                # through to reaction_barrier_comparison as its plot argument
                                # (default False - only the aggregate violin / scaling plots
                                # below are produced).

    Any omitted field is set to None.
    """


    # input YAML config path from a single command-line argument
    parser = argparse.ArgumentParser(
        description="Run an MLIP reaction-barrier comparison over several IRC trajectories, "
                    "driven by an input YAML config file (format described above).",
        epilog="Example: rmlt run-multi-react-barrier inputs/irc/multi_react_barrier_eg.yaml",
    )
    parser.add_argument(
        "input_yaml",
        help="Path to the input YAML config file "
             "(e.g. inputs/irc/multi_react_barrier_eg.yaml).",
    )
    args = parser.parse_args()

    # load + process the input YAML (see react_ml_test_public/helpers/inputs.py); this is the
    # same resolver run_reaction_barrier_comparison.py uses, and also returns test_trj_fpaths
    resolved = rmlt_inputs.resolve_reaction_barrier_comparison_yaml(args.input_yaml, parser)

    # ===== model / fine-tuning selection =====
    model_dir = resolved["model_dir"]
    model_names = resolved["model_names"]
    save_name = resolved["save_name"]
    alpha_list = resolved["alpha_list"]
    linestyle_list = resolved["linestyle_list"]
    fair_chem_task = resolved["fair_chem_task"]
    fair_chem_setting = resolved["fair_chem_setting"]

    # ===== data / output directories =====
    data_dir = resolved["data_dir"]
    out_dir = resolved["out_dir"]
    plot_dir = resolved["plot_dir"]
    csv_dir = resolved["csv_dir"]
    logger.info(f"Using data dir: {data_dir}, out dir: {out_dir} (plots: {plot_dir}, csv: {csv_dir})...")

    # ===== IRC trajectory comparison =====
    test_trj_fpaths = resolved["test_trj_fpaths"]
    custom_xyz_parser = resolved["custom_xyz_parser"]
    include_forces = resolved["include_forces"]
    ref_prod_index = resolved["ref_prod_index"]
    force_units = resolved["force_units"]
    show_legend = resolved["show_legend"]
    ref_point_labels = resolved["ref_point_labels"]
    cust_legend_colours = resolved["cust_legend_colours"]
    cust_legend_labels = resolved["cust_legend_labels"]
    cust_legend_linestyles = resolved["cust_legend_linestyles"]
    energy_units = resolved["energy_units"]
    energies_ylim = resolved["energies_ylim"]
    forces_ylim = resolved["forces_ylim"]
    cuda_device = resolved["cuda_device"]
    custom_colour_dict = resolved["custom_colour_dict"]
    use_react_profile_format = resolved["use_react_profile_format"]
    profile_segment_halfwidth = resolved["profile_segment_halfwidth"]
    split_violin_by_config_type = resolved["split_violin_by_config_type"]
    plot_each_barrier = resolved["plot_each_barrier"]
    del resolved

    # resolve the requested device (i.e. fall back to 'cpu' if the GPU is not available)
    cuda_device = resolve_cuda_device(cuda_device)
    if cuda_device != "cpu":
        torch.set_default_dtype(torch.float32)

    # base plot_settings the aggregate violin plots specialise (see run_multi_react_barrier);
    # the per-trajectory energy-error / reaction-profile / force-error plots these would
    # otherwise apply to are skipped unless plot_each_barrier is set (reaction_barrier_comparison
    # is called with plot=plot_each_barrier)
    plot_settings = PlotSettings(energies_ylim=energies_ylim,
                                forces_ylim=forces_ylim,
                                show_legend=show_legend,
                                cust_legend_colours=cust_legend_colours,
                                cust_legend_labels=cust_legend_labels,
                                cust_legend_linestyles=cust_legend_linestyles,
                                custom_colour_dict=custom_colour_dict,
                                use_react_profile_format=use_react_profile_format,
                                profile_segment_halfwidth=profile_segment_halfwidth)

    run_multi_react_barrier(test_trj_fpaths,
                            model_names,
                            plot_settings,
                            save_name=save_name,
                            alpha_list=alpha_list,
                            linestyle_list=linestyle_list,
                            include_forces=include_forces,
                            ref_prod_index=ref_prod_index,
                            custom_xyz_parser=custom_xyz_parser,
                            energy_units=energy_units,
                            force_units=force_units,
                            ref_point_labels=ref_point_labels,
                            cuda_device=cuda_device,
                            plot_dir=plot_dir,
                            csv_dir=csv_dir,
                            model_dir=model_dir,
                            fair_chem_task=fair_chem_task,
                            fair_chem_setting=fair_chem_setting,
                            split_violin_by_config_type=split_violin_by_config_type,
                            plot_each_barrier=plot_each_barrier)
