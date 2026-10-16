import argparse
import os
from typing import Callable, List

import numpy as np
import torch

from react_ml_test_public.helpers import inputs as rmlt_inputs
from react_ml_test_public.helpers.io import resolve_cuda_device
from react_ml_test_public.helpers.logging import logger
from react_ml_test_public.helpers.model_eval import _infer_base_model_grouping, reaction_barrier_comparison
from react_ml_test_public.helpers.plot import violin_plot
from react_ml_test_public.helpers.settings import CSV_DIR, KCAL_PER_MOL, PLOT_DIR, PlotSettings, get_model_colour
from react_ml_test_public.helpers.stats import calculate_rmse_on_errors
from react_ml_test_public.helpers.utils import get_formatted_model_name


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
                            fair_chem_setting: str = None):
    """
    Run a reaction barrier comparison for all test_trj_fpath in test_trj_fpaths, report the absolute energy and force RMSEs over all profiles for each model,
    and report the RMSE in predicted barrier height for each model. These RMSE values are output to the terminal as a table, and also three plots are produces
    1) a overall violin plot of absolute energy errors for each model, 2) an overall violin plot of absolute force errors for each model,
    and 3) the same but for the reaction barrier energy errors.
    """
    # one reaction_barrier_comparison call per trajectory - plot=False skips its per-trajectory
    # energy-error / reaction-profile / force-error plots (we draw our own aggregate violin
    # plots below instead), but it still writes that trajectory's CSV and hands back each
    # model's per-config errors and barrier-height error, which we collect here and aggregate
    # "over all profiles" below
    per_model_energy_errors = {model_name: [] for model_name in model_names}
    per_model_force_errors = {model_name: [] for model_name in model_names}
    per_model_barrier_errors = {model_name: [] for model_name in model_names}

    for test_trj_fpath in test_trj_fpaths:
        model_error_values = reaction_barrier_comparison(model_names,
                                                        test_trj_fpath,
                                                        plot_settings,
                                                        plot=False,
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

    # combine each model's per-config errors across every trajectory into one array, so the
    # RMSEs / violins below are "over all profiles" rather than per trajectory
    combined_energy_errors = {model_name: np.concatenate(errors)
                              for model_name, errors in per_model_energy_errors.items()}
    combined_force_errors = ({model_name: np.concatenate(errors) for model_name, errors in per_model_force_errors.items()}
                            if include_forces else None)
    combined_barrier_errors = {model_name: np.array(errors) for model_name, errors in per_model_barrier_errors.items()}

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

    # ===== violin plots of the combined per-profile / per-config errors =====
    colours = [get_model_colour(model_name) for model_name in model_names]
    num_base_models, num_base_model_comps = _infer_base_model_grouping(model_names)
    # alpha_list is repeated once per base model, exactly as reaction_barrier_comparison does,
    # so each model's violin gets the same alpha it was plotted with in the per-trajectory plots
    custom_alpha_list = (alpha_list if alpha_list is not None else [1.0]) * num_base_models
    violin_settings = plot_settings._replace(custom_colour_list=colours,
                                            custom_alpha_list=custom_alpha_list,
                                            num_base_models=num_base_models,
                                            num_base_model_comps=num_base_model_comps)

    plot_dir_irc = f"{plot_dir}/irc"
    os.makedirs(plot_dir_irc, exist_ok=True)
    save_name_prefix = f"{save_name}_" if save_name else ""

    violin_plot([[combined_energy_errors[model_name] for model_name in model_names]],
                formatted_model_names,
                ["All Profiles"],
                save_fp=f"{plot_dir_irc}/{save_name_prefix}multi_energy_errors_violin.png",
                plot_settings=violin_settings._replace(y_label=f"Absolute Energy Error ({energy_units})"))

    if include_forces:
        violin_plot([[combined_force_errors[model_name] for model_name in model_names]],
                    formatted_model_names,
                    ["All Profiles"],
                    save_fp=f"{plot_dir_irc}/{save_name_prefix}multi_force_errors_violin.png",
                    plot_settings=violin_settings._replace(y_label=f"Absolute Force Error ({force_units})"))
    else:
        logger.info("include_forces is False - skipping the force-error violin plot.")

    violin_plot([[combined_barrier_errors[model_name] for model_name in model_names]],
                formatted_model_names,
                ["All Profiles"],
                save_fp=f"{plot_dir_irc}/{save_name_prefix}multi_barrier_energy_errors_violin.png",
                plot_settings=violin_settings._replace(y_label=f"Barrier Energy Error ({energy_units})"))

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
                                # per-trajectory plots themselves are skipped - see plot=False
                                # in run_multi_react_barrier).
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
        show_legend:            # whether to show the legend. Currently unused: the
                                # per-trajectory plots it would apply to are skipped (see
                                # plot=False in run_multi_react_barrier); kept for parity with
                                # run_reaction_barrier_comparison.py's YAML format.
        ref_point_labels:       # labels for the RS / TS / PS reference points (or empty).
        fair_chem_task:         # FairChem only: the task a FAIRCHEM_MODELS entry runs as, e.g.
                                # "omol" for molecules. Ignored by the other model loaders.
        fair_chem_setting:      # FairChem only: e.g. "turbo", fairchem's inference settings.
                                # Ignored by the other model loaders.
        cuda_device:            # GPU device for the MLIP calculators (default "cuda:0",
                                # falls back to "cpu" if that device is not found).

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
    energy_units = resolved["energy_units"]
    energies_ylim = resolved["energies_ylim"]
    forces_ylim = resolved["forces_ylim"]
    cuda_device = resolved["cuda_device"]
    del resolved

    # resolve the requested device (i.e. fall back to 'cpu' if the GPU is not available)
    cuda_device = resolve_cuda_device(cuda_device)
    if cuda_device != "cpu":
        torch.set_default_dtype(torch.float32)

    # base plot_settings the aggregate violin plots specialise (see run_multi_react_barrier);
    # the per-trajectory energy-error / reaction-profile / force-error plots these would
    # otherwise apply to are skipped (reaction_barrier_comparison is called with plot=False)
    plot_settings = PlotSettings(energies_ylim=energies_ylim,
                                forces_ylim=forces_ylim,
                                show_legend=show_legend)

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
                            fair_chem_setting=fair_chem_setting)
