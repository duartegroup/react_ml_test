import argparse

import torch

from react_ml_test_public.helpers import inputs as rmlt_inputs
from react_ml_test_public.helpers.io import resolve_cuda_device
from react_ml_test_public.helpers.model_eval import reaction_barrier_comparison

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
        run_names:              # multiple run names to detect and process (defaults to [''],
                                # i.e. a single unnamed run using model_names as listed).
        run_type:               # "fine_tuned" / "from_scratch": how a named run's models
                                # are discovered by glob (defaults to None = no discovery).
        from_scratch_name:      # base model name used in from-scratch model names
                                # (only needed when run_type is "from_scratch").
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
        cuda_device:            # GPU device for the MLIP calculators (default "cuda:0",
                                # falls back to "cpu" if that device is not found).

    Any omitted field is set to None, except run_names, which defaults to [''].
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
    model_names = resolved["model_names"]
    save_name = resolved["save_name"]
    run_names = resolved["run_names"]
    run_type = resolved["run_type"]
    from_scratch_name = resolved["from_scratch_name"]
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
    cuda_device = resolved["cuda_device"]
    del resolved

    # resolve the requested device (i.e. fall back to 'cpu' if the GPU is not available)
    cuda_device = resolve_cuda_device(cuda_device)
    if cuda_device != "cpu":
        torch.set_default_dtype(torch.float32)

    # run the comparison (see helpers/model_eval.py): writes the energy-error /
    # reaction-profile / force-error plots and the per-model CSV
    reaction_barrier_comparison(model_names,
                                test_trj_fpath,
                                run_names=run_names,
                                save_name=save_name,
                                run_type=run_type,
                                from_scratch_name=from_scratch_name,
                                num_base_models=num_base_models,
                                num_base_model_comps=num_base_model_comps,
                                alpha_list=alpha_list,
                                linestyle_list=linestyle_list,
                                include_forces=include_forces,
                                ref_prod_index=ref_prod_index,
                                custom_xyz_parser=custom_xyz_parser,
                                xlabel=xlabel,
                                energy_units=energy_units,
                                force_units=force_units,
                                energies_ylim=energies_ylim,
                                forces_ylim=forces_ylim,
                                grid_lines=grid_lines,
                                show_legend=show_legend,
                                cust_legend_colours=cust_legend_colours,
                                cust_legend_labels=cust_legend_labels,
                                cust_legend_linestyles=cust_legend_linestyles,
                                ref_point_labels=ref_point_labels,
                                cuda_device=cuda_device)
