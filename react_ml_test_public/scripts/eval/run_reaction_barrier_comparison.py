import argparse

import torch

from react_ml_test_public.helpers import inputs as rmlt_inputs
from react_ml_test_public.helpers.io import resolve_cuda_device
from react_ml_test_public.helpers.logging import logger
from react_ml_test_public.helpers.model_eval import reaction_barrier_comparison
from react_ml_test_public.helpers.settings import PlotSettings

if __name__ == "__main__":

    """Run an MLIP reaction-barrier comparison along an IRC trajectory.

    The path to the input YAML file is given as a single command-line argument:

        rmlt run-reaction-barrier-comparison inputs/irc/<test_name>.yaml

    (or equivalently: python -m react_ml_test_public.scripts.eval.run_reaction_barrier_comparison
    inputs/irc/<test_name>.yaml)

    Input YAML format (see inputs/IRC/guajadial_DA_IRC_barrier.yaml for a full example):

    Fields (path fields may reference environment variables, e.g. model_dir: "$DATA/models"):
        data_dir:               # data root directory for relative test_trj_fpath paths
                                # (default DATA_DIR from settings.py).
        out_dir:                # output root directory, i.e. plots go to <out_dir>/plt and
                                # CSVs to <out_dir>/csv (default OUT_DIR from settings.py).
        model_dir:              # directory the listed models are resolved in, when they live
                                # outside the default MODEL_DIR (default MODEL_DIR).
        n:                      # training set size.
        train_name:             # fine-tuning training set name.
        sample_str:             # sampling-strategy suffix used in fine-tuned names (e.g. "ts_gauss_").
        energy_str:             # energy label string used in fine-tuned names (e.g. "energy_").
        rep_num:                # fine-tuned model repetition number.
        omol_rep_num:           # OMOL model repetition number.
        omol_energy_str:        # OMOL energy suffix/prefix string.
        model_names:            # list of model names to compare (supports ${var} templating).
                                # The number of base models and the number of comparisons per
                                # base model are inferred from this list: a name counts as a
                                # base model when no other name in the list is a prefix of it,
                                # and every other name is a comparison of whichever base name is
                                # the longest matching prefix of it (e.g. a fine-tuned name).
        save_name:              # output name.
        alpha_list:             # alpha values used for plotting.
        linestyle_list:         # line style per base-model comparison.

        test_trj_fpath:         # path to the IRC trajectory .xyz to compare against, taken to be
                                # relative to data_dir when it does not resolve from the cwd.
                                # Optionally templated on another field of this YAML with a
                                # ${var} reference, e.g. "test/.../wB97M-V/${save_name}.xyz"
                                # (filled in from save_name, or any other field set above).
        custom_xyz_parser:      # "orca_props_parser" / "autode_props_parser" /
                                # "hartree_props_parser" (REF_energy=<float> in Hartree) / empty.
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
                                # traditional organic-chemistry reaction profile - a thick
                                # horizontal energy-level marker over each point, joined by thin
                                # lines - instead of straight interpolated line segments
                                # (default False).
        profile_segment_halfwidth: # half-width (in index units) of each horizontal energy-level
                                # marker, when use_react_profile_format is set (default 0.3).

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
    model_dir = resolved["model_dir"]
    model_names = resolved["model_names"]
    save_name = resolved["save_name"]
    alpha_list = resolved["alpha_list"]
    linestyle_list = resolved["linestyle_list"]

    # ===== data / output directories =====
    data_dir = resolved["data_dir"]
    out_dir = resolved["out_dir"]
    plot_dir = resolved["plot_dir"]
    csv_dir = resolved["csv_dir"]
    logger.info(f"Using data dir: {data_dir}, out dir: {out_dir} (plots: {plot_dir}, csv: {csv_dir})...")

    # ===== IRC trajectory comparison =====
    test_trj_fpath = resolved["test_trj_fpath"]
    custom_xyz_parser = resolved["custom_xyz_parser"]
    include_forces = resolved["include_forces"]
    ref_prod_index = resolved["ref_prod_index"]
    force_units = resolved["force_units"]
    show_legend = resolved["show_legend"]
    cust_legend_colours = resolved["cust_legend_colours"]
    cust_legend_labels = resolved["cust_legend_labels"]
    cust_legend_linestyles = resolved["cust_legend_linestyles"]
    ref_point_labels = resolved["ref_point_labels"]
    fair_chem_task = resolved["fair_chem_task"]
    fair_chem_setting = resolved["fair_chem_setting"]
    energy_units = resolved["energy_units"]
    energies_ylim = resolved["energies_ylim"]
    forces_ylim = resolved["forces_ylim"]
    cuda_device = resolved["cuda_device"]
    custom_colour_dict = resolved["custom_colour_dict"]
    use_react_profile_format = resolved["use_react_profile_format"]
    profile_segment_halfwidth = resolved["profile_segment_halfwidth"]
    del resolved

    # resolve the requested device (i.e. fall back to 'cpu' if the GPU is not available)
    cuda_device = resolve_cuda_device(cuda_device)
    if cuda_device != "cpu":
        torch.set_default_dtype(torch.float32)

    # plot appearance settings for the energy-error / reaction-profile / force-error plots
    plot_settings = PlotSettings(x_label='IRC Image',
                                energies_ylim=energies_ylim,
                                forces_ylim=forces_ylim,
                                grid_lines=False,
                                show_legend=show_legend,
                                cust_legend_colours=cust_legend_colours,
                                cust_legend_labels=cust_legend_labels,
                                cust_legend_linestyles=cust_legend_linestyles,
                                custom_colour_dict=custom_colour_dict,
                                use_react_profile_format=use_react_profile_format,
                                profile_segment_halfwidth=profile_segment_halfwidth,
                                line_width=2.0)

    # run the comparison (see helpers/model_eval.py): writes the energy-error /
    # reaction-profile / force-error plots and the per-model CSV
    reaction_barrier_comparison(model_names,
                                test_trj_fpath,
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
