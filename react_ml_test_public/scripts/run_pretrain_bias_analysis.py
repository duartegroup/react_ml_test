import os

import colorsys
from itertools import combinations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from scipy.stats import linregress, pearsonr

from react_ml_test_public.helpers import inputs as rmlt_inputs

from react_ml_test_public.helpers.io import (
    flexible_ase_read,
    get_ase_atoms_forces,
    get_name_from_path,
    load_data_cache,
    save_npz_data_cache,
)
from react_ml_test_public.helpers.model_eval import (
    get_model_predictions,
    get_test_and_pred_data,
)
from react_ml_test_public.helpers.plot import bar_plot, correlation_scatter_plot, index_plot
from react_ml_test_public.helpers.sampling import remove_duplicates
from react_ml_test_public.helpers.settings import (
    ANTHRACENE_DA_DATA_DIR,
    CPD_MVK_DA_DATA_DIR,
    DATASETS_DIR,
    DPI,
    EV_PER_AMSTRONG,
    GUAJADIAL_DA_DATA_DIR,
    MEV,
    MEV_PER_AMSTRONG,
    NPZ_DIR,
    PAPER_MODEL_COLOURS,
    PAPER_MODELS,
    PLOT_DIR,
    SIMILARITIES_DIR,
    SINGLE_COLUMN_FIG_SIZE,
    PlotSettings,
    eV_to_mEV,
)
from react_ml_test_public.helpers.stats import (
    calculate_rmse_on_errors,
    max_norm,
    mean_center,
    min_max_norm,
)
from react_ml_test_public.helpers.utils import (
    atoms_hash,
    convert_units,
    get_ase_props,
    get_formatted_model_name,
    unflatten_array_by_lengths,
)


def _format_value_for_table(value):
    """Format values consistently for end-of-script summary tables."""
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, np.floating):
        return f"{float(value):.4f}"
    if isinstance(value, (list, tuple)):
        return str(value)
    return str(value)


def print_summary_table(title, rows, columns):
    """Print a simple aligned table without adding external dependencies."""
    if not rows:
        return

    str_rows = [
        [_format_value_for_table(row.get(column, "")) for column in columns]
        for row in rows
    ]
    widths = [
        max(len(column), *(len(row[i]) for row in str_rows))
        for i, column in enumerate(columns)
    ]

    print(f"\n{title}")
    print("-" * (sum(widths) + 3 * (len(columns) - 1)))
    print(" | ".join(column.ljust(widths[i]) for i, column in enumerate(columns)))
    print("-" * (sum(widths) + 3 * (len(columns) - 1)))
    for row in str_rows:
        print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(
        *(round(c * 255) for c in rgb)
    )

def generate_shades(hex_color, light_delta=0.18, dark_delta=0.15):
    r, g, b = hex_to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    light = colorsys.hls_to_rgb(h, min(1, l + light_delta), s)
    dark = colorsys.hls_to_rgb(h, max(0, l - dark_delta), s)

    return [
        rgb_to_hex(light),
        hex_color.upper(),
        rgb_to_hex(dark),
    ]


def corr_plot(x_series, y_series, alphas, colours, 
              x_label=None, 
              y_label=None,
              labels=None,
              invert_y: bool = False, plot_x_y_line: bool = False,
              save_fp: str = None,
              log_log: bool = False,
              beta_factor: float = None,
              mag_value: float = None, 
              plot_legend: bool = False,
              label_fmtr: str = '.2f'):

    labels = labels or [f"Series {i+1}" for i in range(len(x_series))]
    plt.figure(figsize=SINGLE_COLUMN_FIG_SIZE, dpi=DPI)

    all_x_plot, all_y_plot = [], []

    betas, pred_mags = [], []
    fmt_labels = []
    for x, y, a, c, lab in zip(x_series, y_series, alphas, colours, labels):
        x = np.asarray(x)
        y = np.asarray(y)

        if not np.all(np.isfinite(x)):
            print(f"Invalid values detected in x for {lab}. Returning early.")
            return [None] * len(x_series), [None] * len(x_series)

        if not np.all(np.isfinite(y)):
            print(f"Invalid values detected in y for {lab}. Returning early.")
            return [None] * len(x_series), [None] * len(x_series)

        if np.all(x == x[0]):
            print(f"x is constant for {lab}; cannot fit linear regression. Returning early.")
            return [None] * len(x_series), [None] * len(x_series)

        if log_log and (np.any(x <= 0) or np.any(y <= 0)):
            print(f"Non-positive values detected for {lab}; cannot use log-log plot. Returning early.")
            return [None] * len(x_series), [None] * len(x_series)

        # Compute linear relationship on original, untransformed data
        r = np.corrcoef(x, y)[0, 1]
        beta, intercept = np.polyfit(x, y, 1)

        # get explained error magnitude
        pred_mag_error = beta * mag_value #+ intercept #if mag_value is not None else None
        print(f"Eq: y = {beta}x + {intercept}, with bias predicted error: beta*u = ", pred_mag_error)
        # print("Abs force mag error as pred by x: ", exp_error)
        pred_mags.append(pred_mag_error)

        if invert_y:
            r = -r
            beta = -beta

        beta_adj = beta_factor * beta if beta_factor is not None else beta
        # print("Beta adjusted: ", beta_adj)

        # Transform only for plotting
        x_plot = np.log(x) if log_log else x
        y_plot = np.log(y) if log_log else y

        all_x_plot.append(x_plot)
        all_y_plot.append(y_plot)

        # label = rf"{lab} ($r$={r:.2f}, $\beta_{ {beta_factor} }$={beta_adj:.2g})" if beta_factor is not None else f"{lab} (r={r:.2f}, β={beta:.2f})"
        label = rf"{lab:<15} ($r$={r:.2f}, y={beta:{label_fmtr}}x {'+' if n >= 0 else '-'} {abs(intercept):.0f})"
        plt.scatter(
            x_plot, y_plot,
            s=0.5,
            color=c,
            alpha=a,
            label=label
        )
        fmt_labels.append(label)

        # Plot regression line transformed for display
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = beta * x_fit + intercept

        x_fit_plot = np.log(x_fit) if log_log else x_fit
        y_fit_plot = np.log(y_fit) if log_log else y_fit

        plt.plot(x_fit_plot, y_fit_plot, color=c, alpha=min(a + 0.3, 1), lw=1)
        betas.append(beta)

    all_x_plot = np.concatenate(all_x_plot)
    all_y_plot = np.concatenate(all_y_plot)

    if plot_x_y_line:
        lo = min(all_x_plot.min(), all_y_plot.min())
        hi = max(all_x_plot.max(), all_y_plot.max())
        plt.plot([lo, hi], [lo, hi], "k-", lw=1)

    if invert_y:
        plt.gca().invert_yaxis()

    log_str = 'Log ' if log_log else ''

    if plot_legend:

        legend_handles = [
            Line2D([0], [0], color=c, lw=2) for a, c, in zip(alphas, colours)
        ]

        plt.legend(
            handles=legend_handles,
            labels=fmt_labels,
            frameon=False, 
            loc="upper left",
        )

    # plt.legend(frameon=False, loc="best")
    plt.gca().set_position([0.18, 0.15, 0.65, 0.65])
    if x_label is not None:
        plt.xlabel(f"{log_str}{x_label}", fontsize=10)
    if y_label is not None:
        plt.ylabel(f"{log_str}{y_label}", fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    if save_fp is not None:
        plt.savefig(save_fp, dpi=600, bbox_inches="tight")
        print(f"Saved correlation plot to {save_fp}")
    
    plt.close()

    return betas, pred_mags
    # return betas, pred_mags


def pairwise_x_y1_correlations(model_data_dict, x_index: int = 0, y_index: int = 2):
    """
    model_data_dict:
    {
        model_name: [
            (subset_name, x1...xi, y1...yi),
            (subset_name, x1...xi, y1...yi),
            ...
        ],
        ...
    }

    For each model pair and each subset present in both models:
        x_d = x_a - x_b
        y_d = y1_a - y1_b

    Fits:
        y_d = alpha + beta * x_d

    Prints Pearson r and beta (slope).
    """

    # Convert each model's list of tuples into a subset lookup
    model_subset_data = {
        model_name: {
            subset_info[0]: subset_info[1:]
            for subset_info in subset_list
        }
        for model_name, subset_list in model_data_dict.items()
    }

    for model_a, model_b in combinations(model_subset_data.keys(), 2):

        subsets_a = set(model_subset_data[model_a])
        subsets_b = set(model_subset_data[model_b])
        common_subsets = sorted(subsets_a & subsets_b)

        for subset_name in common_subsets:

            x_a, yi_a = model_subset_data[model_a][subset_name][x_index], model_subset_data[model_a][subset_name][y_index]
            x_b, yi_b = model_subset_data[model_b][subset_name][x_index], model_subset_data[model_b][subset_name][y_index]

            x_d = np.asarray(x_a) - np.asarray(x_b)            
            y_d = np.asarray(yi_a) - np.asarray(yi_b)

            mask = np.isfinite(x_d) & np.isfinite(y_d)
            x_d = x_d[mask]
            y_d = y_d[mask]

            if len(x_d) < 2:
                print(
                    f"{model_a} vs {model_b} | {subset_name}: "
                    "insufficient data"
                )
                continue

            if not np.all(np.isfinite(x_d)):
                print(f"Invalid values detected in x for {model_a} vs {model_b}. Skipping.")
                continue

            if not np.all(np.isfinite(y_d)):
                print(f"Invalid values detected in y for {model_a} vs {model_b}. Skipping.")
                continue

            if np.all(x_d == x_d[0]):
                print(f"x is constant for {model_a} vs {model_b}; cannot fit linear regression. Skipping.")
                continue

            r, _ = pearsonr(x_d, y_d)
            beta = linregress(x_d, y_d).slope

            print(
                f"{model_a} vs {model_b} | {subset_name}: "
                f"r = {r:.4f}, beta = {beta:.4f}"
            )

def category_scatter_plot(all_series: list[np.array], 
                          point_markers: list[str],
                          subset_names: list[str],
                          base_model_names: list[str],
                          save_name: str,
                          y_label: str,
                          log_y: bool = False,
                          ylim = None,
                          show_legend=False,
                          remove_negs=False
                          ):

    # subset_names = subset_names[1:] 
    x = np.array(range(len(subset_names)))

    # scatter plot
    major_offset = 0.1
    minor_offset = 0.05
    plt.figure(figsize=SINGLE_COLUMN_FIG_SIZE, dpi=DPI)
    normalise = False
    line_styles = ['dotted', '-', '-', 'dotted', '-', '-']
    marker_styles = [0, 1, 2, 0, 1, 2]
    n = 3
    num_base_model_comps = 3
    major_offsets = [(k - (n - 1) / 2) * 0.16 for k in range(n)]
    for i, model_series in enumerate(all_series):
        colour = PAPER_MODEL_COLOURS[base_model_names[i]]
        for j, series in enumerate(model_series):

            neg_mask = series > 0
            # if any(x < 0 for x in series):
            #     neg_values = True

            style = marker_styles[j]
            if style == 0:
                # hollow
                face = 'none'
                edge = colour
            elif style == 1:
                # filled, 0.4 alpha
                face = to_rgba(colour, 0.4)
                edge = to_rgba(colour, 0.4)
            elif style == 2:
                # filled, 0.9 alpha
                face = to_rgba(colour, 0.9)
                edge = to_rgba(colour, 0.9)
            
            x_pos = x + major_offsets[i]

            if remove_negs:
                series = series[neg_mask]
                x_pos = x_pos[neg_mask]
            plt.scatter(x_pos, series, marker=point_markers[j], facecolors=face, edgecolors=edge)  # c=plot_colour

    if ylim is not None:
        plt.ylim(ylim)
    plt.yticks(fontsize=10)
    plt.margins(y=0.2)

    if log_y:
        # if neg_values:
        # plt.yscale('symlog', linthresh=0.01)
        # else:
        plt.yscale("log")

    if show_legend:
        plt.legend()

    plt.gca().set_position([0.18, 0.15, 0.75, 0.72])
    plt.gca().set_xticks(x)
    plt.gca().set_xticklabels(subset_names, fontsize=10)
    plt.xlabel('Config Type', fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    # plt.tight_layout()

    # save plot
    save_fp = f'{PLOT_DIR}/descriptors/{save_name}model_pred_analysis.png'
    plt.savefig(save_fp, dpi=600)
    print(f"Saved plot to {save_fp}")
    plt.close()
 


if __name__ == "__main__":

    """Run a pretraining-bias analysis.

    Compares zero-shot / fine-tuned / fully fine-tuned MLIP force-norm errors against
    lower-level-of-theory (LoT) reference force errors and MACE descriptor similarities,
    across config-type / reaction-coordinate / atom-type subsets of a test set.

    The path to the input YAML file is given as a single command-line argument:

        rmlt run-pretrain-bias-analysis inputs/pretrain_bias/CPD_MVK_DA_pretrain_bias.yaml

    (or equivalently: python -m react_ml_test_public.scripts.run_pretrain_bias_analysis
    inputs/pretrain_bias/CPD_MVK_DA_pretrain_bias.yaml)

    Input YAML format (see inputs/pretrain_bias/CPD_MVK_DA_pretrain_bias.yaml for a full
    example). Fields:

    Test system:
        test_name:              # test set name (used in output / cache filenames).
        test_fpath:             # path to the DFT reference test-set .xyz.
        react_coords:           # [[i, j], ...] reactive atom pairs defining the
                                # reaction-coordinate distance.
        ref_ts_fpath:           # path to the reference (vacuum) TS .xyz.
        ts_dist_threshold:      # +/- window (Angstrom) classifying RS / TS / PS.
        input_lot_test_fpaths:  # one lower-level-of-theory .xyz per base model, with
                                # identical structures in the same order as test_fpath.
        n_solute_atoms:         # number of solute atoms (the leading atoms of each config).

    Model / fine-tuning selection:
        model_names:            # list of base (zero-shot) models to analyse.
        model_plt_labels:       # short model labels used in correlation-plot annotations.
        model_ft_n:             # fine-tuning set size (used in plot labels).
        model_full_ft_n:        # full-fine-tuning set size (used in plot labels).
        n:                      # fine-tuning training-set size.
        ft_train_name:          # fine-tuning training-set name.
        sample_str:             # sampling-strategy suffix in fine-tuned names.
        energy_str:             # energy-label string in fine-tuned names (e.g. "energy_").
        rep_num:                # fine-tuned model repetition number.
        full_ft_rep_num:        # full-fine-tuned model repetition number.

    Optional relative baselines (default off; if set, lists need one entry per model):
        from_scratch_names:     # base model names for from-scratch-trained baselines.
        from_scratch_energy_str:# energy-label string of those from-scratch models.
        zero_shot_ref_names:    # use zero-shot base models as the relative baseline.
        ft_ref_names:           # use this fine-tuned reference base model instead.
        rel_comp_str:           # string added to output names, e.g. "vs_scratch_".
        abs_y:                  # take absolute values of relative errors.

    Subset (atom-type spec) selection:
        atom_type_specs:        # list of {config_type, react_coord_type, atom_type_ids}.
                                # config_type: ~ / "vacuum" / "solvated".
                                # react_coord_type: ~ / "RS" / "TS" / "PS".
                                # atom_type_ids: ~ (all) / "solute" / "solvent" / [i, ...].
        subset_names:           # labels for each entry of atom_type_specs.
        subsets_name:           # short name for this subset scheme (used in output names).
        atom_id_cap:            # upper atom index for "solvent" selections (default 500).

    Plot settings (all optional):
        plot_settings:
            show_legend:        # whether to draw legends on the correlation plots.
            per_element:        # use per-element MACE descriptor similarity caches.
            save_corr_plots:    # whether to save the correlation scatter plots.
            corr_plot_alphas:   # scatter alpha per stage (zero-shot / FT / full FT).
            r_plot:             # {y_label: ..., ylim: [lo, hi]} for the Pearson's-r plots.
            beta_plot:          # {y_label: ..., ylim: [lo, hi]} for the beta plots.
            mag_plot:           # {y_label: ..., ylim: [lo, hi], log_y: bool, remove_negs: bool}.

    Any omitted field is set to None (or its default, where stated above).
    """

    import argparse

    # input YAML config path from a single command-line argument
    parser = argparse.ArgumentParser(
        description="Run an MLIP pretraining-bias analysis driven by an input YAML "
                    "config file (format described above).",
        epilog="Example: rmlt run-pretrain-bias-analysis "
               "inputs/pretrain_bias/CPD_MVK_DA_pretrain_bias.yaml",
    )
    parser.add_argument(
        "input_yaml",
        help="Path to the input YAML config file "
             "(e.g. inputs/pretrain_bias/CPD_MVK_DA_pretrain_bias.yaml).",
    )
    args = parser.parse_args()

    # ===== load + process the input YAML (see react_ml_test_public/helpers/inputs.py) =====
    resolved = rmlt_inputs.resolve_pretrain_bias_analysis_yaml(args.input_yaml, parser)

    # test system
    test_name = resolved["test_name"]
    test_fpath = resolved["test_fpath"]
    react_coords = resolved["react_coords"]
    ts_dist_threshold = resolved["ts_dist_threshold"]
    n_solute_atoms = resolved["n_solute_atoms"]
    model_ft_n = resolved["model_ft_n"]
    model_full_ft_n = resolved["model_full_ft_n"]
    ref_ts_config = resolved["ref_ts_config"]
    input_lot_test_fpaths = resolved["input_lot_test_fpaths"]

    # base model / fine-tuning selection
    model_names = resolved["model_names"]
    model_plt_labels = resolved["model_plt_labels"]
    n = resolved["n"]
    ft_train_name = resolved["ft_train_name"]
    sample_str = resolved["sample_str"]
    energy_str = resolved["energy_str"]
    rep_num = resolved["rep_num"]
    full_ft_rep_num = resolved["full_ft_rep_num"]

    # optional relative baselines
    from_scratch_names = resolved["from_scratch_names"]
    from_scratch_energy_str = resolved["from_scratch_energy_str"]
    zero_shot_ref_names = resolved["zero_shot_ref_names"]
    ft_ref_names = resolved["ft_ref_names"]
    rel_comp_str = resolved["rel_comp_str"]
    abs_y = resolved["abs_y"]

    # fine-tuned model names
    ft_model_names = resolved["ft_model_names"]
    full_ft_model_names = resolved["full_ft_model_names"]
    rel_comp_model_names = resolved["rel_comp_model_names"]
    full_rel_comp_model_names = resolved["full_rel_comp_model_names"]

    # atom-type subsets
    atom_type_specs = resolved["atom_type_specs"]
    subset_names = resolved["subset_names"]
    subsets_name = resolved["subsets_name"]

    # plot settings
    show_legend = resolved["show_legend"]
    per_element = resolved["per_element"]
    save_corr_plots = resolved["save_corr_plots"]
    corr_plot_alphas = resolved["corr_plot_alphas"]
    scatter_r_y_label = resolved["scatter_r_y_label"]
    scatter_r_ylim = resolved["scatter_r_ylim"]
    scatter_beta_y_label = resolved["scatter_beta_y_label"]
    scatter_beta_ylim = resolved["scatter_beta_ylim"]
    scatter_mag_y_label = resolved["scatter_mag_y_label"]
    scatter_mag_ylim = resolved["scatter_mag_ylim"]
    mag_log_y = resolved["mag_log_y"]
    mag_remove_negs = resolved["mag_remove_negs"]
    del resolved

    abs_func = np.abs if abs_y else (lambda x: x)


    all_subset_data = {}
    all_config_sim_scores = []
    all_lot_force_maes = []
    all_lot_r_series = []
    all_sim_r_series = []
    all_lot_beta_series = []
    all_sim_beta_series = []
    all_lot_mag_series = []
    all_sim_mag_series = []
    diagnostic_summary_rows = []
    subset_summary_rows = []
    rmse_summary_rows = []
    correlation_summary_rows = []
    for model_name, ft_model_name, full_ft_model_name, rel_comp_model_name, full_rel_comp_model_name, input_lot_test_fpath in zip(model_names, ft_model_names, full_ft_model_names, rel_comp_model_names, full_rel_comp_model_names, input_lot_test_fpaths):
        
        # get test fpath and sort by structure hash
        test_configs = flexible_ase_read(test_fpath)
        test_sorted_indices = sorted(range(len(test_configs)), key=lambda i: atoms_hash(test_configs[i]))
        test_configs = [test_configs[i] for i in test_sorted_indices]

        if not all([x is None for x in rel_comp_model_names]):
            pred_models = [model_name, ft_model_name, full_ft_model_name, rel_comp_model_name, full_rel_comp_model_name]
        else:
            pred_models = [model_name, ft_model_name, full_ft_model_name]
        print("PRED MODELS: ", pred_models)

        # calc model errors + sort
        all_test_energies, all_test_forces, all_pred_energies_dict, all_pred_forces_dict = get_test_and_pred_data(pred_models,
                                                                                                                   [test_fpath])
        
        # energies
        test_energies = np.array([all_test_energies[0][i] for i in test_sorted_indices])       
        zero_shot_pred_energies = np.array([all_pred_energies_dict[model_name][0][i] for i in test_sorted_indices])
        ft_pred_energies = np.array([all_pred_energies_dict[ft_model_name][0][i] for i in test_sorted_indices])
        full_ft_pred_energies = np.array([all_pred_energies_dict[full_ft_model_name][0][i] for i in test_sorted_indices])

        zero_shot_energy_errors = np.abs(test_energies - zero_shot_pred_energies)
        ft_energy_errors = np.abs(test_energies - ft_pred_energies)
        full_ft_energy_errors = np.abs(test_energies - full_ft_pred_energies)

        # forces
        test_forces = all_test_forces[0]
        # test_force_norms = np.linalg.norm(all_test_forces[0], axis=1)
        # zero_shot_force_norms = np.linalg.norm(all_pred_forces_dict[model_name][0], axis=1)
        # ft_force_norms = np.linalg.norm(all_pred_forces_dict[ft_model_name][0], axis=1)
        # full_ft_force_norms = np.linalg.norm(all_pred_forces_dict[full_ft_model_name][0], axis=1)

        # zero_shot_force_norm_errors = np.abs(test_force_norms - zero_shot_force_norms)
        # ft_force_norm_errors = np.abs(test_force_norms - ft_force_norms)
        # full_ft_force_norm_errors = np.abs(test_force_norms - full_ft_force_norms)

        # force magnitudes!
        test_forces = all_test_forces[0]
        zero_shot_force_norm_errors =  np.linalg.norm(test_forces - all_pred_forces_dict[model_name][0], axis=1)
        ft_force_norm_errors = np.linalg.norm(test_forces - all_pred_forces_dict[ft_model_name][0], axis=1)
        full_ft_force_norm_errors = np.linalg.norm(test_forces - all_pred_forces_dict[full_ft_model_name][0], axis=1)

        # control for reference trained potential to isolate effect of models / pretraining data!
        if not all([x is None for x in rel_comp_model_names]):
            scratch_pred_energies = np.array([all_pred_energies_dict[rel_comp_model_name][0][i] for i in test_sorted_indices])
            full_scratch_pred_energies = np.array([all_pred_energies_dict[full_rel_comp_model_name][0][i] for i in test_sorted_indices])
            scratch_energy_errors = test_energies - scratch_pred_energies
            full_scratch_energy_errors = test_energies - full_scratch_pred_energies

            # forces
            # scratch_force_norms = np.linalg.norm(all_pred_forces_dict[rel_comp_model_name][0], axis=1)
            # full_scratch_force_norms = np.linalg.norm(all_pred_forces_dict[full_rel_comp_model_name][0], axis=1)
            
            # scratch_force_norm_errors = np.abs(test_force_norms - scratch_force_norms)
            # full_scratch_force_norm_errors = np.abs(test_force_norms - full_scratch_force_norms)

            # force magnitudes!
            scratch_force_norm_errors = np.linalg.norm(test_forces - all_pred_forces_dict[rel_comp_model_name][0], axis=1)
            full_scratch_force_norm_errors = np.linalg.norm(test_forces - all_pred_forces_dict[full_rel_comp_model_name][0], axis=1)

            # get relative errors
            ft_force_norm_errors = abs_func(ft_force_norm_errors - scratch_force_norm_errors)
            full_ft_force_norm_errors = abs_func(full_ft_force_norm_errors - full_scratch_force_norm_errors)            

        # get input lot energies and forces (ensuring they are ordered the same as test set reference)
        input_lot_configs = flexible_ase_read(input_lot_test_fpath)
        input_lot_sorted_indices = sorted(range(len(input_lot_configs)), key=lambda i: atoms_hash(input_lot_configs[i]))
        input_lot_configs = [input_lot_configs[i] for i in input_lot_sorted_indices]

        # check datasets are identical and ordered in the same way
        input_lot_atoms_hashes = [atoms_hash(a) for a in input_lot_configs]
        test_atoms_hashes = [atoms_hash(b) for b in test_configs]
        assert all(a == b for a, b in zip(input_lot_atoms_hashes, test_atoms_hashes)), f'Structures are not identical and ordered in the same way! Len A: {len(input_lot_atoms_hashes)}, Len B: {len(test_atoms_hashes)}'
        input_lot_config_lens = [len(config) for config in input_lot_configs]

        # energies
        input_lot_energies = get_ase_props(input_lot_configs, 'energy')
        lot_energy_errors = input_lot_energies - test_energies

        # forces
        input_lot_forces = convert_units(get_ase_props(input_lot_configs, 'forces'), EV_PER_AMSTRONG, MEV_PER_AMSTRONG)
        # lot_force_norms = np.linalg.norm(input_lot_forces, axis=1)
        config_lot_forces = unflatten_array_by_lengths(input_lot_forces, input_lot_config_lens)

        # load desc similarities
        per_element_str = "_per_element" if per_element else ""
        test_sim_fpath = f"{SIMILARITIES_DIR}/{test_name}_{model_name}_mace_desc_cosine_sims{per_element_str}_z=8.npz"
        cache_data = load_data_cache(test_sim_fpath)
        best_sim_values, best_sim_indices, test_config_lens = cache_data['best_sim_values'].flatten(), cache_data['best_sim_indices'], cache_data['config_lens']
        # test_config_lens = [test_config_lens[i] for i in test_sorted_indices]
        diagnostic_summary_rows.append({
            "Model": model_name,
            "Subset": "All",
            "Metric": "Best Sim Values shape",
            "Value": best_sim_values.shape,
        })

        # per config sims and forces
        config_best_sims = unflatten_array_by_lengths(best_sim_values, test_config_lens)
        config_test_forces = unflatten_array_by_lengths(test_forces, test_config_lens)
        config_zero_shot_force_norm_errors = unflatten_array_by_lengths(zero_shot_force_norm_errors, test_config_lens)
        config_ft_force_norm_errors = unflatten_array_by_lengths(ft_force_norm_errors, test_config_lens)
        config_full_ft_force_norm_errors = unflatten_array_by_lengths(full_ft_force_norm_errors, test_config_lens)
        # config_scratch_force_norm_errors = unflatten_array_by_lengths(scratch_force_norm_errors, test_config_lens)
        # config_full_scratch_force_norm_errors = unflatten_array_by_lengths(full_scratch_force_norm_errors, test_config_lens)
        diagnostic_summary_rows.extend([
            {"Model": model_name, "Subset": "All", "Metric": "Config best sims count", "Value": len(config_best_sims)},
            {"Model": model_name, "Subset": "All", "Metric": "Config pred force norm errors count", "Value": len(config_zero_shot_force_norm_errors)},
            {"Model": model_name, "Subset": "All", "Metric": "Pred energy errors shape", "Value": zero_shot_energy_errors.shape},
        ])

        # sort by atoms structure hash
        config_best_sims = [config_best_sims[i] for i in test_sorted_indices]
        config_test_forces = [config_test_forces[i] for i in test_sorted_indices]
        config_zero_shot_force_norm_errors = [config_zero_shot_force_norm_errors[i] for i in test_sorted_indices]
        config_ft_force_norm_errors = [config_ft_force_norm_errors[i] for i in test_sorted_indices]
        config_full_ft_force_norm_errors = [config_full_ft_force_norm_errors[i] for i in test_sorted_indices]
        # config_scratch_force_norm_errors = [config_scratch_force_norm_errors[i] for i in test_sorted_indices]
        # config_full_scratch_force_norm_errors = [config_full_scratch_force_norm_errors[i] for i in test_sorted_indices]

        # lot_force_norms = np.linalg.norm(input_lot_forces, axis=1)
        # AS LONG AS THE SPLITTING HERE WORKS - we are good
        config_lot_force_norm_errors = [np.linalg.norm(config_test_forces[i] - config_lot_forces[i], axis=1) for i in range(len(config_test_forces))]

        # split by config types
        subset_data = []
        subset_results = []
        subset_betas = []
        subset_mags = []
        subset_lot_force_maes = []
        for (config_type, react_coord_type, atom_type_ids), subset_name in zip(atom_type_specs, subset_names):

            # filter based on config type
            subset_ids = [config_id for config_id, config in enumerate(test_configs) if config.info['config_type'] == config_type] if config_type is not None else [i for i, _ in enumerate(test_configs)]
            total_num_subset_configs = len(subset_ids)
            print(f"Subset: {subset_name} with {len(subset_ids)} configurations...")
                        
            # filter based on react coord type (RS / TS / PS), if specified
            if react_coord_type is not None:
                ref_ts_dist = sum([ref_ts_config.get_distance(i, j) for i, j in react_coords])
                config_r_coords = [[config.get_distance(i, j) for i, j in react_coords] for config in test_configs]
                config_ts_dists = [sum([config.get_distance(i, j) for i, j in react_coords]) for config in test_configs]
                half_threshold = ts_dist_threshold / 2

                if react_coord_type == 'RS':
                    subset_ids = [i for i in subset_ids if config_ts_dists[i] > ref_ts_dist + half_threshold]
                elif react_coord_type == 'TS':
                    subset_ids = [i for i in subset_ids if np.abs(config_ts_dists[i] - ref_ts_dist) <= half_threshold]
                elif react_coord_type == 'PS':
                    subset_ids = [i for i in subset_ids if config_ts_dists[i] < ref_ts_dist - half_threshold]
                num_react_coord_configs = len(subset_ids)
                subset_r_coords = [config_r_coords[i] for i in subset_ids]
            if react_coord_type is None:
                num_react_coord_configs = ""
                subset_r_coords = ""

            subset_configs = [test_configs[config_id] for config_id in subset_ids]
            subset_config_lens = [test_config_lens[config_id] for config_id in subset_ids]

            subset_zero_shot_energy_errors = [zero_shot_energy_errors[config_id] for config_id in subset_ids]
            subset_ft_energy_errors = [ft_energy_errors[config_id] for config_id in subset_ids]
            subset_full_ft_energy_errors = [full_ft_energy_errors[config_id] for config_id in subset_ids]

            subset_config_zero_shot_force_norm_errors = [config_zero_shot_force_norm_errors[config_id] for config_id in subset_ids]
            subset_config_ft_force_norm_errors = [config_ft_force_norm_errors[config_id] for config_id in subset_ids]
            subset_config_full_ft_force_norm_errors = [config_full_ft_force_norm_errors[config_id] for config_id in subset_ids]
            # subset_config_scratch_force_norm_errors = [config_scratch_force_norm_errors[config_id] for config_id in subset_ids]
            # subset_config_full_scratch_force_norm_errors = [config_full_scratch_force_norm_errors[config_id] for config_id in subset_ids]

            subset_config_best_sims = [config_best_sims[config_id] for config_id in subset_ids]

            subset_lot_energy_errors = [lot_energy_errors[config_id] for config_id in subset_ids]
            subset_config_lot_force_norm_errors = [config_lot_force_norm_errors[config_id] for config_id in subset_ids]
            subset_summary_rows.append({
                "Model": model_name,
                "Subset": subset_name,
                "Total Configs": total_num_subset_configs,
                "React Coord Type": react_coord_type or "",
                "Filtered Configs": len(subset_configs),
            })  

            # get force errors and similarities for atom ids only
            if atom_type_ids is not None:
                subset_config_zero_shot_force_norm_errors = [force_norm_errors[[i for i in atom_type_ids if i < len(force_norm_errors)]] for force_norm_errors in subset_config_zero_shot_force_norm_errors]
                subset_config_ft_force_norm_errors = [force_norm_errors[[i for i in atom_type_ids if i < len(force_norm_errors)]] for force_norm_errors in subset_config_ft_force_norm_errors]
                subset_config_full_ft_force_norm_errors = [force_norm_errors[[i for i in atom_type_ids if i < len(force_norm_errors)]] for force_norm_errors in subset_config_full_ft_force_norm_errors]
                # subset_config_scratch_force_norm_errors = [force_norm_errors[[i for i in atom_type_ids if i < len(force_norm_errors)]] for force_norm_errors in subset_config_scratch_force_norm_errors]
                # subset_config_full_scratch_force_norm_errors = [force_norm_errors[[i for i in atom_type_ids if i < len(force_norm_errors)]] for force_norm_errors in subset_config_full_scratch_force_norm_errors]

                subset_config_best_sims = [best_sims[[i for i in atom_type_ids if i < len(best_sims)]] for best_sims in subset_config_best_sims]
                subset_config_lens = [len(force_norm_errors) for force_norm_errors in subset_config_zero_shot_force_norm_errors] if atom_type_ids is not None else subset_config_lens

                subset_config_lot_force_norm_errors = [lot_force_errors[[i for i in atom_type_ids if i < len(lot_force_errors)]] for lot_force_errors in subset_config_lot_force_norm_errors]

            subset_zero_shot_force_errors = np.concatenate(subset_config_zero_shot_force_norm_errors)
            subset_ft_force_errors = np.concatenate(subset_config_ft_force_norm_errors)
            subset_full_ft_force_errors = np.concatenate(subset_config_full_ft_force_norm_errors)
            subset_best_sims = np.concatenate(subset_config_best_sims)
            subset_lot_force_errors = np.concatenate(subset_config_lot_force_norm_errors)

            # get per config similarity means 
            subset_config_mean_best_sims = np.array([np.mean(config_best_sim_values) for config_best_sim_values in subset_config_best_sims])
            # print(f"Subset config mean best sims shape: {subset_config_mean_best_sims.shape}")
            subset_mean_best_sim = np.mean(subset_config_mean_best_sims)
            subset_best_best_dissim_range = (1 - subset_best_sims).max() - (1 - subset_best_sims).min()
            rmse_summary_rows.append({
                "Model": model_name,
                "Subset": subset_name,
                "Metric": "Config Mean Best Similarity",
                "Value": subset_mean_best_sim,
            })

            subset_zero_shot_energy_error_rmse = calculate_rmse_on_errors(subset_zero_shot_energy_errors)
            subset_zero_shot_force_error_rmse = calculate_rmse_on_errors(subset_zero_shot_force_errors)
            subset_ft_force_error_rmse = calculate_rmse_on_errors(subset_ft_force_errors)
            subset_full_ft_force_error_rmse = calculate_rmse_on_errors(subset_full_ft_force_errors)
            subset_lot_force_error_rmse = calculate_rmse_on_errors(subset_lot_force_errors)
            subset_lot_force_error_mae = np.mean(np.abs(subset_lot_force_errors))
            subset_lot_force_error_range = subset_lot_force_errors.max() - subset_lot_force_errors.min()
            rmse_summary_rows.extend([
                {"Model": model_name, "Subset": subset_name, "Metric": "Energy Error RMSE", "Value": subset_zero_shot_energy_error_rmse},
                {"Model": model_name, "Subset": subset_name, "Metric": "Zero Shot Force Error RMSE", "Value": subset_zero_shot_force_error_rmse},
                {"Model": model_name, "Subset": subset_name, "Metric": "FT Force Error RMSE", "Value": subset_ft_force_error_rmse},
                {"Model": model_name, "Subset": subset_name, "Metric": "Full FT Force Error RMSE", "Value": subset_full_ft_force_error_rmse},
                {"Model": model_name, "Subset": subset_name, "Metric": "LoT Mean Absolute Force Error", "Value": subset_lot_force_error_mae},
            ])

            # correlate various factors across all datasets
            # 1. lot error (to zero-shot errors) + to fine-tuned errors
            lot_zero_shot_energy_corr = np.nan_to_num(np.corrcoef(subset_zero_shot_energy_errors, subset_lot_energy_errors)[0, 1], nan=0.0)
            lot_zero_shot_force_corr = np.nan_to_num(np.corrcoef(subset_zero_shot_force_errors, subset_lot_force_errors)[0, 1], nan=0.0)
            lot_ft_energy_corr = np.nan_to_num(np.corrcoef(subset_ft_energy_errors, subset_lot_energy_errors)[0, 1], nan=0.0)
            lot_full_ft_energy_corr = np.nan_to_num(np.corrcoef(subset_full_ft_energy_errors, subset_lot_energy_errors)[0, 1], nan=0.0)
            lot_ft_force_corr = np.nan_to_num(np.corrcoef(subset_ft_force_errors, subset_lot_force_errors)[0, 1], nan=0.0)
            lot_full_ft_force_corr = np.nan_to_num(np.corrcoef(subset_full_ft_force_errors, subset_lot_force_errors)[0, 1], nan=0.0)

            colour = PAPER_MODEL_COLOURS[model_name]
            plt_save_fp_base = f"{PLOT_DIR}/descriptors/{model_name}_{test_name}"

            correlation_summary_rows.extend([
                # {"Model": model_name, "Subset": subset_name, "Metric": "LoT Zero Shot Energy R", "Value": lot_zero_shot_energy_corr},
                {"Model": model_name, "Subset": subset_name, "Metric": "LoT Zero Shot Force R", "Value": lot_zero_shot_force_corr},
                # {"Model": model_name, "Subset": subset_name, "Metric": "LoT FT Energy R", "Value": lot_ft_energy_corr},
                {"Model": model_name, "Subset": subset_name, "Metric": "LoT FT Force R", "Value": lot_ft_force_corr},
                # {"Model": model_name, "Subset": subset_name, "Metric": "LoT Full FT Energy R", "Value": lot_full_ft_energy_corr},
                {"Model": model_name, "Subset": subset_name, "Metric": "LoT Full FT Force R", "Value": lot_full_ft_force_corr},
            ])

            # 2. desc error (to zero-shot errors) + to fine-tuned errors
            sim_zero_shot_energy_corr = np.corrcoef(subset_zero_shot_energy_errors, 1 - subset_config_mean_best_sims)[0, 1]
            sim_zero_shot_force_corr = np.corrcoef(subset_zero_shot_force_errors, 1 - subset_best_sims)[0, 1]
            sim_ft_energy_corr = np.corrcoef(subset_ft_energy_errors, 1 - subset_config_mean_best_sims)[0, 1]
            sim_full_ft_energy_corr = np.corrcoef(subset_full_ft_energy_errors, 1 - subset_config_mean_best_sims)[0, 1]
            sim_ft_force_corr = np.corrcoef(subset_ft_force_errors, 1 - subset_best_sims)[0, 1]
            sim_full_ft_force_corr = np.corrcoef(subset_full_ft_force_errors, 1 - subset_best_sims)[0, 1]

            zero_shot_lot_beta, ft_lot_beta, full_ft_lot_beta = None, None, None 
            zero_shot_lot_mag, ft_lot_mag, full_ft_lot_mag = None, None, None 
            zero_shot_sim_beta, ft_sim_beta, full_ft_sim_beta = None, None, None 
            zero_shot_sim_mag, ft_sim_mag, full_ft_sim_mag = None, None, None 
            if save_corr_plots:
                corr_plot_datasets = [
                                    (subset_zero_shot_force_errors, subset_lot_force_errors, 'zero_shot_lot', 'lot'),
                                    (subset_ft_force_errors, subset_lot_force_errors, 'ft_lot', 'lot'),
                                    (subset_full_ft_force_errors, subset_lot_force_errors, 'full_ft_lot', 'lot'),
                                    (subset_zero_shot_force_errors, 1 - subset_best_sims, 'zero_shot_sim', 'sim'),
                                    (subset_ft_force_errors, 1 - subset_best_sims, 'ft_sim', 'sim'),
                                    (subset_full_ft_force_errors, 1 - subset_best_sims, 'full_ft_sim', 'sim'),
                                    ]

                labels = [rf"$\mathbf{{{model_plt_labels[model_name]}}}$", 
                          rf"$\mathbf{{{model_plt_labels[model_name]}_{{{model_ft_n}}}}}$", 
                          rf"$\mathbf{{{model_plt_labels[model_name]}_{{{model_full_ft_n}}}}}$"]

                # combined lot
                (zero_shot_lot_beta, ft_lot_beta, full_ft_lot_beta), \
                (zero_shot_lot_mag, ft_lot_mag, full_ft_lot_mag) = corr_plot(
                        [x[1] for x in corr_plot_datasets[0:3]], 
                        [x[0] for x in corr_plot_datasets[0:3]], 
                        alphas=corr_plot_alphas, 
                        # alphas=[0.3, 0.3, 0.3], 
                        colours=generate_shades(PAPER_MODEL_COLOURS[model_name]), 
                        x_label=None, #r"Lot Force Norm Error (meV / $\AA$)", 
                        y_label=None, # r"Pred. Force Norm Error (meV / $\AA$)", 
                        # colours=list(plt.get_cmap("tab10").colors)[0:3], 
                        labels=labels,
                        save_fp=f"{plt_save_fp_base}_{subset_name}_{rel_comp_str}combined_force_lot_corr.png",
                        log_log=False,
                        plot_legend=show_legend,
                        mag_value=subset_lot_force_error_mae,
                        label_fmtr=".2f"
                        # mag_value=subset_lot_force_error_range#subset_lot_force_error_mae
                )

                # combined sim
                (zero_shot_sim_beta, ft_sim_beta, full_ft_sim_beta), \
                (zero_shot_sim_mag, ft_sim_mag, full_ft_sim_mag) = corr_plot(
                        [x[1] for x in corr_plot_datasets[3:6]], 
                        [x[0] for x in corr_plot_datasets[3:6]], 
                        alphas=corr_plot_alphas, 
                        # alphas=[0.3, 0.3, 0.3], 
                        colours=generate_shades(PAPER_MODEL_COLOURS[model_name]), 
                        # colours=list(plt.get_cmap("tab10").colors)[0:3], 
                        x_label=None, #r'1 - sim_a',
                        y_label=None, #r"Pred. Force Norm Error (meV / $\AA$)", 
                        labels=labels,
                        save_fp=f"{plt_save_fp_base}_{subset_name}_{rel_comp_str}combined_force_sim_corr.png",
                        log_log=False,
                        beta_factor=0.001,
                        plot_legend=show_legend,
                        mag_value=1 - subset_mean_best_sim,
                        label_fmtr=".0f"
                        # mag_value=subset_best_best_dissim_range # 1 - subset_mean_best_sim
                )

            # plot absolute mean cosine correlations

            correlation_summary_rows.extend([
                {"Model": model_name, "Subset": subset_name, "Metric": "Sim Zero Shot Force R", "Value": sim_zero_shot_force_corr},
                {"Model": model_name, "Subset": subset_name, "Metric": "Sim FT Force R", "Value": sim_ft_force_corr},
                {"Model": model_name, "Subset": subset_name, "Metric": "Sim Full FT Force R", "Value": sim_full_ft_force_corr},
            ])

            correlation_summary_rows.extend([
                {"Model": model_name, "Subset": subset_name, "Metric": "LoT Zero Shot Force Beta", "Value": zero_shot_lot_beta},
                {"Model": model_name, "Subset": subset_name, "Metric": "LoT FT Force Beta", "Value": ft_lot_beta},
                {"Model": model_name, "Subset": subset_name, "Metric": "LoT Full FT Force Beta", "Value": full_ft_lot_beta},
                {"Model": model_name, "Subset": subset_name, "Metric": "Sim Zero Shot Force Beta", "Value": zero_shot_sim_beta},
                {"Model": model_name, "Subset": subset_name, "Metric": "Sim FT Force Beta", "Value": ft_sim_beta},
                {"Model": model_name, "Subset": subset_name, "Metric": "Sim Full FT Force Beta", "Value": full_ft_sim_beta},
            ])

            correlation_summary_rows.extend([
                {"Model": model_name, "Subset": subset_name, "Metric": "LoT Zero Shot Force Error Mag", "Value": zero_shot_lot_mag},
                {"Model": model_name, "Subset": subset_name, "Metric": "LoT FT Force Error Mag", "Value": ft_lot_mag},
                {"Model": model_name, "Subset": subset_name, "Metric": "LoT Full FT Force Error Mag", "Value": full_ft_lot_mag},
                {"Model": model_name, "Subset": subset_name, "Metric": "Sim Zero Shot Force Error Mag", "Value": zero_shot_sim_mag},
                {"Model": model_name, "Subset": subset_name, "Metric": "Sim FT Force Error Mag", "Value": ft_sim_mag},
                {"Model": model_name, "Subset": subset_name, "Metric": "Sim Full FT Force Error Mag", "Value": full_ft_sim_mag},
            ])
            
            subset_data.append((subset_name, subset_lot_force_errors, 1 - subset_best_sims, subset_ft_force_errors, subset_full_ft_force_errors))
            subset_results.append((subset_name, lot_zero_shot_force_corr, lot_ft_force_corr, lot_full_ft_force_corr, sim_zero_shot_force_corr, sim_ft_force_corr, sim_full_ft_force_corr, subset_mean_best_sim))
            subset_betas.append((zero_shot_lot_beta, ft_lot_beta, full_ft_lot_beta, zero_shot_sim_beta, ft_sim_beta, full_ft_sim_beta))
            subset_mags.append((zero_shot_lot_mag, ft_lot_mag, full_ft_lot_mag, zero_shot_sim_mag, ft_sim_mag, full_ft_sim_mag))

            subset_lot_force_maes.append(subset_lot_force_error_mae)

        model_lot_r_series = [
                np.array([x[1] for x in subset_results]),
                np.array([x[2] for x in subset_results]),
                np.array([x[3] for x in subset_results]),
            ]

        model_sim_r_series = [
                np.array([x[4] for x in subset_results]),
                np.array([x[5] for x in subset_results]),
                np.array([x[6] for x in subset_results]),
            ]

        model_lot_beta_series = [
            np.array([x[0] for x in subset_betas]), 
            np.array([x[1] for x in subset_betas]), 
            np.array([x[2] for x in subset_betas]), 
        ]

        model_sim_beta_series = [
            np.array([x[3] for x in subset_betas]), 
            np.array([x[4] for x in subset_betas]), 
            np.array([x[5] for x in subset_betas]), 
        ]

        model_lot_mag_series = [
            np.array([x[0] for x in subset_mags]), 
            np.array([x[1] for x in subset_mags]), 
            np.array([x[2] for x in subset_mags]), 
        ]

        model_sim_mag_series = [
            np.array([x[3] for x in subset_mags]), 
            np.array([x[4] for x in subset_mags]), 
            np.array([x[5] for x in subset_mags]),
        ]

        all_config_sim_scores.append([x[7] for x in subset_results])
        all_lot_force_maes.append(subset_lot_force_maes)

        all_subset_data[model_name] = subset_data

        all_lot_r_series.append(model_lot_r_series)
        all_sim_r_series.append(model_sim_r_series)

        all_lot_beta_series.append(model_lot_beta_series)
        all_sim_beta_series.append(model_sim_beta_series)

        all_lot_mag_series.append(model_lot_mag_series)
        all_sim_mag_series.append(model_sim_mag_series)

    # series_labels = ['Zero Shot Force R', 'FT Force R', 'Full FT Force R']

    print_summary_table(
        "Subset Summary",
        subset_summary_rows,
        ["Model", "Subset", "Total Configs", "React Coord Type", 
        "Filtered Configs"],
    )

    print_summary_table(
        "RMSE / Similarity Summary",
        rmse_summary_rows,
        ["Model", "Subset", "Metric", "Value"],
    )
    print_summary_table(
        "Correlation Summary",
        correlation_summary_rows,
        ["Model", "Subset", "Metric", "Value"],
    )

    # Pairwise comparison
    # print("=== PAIRWISE ANALYSIS SIM ===")
    # pairwise_x_y1_correlations(all_subset_data, x_index=1, y_index=2)
    # print("=============================")

    # plot r values
    save_name = f"{test_name}_lot_r_{subsets_name}_{rel_comp_str}"
    category_scatter_plot(all_lot_r_series, ['o', 'o', 'o'], subset_names, model_names, save_name,
                          y_label=scatter_r_y_label, ylim=scatter_r_ylim)

    save_name = f"{test_name}_sim_r_{subsets_name}_{rel_comp_str}"
    category_scatter_plot(all_sim_r_series, ['^', '^', '^'], subset_names, model_names, save_name,
                          y_label=scatter_r_y_label, ylim=scatter_r_ylim)

    # plot betas
    save_name = f"{test_name}_lot_betas_{subsets_name}_{rel_comp_str}"
    category_scatter_plot(all_lot_beta_series, ['o', 'o', 'o'], subset_names, model_names, save_name,
                          y_label=scatter_beta_y_label, ylim=scatter_beta_ylim, log_y=False)

    save_name = f"{test_name}_sim_betas_{subsets_name}_{rel_comp_str}"
    category_scatter_plot(all_sim_beta_series, ['^', '^', '^'], subset_names, model_names, save_name,
                          y_label=scatter_beta_y_label, ylim=scatter_beta_ylim, log_y=False)

    # plot beta * u magnitudes
    save_name = f"{test_name}_lot_mags_{subsets_name}_{rel_comp_str}"
    category_scatter_plot(all_lot_mag_series, ['o', 'o', 'o'], subset_names, model_names, save_name,
                          y_label=scatter_mag_y_label, ylim=scatter_mag_ylim,
                          log_y=mag_log_y, remove_negs=mag_remove_negs)

    save_name = f"{test_name}_sim_mags_{subsets_name}_{rel_comp_str}"
    category_scatter_plot(all_sim_mag_series, ['^', '^', '^'], subset_names, model_names, save_name,
                          y_label=scatter_mag_y_label, ylim=scatter_mag_ylim,
                          log_y=mag_log_y, remove_negs=mag_remove_negs)

    # bar plot of lot MAEs
    save_fp = f'{PLOT_DIR}/descriptors/{test_name}_model_lot_maes.png'
    plot_settings = PlotSettings(custom_colour_list=[PAPER_MODEL_COLOURS[model_name] for model_name in model_names], 
                                 show_legend=False, x_label='Config Type', y_label=r'$\bar{LoT_a}$'
                                )
    bar_plot(all_lot_force_maes, subset_names, model_names, save_fp=save_fp, plot_settings=plot_settings, show_x_tick_labels=True)

    # bar plot of absolute similarities!
    save_fp = f'{PLOT_DIR}/descriptors/{test_name}_model_abs_sims.png'
    plot_settings = PlotSettings(custom_colour_list=[PAPER_MODEL_COLOURS[model_name] for model_name in model_names], 
                                 show_legend=False, x_label='Config Type', y_label=r'$\bar{sim_a}$'
                                )
    bar_plot(all_config_sim_scores, subset_names, model_names, save_fp=save_fp, plot_settings=plot_settings, show_x_tick_labels=True)
