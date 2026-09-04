"""
Shared YAML input loading / processing for the YAML-driven workflow scripts.

Each workflow script that takes a single input YAML config file has one loader
function in this module that takes in and processes the YAML for that script:

    resolve_test_set_comparison_yaml(...)          -> run_test_set_comparison.py
    resolve_pretrain_bias_analysis_yaml(...)       -> run_pretrain_bias_analysis.py
    resolve_reaction_barrier_comparison_yaml(...)  -> run_reaction_barrier_comparison.py

Each function performs the shared file checks / parsing / ``${var}`` substitution
(via ``_load_yaml_config``) followed by that script's own field resolution and
validation, then returns a flat dict of the resolved config variables. The
corresponding script's ``__main__`` block declares those variables from the returned
dict, so the YAML processing itself stays out of the scripts.

``parser`` (an optional argparse.ArgumentParser) reports errors through
``parser.error(...)``; when None a ValueError is raised instead (useful for testing).
"""

import os

import yaml

from react_ml_test_public.helpers.io import (
    autode_props_parser,
    flexible_ase_read,
    orca_props_parser,
    substitute_yaml_variables,
)
from react_ml_test_public.helpers.settings import (
    KCAL_PER_MOL,
    KJ_PER_MOL,
    MEV,
    MEV_PER_ATOM,
    PAPER_MODEL_COLOURS,
)


def _error(parser, message):
    """Report a config error via ``parser.error``, or raise ValueError when no parser."""
    if parser is not None:
        parser.error(message)
    raise ValueError(message)


def _load_yaml_config(input_yaml_fpath, parser=None):
    """Load, substitute and validate a script input YAML file; returns its mapping dict."""
    if not input_yaml_fpath.endswith((".yaml", ".yml")):
        _error(parser, f"Input file must be a YAML file (.yaml / .yml), got: {input_yaml_fpath!r}")
    if not os.path.isfile(input_yaml_fpath):
        _error(parser, f"Input YAML file not found: {input_yaml_fpath!r}")
    with open(input_yaml_fpath, "r") as fh:
        input_config = yaml.safe_load(fh) or {}
        input_config = substitute_yaml_variables(input_config, input_config)
    if not isinstance(input_config, dict):
        _error(
            parser,
            f"Input YAML must be a mapping of config fields (see the script format), got a "
            f"{type(input_config).__name__}: {input_yaml_fpath!r}",
        )
    return input_config


def _parse_ylim(value):
    """Parse a [lo, hi] y-limit from a config value (or return None)."""
    if value is None:
        return None
    try:
        value = [float(x) for x in value]
    except (TypeError, ValueError):
        return None
    return (value[0], value[1]) if len(value) == 2 else None


def _as_bool(value, default=False):
    """Coerce a config value to a bool (handles quoted strings too)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def resolve_test_set_comparison_yaml(input_yaml_fpath: str, parser=None):
    """Load + process the input YAML for run_test_set_comparison.py.

    Returns a flat dict of the config variables that script's ``__main__`` declares.
    """
    input_config = _load_yaml_config(input_yaml_fpath, parser)
    return {
        "model_names": input_config.get("model_names", None),
        "n": input_config.get("n", None),
        "ft_train_name": input_config.get("ft_train_name", None),
        "test_name": input_config.get("test_name", None),
        "rep_num": input_config.get("rep_num", None),
        "omol_rep_num": input_config.get("omol_rep_num", None),
        "omol_energy_str": input_config.get("omol_energy_str", None),
        "test_set_ids": input_config.get("test_set_ids", None),
        "save_name": input_config.get("save_name", None),
        "run_names": input_config.get("run_names", None),
        "plot": input_config.get("plot", None),
        "num_base_models": input_config.get("num_base_models", None),
        "num_base_model_comps": input_config.get("num_base_model_comps", None),
        "alpha_list": input_config.get("alpha_list", None),
    }


def resolve_reaction_barrier_comparison_yaml(input_yaml_fpath: str, parser=None):
    """Load + process the input YAML for run_reaction_barrier_comparison.py.

    Returns a flat dict of the config variables that script's ``__main__`` declares.
    """
    input_config = _load_yaml_config(input_yaml_fpath, parser)

    # ===== model / fine-tuning selection =====
    n = input_config.get("n", None)
    train_name = input_config.get("train_name", None)
    sample_str = input_config.get("sample_str", None)
    energy_str = input_config.get("energy_str", None)
    rep_num = input_config.get("rep_num", None)
    omol_rep_num = input_config.get("omol_rep_num", None)
    omol_energy_str = input_config.get("omol_energy_str", None)
    model_names = input_config.get("model_names", None)
    save_name = input_config.get("save_name", None)
    num_base_models = input_config.get("num_base_models", None)
    num_base_model_comps = input_config.get("num_base_model_comps", None)
    alpha_list = input_config.get("alpha_list", None)
    linestyle_list = input_config.get("linestyle_list", None)
    if not model_names:
        _error(parser, "model_names must be a non-empty list of model names in the input YAML.")
    if num_base_models is None or num_base_model_comps is None:
        _error(parser, "num_base_models and num_base_model_comps must be set in the input YAML.")
    alpha_list = alpha_list if alpha_list is not None else [1.0]
    linestyle_list = linestyle_list if linestyle_list is not None else ["-"]
    run_names = input_config.get("run_names", None)
    if run_names is None:
        run_names = [save_name] if save_name is not None else []

    # ===== IRC trajectory comparison =====
    test_trj_fpath = input_config.get("test_trj_fpath", None)
    if test_trj_fpath is None:
        _error(parser, "test_trj_fpath must be set in the input YAML "
                       "(path to the IRC trajectory .xyz to compare against).")
    custom_xyz_parser_name = input_config.get("custom_xyz_parser", None)
    include_forces = input_config.get("include_forces", False)
    ref_prod_index = input_config.get("ref_prod_index", -1)
    xlabel = input_config.get("xlabel", "Index")
    force_units = input_config.get("force_units", "meV / A")
    grid_lines = input_config.get("grid_lines", False)
    show_legend = input_config.get("show_legend", False)
    cust_legend_colours = input_config.get("cust_legend_colours", ["black"])
    cust_legend_labels = input_config.get("cust_legend_labels", ["DFT"])
    cust_legend_linestyles = input_config.get("cust_legend_linestyles", ["dashed"])
    ref_point_labels = input_config.get("ref_point_labels", None)

    energies_ylim = _parse_ylim(input_config.get("energies_ylim", None))
    forces_ylim = _parse_ylim(input_config.get("forces_ylim", None))

    # map parser / unit names given as strings in the YAML onto the python objects they refer to
    _xyz_parsers = {
        "orca_props_parser": orca_props_parser,
        "autode_props_parser": autode_props_parser,
    }
    if isinstance(custom_xyz_parser_name, str):
        custom_xyz_parser = _xyz_parsers.get(custom_xyz_parser_name, None)
        if custom_xyz_parser is None:
            _error(parser, f"Unknown custom_xyz_parser: {custom_xyz_parser_name!r} "
                           f"(choose from {sorted(_xyz_parsers)}).")
    else:
        custom_xyz_parser = custom_xyz_parser_name  # e.g. None

    _energy_units = {
        "mev": MEV,
        "mev/atom": MEV_PER_ATOM,
        "kj/mol": KJ_PER_MOL,
        "kcal/mol": KCAL_PER_MOL,
    }
    energy_units_name = input_config.get("energy_units", None)
    if energy_units_name is None:
        energy_units = KCAL_PER_MOL
    else:
        energy_units = _energy_units.get("".join(energy_units_name.lower().split()), energy_units_name)

    return {
        "n": n,
        "train_name": train_name,
        "sample_str": sample_str,
        "energy_str": energy_str,
        "rep_num": rep_num,
        "omol_rep_num": omol_rep_num,
        "omol_energy_str": omol_energy_str,
        "model_names": model_names,
        "save_name": save_name,
        "run_names": run_names,
        "num_base_models": num_base_models,
        "num_base_model_comps": num_base_model_comps,
        "alpha_list": alpha_list,
        "linestyle_list": linestyle_list,
        "test_trj_fpath": test_trj_fpath,
        "custom_xyz_parser": custom_xyz_parser,
        "include_forces": include_forces,
        "ref_prod_index": ref_prod_index,
        "xlabel": xlabel,
        "energy_units": energy_units,
        "force_units": force_units,
        "energies_ylim": energies_ylim,
        "forces_ylim": forces_ylim,
        "grid_lines": grid_lines,
        "show_legend": show_legend,
        "cust_legend_colours": cust_legend_colours,
        "cust_legend_labels": cust_legend_labels,
        "cust_legend_linestyles": cust_legend_linestyles,
        "ref_point_labels": ref_point_labels,
    }


def resolve_pretrain_bias_analysis_yaml(input_yaml_fpath: str, parser=None):
    """Resolve and validate a run_pretrain_bias_analysis YAML config into a flat dict of
    variables matching the script's ``__main__`` expectations.

    ``parser`` (optional argparse.ArgumentParser) reports errors the same way as the other
    YAML-driven scripts; when None a ValueError is raised instead (useful for testing).
    """
    input_config = _load_yaml_config(input_yaml_fpath, parser)

    def _error(message):
        if parser is not None:
            parser.error(message)
        raise ValueError(message)

    def _need(key):
        value = input_config.get(key, None)
        if value is None:
            _error(f"Missing required field '{key}' in input YAML.")
        return value

    def _parse_ylim(value):
        """Parse a [lo, hi] y-limit from a config value (or return None)."""
        if value is None:
            return None
        try:
            value = [float(x) for x in value]
        except (TypeError, ValueError):
            return None
        return (value[0], value[1]) if len(value) == 2 else None

    def _as_bool(value, default=False):
        """Coerce a config value to a bool (handles quoted strings too)."""
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)

    # ---- test system ----
    test_name = _need("test_name")
    test_fpath = _need("test_fpath")
    react_coords = [tuple(pair) for pair in _need("react_coords")]
    ts_dist_threshold = float(input_config.get("ts_dist_threshold", 2.0))
    n_solute_atoms = int(_need("n_solute_atoms"))
    ref_ts_config = flexible_ase_read(_need("ref_ts_fpath"))[0]

    input_lot_test_fpaths = input_config.get("input_lot_test_fpaths", None)
    if input_lot_test_fpaths is None:
        _error("Missing required field 'input_lot_test_fpaths' in input YAML.")
    if isinstance(input_lot_test_fpaths, str):
        input_lot_test_fpaths = [input_lot_test_fpaths]
    num_models = len(input_lot_test_fpaths)

    # ---- base model / fine-tuning selection ----
    n = int(_need("n"))
    model_names = _need("model_names")
    if len(model_names) != num_models:
        _error(f"'model_names' must have one entry per 'input_lot_test_fpaths' entry "
               f"({num_models}), got {len(model_names)}.")
    missing_colours = [name for name in model_names if name not in PAPER_MODEL_COLOURS]
    if missing_colours:
        _error(f"Model names missing from PAPER_MODEL_COLOURS (needed for plot colours): "
               f"{missing_colours}")
    model_plt_labels = input_config.get("model_plt_labels", None)
    if model_plt_labels is None:
        model_plt_labels = {name: name for name in model_names}
    model_ft_n = input_config.get("model_ft_n", n)
    model_full_ft_n = input_config.get("model_full_ft_n", n)
    ft_train_name = _need("ft_train_name")
    sample_str = str(input_config.get("sample_str", ""))
    energy_str = str(input_config.get("energy_str", "energy_"))
    rep_num = int(input_config.get("rep_num", 1))
    full_ft_rep_num = int(input_config.get("full_ft_rep_num", 1))

    # ---- optional relative baselines (default off) ----
    from_scratch_names = input_config.get("from_scratch_names", None)
    from_scratch_energy_str = str(input_config.get("from_scratch_energy_str", ""))
    zero_shot_ref_names = input_config.get("zero_shot_ref_names", None)
    ft_ref_names = input_config.get("ft_ref_names", None)
    rel_comp_str = str(input_config.get("rel_comp_str", ""))
    abs_y = _as_bool(input_config.get("abs_y", True), default=True)
    _rel_baselines = [from_scratch_names, zero_shot_ref_names, ft_ref_names]
    if sum(baseline is not None for baseline in _rel_baselines) > 1:
        _error("Only one of 'from_scratch_names' / 'zero_shot_ref_names' / 'ft_ref_names' "
               "may be set at a time.")
    for _key in ("from_scratch_names", "zero_shot_ref_names", "ft_ref_names"):
        _value = input_config.get(_key, None)
        if _value is not None and (len(_value) != num_models or any(x is None for x in _value)):
            _error(f"'{_key}' must have one non-null entry per base model ({num_models}).")

    # ---- fine-tuned model names ----
    ft_model_names, full_ft_model_names = [], []
    rel_comp_model_names, full_rel_comp_model_names = [], []
    for i, model_name in enumerate(model_names):
        ft_model_names.append(f"{model_name}_{ft_train_name}_{n}_config_strat_{sample_str}"
                              f"fine_tuned_{energy_str}rep_{rep_num}")
        full_ft_model_names.append(f"{model_name}_{ft_train_name}_fine_tuned_"
                                   f"{energy_str}rep_{full_ft_rep_num}")

        if from_scratch_names is not None:
            rel_comp_model_names.append(f"{from_scratch_names[i]}_from_scratch_{ft_train_name}_"
                                        f"{n}_config_strat_{sample_str}{from_scratch_energy_str}rep_1")
            full_rel_comp_model_names.append(f"{from_scratch_names[i]}_{ft_train_name}_"
                                             f"from_scratch_{from_scratch_energy_str}rep_{rep_num}")
            print("Adding from scratch model as relative baseline...")
        elif zero_shot_ref_names is not None:
            rel_comp_model_names.append(zero_shot_ref_names[i])
            full_rel_comp_model_names.append(zero_shot_ref_names[i])
            print("Adding zero-shot model as relative baseline...")
        elif ft_ref_names is not None:
            rel_comp_model_names.append(f"{ft_ref_names[i]}_{ft_train_name}_{n}_config_strat_"
                                        f"{sample_str}fine_tuned_{energy_str}rep_{rep_num}")
            full_rel_comp_model_names.append(f"{ft_ref_names[i]}_{ft_train_name}_fine_tuned_"
                                             f"{energy_str}rep_{full_ft_rep_num}")
            print("Adding fine-tuned reference model as relative baseline...")
        else:
            rel_comp_model_names.append(None)
            full_rel_comp_model_names.append(None)

    # ---- atom-type subset spec ----
    atom_type_specs_raw = _need("atom_type_specs")
    if not isinstance(atom_type_specs_raw, list):
        _error("'atom_type_specs' must be a list of "
               "{config_type, react_coord_type, atom_type_ids} mappings.")
    atom_id_cap = int(input_config.get("atom_id_cap", 500))
    atom_type_specs = []
    for _spec in atom_type_specs_raw:
        if not isinstance(_spec, dict):
            _error("Each entry of 'atom_type_specs' must be a mapping with "
                   "'config_type', 'react_coord_type' and 'atom_type_ids' keys.")
        config_type = _spec.get("config_type", None)
        react_coord_type = _spec.get("react_coord_type", None)
        if react_coord_type not in (None, "RS", "TS", "PS"):
            _error(f"Unrecognised react_coord_type: {react_coord_type!r} "
                   f"(expected None / 'RS' / 'TS' / 'PS').")
        atom_type_ids = _spec.get("atom_type_ids", None)
        if atom_type_ids == "solute":
            atom_type_ids = list(range(n_solute_atoms))
        elif atom_type_ids == "solvent":
            atom_type_ids = list(range(n_solute_atoms, atom_id_cap))
        elif atom_type_ids is not None and not isinstance(atom_type_ids, list):
            _error(f"Unrecognised atom_type_ids: {atom_type_ids!r} "
                   f"(expected None / 'solute' / 'solvent' / a list of atom indices).")
        atom_type_specs.append((config_type, react_coord_type, atom_type_ids))
    subset_names = _need("subset_names")
    if len(subset_names) != len(atom_type_specs):
        _error(f"'subset_names' must have one label per entry of 'atom_type_specs' "
               f"({len(atom_type_specs)}), got {len(subset_names)}.")
    subsets_name = str(_need("subsets_name"))

    # ---- plot settings ----
    plot_settings = input_config.get("plot_settings", None) or {}
    show_legend = _as_bool(plot_settings.get("show_legend", True), default=True)
    per_element = _as_bool(plot_settings.get("per_element", True), default=True)
    save_corr_plots = _as_bool(plot_settings.get("save_corr_plots", True), default=True)
    corr_plot_alphas = plot_settings.get("corr_plot_alphas", [0.1, 0.1, 0.1])

    r_plot_settings = plot_settings.get("r_plot", None) or {}
    scatter_r_y_label = r_plot_settings.get("y_label", "Pearson's r")
    scatter_r_ylim = _parse_ylim(r_plot_settings.get("ylim", [-0.25, 1.0]))

    beta_plot_settings = plot_settings.get("beta_plot", None) or {}
    scatter_beta_y_label = beta_plot_settings.get("y_label", "Beta")
    scatter_beta_ylim = _parse_ylim(beta_plot_settings.get("ylim", None))

    mag_plot_settings = plot_settings.get("mag_plot", None) or {}
    mag_log_y = _as_bool(mag_plot_settings.get("log_y", False), default=False)
    mag_remove_negs = _as_bool(mag_plot_settings.get("remove_negs", mag_log_y),
                               default=mag_log_y)
    scatter_mag_y_label = mag_plot_settings.get("y_label", "Beta * u")
    scatter_mag_ylim = _parse_ylim(mag_plot_settings.get("ylim", None))
    if scatter_mag_ylim is None:
        scatter_mag_ylim = (1.0, 1201.0) if mag_log_y else (-200.0, 1000.0)
    return {
        "test_name": test_name,
        "test_fpath": test_fpath,
        "react_coords": react_coords,
        "ts_dist_threshold": ts_dist_threshold,
        "n_solute_atoms": n_solute_atoms,
        "model_ft_n": model_ft_n,
        "model_full_ft_n": model_full_ft_n,
        "ref_ts_config": ref_ts_config,
        "input_lot_test_fpaths": input_lot_test_fpaths,
        "model_names": model_names,
        "model_plt_labels": model_plt_labels,
        "n": n,
        "ft_train_name": ft_train_name,
        "sample_str": sample_str,
        "energy_str": energy_str,
        "rep_num": rep_num,
        "full_ft_rep_num": full_ft_rep_num,
        "from_scratch_names": from_scratch_names,
        "from_scratch_energy_str": from_scratch_energy_str,
        "zero_shot_ref_names": zero_shot_ref_names,
        "ft_ref_names": ft_ref_names,
        "rel_comp_str": rel_comp_str,
        "abs_y": abs_y,
        "ft_model_names": ft_model_names,
        "full_ft_model_names": full_ft_model_names,
        "rel_comp_model_names": rel_comp_model_names,
        "full_rel_comp_model_names": full_rel_comp_model_names,
        "atom_type_specs": atom_type_specs,
        "subset_names": subset_names,
        "subsets_name": subsets_name,
        "show_legend": show_legend,
        "per_element": per_element,
        "save_corr_plots": save_corr_plots,
        "corr_plot_alphas": corr_plot_alphas,
        "scatter_r_y_label": scatter_r_y_label,
        "scatter_r_ylim": scatter_r_ylim,
        "scatter_beta_y_label": scatter_beta_y_label,
        "scatter_beta_ylim": scatter_beta_ylim,
        "scatter_mag_y_label": scatter_mag_y_label,
        "scatter_mag_ylim": scatter_mag_ylim,
        "mag_log_y": mag_log_y,
        "mag_remove_negs": mag_remove_negs,
    }
