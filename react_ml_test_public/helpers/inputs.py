"""
Shared YAML input loading / processing for the YAML-driven workflow scripts.

Each workflow script that takes a single input YAML config file has one loader
function in this module that takes in and processes the YAML for that script:

    resolve_test_set_comparison_yaml(...)          -> run_test_set_comparison.py
    resolve_pretrain_bias_analysis_yaml(...)       -> run_pretrain_bias_analysis.py
    resolve_reaction_barrier_comparison_yaml(...)  -> run_reaction_barrier_comparison.py
    resolve_mlip_ts_opt_yaml(...)                  -> run_mlip_ts_opt.py
    resolve_mlip_md_yaml(...)                      -> run_mlip_md.py
    resolve_trajectory_analysis_yaml(...)          -> run_trajectory_analysis.py
    resolve_save_dft_inputs_yaml(...)              -> save_dft_inputs_from_xyz.py

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
    hartree_props_parser,
    orca_props_parser,
    substitute_yaml_variables,
)
from react_ml_test_public.helpers.settings import (
    DATA_DIR,
    DFT_DIR,
    KCAL_PER_MOL,
    KJ_PER_MOL,
    MEV,
    MEV_PER_ATOM,
    OPT_DIR,
    OUT_DIR,
    PAPER_MODEL_COLOURS,
    TRAJ_DIR,
    expand_env_vars,
    resolve_data_path,
    resolve_model_dir,
    resolve_out_dirs,
    resolve_out_root,
    resolve_traj_dir,
)


def _error(parser, message):
    """Report a config error via ``parser.error``, or raise ValueError when no parser."""
    if parser is not None:
        parser.error(message)
    raise ValueError(message)


# path fields whose $VAR / ${VAR} references are expanded from the environment while the
# config is loaded, so an unset one is reported as a config error (see expand_env_vars);
# path fields a script reads itself are expanded by the resolve_* helpers instead
_ENV_EXPANDED_FIELDS = ("data_dir", "out_dir", "model_dir")


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

    # environment variables are expanded after the ${...} templating above, so a
    # ${data_dir}-style reference to another field in this YAML has already been resolved
    # and is not mistaken for an unset environment variable
    for field in _ENV_EXPANDED_FIELDS:
        value = input_config.get(field)
        if value is not None:
            try:
                input_config[field] = expand_env_vars(value)
            except ValueError as exc:
                _error(parser, str(exc))
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
        "model_dir": resolve_model_dir(input_config.get("model_dir", None)),
        "cuda_device": str(input_config.get("cuda_device", "cuda:0")),
    }


def resolve_reaction_barrier_comparison_yaml(input_yaml_fpath: str, parser=None):
    """Load + process the input YAML for run_reaction_barrier_comparison.py.

    Returns a flat dict of the config variables that script's ``__main__`` declares.

    Data / output directories may be set per-YAML with the optional ``data_dir`` and ``out_dir``
    fields, which default to ``DATA_DIR`` / ``OUT_DIR`` from settings.py. A relative
    ``test_trj_fpath`` that does not resolve from the current working directory is taken to be
    relative to ``data_dir``, and plots / CSVs are written under ``out_dir`` (see
    ``resolve_out_dirs``).
    """
    input_config = _load_yaml_config(input_yaml_fpath, parser)

    # ===== data / output directories =====
    # defaults come from settings.py, so a YAML only needs to set these when it should read /
    # write somewhere other than the DATA_DIR / OUT_DIR the run was started from
    data_dir = input_config.get("data_dir", DATA_DIR)
    out_dir, plot_dir, csv_dir = resolve_out_dirs(input_config.get("out_dir", OUT_DIR))
    model_dir = resolve_model_dir(input_config.get("model_dir", None))

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
    # run_names defaults to [''] - a single unnamed run, in which case the models are
    # taken from model_names as listed. A named run instead triggers the discovery of
    # that run's models by glob, as selected by run_type.
    run_names = input_config.get("run_names", None)
    if not run_names:
        run_names = ['']
    run_type = input_config.get("run_type", None)
    from_scratch_name = input_config.get("from_scratch_name", None)

    # ===== IRC trajectory comparison =====
    test_trj_fpath = resolve_data_path(input_config.get("test_trj_fpath", None), data_dir)
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
        "hartree_props_parser": hartree_props_parser,
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
        "model_dir": model_dir,
        "model_names": model_names,
        "save_name": save_name,
        "run_names": run_names,
        "run_type": run_type,
        "from_scratch_name": from_scratch_name,
        "num_base_models": num_base_models,
        "num_base_model_comps": num_base_model_comps,
        "alpha_list": alpha_list,
        "linestyle_list": linestyle_list,
        "data_dir": data_dir,
        "out_dir": out_dir,
        "plot_dir": plot_dir,
        "csv_dir": csv_dir,
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
        "cuda_device": str(input_config.get("cuda_device", "cuda:0")),
    }


def resolve_pretrain_bias_analysis_yaml(input_yaml_fpath: str, parser=None):
    """Resolve and validate a run_pretrain_bias_analysis YAML config into a flat dict of
    variables matching the script's ``__main__`` expectations.

    ``parser`` (optional argparse.ArgumentParser) reports errors the same way as the other
    YAML-driven scripts; when None a ValueError is raised instead (useful for testing).
    """
    input_config = _load_yaml_config(input_yaml_fpath, parser)
    model_dir = resolve_model_dir(input_config.get("model_dir", None))

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
        "model_dir": model_dir,
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
        "cuda_device": str(input_config.get("cuda_device", "cuda:0")),
    }


def resolve_mlip_ts_opt_yaml(input_yaml_fpath: str, parser=None):
    """Load + process the input YAML for run_mlip_ts_opt.py.

    Returns a flat dict of the config variables that script's ``__main__`` declares.
    """
    input_config = _load_yaml_config(input_yaml_fpath, parser)

    def _need(key):
        value = input_config.get(key, None)
        if value is None:
            _error(parser, f"Missing required field '{key}' in input YAML.")
        return value

    # ===== TS guess (input config) =====
    save_name = str(_need("save_name"))
    react_coords = [tuple(pair) for pair in _need("react_coords")]
    num_solute_atoms = int(_need("num_solute_atoms"))
    input_config_fp = str(_need("input_config_fp"))
    ref_config_fp = input_config.get("ref_config_fp", None)
    reverse_displace = _as_bool(input_config.get("reverse_displace", False), default=False)

    # ===== model / fine-tuning selection =====
    model_names = _need("model_names")
    if not isinstance(model_names, list) or not model_names:
        _error(parser, "'model_names' must be a non-empty list of model names in the input YAML.")
    if any("${" in str(model_name) for model_name in model_names):
        _error(parser, "Unresolved '${var}' reference(s) in 'model_names'; define the "
                       "referenced variable(s) in the input YAML.")
    system_ft_name = str(input_config.get("system_ft_name", None) or "")
    n = input_config.get("n", None)
    n = int(n) if n is not None else None
    sample_str = str(input_config.get("sample_str", "") or "")
    energy_str = str(input_config.get("energy_str", "energy_") or "")
    rep_num = int(input_config.get("rep_num", 1))
    omol_rep_num = int(input_config.get("omol_rep_num", 1))
    omol_energy_str = str(input_config.get("omol_energy_str", "energy_") or "")

    # ===== solvent (optional, off by default) =====
    include_solvent = _as_bool(input_config.get("include_solvent", False), default=False)
    if include_solvent:
        solvent_xyz_fpath = str(_need("solvent_xyz_fpath"))
        solvent_name = str(_need("solvent_name"))
        solvent_density = float(_need("solvent_density"))
        atoms_per_solv_mol = int(_need("atoms_per_solv_mol"))
        solvation_box_size = float(_need("solvation_box_size"))
    else:
        solvent_xyz_fpath = None
        solvent_name = None
        solvent_density = None
        atoms_per_solv_mol = None
        solvation_box_size = None
    solv_shell_cutoff = float(input_config.get("solv_shell_cutoff", 0.0))

    # ===== optimiser / run settings =====
    out_dir = input_config.get("out_dir", None) or OPT_DIR
    model_dir = resolve_model_dir(input_config.get("model_dir", None))
    fmax = float(input_config.get("fmax", 0.001))
    displace_mag = float(input_config.get("displace_mag", 0.05))
    enable_cueq = _as_bool(input_config.get("enable_cueq", False), default=False)
    override_cache = _as_bool(input_config.get("override_cache", False), default=False)
    cuda_device = str(input_config.get("cuda_device", "cuda:0"))

    return {
        "save_name": save_name,
        "react_coords": react_coords,
        "num_solute_atoms": num_solute_atoms,
        "input_config_fp": input_config_fp,
        "ref_config_fp": ref_config_fp,
        "reverse_displace": reverse_displace,
        "model_dir": model_dir,
        "model_names": model_names,
        "system_ft_name": system_ft_name,
        "n": n,
        "sample_str": sample_str,
        "energy_str": energy_str,
        "rep_num": rep_num,
        "omol_rep_num": omol_rep_num,
        "omol_energy_str": omol_energy_str,
        "include_solvent": include_solvent,
        "solvent_xyz_fpath": solvent_xyz_fpath,
        "solvent_name": solvent_name,
        "solvent_density": solvent_density,
        "atoms_per_solv_mol": atoms_per_solv_mol,
        "solvation_box_size": solvation_box_size,
        "solv_shell_cutoff": solv_shell_cutoff,
        "out_dir": out_dir,
        "fmax": fmax,
        "displace_mag": displace_mag,
        "enable_cueq": enable_cueq,
        "override_cache": override_cache,
        "cuda_device": cuda_device,
    }


def resolve_mlip_md_yaml(input_yaml_fpath: str, parser=None):
    """Load + process the input YAML for run_mlip_md.py.

    The YAML specifies the test system / starting configs, MD settings, the model list
    (full model names, supporting ``${var}`` templating from the fine-tuning naming
    components so zero-shot and fine-tuned models can be mixed) and the remaining run
    options. The script builds the ``formatted_model_names`` (trajectory-name labels)
    from these. Returns a flat dict of the config variables that the script's
    ``__main__`` declares.

    Data / output directories may be set per-YAML with the optional ``data_dir`` and ``out_dir``
    fields, which default to ``DATA_DIR`` / ``OUT_DIR`` from settings.py. A relative
    ``init_configs_fpath`` that does not resolve from the current working directory is taken to
    be relative to ``data_dir``, and trajectories are written under ``out_dir`` (see
    ``resolve_traj_dir``, i.e. ``<out_dir>/trj``).
    """
    input_config = _load_yaml_config(input_yaml_fpath, parser)

    def _need(key):
        value = input_config.get(key, None)
        if value is None:
            _error(parser, f"Missing required field '{key}' in input YAML.")
        return value

    def _parse_box_dim(value):
        if value is None:
            return None
        try:
            value = [float(x) for x in value]
        except (TypeError, ValueError):
            _error(parser, "'box_dim' must be a list of 3 box lengths in Angstrom.")
        if len(value) != 3:
            _error(parser, "'box_dim' must be a list of 3 box lengths in Angstrom.")
        return value

    # ===== data / output directories =====
    # defaults come from settings.py, so a YAML only needs to set these when it should read /
    # write somewhere other than the DATA_DIR / OUT_DIR the run was started from
    data_dir = input_config.get("data_dir", DATA_DIR)
    out_dir = resolve_out_root(input_config.get("out_dir", OUT_DIR))
    traj_dir = resolve_traj_dir(out_dir)
    model_dir = resolve_model_dir(input_config.get("model_dir", None))

    # ===== system / starting configs =====
    # init_configs_fpath is optional: when unset, each model's optimised TS (out/opt) is used
    init_configs_fpath = input_config.get("init_configs_fpath", None)
    if init_configs_fpath is not None:
        init_configs_fpath = resolve_data_path(str(init_configs_fpath), data_dir)
        if not os.path.isfile(init_configs_fpath):
            _error(parser, f"init_configs_fpath not found: {init_configs_fpath!r} "
                           f"(relative paths are taken as relative to data_dir: {data_dir!r}).")
    system_name = str(_need("system_name"))
    include_solvent = _as_bool(input_config.get("include_solvent", False), default=False)
    if include_solvent:
        solvent_name = str(_need("solvent_name"))
    else:
        solvent_name = None

    # ===== MD settings =====
    fs = int(input_config.get("fs", 1000))
    temp = int(input_config.get("temp", 300))
    dt = float(input_config.get("dt", 0.5))
    interval = int(input_config.get("interval", 20))
    md_method = str(input_config.get("md_method", "ase"))
    if md_method not in ("ase", "openmm"):
        _error(parser, f"Unrecognised md_method: {md_method!r} (choose from 'ase' / 'openmm').")
    num_init_configs = int(input_config.get("num_init_configs", 1))
    md_runs_per_config = int(input_config.get("md_runs_per_config", 1))
    box_dim = _parse_box_dim(input_config.get("box_dim", None))

    # ===== models (explicit model_names list + fine-tuning naming components) =====
    model_names = _need("model_names")
    if not isinstance(model_names, list) or not model_names:
        _error(parser, "'model_names' must be a non-empty list of model names in the input YAML.")
    if any("${" in str(model_name) for model_name in model_names):
        _error(parser, "Unresolved '${var}' reference(s) in 'model_names'; define the "
                       "referenced variable(s) in the input YAML.")
    system_ft_name = str(input_config.get("system_ft_name", "") or "")
    n = input_config.get("n", None)
    n = int(n) if n is not None else None
    sample_str = str(input_config.get("sample_str", "") or "")
    energy_str = str(input_config.get("energy_str", "energy_") or "")
    rep_num = int(input_config.get("rep_num", 1))
    omol_rep_num = int(input_config.get("omol_rep_num", 1))
    omol_energy_str = str(input_config.get("omol_energy_str", "energy_") or "")

    # ===== run options =====
    enable_cueq = _as_bool(input_config.get("enable_cueq", False), default=False)
    override_saved = _as_bool(input_config.get("override_saved", False), default=False)
    cuda_device = str(input_config.get("cuda_device", "cuda:0"))

    return {
        "data_dir": data_dir,
        "out_dir": out_dir,
        "traj_dir": traj_dir,
        "init_configs_fpath": init_configs_fpath,
        "system_name": system_name,
        "include_solvent": include_solvent,
        "solvent_name": solvent_name,
        "fs": fs,
        "temp": temp,
        "dt": dt,
        "interval": interval,
        "md_method": md_method,
        "num_init_configs": num_init_configs,
        "md_runs_per_config": md_runs_per_config,
        "box_dim": box_dim,
        "model_dir": model_dir,
        "model_names": model_names,
        "system_ft_name": system_ft_name,
        "n": n,
        "sample_str": sample_str,
        "energy_str": energy_str,
        "rep_num": rep_num,
        "omol_rep_num": omol_rep_num,
        "omol_energy_str": omol_energy_str,
        "enable_cueq": enable_cueq,
        "override_saved": override_saved,
        "cuda_device": cuda_device,
    }


def resolve_trajectory_analysis_yaml(input_yaml_fpath: str, parser=None):
    """Load + process the input YAML for run_trajectory_analysis.py.

    The YAML specifies the trajectory set to analyse (from run_mlip_md.py output), the
    reaction-coordinate atom pairs + product / reactant ``config_defs`` used to classify
    final frames, the model labels (trajectory-name stubs) to plot and the plot settings.
    Returns a flat dict of the config variables that the script's ``__main__`` declares.
    """
    input_config = _load_yaml_config(input_yaml_fpath, parser)

    def _need(key):
        value = input_config.get(key, None)
        if value is None:
            _error(parser, f"Missing required field '{key}' in input YAML.")
        return value

    def _parse_pair(key):
        value = _need(key)
        try:
            pair = tuple(int(x) for x in value)
        except (TypeError, ValueError):
            _error(parser, f"'{key}' must be a pair of atom indices [i, j].")
        if len(pair) != 2:
            _error(parser, f"'{key}' must be a pair of atom indices [i, j].")
        return pair

    def _parse_range(value, key="traj_trunc / ranges"):
        if value is None:
            return None
        try:
            value = [float(x) for x in value]
        except (TypeError, ValueError):
            _error(parser, f"'{key}' must be a [lo, hi] pair.")
        if len(value) != 2:
            _error(parser, f"'{key}' must be a [lo, hi] pair.")
        return tuple(value)

    # ===== trajectory source (run_mlip_md.py output) =====
    traj_dir = str(input_config.get("traj_dir", "") or TRAJ_DIR)
    traj_type = str(_need("traj_type"))
    run_name = str(_need("run_name"))
    ts_ref_fp = str(_need("ts_ref_fp"))

    # ===== reaction coordinate / config definitions =====
    num_solute_atoms = int(_need("num_solute_atoms"))
    threshold = float(_need("threshold"))
    r1_atoms = _parse_pair("r1_atoms")
    r2_atoms = _parse_pair("r2_atoms")
    r3_atoms = _parse_pair("r3_atoms") if input_config.get("r3_atoms", None) is not None else None
    r_atom_pairs = {"r1": r1_atoms, "r2": r2_atoms, "r3": r3_atoms}

    config_defs_raw = _need("config_defs")
    if not isinstance(config_defs_raw, list) or not config_defs_raw:
        _error(parser, "'config_defs' must be a non-empty list of reactant / product "
                       "config definitions (each with a 'name' and 'r1'/'r2'/'r3' flags).")
    config_defs = []
    for spec in config_defs_raw:
        if not isinstance(spec, dict) or not spec.get("name"):
            _error(parser, "Each entry of 'config_defs' must be a mapping with a 'name' "
                           "and 'r1'/'r2'/'r3' bond flags.")
        name = str(spec["name"])
        bonds = []
        for r_key, r_pair in r_atom_pairs.items():
            if spec.get(r_key) is None:
                continue
            if r_pair is None:
                _error(parser, f"config_def '{name}' references '{r_key}' but "
                               f"'{r_key}_atoms' is not set in the input YAML.")
            bonds.append((r_pair, bool(spec[r_key])))
        if not bonds:
            _error(parser, f"config_def '{name}' must define at least one bond flag.")
        config_defs.append({"name": name, "bonds": bonds})

    # ===== models (trajectory-name labels, supporting ${var} templating) =====
    model_names = _need("model_names")
    if not isinstance(model_names, list) or not model_names:
        _error(parser, "'model_names' must be a non-empty list of trajectory model labels "
                       "in the input YAML.")
    if any("${" in str(model_name) for model_name in model_names):
        _error(parser, "Unresolved '${var}' reference(s) in 'model_names'; define the "
                       "referenced variable(s) in the input YAML.")
    system_ft_name = str(input_config.get("system_ft_name", "") or "")
    n = input_config.get("n", None)
    n = int(n) if n is not None else None
    sample_str = str(input_config.get("sample_str", "") or "")
    energy_str = str(input_config.get("energy_str", "energy_") or "")
    omol_energy_str = str(input_config.get("omol_energy_str", "energy_") or "")

    # ===== plot settings =====
    save_name = str(_need("save_name"))
    traj_trunc = _parse_range(input_config.get("traj_trunc", None))
    r1_range = _parse_range(input_config.get("r1_range", None))
    r2_range = _parse_range(input_config.get("r2_range", None))
    show_legend = _as_bool(input_config.get("show_legend", False), default=False)

    return {
        "traj_dir": traj_dir,
        "traj_type": traj_type,
        "run_name": run_name,
        "ts_ref_fp": ts_ref_fp,
        "num_solute_atoms": num_solute_atoms,
        "threshold": threshold,
        "r1_atoms": r1_atoms,
        "r2_atoms": r2_atoms,
        "r3_atoms": r3_atoms,
        "config_defs": config_defs,
        "model_names": model_names,
        "system_ft_name": system_ft_name,
        "n": n,
        "sample_str": sample_str,
        "energy_str": energy_str,
        "omol_energy_str": omol_energy_str,
        "save_name": save_name,
        "traj_trunc": traj_trunc,
        "r1_range": r1_range,
        "r2_range": r2_range,
        "show_legend": show_legend,
    }


# The keys of ORCA's %basis block, spelled as ORCA documents them (see the "Choice of
# Basis Set" section of the manual); a key given in any other case is normalised onto
# these, so "auxj" in a YAML becomes "AuxJ" in the generated input
_ORCA_BASIS_BLOCK_KEYS = (
    "Basis", "AuxJ", "AuxJK", "AuxC", "CABS",
    "NewGTO", "NewECP", "NewAuxGTO",
    "Decontract", "DecontractBas", "DecontractAuxJ", "DecontractAuxJK",
    "DecontractAuxC", "DecontractCABS",
)
_ORCA_BASIS_BLOCK_KEYS_BY_LOWER = {key.lower(): key for key in _ORCA_BASIS_BLOCK_KEYS}


def _orca_basis_pair(key, value, parser=None):
    """Normalise one ORCA ``%basis`` block key / value pair from the input YAML."""
    key = str(key).strip()
    if len(key.split()) != 1:
        _error(parser, f"'basis' block keys must be a single word, got: {key!r}.")
    # normalise a known key onto ORCA's own spelling (e.g. "auxj" -> "AuxJ")
    key = _ORCA_BASIS_BLOCK_KEYS_BY_LOWER.get(key.lower(), key)

    # strip the quotation marks ORCA takes around a basis set name, if the YAML has them
    value = str(value).strip()
    if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    if not value:
        _error(parser, f"Missing value for the 'basis' block key {key!r}.")

    return key, value


def _parse_orca_basis_entry(entry, parser=None):
    """Parse one ``basis`` field entry into ORCA ``%basis`` block ``(key, value)`` pairs.

    An entry is either a mapping of block keys to their values, or a string of a block
    key and its value, e.g. ``'AuxJ "aug-cc-pVTZ/JK"'`` - a string holding only a basis
    set name is taken as the orbital ``Basis``, so ``basis: def2-TZVPPD`` also works.
    """
    if isinstance(entry, dict):
        return [_orca_basis_pair(key, value, parser) for key, value in entry.items()]

    if isinstance(entry, str):
        parts = entry.strip().split(None, 1)
        if not parts:
            _error(parser, "'basis' entries must not be empty.")
        # a bare basis set name sets the orbital basis set
        key, value = ("Basis", parts[0]) if len(parts) == 1 else tuple(parts)
        return [_orca_basis_pair(key, value, parser)]

    _error(parser, f"'basis' entries must be strings or mappings of block keys to values, "
                   f"got a {type(entry).__name__}: {entry!r}.")


def _parse_orca_basis(basis_config, parser=None):
    """Parse the optional ``basis`` field of the save_dft_inputs_from_xyz YAML.

    Returns a list of ORCA ``%basis`` block ``(key, value)`` pairs (see
    ``helpers.io.format_orca_basis_block``), or None when the field is unset, in which
    case no block is written into the generated input.

    Accepted spellings, as a single entry or as a list of them:

        basis: AuxJ "aug-cc-pVTZ/JK"        # block key + basis set name
        basis: "AuxJ aug-cc-pVTZ/JK"        # the same, quoted as a whole
        basis: def2-TZVPPD                  # no key: the orbital (Basis) set
        basis:                              # a mapping of as many keys as needed
          Basis: def2-TZVPPD
          AuxJ: "aug-cc-pVTZ/JK"
    """
    if basis_config is None:
        return None

    entries = basis_config if isinstance(basis_config, list) else [basis_config]
    if not entries:
        _error(parser, "'basis' must not be an empty list.")

    basis_entries = []
    for entry in entries:
        basis_entries.extend(_parse_orca_basis_entry(entry, parser))
    return basis_entries


def resolve_save_dft_inputs_yaml(input_yaml_fpath: str, parser=None):
    """Load + process the input YAML for save_dft_inputs_from_xyz.py.

    The YAML specifies the .xyz configuration file to convert, the level of theory
    (``method: orca`` / ``vasp``) with that method's settings, and where the input files
    are written. Returns a flat dict of the config variables that the script's ``__main__``
    declares.

    A relative ``input_fpath`` that does not resolve from the current working directory is
    taken to be relative to the optional ``data_dir`` field (default ``DATA_DIR``), so it
    can be spelled from either the repository root or the data root, and ``out_dir`` (which
    defaults to ``DFT_DIR``, i.e. ``out/dft``) is the directory the per-config input
    directories are created under. The ORCA settings (``n_cores``, ``orca_keywords``,
    ``max_core``) are returned for the script to apply to the mlptrain / autode configs,
    which is where the input-file generation reads them from; each defaults to None, which
    leaves that config value alone.

    The optional ``basis`` field holds ORCA block input that the ``orca_keywords`` simple
    input line cannot express - most usefully the auxiliary basis slots of ``%basis`` -
    and is returned as its ``(key, value)`` pairs for the script to write into each
    generated input file (see ``_parse_orca_basis``).
    """
    input_config = _load_yaml_config(input_yaml_fpath, parser)

    def _need(key):
        value = input_config.get(key, None)
        if value is None:
            _error(parser, f"Missing required field '{key}' in input YAML.")
        return value

    # ===== method / output directory =====
    method = str(_need("method")).lower()
    if method not in ("orca", "vasp"):
        _error(parser, f"Unrecognised method: {method!r} (choose from 'orca' / 'vasp').")
    out_dir = resolve_out_root(input_config.get("out_dir", None) or DFT_DIR)

    # ===== input config file =====
    data_dir = input_config.get("data_dir", DATA_DIR)
    input_fpath = resolve_data_path(str(_need("input_fpath")), data_dir)
    if not os.path.isfile(input_fpath):
        _error(parser, f"input_fpath not found: {input_fpath!r} "
                       f"(relative paths are taken as relative to data_dir: {data_dir!r}).")

    # ===== ORCA settings =====
    if method == "orca":
        # the charge / multiplicity of every config in the .xyz, as ORCA input needs them
        charge = int(_need("charge"))
        mult = int(_need("mult"))
        # block input ORCA needs but mlptrain's orca_keywords (simple input) cannot hold,
        # written into each generated input as a %basis block (see the script)
        basis = _parse_orca_basis(input_config.get("basis", None), parser)
    else:
        charge, mult, basis = None, None, None
        if input_config.get("basis", None) is not None:
            _error(parser, "'basis' is only used by the ORCA method, but the input YAML "
                           f"sets it with method: {method!r}.")

    n_cores = input_config.get("n_cores", None)
    n_cores = int(n_cores) if n_cores is not None else None
    orca_keywords = input_config.get("orca_keywords", None)
    if orca_keywords is not None and not isinstance(orca_keywords, list):
        _error(parser, "'orca_keywords' must be a list of ORCA keyword strings, "
                       "e.g. [\"WB97M-D3BJ\", \"def2-TZVPPD\", \"TightSCF\", \"EnGrad\"].")
    max_core = input_config.get("max_core", None)
    max_core = float(max_core) if max_core is not None else None
    extra_name_tag = input_config.get("extra_name_tag", None)
    extra_name_tag = str(extra_name_tag) if extra_name_tag is not None else None

    # ===== VASP settings =====
    if method == "vasp":
        box_size = float(_need("box_size"))
        incar_settings = input_config.get("incar_settings", None) or {}
        if not isinstance(incar_settings, dict):
            _error(parser, "'incar_settings' must be a mapping of INCAR parameters, "
                           "e.g. {ENCUT: 550, EDIFF: 1E-5}.")
    else:
        box_size, incar_settings = None, None

    return {
        "input_fpath": input_fpath,
        "charge": charge,
        "mult": mult,
        "method": method,
        "out_dir": out_dir,
        "n_cores": n_cores,
        "orca_keywords": orca_keywords,
        "basis": basis,
        "max_core": max_core,
        "extra_name_tag": extra_name_tag,
        "box_size": box_size,
        "incar_settings": incar_settings,
    }
