"""
Analyse the spread of one or more datasets about a reaction coordinate.

Plots SOAP-descriptor PCA projections, absolute / relative energy distributions, force
distributions and the reaction-coordinate (r1 / r2) distributions of each dataset, so
dataset subsets (e.g. train / valid / test, or vacuum / solvated) can be compared.
"""
import argparse
import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from react_ml_test_public.helpers import inputs as rmlt_inputs
from react_ml_test_public.helpers.descriptors import get_soap_descriptors
from react_ml_test_public.helpers.dim_red import run_pca
from react_ml_test_public.helpers.io import (
    flexible_ase_read,
    get_ase_atoms_energy,
    get_ase_atoms_forces,
    get_name_from_path,
)
from react_ml_test_public.helpers.logging import logger
from react_ml_test_public.helpers.plot import plot_2d
from react_ml_test_public.helpers.settings import (
    DPI,
    EV,
    KCAL_PER_MOL,
    MEV_PER_ATOM,
    SINGLE_COLUMN_FIG_SIZE,
    PlotSettings,
    eV_to_kcal_mol,
    eV_to_mEV,
)
from react_ml_test_public.helpers.utils import convert_units, get_ase_props


if __name__ == "__main__":

    """Analyse one or more datasets about a reaction coordinate.

    The path to the input YAML file is given as a single command-line argument:

        rmlt run-ds-analysis inputs/dataset/guajadial_DA_train_subsets.yaml

    Input YAML format (see inputs/dataset/guajadial_DA_train_subsets.yaml for a full
    example):

    Fields (path fields may reference environment variables, e.g. out_dir: "$DATA/out"):
        data_dir:              # data root directory for relative dataset / TS paths
                               # (default DATA_DIR from settings.py).
        out_dir:               # output root directory, i.e. plots go to <out_dir>/plt
                               # (default OUT_DIR from settings.py).
        input_ds_fps:          # list of dataset .xyz files to analyse, plotted alongside
                               # each other.
        dataset_labels:        # list of legend labels, one per dataset (defaults to the
                               # file name of each).
        custom_xyz_parsers:    # optional per-dataset config-property parsers, one entry
                               # per dataset, each "orca_props_parser" /
                               # "autode_props_parser" / "hartree_props_parser" or null
                               # for the default parsing (default: null for every dataset).
        filter_config_type:    # only analyse configs of this config_type, i.e. 'vacuum' /
                               # 'solvated' (default None = no filtering).
        ts_ref_fpath:          # optional path to the DFT reference TS the reaction-coordinate
                               # distances are measured from; without it the TS plots and the
                               # TS-distance histogram are skipped (default None).
        react_coords:          # list of reactive atom pairs [[i, j], ...] defining the
                               # r1 / r2 distances.
        calc_soap:             # whether to calculate SOAP descriptors and plot their PCA
                               # projections (needs dscribe; default false).
        test_elements:         # list of element symbols the SOAP descriptors are
                               # calculated for, e.g. [C, H, O] (default: every element).
        include_energies:      # whether to calculate / plot energies (default true).
        include_forces:        # whether to calculate / plot forces (default true).
        energy_units:          # "meV / atom" or "kcal / mol" (default "meV / atom").
        save_name:             # base name of the output plots under <out_dir>/plt.
        plot_2d_xlim:          # optional [lo, hi] r1 (x) plot limits.
        plot_2d_ylim:          # optional [lo, hi] r2 (y) plot limits.

    Any omitted field is set to None, or the documented default.
    """

    # input YAML config path from a single command-line argument
    parser = argparse.ArgumentParser(
        description="Analyse one or more datasets about a reaction coordinate, driven by an "
                    "input YAML config file (format described above).",
        epilog="Example: rmlt run-ds-analysis inputs/dataset/guajadial_DA_train_subsets.yaml",
    )
    parser.add_argument(
        "input_yaml",
        help="Path to the input YAML config file "
             "(e.g. inputs/dataset/guajadial_DA_train_subsets.yaml).",
    )
    args = parser.parse_args()

    # load + process the input YAML (see react_ml_test_public/helpers/inputs.py)
    resolved = rmlt_inputs.resolve_ds_analysis_yaml(args.input_yaml, parser)

    # ===== data / output directories =====
    data_dir = resolved["data_dir"]
    out_dir = resolved["out_dir"]
    plot_dir = resolved["plot_dir"]
    logger.info(f"Using data dir: {data_dir}, out dir: {out_dir} (plots: {plot_dir})...")

    # ===== datasets to analyse =====
    input_ds_fps = resolved["input_ds_fps"]
    dataset_labels = resolved["dataset_labels"]
    custom_xyz_parsers = resolved["custom_xyz_parsers"]
    filter_config_type = resolved["filter_config_type"]

    # ===== reaction coordinate reference =====
    ts_ref_fpath = resolved["ts_ref_fpath"]
    react_coords = resolved["react_coords"]
    test_elements = resolved["test_elements"]

    # ===== what to calculate =====
    include_energies = resolved["include_energies"]
    include_forces = resolved["include_forces"]
    energy_units = resolved["energy_units"]
    calc_soap = resolved["calc_soap"]

    # ===== plot settings =====
    save_name = resolved["save_name"]
    plot_2d_xlim = resolved["plot_2d_xlim"]
    plot_2d_ylim = resolved["plot_2d_ylim"]
    del resolved

    dpi = DPI
    fig_size = SINGLE_COLUMN_FIG_SIZE
    os.makedirs(f"{plot_dir}/2D", exist_ok=True)
    os.makedirs(f"{plot_dir}/ds_analysis", exist_ok=True)

    input_ds_names = [get_name_from_path(input_ds_fp) for input_ds_fp in input_ds_fps]

    # load ts
    if ts_ref_fpath is not None:
        ts_ref_config = flexible_ase_read(ts_ref_fpath)[0]
        ts_ref_bond_distances = {ri: ts_ref_config.get_distance(*ri) for ri in react_coords}
        ts_ref_bond_dist_tuple = tuple([dist for dist in ts_ref_bond_distances.values()])
        logger.info(f'TS ref bond distances: {ts_ref_bond_distances}')
    else:
        logger.info('TS ref path not provided. Skipping plotting of TS distance distributions.')


    all_configs, all_soap_desc_2D, all_energies, all_energy_deltas, all_force_deltas, all_force_norms, all_force_comps, all_r_coord_dists = [], [], [], [], [], [], [], []
    for i, (input_ds_fp, input_ds_name) in enumerate(zip(input_ds_fps, input_ds_names)):

        logger.info(f"Dataset Name: {input_ds_name}")

        # load train configs
        configs = flexible_ase_read(input_ds_fp, cust_props_parser=custom_xyz_parsers[i] if custom_xyz_parsers is not None else None)

        # === FILTER CONFIG TYPES (filter_config_type in the input YAML) ===
        if filter_config_type is None:
            filter_ids = [config_id for config_id, config in enumerate(configs)]                                    # no filter
        else:
            filter_ids = [config_id for config_id, config in enumerate(configs) if config.info.get('config_type') == filter_config_type]    # this config type only
        # ==============================

        configs = [configs[config_id] for config_id in filter_ids]
        logger.info(f'Num Filtered Configs: {len(configs)}')
        all_configs.append(configs)

        config_types = [config.info.get('config_type') for config in configs]
        atom_nums = [len(config) for config in configs]
        atom_num_counts, type_counts = {}, {}
        for config_type in config_types:
            type_counts[config_type] = type_counts.get(config_type, 0) + 1
        for atom_num in atom_nums:
            atom_num_counts[atom_num] = atom_num_counts.get(atom_num, 0) + 1

        logger.info(f'Num Configs in Dataset: {len(configs)}')
        logger.info(f'Num Atoms in Dataset: {sum([len(config) for config in configs])}')
        logger.info(f"Atom Num Counts: {atom_num_counts}")
        logger.info(f'Config Types in Dataset: {set(config_types)}')
        logger.info(f"Config Type Counts: {type_counts}")

        if calc_soap:
            # 1. get SOAP descriptors of training set
            start_time = time()
            average_soap = get_soap_descriptors(configs,
                                                specific_elements=test_elements,
                                                average='outer',
                                                compress_dim=None,
                                                flatten_output=True,
                                                cache_name=input_ds_name)
            logger.info(f"Loaded / calculated {len(configs)} average train SOAP descriptors in {time() - start_time:.4f}s...")
            soap_desc_2D = run_pca(average_soap, n_components=2)
            all_soap_desc_2D.append(soap_desc_2D)

        # configs grouped by atom count, i.e. by the species count that their relative
        # energies and forces are measured against
        atom_num_groups = {}
        for config in configs:
            if atom_num_groups.get(len(config)) is None:
                atom_num_groups[len(config)] = [config]
            else:
                atom_num_groups[len(config)].append(config)

        if include_energies:

            # get absolute energies
            energies = np.array([get_ase_atoms_energy(config) for config in configs])
            # print('Energies: ', energies)
            all_energies.append(energies)

            # 2. get relative energies
            atom_num_group_energy_mins = {num_atoms: np.min(get_ase_props(group_configs, prop='energy'))
                                            for num_atoms, group_configs in atom_num_groups.items()}
            energy_deltas = np.array([abs(get_ase_atoms_energy(config) - atom_num_group_energy_mins[len(config)])
                            for config in configs])
            # print('Energy deltas: ', energy_deltas)
            all_energy_deltas.append(energy_deltas)

        if include_forces:

            atom_num_group_force_mins = {num_atoms: np.min([np.mean(np.linalg.norm(config_forces, axis=0) * eV_to_mEV)
                                                            for config_forces in get_ase_props(group_configs, prop='forces')])
                                        for num_atoms, group_configs in atom_num_groups.items()}
            force_deltas = [abs(np.mean(np.linalg.norm(get_ase_props([config], prop='forces'), axis=0) * eV_to_mEV) - atom_num_group_force_mins[len(config)]) for config in configs]
            all_force_deltas.append(force_deltas)

            # 4. get force norms
            force_norms = np.array([np.linalg.norm(force_vec, axis=0) * eV_to_mEV for config in configs for force_vec in get_ase_atoms_forces(config)])
            all_force_norms.append(force_norms)

            # 5. get force components
            force_comps = np.concatenate([force_vec.flatten() * eV_to_mEV for config in configs for force_vec in get_ase_atoms_forces(config)])
            logger.info(f"Force Comps Shape: {force_comps.shape}")
            all_force_comps.append(force_comps)

        # 6. get distances from TS
        r_coord_dists = [[config.get_distance(*ri, mic=True) for ri in react_coords] for config in configs]
        all_r_coord_dists.append(r_coord_dists)

        logger.info("============================================")

    # apply unit conversions for energy
    if include_energies:
        if energy_units == KCAL_PER_MOL:
            all_energies = [energies * eV_to_kcal_mol  for energies in all_energies]
            all_energy_deltas = [energies * eV_to_kcal_mol  for energies in all_energy_deltas]
            print('Converted Energies: ', all_energies)
            print('Converted Energy deltas: ', all_energy_deltas)
        elif energy_units == MEV_PER_ATOM:
            all_energies = [convert_units(energies, EV, MEV_PER_ATOM, [len(config) for config in all_configs[i]])  for i, energies in enumerate(all_energies)]
            all_energy_deltas = [convert_units(energies, EV, MEV_PER_ATOM, [len(config) for config in all_configs[i]])  for i, energies in enumerate(all_energy_deltas)]

    if not include_energies:
        all_energy_deltas = [None for _ in range(len(input_ds_names))]

    if calc_soap:
        # 1. PCA SOAP plot
        for input_ds_name, soap_desc_2D, energy_deltas, configs in zip(input_ds_names, all_soap_desc_2D, all_energy_deltas, all_configs):

            plot_settings = PlotSettings(x_label='PC1',
                                        y_label='PC2',
                                        colour_bar_label=f'∆ Energy ({energy_units})',
                                        title=input_ds_name,
                                        dpi=dpi,
                                        fig_size=fig_size,
                                        show_legend=False)

            # concat_pca_soap_descs = np.concatenate(train_config_pca_soap_descs, axis=0)
            soap_pca_1, soap_pca_2 = soap_desc_2D[:, 0], soap_desc_2D[:, 1]

            plot_2d([soap_desc_2D],
                f"{plot_dir}/2D/{input_ds_name}_soap_PCA.png",
                colour_maps=[energy_deltas] if energy_deltas is not None else None,
                plot_settings=plot_settings,
                )

            # if include_energies:
            #     properties = {"PCA-1": soap_pca_1, "PCA-2": soap_pca_2, "score": energy_deltas}
            #     widget = chemiscope.show(configs, properties=properties)
            #     widget.save(f"out/chemiscope/{input_ds_name}_soap_pca.json")

        # 1.2 combined PCA plot
        plot_settings = PlotSettings(x_label='PC1',
                                    y_label='PC2',
                                    title=input_ds_name,
                                    dpi=dpi,
                                    fig_size=fig_size,
                                    show_legend=False)

        plot_2d(all_soap_desc_2D,
                f"{plot_dir}/ds_analysis/{save_name}_2D_soap_PCA.png",
                plot_settings=plot_settings)

    # 2. plot energies
    plt.figure(figsize=fig_size, dpi=dpi)
    for input_ds_name, label, energies, configs in zip(input_ds_names, dataset_labels, all_energies, all_configs):

        indices = np.arange(0, len(configs))
        plt.plot(indices, energies, label=label, linewidth=0.6)

    plt.legend(loc='upper right')
    plt.xlabel('Index')
    plt.ylabel(f'Energy ({energy_units})')
    plt.tight_layout()

    # plot relative energies
    if include_energies:
        plt.figure(figsize=fig_size, dpi=dpi)
        for input_ds_name, label, energy_deltas, configs in zip(input_ds_names, dataset_labels, all_energy_deltas, all_configs):

            indices = np.arange(0, len(configs))
            # plt.scatter(indices, energy_deltas, s=1)
            plt.plot(indices, energy_deltas, label=label, linewidth=0.6)

            # save chemiscope energies
            # properties = {"Index": indices, f"Rel. Energy ({energy_units}))": energy_deltas}
            # chemiscope_out_fp = f"out/chemiscope/{input_ds_name}_rel_energies.json"
            # chemiscope.write_input(chemiscope_out_fp,
            #                     configs,
            #                     properties=properties)
            # print(f'Saved chemiscope plot to {chemiscope_out_fp}...')

        plt.legend(loc='upper right')
        plt.xlabel('Index')
        plt.ylabel(f'Rel. Energy ({energy_units})')
        # plt.yscale("log")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/ds_analysis/{save_name}_rel_energies.png", dpi=dpi)

    # 4. plot raw force components
    if include_forces:
        plt.figure(figsize=fig_size, dpi=dpi)
        for input_ds_name, label, force_comps in zip(input_ds_names, dataset_labels, all_force_comps):
            plt.plot(np.arange(0, len(force_comps)), force_comps, label=label, linewidth=0.6, alpha=0.6)
        # plt.legend(loc='upper right')
        plt.xlabel('Component Index')
        plt.ylabel(r'Force Component (meV / $\AA$)')
        # plt.yscale("log")
        plt.tight_layout()
        save_fp = f"{plot_dir}/ds_analysis/{save_name}_force_comps.png"
        plt.savefig(save_fp, dpi=dpi)
        print(f"Saved {save_fp}...")

    if ts_ref_fpath is not None:
        print('React Coords Num', len(react_coords))
        if len(react_coords) == 2:
            # 5. TS dist plot
            for input_ds_name, energy_deltas, r_coord_dists, configs in zip(input_ds_names, all_energy_deltas, all_r_coord_dists, all_configs):

                points = np.array([np.array([x, y]) for x, y in r_coord_dists])
                plot_settings = PlotSettings(x_label=r'r1 / $\AA$', y_label=r'r2 / $\AA$', colour_bar_label='∆ Energy (meV / atom)', title=input_ds_name,
                                            dpi=dpi, fig_size=fig_size, show_legend=False, xlim=plot_2d_xlim, ylim=plot_2d_ylim)
                plot_2d([points],
                        f"{plot_dir}/2D/{input_ds_name}_2D_react_coords.png",
                        colour_maps=[energy_deltas] if energy_deltas is not None else None,
                        plot_settings=plot_settings,
                        ref_points=[ts_ref_bond_dist_tuple],
                        # ref_point_labels=['TS'],
                        )

                # properties = {r'$r_{1}$ / $\AA$': np.array(r1_dists), r'$r_{2}$ / $\AA$':np.array(r2_dists)}
                # widget = chemiscope.show(configs, properties=properties)
                # widget.save(f"out/chemiscope/{input_ds_name}_ts_dist.json")

            # combined TS dist plot
            all_points = [np.array([np.array([x, y]) for x, y in r_coord_dists]) for r_coord_dists in all_r_coord_dists]
            plot_settings = PlotSettings(x_label=r'r1 / $\AA$', y_label=r'r2 / $\AA$', title=input_ds_name, dpi=dpi, fig_size=fig_size, show_legend=False)
            plot_2d(all_points,
                    f"{plot_dir}/ds_analysis/{save_name}_2D_react_coords.png",
                    plot_settings=plot_settings,
                    ref_points=[ts_ref_bond_dist_tuple],
                    ref_point_labels=['TS'],
                    )

        # train_ts_distances = np.array([abs(react_coord_dists[i] - ts_ref_bond_distances[r1]) for i in range(len(configs))])

        # plot histogram of TS dist on reaction coordinate distribution
        plt.figure(figsize=fig_size, dpi=dpi)
        colors = list(plt.get_cmap("tab10").colors)
        for i, (ds_label, input_ds_name, r_coord_dists) in enumerate(zip(dataset_labels, input_ds_names, all_r_coord_dists)):
            ts_dists = [sum([d - ts_ref_bond_dist_tuple[i] for i, d in enumerate(r_coord_dist)]) for r_coord_dist in r_coord_dists]
            plt.hist(ts_dists, bins=list(reversed([-0.2 * i for i in range(10)])) + [0.2 * i for i in range(50)], alpha=0.5, color=colors[i], label=ds_label)

            # plot kde
            ts_dists = np.array(ts_dists)
            bin_edges = np.histogram_bin_edges(ts_dists, bins=40)
            bin_width = bin_edges[1] - bin_edges[0]
            xs = np.linspace(ts_dists.min(), ts_dists.max(), 500)
            kde = gaussian_kde(ts_dists, bw_method="scott")
            kde_counts = kde(xs) * len(ts_dists) * bin_width
            plt.plot(xs, kde_counts, color=colors[i], linewidth=1)

        plt.xlabel(r'TS Dist. $\AA$', fontsize=10)
        plt.ylabel(r'Count', fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.legend()
        plt.tight_layout()

        save_fp = f"{plot_dir}/ds_analysis/{save_name}_data_hist.png"
        plt.savefig(save_fp, dpi=dpi)
        print(f"Saved {save_fp}...")
