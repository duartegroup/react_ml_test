import glob
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from react_ml_test_public.helpers.io import flexible_ase_read
from react_ml_test_public.helpers.logging import logger
from react_ml_test_public.helpers.plot import DPI, SINGLE_COLUMN_FIG_SIZE
from react_ml_test_public.helpers.settings import (
    ANTHRACENE_DA_DATA_DIR,
    CPD_MVK_DA_DATA_DIR,
    GUAJADIAL_DA_DATA_DIR,
    MODELS,
    PAPER_MODEL_COLOURS,
    PAPER_MODELS,
    PLOT_DIR,
    TRAJ_DIR,
    BondDef,
    ConfigDef,
)
from react_ml_test_public.helpers.stats import aligned_rmsd, calculate_rmse
from react_ml_test_public.helpers.traj_eval import (
    get_config_def_indices,
    print_atom_distance_table,
)
from react_ml_test_public.helpers.utils import get_formatted_model_name, natural_sort_key

if __name__ == "__main__":

    config_react_coord_defs = None
    traj_dir = TRAJ_DIR

    # === SPECIFY CONFIGURATION SET ===
    # config_set_fpath = "out/trj/final_configs/final_configs_init_configs_downhill_traj_endo_in_water_GO-MACE-23.xyz"
    # config_set_fpath = "out/trj/final_configs/final_configs_init_configs_downhill_traj_endo_in_water_MACE-OFF23.xyz"
    # config_set_fpath = "out/trj/final_configs/final_configs_cis_endo_TS_PBE_GO-MACE-23"
    # config_fnames = ["final_configs_init_configs_uphill_vac_wB97X_GO-MACE-23", "final_configs_init_configs_uphill_vac_wB97X_MACE-OFF23_small"]
    # config_set_fpath = "out/trj/final_configs/final_configs_cis_endo_TS_wB97M_GO-MACE-23.xyz"
    # config_set_fpath = "out/trj/final_configs/final_configs_cis_endo_TS_wB97M_MACE-OFF23.xyz"

    # vacuum
    # traj_name_stubs = ["cis_endo_TS_wB97M_MACE-OFF23_medium_300"]
    # save_name = "cis_endo_TS_wB97M_MACE-OFF23_medium"

    # traj_name_stubs = ["cis_endo_TS_wB97M_GO-MACE-23_300"]
    # save_name = "cis_endo_TS_wB97M_GO-MACE-23"

    # traj_name_stubs = ["cis_endo_TS_wB97M_MACE-OFF23_medium_endo_DA_1.0_train_from_scratch_300"]
    # save_name = "cis_endo_TS_wB97M_MACE-OFF23_medium_endo_DA_1.0_train_from_scratch"

    # ===== SPECIFY SYSTEM =====

    # --- CPD + MVK ---
    # traj_type, run_name, ts_ref_fp = "CPD_MVK_DA_vac_opt_TS", '300K_0.5ps', f'{CPD_MVK_DA_DATA_DIR}/TS/endo_DA_vac_TS_wB97M.xyz'   # vacuum, from model optimised
    # traj_type, run_name, ts_ref_fp = "CPD_MVK_DA_vac_opt_TS", '300K_1.0ps', f'{CPD_MVK_DA_DATA_DIR}/TS/endo_DA_vac_TS_wB97M.xyz'   # vacuum, from model optimised
    traj_type, run_name, ts_ref_fp = "CPD_MVK_DA_water_opt_TS", '300K_1.0ps', f'{CPD_MVK_DA_DATA_DIR}/TS/CPD_MVK_TS_opt_water_wB97M.xyz'   # explicit water, from model optimised
    r1_atoms, r2_atoms, r3_atoms = (1, 12), (6, 11), (2, 17)
    num_solute_atoms = 22

    threshold = 1.8
    config_def_R = ConfigDef('RS', [BondDef(*r1_atoms, bonded=False, threshold=threshold), BondDef(*r2_atoms, bonded=False, threshold=threshold)])
    config_def_P = ConfigDef('PS', [BondDef(*r1_atoms, bonded=True, threshold=threshold), BondDef(*r2_atoms, bonded=True, threshold=threshold)])
    config_def_PS2 = ConfigDef('PS2', [BondDef(*r3_atoms, bonded=True, threshold=threshold)])
    config_def_PS3 = ConfigDef('PS3', [BondDef(*r1_atoms, bonded=True, threshold=threshold), BondDef(*r2_atoms, bonded=False, threshold=threshold), BondDef(*r3_atoms, bonded=False, threshold=threshold)])
    config_defs=[config_def_R, config_def_P, config_def_PS2, config_def_PS3]

    # --- anthracene ---
    # traj_type, run_name, ts_ref_fp = "anthracene_vac_opt_TS", '300K_0.5ps', f'{ANTHRACENE_DA_DATA_DIR}/TS/anthracene_DA_vac_TS_wB97M.xyz'   # vacuum
    # traj_type, run_name, ts_ref_fp = "anthracene_DA_vac_opt_TS", '300K_1.0ps', f'{ANTHRACENE_DA_DATA_DIR}/TS/anthracene_DA_vac_TS_wB97M.xyz'   # vacuum
    # traj_type, run_name, ts_ref_fp = "anthracene_DA_toluene_opt_TS", '300K_1.0ps', f'{ANTHRACENE_DA_DATA_DIR}/TS/anthracene_DA_TS_opt_SMD_toluene_wB97M.xyz'   # explicit toluene
    # r1_atoms, r2_atoms = (2, 20), (5, 22)
    # num_solute_atoms = 41

    # 1, 4 TS
    # traj_type, run_name, ts_ref_fp = "anthracene_1_4_MMFF_vac_opt_TS", '300K_0.5ps', f'{ANTHRACENE_DA_DATA_DIR}/TS/anthracene_1_4_DA_vac_TS_wB97M.xyz'   # vacuum
    # traj_type, run_name, ts_ref_fp = "anthracene_1_4_DA_vac_opt_TS", '300K_1.0ps', f'{ANTHRACENE_DA_DATA_DIR}/TS/anthracene_1_4_DA_vac_TS_wB97M.xyz'   # vacuum
    # traj_type, run_name, ts_ref_fp = "anthracene_1_4_DA_toluene_opt_TS", '300K_1.0ps', f'{ANTHRACENE_DA_DATA_DIR}/TS/anthracene_1_4_DA_TS_opt_SMD_toluene_wB97M.xyz'   # vacuum
    # r1_atoms, r2_atoms = (16, 20), (15, 22)
    # num_solute_atoms = 41

    # threshold = 1.8
    # config_def_R = ConfigDef('R', [BondDef(*r1_atoms, bonded=False, threshold=threshold), BondDef(*r2_atoms, bonded=False, threshold=threshold)])
    # config_def_P = ConfigDef('P', [BondDef(*r1_atoms, bonded=True, threshold=threshold), BondDef(*r2_atoms, bonded=True, threshold=threshold)])
    # config_defs=[config_def_R, config_def_P]

    # --- guajadial ---
    # traj_type, run_name, ts_ref_fp = "guajadial_MMFF_vac_opt_TS", '300K_0.5ps', f'{GUAJADIAL_DA_DATA_DIR}/TS/guajadial_DA_vac_TS_wB97M.xyz'   # vacuum
    # traj_type, run_name, ts_ref_fp = "guajadial_water_cutoff_3_0_PBC_opt_TS", '300K_1.0ps', f'{GUAJADIAL_DA_DATA_DIR}/TS/guajadial_DA_vac_TS_wB97M.xyz'   # explicit
    # traj_type, run_name, ts_ref_fp = "guajadial_DA_vac_opt_TS", '300K_1.0ps', f'{GUAJADIAL_DA_DATA_DIR}/TS/guajadial_DA_vac_TS_wB97M.xyz'   # explicit
    # traj_type, run_name, ts_ref_fp = "guajadial_DA_vac_opt_TS", '300K_2.0ps', f'{GUAJADIAL_DA_DATA_DIR}/TS/guajadial_DA_vac_TS_wB97M.xyz'   # explicit
    # traj_type, run_name, ts_ref_fp = "guajadial_DA_vac_opt_TS", '300K_', f'{GUAJADIAL_DA_DATA_DIR}/TS/guajadial_DA_vac_TS_wB97M.xyz'   # explicit
    # traj_type, run_name, ts_ref_fp = "guajadial_DA_water_opt_TS", '300K_', f'{GUAJADIAL_DA_DATA_DIR}/TS/guajadial_TS_opt_water_wB97M.xyz'   # explicit 
    # r1_atoms, r2_atoms = (13, 31), (6, 32)
    # num_solute_atoms = 69

    # threshold = 1.8
    # config_def_R = ConfigDef('R', [BondDef(*r1_atoms, bonded=False, threshold=threshold), BondDef(*r2_atoms, bonded=False, threshold=threshold)])
    # config_def_P = ConfigDef('P', [BondDef(*r1_atoms, bonded=True, threshold=threshold), BondDef(*r2_atoms, bonded=True, threshold=threshold)])
    # config_defs=[config_def_R, config_def_P]
    # ------------------

    config_def_r1 = ConfigDef('r1', [BondDef(*r1_atoms, bonded=True, threshold=threshold)])
    config_def_r2 = ConfigDef('r2', [BondDef(*r2_atoms, bonded=True, threshold=threshold)])
    config_react_coord_defs = [config_def_r1, config_def_r2]

    # ===== SELECT MODELS =====

    # --- full zero-shot model comparison ---
    model_names = [
            'GO-MACE-23', 
            'MACE-OFF23_medium', 
            'MACE-MP-0b3_medium',
            # 'EGRET_1T', 
            # 'MACE-NEB',
            'ani-1xnr',
            'MACE-omol-0-extra-large-1024',
            "MACE-OFF23_CPD_MVK_DA_train_42_config_energy_ts_gauss_ft",
    ]

    traj_name_stubs = [f"{traj_type}_{model_name}_{run_name}" for model_name in model_names]
    save_name = f"{traj_type}_zero_shot"
    labels = [model_name for model_name in model_names]
    print("Labels: ", labels)
    colours = [PAPER_MODEL_COLOURS.get(label) if PAPER_MODEL_COLOURS.get(label) is not None else "grey" for label in labels]
    traj_trunc = (3.5, 3.0)  # (r1_dist, r2_dist) for reactant cutoff

    # --- selected fine-tuning models comparison for paper ---

    # # CPD + MVK
    # n, system_ft_name, rep_num, omol_rep_num, omol_energy_str = 42, "CPD_MVK_DA_train", 1, 1, 'energy_'
    # # sample_str = 'ts_gauss_'
    # sample_str = ''

    # # anthracene
    # # n, system_ft_name, rep_num, omol_rep_num, omol_energy_str = 46, "anthracene_DA_train", 2, 3, 'energy_'
    # # sample_str = ''

    # # guajadial
    # # n, system_ft_name, rep_num, omol_rep_num, omol_energy_str = 50, "guajadial_DA_train", 1, 1, ''
    # # sample_str = ''

    # model_names = [
    #     f"GO-MACE-23_{system_ft_name}_{n}_config_energy_{sample_str}ft",
    #     f"MACE-OFF23_{system_ft_name}_{n}_config_energy_{sample_str}ft",
    #     f"MACE-MP-0b3_medium_{system_ft_name}_{n}_config_energy_{sample_str}ft",
    #     # f"MACE-omol-0-extra-large-1024",
    #     f"MACE-omol-0-extra-large-1024_{system_ft_name}_{n}_config_{omol_energy_str}{sample_str}ft",
    # ]
    # traj_name_stubs = [f"{traj_type}_{model_name}_{run_name}" for model_name in model_names]
    # save_name = f"{traj_type}_paper_selected_ft_models_{n}_config"
    # labels = [f"{get_formatted_model_name(model_name)}" for model_name in model_names]
    # colours = list(PAPER_MODEL_COLOURS.values())[0:len(model_names)]
    # print("Colours: ", colours)
    # traj_trunc = (3.5, 3.5)  # (r1_dist, r2_dist) for reactant cutoff
    # # traj_trunc = None  # (r1_dist, r2_dist) for reactant cutoff

    # --- full positive control fine-tuned for paper ---
    # # model_names = [
    # #                 f"GO-MACE-23_endo_DA_final_train_clean_vac_solv_only_force_filtered_fine_tuned_rep_1",
    # #                 f"MACE-OFF23_medium_endo_DA_final_train_clean_vac_solv_only_force_filtered_fine_tuned_rep_1", 
    # #                 f"MACE-MP0-128-L1_endo_DA_final_train_clean_vac_solv_only_force_filtered_fine_tuned_rep_1",
    # #                 ]  # for old runs without TS opt
    # model_names = ['GO-MACE-23_full_ft', 
    #                'MACE-OFF23_full_ft', 
    #                'MACE-MP0-128-L1_full_ft']
    # traj_name_stubs = [f"{traj_type}_{model_name}_{run_name}" for model_name in model_names]
    # save_name = f"{traj_type}_paper_positive_control_ft_models"
    # # labels = [f"{get_formatted_model_name(model_name)} Full FT" for model_name in ['GO-MACE-23', 'MACE-MP0']]
    # labels = [f"{get_formatted_model_name(model_name)} Full FT" for model_name in PAPER_MODELS]

    # run_name = "endo_DA_train_random_10_config"
    # run_name = "endo_DA_train_random_20_config"
    # run_name = "endo_DA_vac_TS_wB97M"
    # traj_name_stubs = [f"{traj_type}_{base_model_name}_{run_name}"]
    # save_name = f"{traj_type}_{base_model_name}_{run_name}"
    # labels = [base_model_name + f'_{run_name}']

    # ====================================================

    ts_ref_config = flexible_ase_read(ts_ref_fp)[0]
    ts_ref_bond_distances = {r1_atoms: ts_ref_config.get_distance(*r1_atoms), r2_atoms: ts_ref_config.get_distance(*r2_atoms)}
    ts_ref_point = tuple([dist for dist in ts_ref_bond_distances.values()])
    logger.info(f'TS ref bond distance: {ts_ref_point}')

    # get all trajectories
    all_model_trajectories = []
    traj_len = None
    for traj_name_stub in traj_name_stubs:

        # get all files in dir (and sort by natural ascii ordering)
        fpaths = sorted(glob.glob(os.path.join(traj_dir, f"{traj_name_stub}*")), key=natural_sort_key)
        print(f"Found {len(fpaths)} trajectories for name stub: {traj_name_stub}...")
        if len(fpaths) < 1:
            print(f"No files detected in directory {traj_dir} with name stub {traj_name_stub}...")
            continue     
        
        trajectories = []
        for fpath in fpaths:
            # print(fpath)
            trajectories.append(flexible_ase_read(fpath))
        all_model_trajectories.append(trajectories)

        # get initial atom distances (to debug if our r1 and r2 are correct)
        first_traj_config = trajectories[0][0]
        traj_ts_r1_dist = first_traj_config.get_distance(r1_atoms[0], r1_atoms[1], mic=True)
        traj_ts_r2_dist = first_traj_config.get_distance(r2_atoms[0], r2_atoms[1], mic=True)
        print(f"Trajectories TS Distance: {np.array([traj_ts_r1_dist, traj_ts_r2_dist])}")
        traj_lens = [len(traj) for traj in trajectories]
        print(f"Trajectory lengths: ", traj_lens)
        traj_len = traj_lens[0]

        logger.info(f'TS Dist. Error: {sum(abs(np.array(ts_ref_point) - np.array([traj_ts_r1_dist, traj_ts_r2_dist])))}')

        # struct_rmsd = calculate_rmse(first_traj_config.get_positions()[0:num_solute_atoms], 
        #                              ts_ref_config.get_positions()[0:num_solute_atoms])

        struct_rmsd = aligned_rmsd(first_traj_config, ts_ref_config, num_solute_atoms)
        logger.info(f'MLIP OPT TS Structural RMSD: {struct_rmsd:.3f}\n')

    # PLOT SETTINGS
    from matplotlib.ticker import MaxNLocator
    from scipy.stats import gaussian_kde

    r1_range = (4.0, 1.0)
    # r1_range = (3.0, 1.0)
    r2_range = (5.0, 1.0)
    # traj_trunc = (2.8, 3.5)  # (r1_dist, r2_dist) for reactant cutoff
    show_legend = False

    fig = plt.figure(figsize=SINGLE_COLUMN_FIG_SIZE, dpi=DPI)
    save_fp = f"{PLOT_DIR}/trj_atom_dist/{save_name}_atom_distance_plot.png"
    # colours = plt.cm.tab10(range(len(all_model_trajectories) + 1))

    print('COLOURS: ', colours)
    all_r1_first_formed, all_r2_first_formed = [], []
    for trajectories, colour, label in zip(all_model_trajectories, colours, labels):

        print(f'Evaluating {label}...')

        # if specified, determine formed products, print stats and colour on graph
        n = len(trajectories)
        prod_formation_indices = None
        if config_defs is not None:

            final_frames = [trajectory[-1] for trajectory in trajectories]
            prod_formation_indices = get_config_def_indices(final_frames, config_defs)
            print("Product Formation Indices: ", prod_formation_indices)

            # get percentages
            prod_formation_perc = {prod_name: len(indices) / float(n) for prod_name, indices in prod_formation_indices.items()}
            print("Product Formation Percentages: ", prod_formation_perc)

        # create colours for each set of products
        tab10 = plt.get_cmap("tab10")
        colour_dict = None
        if prod_formation_indices is not None:
            colour_dict = {prod_name: tab10(i) for i, prod_name in enumerate(prod_formation_indices.keys())}
        
        used_labels = set()
        r1_end_points, r2_end_points = [], []
        model_r1_first_formed, model_r2_first_formed = [], []
        for i in range(len(trajectories)):

            trajectory = trajectories[i]
            if len(trajectory) < 1:
                print(f"Skipping trajectory {i} which has length {len(trajectory)}")
                continue
            r1_dists = [config.get_distance(r1_atoms[0], r1_atoms[1], mic=True) for config in trajectory]
            r2_dists = [config.get_distance(r2_atoms[0], r2_atoms[1], mic=True) for config in trajectory]
            # print('Trajectory Length: ', len(trajectory)

            # report for each traj frame whether each bond has been formed
            r1_formed = [int(abs(dist) <= threshold) for dist in r1_dists]
            r2_formed = [int(abs(dist) <= threshold) for dist in r2_dists]

            r1_first_formed = next((i for i, x in enumerate(r1_formed) if x == 1), None)
            r2_first_formed = next((i for i, x in enumerate(r2_formed) if x == 1), None)
            model_r1_first_formed.append(r1_first_formed)
            model_r2_first_formed.append(r2_first_formed)

            # truncate reactant trajectories by traj_trunc if specified
            if traj_trunc is not None and i in prod_formation_indices[config_def_R.name]:
                    trunc_index = len(trajectory)
                    for j, (r1_dist, r2_dist) in enumerate(zip(r1_dists, r2_dists)):
                        if r1_dist >= traj_trunc[0] and r2_dist >= traj_trunc[1]:   # TODO: debug, this was switched to or
                            trunc_index = j
                            break
                    r1_dists = r1_dists[0:trunc_index]
                    r2_dists = r2_dists[0:trunc_index]
                    trajectory = trajectory[0:trunc_index]

            plt.plot(r1_dists, r2_dists, linewidth=0.5, alpha=0.08, color=colour, 
                     linestyle='-', label=label if label not in used_labels else None, zorder=0)
            used_labels.add(label)

            r1_end_points.append(trajectory[-1].get_distance(r1_atoms[0], r1_atoms[1], mic=True))
            r2_end_points.append(trajectory[-1].get_distance(r2_atoms[0], r2_atoms[1], mic=True))
        
        all_r1_first_formed.append(model_r1_first_formed)
        all_r2_first_formed.append(model_r2_first_formed)

        # plot dots for end points
        plt.scatter(r1_end_points, r2_end_points, s=0.2, alpha=1.0, color=colour, zorder=1)

    # plot reference points
    plt.scatter([ts_ref_point[0]], [ts_ref_point[1]], color='black', zorder=2, s=2.0)
    # plt.text(ts_ref_point[0], ts_ref_point[1] + 0.6, r'TS$_{DFT}$', fontsize=8, ha='center')

    plt.xlabel(r"r$_{1}$ ($\AA$)", fontsize=10)
    plt.ylabel(r"r$_{2}$ ($\AA$)", fontsize=10)

    # make pretty
    for dir in "top", "right":
        plt.gca().spines[dir].set_visible(False)
    for dir in "left", "bottom":
        plt.gca().spines[dir].set_position(("outward", 10))

    if r1_range is not None:
        plt.xlim(*r1_range)

    if r2_range is not None:
        plt.ylim(*r2_range)

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    if show_legend:
        patches = [Patch(color=col, label=lbl) for col, lbl in zip(colours, labels)]
        plt.legend(handles=patches, loc='upper left', fontsize=5, frameon=False, ncol=3)
    plt.tight_layout()
    # plt.rc('font', family='Helvetica')

    print(f"Saving plot to {save_fp}...")
    plt.savefig(save_fp)

    traj_lens = [len(traj) for traj in trajectories]

    # create plot to be inset - distribution over time 
    plt.figure(figsize=SINGLE_COLUMN_FIG_SIZE, dpi=DPI)
    ps_interval = 0.02
    xs = np.linspace(0, traj_len * ps_interval, 500)
    bin_edges = np.histogram_bin_edges(np.concatenate([[val * ps_interval for val in x if val is not None] for x in all_r2_first_formed]), bins=30)
    bin_width = bin_edges[1] - bin_edges[0]
    for i, (model_r1_first_formed, model_r2_first_formed) in enumerate(zip(all_r1_first_formed, all_r2_first_formed)):
        
        line_styles = ['-', (0, (1.0, 1.0))]
        # line_styles = [(0, (1.0, 1.0)), '-']
        # line_styles = ['dashdot', '-']
        # line_widths = [1, 2]
        line_widths = [2, 1]
        # alphas= [0.4, 0.9]
        alphas= [0.7, 0.4]
        r2_r1_form_diff = np.array([x2 - x1 for x2, x1 in zip(model_r2_first_formed, model_r1_first_formed) 
                                    if x2 is not None and x1 is not None])
        mean_r2_r1_diff = np.mean(r2_r1_form_diff) * ps_interval
        for j, x in enumerate([model_r2_first_formed]):
            
            x = [val * ps_interval for val in x if val is not None]
            print(f"Model {model_names[i]}:")
            print("Mean r2-r1 bond time diff: ", mean_r2_r1_diff)
            # kde = gaussian_kde(x, bw_method="scott")
            # kde_counts = kde(xs) * len(x) * bin_width
            eps = 1e-12
            # kde_counts = np.where(kde_counts > eps, kde_counts, np.nan)
            color=colours[i]
            # plt.plot(xs, kde_counts, linewidth=line_widths[j], color=colours[i], linestyle=line_styles[j], alpha=alphas[j])
            plt.hist(x, color=colours[i], bins=bin_edges, alpha=alphas[j])

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlim((0.0, 1.0))
    # plt.ylim((0.0, 30.0))
    # plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel(r'Time gap (ps)', fontsize=10)
    plt.ylabel(r'Counts', fontsize=10)
    plt.tight_layout()

    save_fp = f"{PLOT_DIR}/trj_atom_dist/{save_name}_inset_time_plot.png"
    print(f"Saving plot to {save_fp}...")
    plt.savefig(save_fp)

