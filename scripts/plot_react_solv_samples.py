import os
import sys

import numpy as np
from matplotlib import patches

sys.path.append('../')
import glob

import matplotlib.pyplot as plt

from react_ml_test.helpers.io import flexible_ase_read, get_name_from_path
from react_ml_test.helpers.logging import logger
from react_ml_test.helpers.plot import DPI, SINGLE_COLUMN_FIG_SIZE
from react_ml_test.helpers.settings import (
    ANTHRACENE_DA_DATA_DIR,
    CPD_MVK_DA_DATA_DIR,
    GUAJADIAL_DA_DATA_DIR,
    PLOT_DIR,
    PlotSettings,
)
from react_ml_test.helpers.utils import natural_sort_key

np.set_printoptions(precision=3, suppress=True)


if __name__ == "__main__":

    # === CHOOSE SYSTEM ===

    # --- CPD + MVK DA ---

    # system_dir = CPD_MVK_DA_DATA_DIR
    # ts_ref_config = flexible_ase_read(f'{ANTHRACENE_DA_DATA_DIR}/TS/endo_DA_vac_TS_wB97M.xyz')[0]
    # r1, r2 = (1, 12), (6, 11)

    # --- ANTHRACENE DA ---

    # system_dir = ANTHRACENE_DA_DATA_DIR
    # ts_ref_config = flexible_ase_read(f'{ANTHRACENE_DA_DATA_DIR}/TS/anthracene_DA_vac_TS_wB97M.xyz')[0]
    # r1, r2 = (2, 20), (5, 22)

    # --- GUAJADIAL DA DIR ---
    system_dir = GUAJADIAL_DA_DATA_DIR
    ts_ref_config = flexible_ase_read(f'{GUAJADIAL_DA_DATA_DIR}/TS/guajadial_DA_vac_TS_wB97M.xyz')[0]
    r1, r2 = (13, 31), (6, 32)

    # grid of solvated / reactive samples
    # solv_fracs = [0.0, 0.25, 0.50, 0.75, 1.0]
    # ds_name_stub = f"{system_dir}/train/subsets/*_n=8_r_dist*_diverse_solv_0_rep_1.xyz"
    # ds_name_stub = f"{system_dir}/train/subsets/*_n=8_r_dist*_diverse_solv_0_rep_2.xyz"
    # ds_name_stub = f"{system_dir}/train/subsets/*_n=8_r_dist*_diverse_solv_0_rep_3.xyz"

    # ds_name_stub = f"{system_dir}/train/subsets/*_n=8_r_dist*_local_solv_0.xyz"
    # ds_name_stub = f"{system_dir}/train/subsets/*_n=8_r_dist*_diverse_solv_25.xyz"
    # ds_name_stub = f"{system_dir}/train/subsets/*_n=8_r_dist*_diverse_solv_75.xyz"
    # ds_name_stub = f"{system_dir}/train/subsets/*_n=8_r_dist*_local_solv_25.xyz"
    # ds_name_stub = f"{system_dir}/train/subsets/*_n=8_r_dist*_diverse_solv_50_rep_1.xyz"
    # ds_name_stub = f"{system_dir}/train/subsets/*_n=8_r_dist*_diverse_solv_50_rep_1.xyz"
    # ds_name_stub = f"{system_dir}/train/subsets/*_n=8_r_dist*_diverse_solv_100.xyz"
    # ds_name_stub = f"{system_dir}/train/subsets/*_n=8_r_dist*_diverse_solv_100_rep_1.xyz"
    ds_name_stub = f"{system_dir}/train/subsets/*_n=8_r_dist*_diverse_solv_0_rep_2.xyz"
    # ds_name_stub = f"{system_dir}/train/subsets/*_n=8_r_dist*_diverse_solv_100_rep_3.xyz"
    # ds_name_stub = f"{system_dir}/train/subsets/*_n=8_r_dist*_diverse_solv_0_rep_3.xyz"
    # ds_name_stub = f"{system_dir}/train/subsets/*_n=16_r_dist*_diverse_solv_0_rep_1.xyz"
    # ds_name_stub = f"{system_dir}/train/subsets/*_n=8_r_dist*_local_solv_100.xyz"
    # ds_name_stub = f"{system_dir}/train/subsets/*_n=20_r_dist*_local_solv_100.xyz"
    ds_fpaths = sorted(glob.glob(ds_name_stub), key=natural_sort_key)
    assert len(ds_fpaths) > 0, f"No files detected with name stub {ds_name_stub}..."
    ds_names = [get_name_from_path(fpath) for fpath in ds_fpaths]
    print('No. Datasets: ', len(ds_fpaths))
    print('DS Names: ', ds_names)

    ts_ref_bond_distances = {r1: ts_ref_config.get_distance(*r1), r2: ts_ref_config.get_distance(*r2)}
    ts_ref_bond_dist_tuple = tuple([dist for dist in ts_ref_bond_distances.values()])
    logger.info(f'TS ref bond distances: {ts_ref_bond_distances}')

    # lim=(0.5, 6.0)
    lim=(-2.0, 6.0)
    custom_markers = None # ['^', '^', 'X', 'X', '*', '*']
    colour_map = list(plt.cm.tab10(np.arange(0, len(ds_fpaths))))

    # circle_radii = [0.5, 2.0, 3.0, 4.0, 6.0]
    circle_radii = [0.5, 1.5, 3.0, 6.0]

    n = 50
    all_points = []
    all_r1_dists, all_r2_dists = [], []
    all_r1_ts_dists, all_r2_ts_dists = [], []
    all_config_types = []
    all_ds_names = []
    for i, (input_ds_fp, dataset_name) in enumerate(zip(ds_fpaths, ds_names)):
        configs = flexible_ase_read(input_ds_fp)
        config_types = [config.info.get('config_type') for config in configs]
        r1_dists = [config.get_distance(*r1, mic=True) for config in configs]
        r2_dists = [config.get_distance(*r2, mic=True) for config in configs]
        r1_ts_dists = np.array(r1_dists) - ts_ref_bond_dist_tuple[0]
        r2_ts_dists = np.array(r2_dists) - ts_ref_bond_dist_tuple[0]
        # points = np.array([np.array([x, y]) for x, y in zip(r1_dists[0:n], r2_dists[0:n])])
        points = np.array([np.array([x, y]) for x, y in zip(r1_ts_dists[0:n], r2_ts_dists[0:n])])
        all_points.append(points)
        all_r1_dists.append(r1_dists)
        all_r2_dists.append(r2_dists)
        all_r1_ts_dists.append(r1_ts_dists)
        all_r2_ts_dists.append(r2_ts_dists)
        all_config_types.append(config_types)
        all_ds_names.append(dataset_name)

        n = len(r1_dists)
        print(f'Dataset {dataset_name} with length {n}')
        # print('Points: ', points)
        # print('R1 Distances from TS', r1_ts_dists)
        # print('R2 Distances from TS', r2_ts_dists)

        properties = {'r1 / A': np.array(r1_dists[0:n]), 'r2 / A':np.array(r2_dists[0:n])}
        # widget = chemiscope.show(configs[0:n], properties=properties)
        # widget.save(f"out/chemiscope/{dataset_name}_ts_dist.json")
    
        plt.figure(figsize=SINGLE_COLUMN_FIG_SIZE, dpi=DPI)
        ax = plt.gca()

        # plot all
        for j, (x, y, ds_name) in enumerate(zip(all_r1_ts_dists, all_r2_ts_dists, all_ds_names)):
            plt.scatter(x, y, alpha=1.0, color=colour_map[j], s=0.6, label=ds_name, zorder=j)

            # Plot circles
            radius = circle_radii[j]
            circle = patches.Circle((0, 0), radius, color=colour_map[j], alpha=0.05, zorder=len(all_ds_names)-j)
            ax.add_patch(circle)
    
        # plot just this subset
        # plt.scatter(r1_dists, r2_dists, alpha=1.0, color=colour_map[i], s=0.6, label=dataset_name, zorder=i)
        # radius = circle_radii[i]
        # circle = patches.Circle((0, 0), radius, color=colour_map[j], alpha=0.05, zorder=len(all_ds_names)-j)
        # ax.add_patch(circle)

        ax.set_aspect('equal')
        plt.xlim(lim)
        plt.ylim(lim)
        # *ts_ref_bond_dist_tuple
        plt.scatter(0, 0, color='black', zorder=0, s=1.0)
        plt.text(0, 0, 'TS', fontsize=4, ha='right')
        plt.legend(fontsize=4, loc="upper left")
        
        save_fp = f"{PLOT_DIR}/grid/{dataset_name}_n={n}_react_solv_grid_samples.png"
        plt.savefig(save_fp, format='png', dpi=600)
        print(f'Saved plot to {save_fp}')

        # plot_settings = PlotSettings(x_label='r1 / A', y_label='r2 / A', 
        #                         custom_markers=custom_markers,
        #                         custom_colour_list=colour_map,
        #                         xlim=lim, ylim=lim, point_size = 0.2)

        # plot_2d(all_points,
        #         f"{PLOT_DIR}/2D/2D_react_coords_training.png", 
        #         plot_settings=plot_settings,
        #         dataset_names=ds_names,
        #         # ref_points=[ts_ref_bond_dist_tuple],
        #         # ref_point_labels=['TS'],
        #         )
