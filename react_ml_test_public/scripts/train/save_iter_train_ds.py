"""
Script for saving iterative training dataset, i.e. samples with replicates selected from dataset with a particular ordering.
"""

import math
import random

from ase.io import write

from react_ml_test_public.helpers.io import flexible_ase_read, get_name_from_path
from react_ml_test_public.helpers.logging import logger
from react_ml_test_public.helpers.sampling import ts_gaussian_importance_sample
from react_ml_test_public.helpers.settings import (
    ANTHRACENE_DA_DATA_DIR,
    CPD_MVK_DA_DATA_DIR,
    DATASETS_DIR,
    GUAJADIAL_DA_DATA_DIR,
    REACTION_DATA_DIR,
)

if __name__ == "__main__":

    # === SPECIFY INPUT TRAINING DATASET ===

    # --- CPD MVK DA ---
    # train_fpath, out_dir = f"{CPD_MVK_DA_DATA_DIR}/train/CPD_MVK_DA_train.xyz", f"{CPD_MVK_DA_DATA_DIR}/train/subsets"  # CPD MVK dataset
    # react_coords = [(1, 12), (6, 11)]
    # ref_ts_fp = f"{CPD_MVK_DA_DATA_DIR}/TS/endo_DA_vac_TS_wB97M.xyz"  # DFT opt TS
    # ts_gauss_sigmas = [1.0, 1.0]
    # ts_gauss_betas = [0.9, 0.6]
    # config_types = ['vacuum', 'reactants_in_water']

    # --- anthracene ---
    train_fpath, out_dir,  = f"{ANTHRACENE_DA_DATA_DIR}/train/anthracene_DA_train.xyz", f"{ANTHRACENE_DA_DATA_DIR}/train/subsets"  # anthracene DA dataset
    react_coords = [(2, 20), (5, 22)] 
    ref_ts_fp = f"{ANTHRACENE_DA_DATA_DIR}/TS/anthracene_DA_vac_TS_wB97M.xyz"
    ts_gauss_sigmas = [1.0, 1.0]
    ts_gauss_betas = [-0.0, 0.0]
    config_types = ['vacuum', 'solvated']

    # --- guajadial ---
    # train_fpath, out_dir = f"{GUAJADIAL_DA_DATA_DIR}/train/guajadial_DA_train.xyz", f"{GUAJADIAL_DA_DATA_DIR}/train/subsets"  # anthracene DA dataset
    # react_coords = [(13, 31), (6, 32)]
    # ref_ts_fp = f"{GUAJADIAL_DA_DATA_DIR}/TS/guajadial_DA_vac_TS_wB97M.xyz"
    # ts_gauss_sigmas = [1.0, 1.0]
    # ts_gauss_betas = [-0.0, 0.0]
    # config_types = ['vacuum', 'solvated']

    # specify rep nums
    rep_nums = [1, 2, 3]
    seeds = [3, 5, 7]
    splits = [2, 4, 6, 8] + [10 + 4*i for i in range(11)]  # up to 50 samples   
    stratify_splits = True

    # select random sampling 
    # random_sample_type, sample_str = 'uniform', 'random'
    random_sample_type, sample_str = 'ts_gaussian_importance', 'ts_gauss'
    # random_sample_type = None
    # --------------------------------------

    # load training set
    train_configs = flexible_ase_read(train_fpath)
    train_name = get_name_from_path(train_fpath)
    logger.info(f'Length of training set: {len(train_configs)}')

    # get configs by config type
    if stratify_splits:
        strat_configs = {config_type: [] for config_type in config_types}
        assert len(config_types) > 0, 'No config_type information found in ase file, cannot stratify splits...'
        for config in train_configs:
            strat_configs[config.info.get('config_type')].append(config)

        strat_configs = list(strat_configs.values())
    else:
        config_types = [None]
        strat_configs = [train_configs]

    # if selected, randomise dataset for each replicate
    rep_configs_list = [[] for _ in range(len(rep_nums))]
    if random_sample_type == 'uniform':
        for i in range(len(rep_nums)):
            random.seed(seeds[i])
            for j in range(len(config_types)):
                rep_configs_list[i].append(random.sample(strat_configs[j], len(strat_configs[j])))
    elif random_sample_type == 'ts_gaussian_importance':
        ref_ts_config = flexible_ase_read(ref_ts_fp)[0]
        for j in range(len(config_types)):
            all_rep_configs_for_strat = ts_gaussian_importance_sample(
                configs=strat_configs[j],
                ref_ts_config=ref_ts_config,
                react_coords=react_coords,
                u=0.0,
                sigma=ts_gauss_sigmas[j],
                num_reps=len(rep_nums),
                seeds=seeds,
                alpha=1.0,
                beta=ts_gauss_betas[j],
                plot=True,
                save_fp=f"{CPD_MVK_DA_DATA_DIR}/train/subsets/{train_name}_ts_gauss_sample_{config_types[j]}.png"
            )
            for i in range(len(rep_nums)):
                rep_configs_list[i].append(all_rep_configs_for_strat[i])

    # splits / slices of dataset
    subset_fpaths = []
    for i, (rep_num, rep_configs) in enumerate(zip(rep_nums, rep_configs_list)):
            
        for split in splits:
            subset_fpath = f"{out_dir}/{train_name}_{split}_config_strat_{sample_str}_rep_{rep_num}.xyz"
            samples_per_config_type = int(split / float(len(config_types)))
            print('Samples Per Config Type: ', samples_per_config_type)
            subset_configs = []
            for strat_rep_configs in rep_configs:
                subset_configs.extend([config for config in strat_rep_configs[0:samples_per_config_type]])
            print('Subset Size: ', len(subset_configs))

            write(subset_fpath, subset_configs, format='extxyz')
            subset_fpaths.append(subset_fpath)

    logger.info(f'Saved {len(subset_fpaths)} subsets...')
    logger.info(f'Subsets: \n{subset_fpaths}')
