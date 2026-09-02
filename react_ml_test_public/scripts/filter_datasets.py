"""Script to filter and / or combine .xyz file datasets."""

import random

import numpy as np
from ase.io import write

from react_ml_test_public.helpers.io import flexible_ase_read, get_ase_atoms_forces
from react_ml_test_public.helpers.sampling import remove_duplicates
from react_ml_test_public.helpers.settings import (
    ANTHRACENE_DA_DATA_DIR,
    CPD_MVK_DA_DATA_DIR,
    DATASETS_DIR,
    GUAJADIAL_DA_DATA_DIR,
    REACTION_DATA_DIR,
    I_Cl_SN2_DATA_DIR,
)
from react_ml_test_public.helpers.utils import get_relative_energies

if __name__ == "__main__":

    type_filter: set = None
    energy_filter: float = None
    force_comp_filter: float = None
    slice_filter: int = None
    out_fp: str = None
    react_coord_dist_filter: float = None
    shuffle_dataset: bool = False
    reverse_selection: bool = False
    ts_ref_fpath = None
    rs = None
    out_format = 'extxyz'

    # === SPECIFY INPUT DATASETS, FILTER AND OUT FPATH ===

    # 1. --- CPD + MVK diels alder ---

    # ts_ref_fpath = f"{ENDO_DA_DATA_DIR}/TS/cis_endo_TS_wB97M.xyz"
    # rs = [(1, 12), (6, 11)]

    # all test sets
    # input_fpaths = [
    #     f"{DATASETS_DIR}/da_paper_data/valid/endo_ace_wB97M_nonpca_orca.xyz",                       # vac no reaction
    #     f"{DATASETS_DIR}/da_paper_data/neb/endo_DA_neb_gas_wB97M.xyz",                              # vac reaction
    #     f"{DATASETS_DIR}/da_paper_data/valid/endo_in_water_ace_wB97M_short_orca_non_periodic.xyz",  # water reaction
    #     f"{TRAIN_SUBSETS_DIR}/endo_DA_valid_size_sorted.xyz",                                       # vac, microsolvated, water, reactants in water
    # ]
    # filter, out_fp = {'vacuum'}, f"{DATASETS_DIR}/da_paper_data/valid/endo_DA_vacuum_test.xyz"
    # filter, out_fp = {'reactants_in_water'}, f"{DATASETS_DIR}/da_paper_data/valid/endo_DA_react_in_water_test.xyz"
    # filter, out_fp = {'water'}, f"{DATASETS_DIR}/da_paper_data/valid/endo_DA_water_test.xyz"
    # filter, out_fp = {'vacuum', 'reactants_in_water'}, f"{DATASETS_DIR}/da_paper_data/valid/endo_DA_all_test.xyz"

    # === valid set ====
    # input_fpaths, save_name = [f"{TRAIN_SUBSETS_DIR}/endo_DA_valid.xyz"], "endo_DA_valid"                                                                      # full diels alder reaction sets
    # type_filter, out_fp = {'vacuum'}, f"{TRAIN_SUBSETS_DIR}/endo_DA_valid_vac_only.xyz"

    # === filter waters out of final validation set + rename ===
    # input_fpaths = [f"{CPD_MVK_DA_DATA_DIR}/valid/endo_DA_final_valid_clean_force_filtered.xyz"]
    # type_filter, out_fp = {'vacuum', 'reactants_in_water'}, f"{CPD_MVK_DA_DATA_DIR}/valid/CPD_MVK_DA_valid.xyz"

    # === filter waters out of final test set + rename ===
    # input_fpaths = [f"{CPD_MVK_DA_DATA_DIR}/test/endo_DA_final_test_combined_force_filtered.xyz"]
    # type_filter, out_fp = {'vacuum', 'reactants_in_water'}, f"{CPD_MVK_DA_DATA_DIR}/test/CPD_MVK_DA_test.xyz"

    # input_fpaths = [f"{CPD_MVK_DA_DATA_DIR}/test/endo_DA_final_test_combined_force_filtered.xyz"]
    # type_filter, out_fp = {'vacuum'}, f"{CPD_MVK_DA_DATA_DIR}/test/CPD_MVK_DA_test_vac_only.xyz"

    # input_fpaths = [f"{CPD_MVK_DA_DATA_DIR}/test/endo_DA_final_test_combined_force_filtered.xyz"]
    # type_filter, out_fp = {'reactants_in_water'}, f"{CPD_MVK_DA_DATA_DIR}/test/CPD_MVK_DA_test_solv_only.xyz"

    # get force filtered + combined PBE test datasets
    # input_fpaths = [
    #             f"{TRAIN_SUBSETS_DIR}/endo_DA_final_test_clean_PBE_def2-SVP.xyz",
    #             f"{ENDO_DA_DATA_DIR}/endo_DA_neb_gas_PBE_def2-SVP.xyz",
    #             f"{ENDO_DA_DATA_DIR}/endo_DA_vac_ace_PBE_def2-SVP.xyz"
    #            ]
    # out_fp = f"{ENDO_DA_DATA_DIR}/endo_DA_PBE_combined_test.xyz"
    # force_comp_filter = 7.0
    # shuffle_dataset = False

    # get combined wB97M dataset for 1-1 comparison with above PBE data:
    # input_fpaths = [
    #             f"{TRAIN_SUBSETS_DIR}/endo_DA_final_test_clean.xyz",
    #             f"{ENDO_DA_DATA_DIR}/endo_DA_neb_gas_wB97M.xyz",
    #             f"{ENDO_DA_DATA_DIR}/endo_DA_vac_ace_wB97M.xyz",
    #             ]
    # out_fp = f"{ENDO_DA_DATA_DIR}/endo_DA_wB97M_combined_test.xyz"
    # force_comp_filter = 7.0
    # shuffle_dataset = False

    # --- make combined test set ---
    # input_fpaths = [f"{TRAIN_SUBSETS_DIR}/endo_DA_final_test_clean.xyz",                                        # training split test set
    #                 f"{ENDO_DA_DATA_DIR}/endo_ace_wB97M_nonpca_orca.xyz",                       # vac no reaction
    #                 f"{ENDO_DA_DATA_DIR}/endo_DA_neb_gas_wB97M.xyz",                              # vac reaction
    #                 f"{ENDO_DA_DATA_DIR}/endo_in_water_ace_wB97M_short_orca_non_periodic.xyz"]  # water reaction
    # out_fp = f"{ENDO_DA_DATA_DIR}/endo_DA_final_test_combined.xyz"

    # --- combine all force filtered datasets for one combined dataset (for descriptor analyses) ---
    # input_fpaths, out_fp = [f"{TRAIN_SUBSETS_DIR}/endo_DA_final_train_clean_vac_solv_only_force_filtered.xyz", 
    #                         f"{TRAIN_SUBSETS_DIR}/endo_DA_final_valid_clean_force_filtered.xyz", 
    #                         f"{ENDO_DA_DATA_DIR}/endo_DA_final_test_combined_force_filtered.xyz"], f"{ENDO_DA_DATA_DIR}/endo_DA_full_dataset_force_filtered.xyz"

    # input_fpaths, out_fp = [f"{ENDO_DA_DATA_DIR}/descriptor_comp/endo_DA_full_dataset_force_filtered.xyz"], f"{ENDO_DA_DATA_DIR}/descriptor_comp/endo_DA_full_dataset_force_filtered_vac_only.xyz"
    # type_filter = {'vacuum'}

    # input_fpaths, out_fp = [f"{ENDO_DA_DATA_DIR}/endo_DA_full_dataset_force_filtered.xyz"], f"{ENDO_DA_DATA_DIR}/endo_DA_full_dataset_force_filtered_solv_only.xyz"
    # type_filter = {'reactants_in_water'}

    # get combined test set vacuum configs and react in water configs only
    # input_fpaths, save_name = [f"{ENDO_DA_DATA_DIR}/endo_DA_final_test_combined_force_filtered.xyz"], "endo_DA_final_test_combined_force_filtered"
    # type_filter, out_fp = {'vacuum'}, f"{ENDO_DA_DATA_DIR}/{save_name}_vac_only.xyz",
    # type_filter, out_fp = {'reactants_in_water'}, f"{ENDO_DA_DATA_DIR}/{save_name}_solv_only.xyz",

    # --- filtered datasets for descriptor comparison ---
    # input_fpaths, out_fp = [f"{TRAIN_SUBSETS_DIR}/endo_DA_final_train_clean_vac_solv_only_force_filtered.xyz"], f"{ENDO_DA_DATA_DIR}/endo_DA_vac_TS_only.xyz"            # 1. vacuum TS only
    # type_filter, react_coord_dist_filter, reverse_selection  = {'vacuum'}, 0.3, False
    
    # input_fpaths, out_fp = [f"{TRAIN_SUBSETS_DIR}/endo_DA_final_train_clean_vac_solv_only_force_filtered.xyz"], f"{ENDO_DA_DATA_DIR}/endo_DA_vac_react_only.xyz"        # 2. vacuum only reactants
    # type_filter, react_coord_dist_filter, reverse_selection  = {'vacuum'}, 7.0, True

    # input_fpaths, out_fp = [f"{TRAIN_SUBSETS_DIR}/endo_DA_final_train_clean_vac_solv_only_force_filtered.xyz"], f"{ENDO_DA_DATA_DIR}/endo_DA_vac_prod_only.xyz"        # 2. vacuum only products
    # type_filter, react_coord_dist_filter, reverse_selection  = {'vacuum'}, -0.5, True

    # input_fpaths, out_fp = [f"{TRAIN_SUBSETS_DIR}/endo_DA_final_train_clean_vac_solv_only_force_filtered.xyz"], f"{ENDO_DA_DATA_DIR}/endo_DA_solv_TS_only.xyz"           # 4. solv TS only
    # input_fpaths, out_fp = [f"{TRAIN_SUBSETS_DIR}/endo_DA_final_train_clean_vac_solv_only_force_filtered.xyz"], f"{ENDO_DA_DATA_DIR}/endo_DA_solv_react_prod_only.xyz"   # 5. solv react and prod only
    # ---------------------------------------------------

    # ==== 2. anthracene DA dataset filtering ====

    ts_ref_fpath = f'{ANTHRACENE_DA_DATA_DIR}/TS/anthracene_DA_vac_TS_wB97M.xyz'
    rs = [(2, 20), (5, 22)]

    # 1. combine additional trajectory configs from AL and take only configurations within TS dist range
    # input_fpaths = [f"{ANTHRACENE_DA_DATA_DIR}/full_al_traj/traj_iter35_{i}_toluene.xyz" for i in range(5)] + \
    #                [f"{ANTHRACENE_DA_DATA_DIR}/full_al_traj/traj_iter49_{i}.xyz" for i in range(5)]

    # # react_coord_dist_filter, out_fp = None, f"{ANTHRACENE_DA_DATA_DIR}/full_al_traj_samples.xyz"
    # react_coord_dist_filter, out_fp = (-0.4, 2.0), f"{ANTHRACENE_DA_DATA_DIR}/full_al_traj_samples_dist_filtered.xyz"

    # 2. force filter inputs
    # force_comp_filter = 7.0
    # input_fpaths, out_fp = [f"{ANTHRACENE_DA_DATA_DIR}/anthracene_DA_vac_al_iter.xyz"], f"{ANTHRACENE_DA_DATA_DIR}/anthracene_DA_vac_al_iter_force_filtered.xyz"
    # input_fpaths, out_fp = [f"{ANTHRACENE_DA_DATA_DIR}/anthracene_DA_in_toluene_al_iter.xyz"], f"{ANTHRACENE_DA_DATA_DIR}/anthracene_DA_in_toluene_al_iter_force_filtered.xyz"
    # input_fpaths, out_fp = [f"{ANTHRACENE_DA_DATA_DIR}/anthracene_DA_al_traj_samples_dist_filtered_orca.xyz"], f"{ANTHRACENE_DA_DATA_DIR}/anthracene_DA_al_traj_samples_dist_filtered_orca_force_filtered.xyz"

    # 3. run get_train_test_split.py on these datasets
    # ...

    # 4. get train, valid and test by config types
    # input_fpaths, out_dir, save_name = [f"{ANTHRACENE_DA_DATA_DIR}/train/anthracene_DA_train.xyz"], f"{ANTHRACENE_DA_DATA_DIR}/train", "anthracene_DA_train"
    # input_fpaths, out_dir, save_name = [f"{ANTHRACENE_DA_DATA_DIR}/valid/anthracene_DA_valid.xyz"], f"{ANTHRACENE_DA_DATA_DIR}/valid", "anthracene_DA_valid"
    # input_fpaths, out_dir, save_name = [f"{ANTHRACENE_DA_DATA_DIR}/test/anthracene_DA_test.xyz"], f"{ANTHRACENE_DA_DATA_DIR}/test", "anthracene_DA_test"

    # # type_filter, out_fp = {'vacuum'}, f"{out_dir}/{save_name}_vac_only.xyz"
    # type_filter, out_fp = {'solvated'}, f"{out_dir}/{save_name}_solv_only.xyz"

    # input_fpaths, out_fp = [f"{ANTHRACENE_DA_DATA_DIR}/anthracene_DA_vac_al_iter_force_filtered.xyz"], f"{ANTHRACENE_DA_DATA_DIR}/anthracene_DA_vac_al_iter_force_filtered_solv_only.xyz"
    
    # exrta: combine trajectories and cut out 
    # shuffle_dataset = False
    # input_fpaths, out_fp = [
    #     f"{ANTHRACENE_DA_DATA_DIR}/react_traj/anthracene_1_4_MMFF_vac_MACE-OFF23_46_reactants_reversed.xyz",
    #     f"{ANTHRACENE_DA_DATA_DIR}/react_traj/anthracene_1_4_MMFF_vac_MACE-OFF23_46_products.xyz"
    # ], f"{ANTHRACENE_DA_DATA_DIR}/react_traj/anthracene_1_4_MMFF_vac_MACE-OFF23_46_reaction.xyz"
    # slice_filter = 2
    # out_format = 'xyz'

    # 5. combine two LoT datasets
    # input_fpaths, type_filter, out_fp = [f"{ANTHRACENE_DA_DATA_DIR}/test/anthracene_DA_test_wB97M_TZVPP.xyz"], {'solvated'}, f"{ANTHRACENE_DA_DATA_DIR}/test/anthracene_DA_test_wB97M_TZVPP_solv_only.xyz"

    # input_fpaths, out_fp = [f"{ANTHRACENE_DA_DATA_DIR}/test/anthracene_DA_test_wB97M_TZVPPD.xyz",
    #                         f"{ANTHRACENE_DA_DATA_DIR}/test/anthracene_DA_test_wB97M_TZVPP_solv_only.xyz"], f"{ANTHRACENE_DA_DATA_DIR}/test/anthracene_DA_test_wB97M_TZVPPD_complete"

    # filter duplicates only
    # input_fpaths, type_filter, out_fp = [f"{ANTHRACENE_DA_DATA_DIR}/test/anthracene_DA_test_wB97M_TZVPP.xyz"], None, f"{ANTHRACENE_DA_DATA_DIR}/test/anthracene_DA_test_wB97M_TZVPP.xyz"
    input_fpaths, type_filter, out_fp = [f"{ANTHRACENE_DA_DATA_DIR}/test/anthracene_DA_test.xyz"], None, f"{ANTHRACENE_DA_DATA_DIR}/test/anthracene_DA_test.xyz"

    # ------------------------------------------

    # --- 2.5 anthracene 1,4 DA dataset construction ---
    # input_name_stub = f"anthracene_1_4_MMFF_toluene_froz_solv_PBC_opt_TS_MACE-OFF23_medium_anthracene_DA_train_46_config_energy_ft_300K_1.0ps*.xyz" # fine_tuned
    # input_fpaths = sorted(glob.glob(os.path.join(TRJ_DIR, input_name_stub)))
    
    # out_dir, save_name = [f"{ANTHRACENE_DA_DATA_DIR}/train/anthracene_DA_train.xyz"], f"{ANTHRACENE_DA_DATA_DIR}/train", "anthracene_1_4_DA_combined_trj"
    # # input_fpaths, out_dir, save_name = [f"{ANTHRACENE_DA_DATA_DIR}/train/anthracene_DA_train.xyz"], f"{ANTHRACENE_DA_DATA_DIR}/train", "anthracene_DA_train"

    # --- 3. guajadial DA dataset filtering ---

    # ts_ref_fpath = f'{GUAJADIAL_DA_DATA_DIR}/TS/guajadial_DA_vac_TS_wB97M.xyz'
    # rs = [(13, 31), (6, 32)]

    # 1. combine additional trajectory configs from AL and take only configurations within TS dist range
    # input_fpaths = [f"{GUAJADIAL_DA_DATA_DIR}/extra_traj/vac_traj_iter{i}_0.xyz" for i in list(range(9)) + list(range(11, 20))] + \
    #                [f"{GUAJADIAL_DA_DATA_DIR}/extra_traj/water_traj_iter{i}_0.xyz" for i in range(9)]

    # # # react_coord_dist_filter, out_fp = None, f"{GUAJADIAL_DA_DATA_DIR}/guajadial_extra_traj_samples.xyz"
    # react_coord_dist_filter, out_fp = (-0.3, 1.0), f"{GUAJADIAL_DA_DATA_DIR}/guajadial_extra_traj_samples_dist_filtered.xyz"

    # 2. force filtering of each part of the overall combined dataset
    # force_comp_filter = 7.0
    # # input_fpaths, out_fp = [f"{GUAJADIAL_DA_DATA_DIR}/guajadial_DA_vac_al_iter.xyz"], f"{GUAJADIAL_DA_DATA_DIR}/guajadial_DA_vac_al_iter_force_filtered.xyz" 
    # # input_fpaths, out_fp = [f"{GUAJADIAL_DA_DATA_DIR}/guajadial_DA_in_water_al_iter.xyz"], f"{GUAJADIAL_DA_DATA_DIR}/guajadial_DA_in_water_al_iter_force_filtered.xyz" 
    # # input_fpaths, out_fp = [f"{GUAJADIAL_DA_DATA_DIR}/guajadial_extra_traj_samples_dist_filtered_vac.xyz"], f"{GUAJADIAL_DA_DATA_DIR}/guajadial_extra_traj_samples_dist_filtered_vac_force_filtered.xyz" 
    # input_fpaths, out_fp = [f"{GUAJADIAL_DA_DATA_DIR}/guajadial_extra_traj_samples_dist_filtered_solv.xyz"], f"{GUAJADIAL_DA_DATA_DIR}/guajadial_extra_traj_samples_dist_filtered_solv_force_filtered.xyz" 

    # 3. use get_train_test_split.py to combine datasets and create train / valid / test set

    # 4. get train, valid and test by config types
    # input_fpaths, out_dir, save_name = [f"{GUAJADIAL_DA_DATA_DIR}/train/guajadial_DA_train.xyz"], f"{GUAJADIAL_DA_DATA_DIR}/train", "guajadial_DA_train"
    # input_fpaths, out_dir, save_name = [f"{GUAJADIAL_DA_DATA_DIR}/valid/guajadial_DA_valid.xyz"], f"{GUAJADIAL_DA_DATA_DIR}/valid", "guajadial_DA_valid"
    # input_fpaths, out_dir, save_name = [f"{GUAJADIAL_DA_DATA_DIR}/test/guajadial_DA_test.xyz"], f"{GUAJADIAL_DA_DATA_DIR}/test", "guajadial_DA_test"
    # input_fpaths, out_dir, save_name = [f"{GUAJADIAL_DA_DATA_DIR}/guajadial_extra_traj_samples_dist_filtered.xyz"], f"{GUAJADIAL_DA_DATA_DIR}", "guajadial_extra_traj_samples_dist_filtered"

    # type_filter, out_fp = {'vacuum'}, f"{out_dir}/{save_name}_vac_only.xyz"
    # type_filter, out_fp = {'solvated'}, f"{out_dir}/{save_name}_solv_only.xyz"

    # 5. combine two LoT datasets
    # input_fpaths, type_filter, out_fp = [f"{GUAJADIAL_DA_DATA_DIR}/test/guajadial_DA_test_wB97M_TZVPP.xyz"], {'solvated'}, f"{GUAJADIAL_DA_DATA_DIR}/test/guajadial_DA_test_wB97M_TZVPP_solv_only.xyz"

    # input_fpaths, out_fp = [f"{GUAJADIAL_DA_DATA_DIR}/test/guajadial_DA_test_wB97M_TZVPPD.xyz",
    #                         f"{GUAJADIAL_DA_DATA_DIR}/test/guajadial_DA_test_wB97M_TZVPP_solv_only.xyz"], f"{GUAJADIAL_DA_DATA_DIR}/test/guajadial_DA_test_wB97M_TZVPPD_complete"


    # --- 4. I Cl SN2 Dataset Filtering ---
    # force_comp_filter = 7.0
    # input_fpaths, out_fp = [f"{I_Cl_SN2_DATA_DIR}/I_Cl_SN2_vac_al_iter.xyz"], f"{I_Cl_SN2_DATA_DIR}/I_Cl_SN2_vac_al_iter_force_filtered.xyz"
    # input_fpaths, out_fp = [f"{I_Cl_SN2_DATA_DIR}/I_Cl_SN2_in_acetone_al_iter.xyz"], f"{I_Cl_SN2_DATA_DIR}/I_Cl_SN2_in_acetone_al_iter_force_filtered.xyz"

    # ------------------------------------------

    # --- FINAL TRAIN, VALID, TEST SETS FILTERING ---
    # input_fpaths, save_name = [f"{TRAIN_SUBSETS_DIR}/endo_DA_final_train_clean.xyz"], "endo_DA_final_train_clean"                                                             # random train set
    # input_fpaths, save_name = [f"{TRAIN_SUBSETS_DIR}/endo_DA_final_valid_clean.xyz"], "endo_DA_final_valid_clean"                                                             # random train set
    # input_fpaths, save_name = [f"{TRAIN_SUBSETS_DIR}/endo_DA_final_test_clean.xyz"], "endo_DA_final_test_clean"                                                             # random train set
    # input_fpaths, save_name = [f"{ENDO_DA_DATA_DIR}/endo_DA_final_test_combined.xyz"], "endo_DA_final_test_combined"                                                             # random train set
    # input_fpaths, save_name = [f"{ENDO_DA_DATA_DIR}/endo_DA_final_test_combined_force_filtered.xyz"], "endo_DA_final_test_combined"                                                             #  final test set
    
    # === SELECT TYPE FILTER AND OUT NAME ===
    # type_filter, out_fp = {'vacuum'}, f"{TRAIN_SUBSETS_DIR}/{save_name}_vacuum.xyz"
    # type_filter, out_fp = {'microsolvated'}, f"{TRAIN_SUBSETS_DIR}/{save_name}_microsolvated.xyz"
    # type_filter, out_fp = {'water'}, f"{ENDO_DA_DATA_DIR}/{save_name}_water.xyz"
    # type_filter, out_fp = {'reactants_in_water'}, f"{TRAIN_SUBSETS_DIR}/{save_name}_react_in_water.xyz"
    # type_filter, out_fp = {'vacuum', 'reactants_in_water', 'water'}, f"{TRAIN_SUBSETS_DIR}/{save_name}_no_microsolv.xyz"
    # type_filter, out_fp = {'vacuum', 'reactants_in_water'}, f"{TRAIN_SUBSETS_DIR}/{save_name}_vac_solv_only.xyz"
    # type_filter, out_fp = {'vacuum', 'microsolvated', 'reactants_in_water'}, f"{TRAIN_SUBSETS_DIR}/{save_name}_no_water.xyz"

    # === ENERGY FILTERING ===
    # energy_filter = 30.0
    # input_fpaths, out_fp = [f"{TRAIN_SUBSETS_DIR}/endo_DA_final_train.xyz"], f"{TRAIN_SUBSETS_DIR}/endo_DA_final_train_clean.xyz"
    # input_fpaths, out_fp = [f"{TRAIN_SUBSETS_DIR}/endo_DA_final_valid.xyz"], f"{TRAIN_SUBSETS_DIR}/endo_DA_final_valid_clean.xyz"
    # input_fpaths, out_fp = [f"{TRAIN_SUBSETS_DIR}/endo_DA_final_test.xyz"], f"{TRAIN_SUBSETS_DIR}/endo_DA_final_test_clean.xyz"

    # input_fpaths = [f"{TRAIN_SUBSETS_DIR}/endo_DA_train_size_sorted_MACE-OFF23_medium_overall_loss_influence_sorted.xyz"]           # MACE-OFF influence sorted train
    # filter, out_fp = {'vacuum'}, f"{TRAIN_SUBSETS_DIR}/endo_DA_train_MACE-OFF23_medium_overall_loss_influence_sorted_vac_only.xyz"

    # input_fpaths = [f"{TRAIN_SUBSETS_DIR}/endo_DA_train_size_sorted_GO-MACE-23_overall_loss_influence_sorted.xyz"]                  # GO-MACE influence sorted train
    # filter, out_fp = {'vacuum'}, f"{TRAIN_SUBSETS_DIR}/endo_DA_train_GO-MACE-23_overall_loss_influence_sorted_vac_only.xyz"

    # input_fpaths = [f"{TRAIN_SUBSETS_DIR}/endo_DA_train_size_sorted_MACE-MP0-128-L1_overall_loss_influence_sorted.xyz"]             # MACE-MP0 influence sorted train
    # filter, out_fp = {'vacuum'}, f"{TRAIN_SUBSETS_DIR}/endo_DA_train_size_sorted_MACE-MP0-128-L1_overall_loss_influence_sorted_vac_only.xyz"

    # === FORCE COMPONENT FILTERING ===
    # force_comp_filter = 7.0
    # input_fpaths, out_fp = [f"{TRAIN_SUBSETS_DIR}/endo_DA_final_train.xyz"], f"{TRAIN_SUBSETS_DIR}/endo_DA_final_train_force_filtered.xyz"
    # input_fpaths, out_fp = [f"{TRAIN_SUBSETS_DIR}/endo_DA_final_train_clean_vac_solv_only.xyz"], f"{TRAIN_SUBSETS_DIR}/endo_DA_final_train_clean_vac_solv_only_force_filtered.xyz"
    # input_fpaths, out_fp = [f"{TRAIN_SUBSETS_DIR}/endo_DA_final_valid_clean.xyz"], f"{TRAIN_SUBSETS_DIR}/endo_DA_final_valid_clean_force_filtered.xyz"
    # input_fpaths, out_fp = [f"{TRAIN_SUBSETS_DIR}/endo_DA_final_test.xyz"], f"{TRAIN_SUBSETS_DIR}/endo_DA_final_test_force_filtered.xyz"
    # input_fpaths, out_fp = [f"{ENDO_DA_DATA_DIR}/endo_DA_final_test_combined.xyz"], f"{ENDO_DA_DATA_DIR}/endo_DA_final_test_combined_force_filtered.xyz"

    # ======================================

    # SCRIPT SETTINGS
    debug = False

    # get distances from TS
    ts_ref_config = flexible_ase_read(ts_ref_fpath)[0]
    ts_ref_bond_distances = {ri: ts_ref_config.get_distance(*ri) for ri in rs}
    print(f'TS ref bond distances: {ts_ref_bond_distances}')

    random.seed(3)
    combined_configs = []
    for input_fp in input_fpaths:

        configs = flexible_ase_read(input_fp)
        print("No. configs: ", len(configs))

        # config type filter
        if type_filter is not None:
            print(f'Using config_type filters: {type_filter}')
            assert all(config.info.get('config_type') is not None for config in configs), f"Config type(s) {type_filter} not labelled in input dataset..."
            init_len = len(configs)
            configs = [config for config in configs if config.info['config_type'] in type_filter]
            print(f'Got {len(configs)} filtered configs from {input_fp} (filtered out {init_len - len(configs)} configs)..')
        
        # energy filter
        if energy_filter is not None:
            print(f'Using energy filter: {energy_filter} eV')
            rel_energies = get_relative_energies(configs)
            init_len = len(configs)
            configs = [config for i, config in enumerate(configs) if rel_energies[i] <= energy_filter]
            print(f'Got {len(configs)} filtered configs from {input_fp} (filtered out {init_len - len(configs)} configs)..')

        # force component filter
        if force_comp_filter is not None:
            print(f'Using force component filter: {force_comp_filter} eV / A')
            force_comps = [np.abs(get_ase_atoms_forces(config)) for config in configs]
            removed_config_tuples = [(config, i) for i, config in enumerate(configs) if np.any(force_comps[i] > force_comp_filter)]
            configs = [config for i, config in enumerate(configs) if np.all(force_comps[i] <= force_comp_filter)]
            print(f'Got {len(configs)} filtered configs from {input_fp} (filtered out {len(removed_config_tuples)} configs)..')

            # debug, save removed configs as well for inspection
            removed_configs = [x[0] for x in removed_config_tuples]
            removed_config_indices = [x[1] for x in removed_config_tuples]
            print(f'Removed indices: {removed_config_indices}')
            if debug:
                removed_out_fp = out_fp.split('.')[0] + '_removed.xyz'
                write(removed_out_fp, removed_configs, format='extxyz')
                print(f"Saved {len(removed_configs)} removed configs to {removed_out_fp} for debug...")

        # react coord distance filter
        if react_coord_dist_filter is not None:
            print(fr'Using reaction coordinate distance filter range: {react_coord_dist_filter} \$AA$')
            ts_dists = [sum([(config.get_distance(*ri, mic=True) - ts_ref_bond_distances[ri]) for ri in rs]) for config in configs]

            if reverse_selection:
                configs = [config for i, config in enumerate(configs) if ts_dists[i] <= react_coord_dist_filter[0] or ts_dists[i] >= react_coord_dist_filter[1]]
            else:
                configs = [config for i, config in enumerate(configs) if react_coord_dist_filter[0] < ts_dists[i] < react_coord_dist_filter[1]]
                
            print(f'Got {len(configs)} filtered configs from {input_fp}..')

        if slice_filter is not None:
            print(f'Slicing every {slice_filter} configs...')
            configs = configs[::slice_filter]
            print(f'Num sliced configs: {len(configs)}')

        combined_configs.extend(configs)

        if shuffle_dataset:
            random.shuffle(combined_configs)

    # remove duplicates
    # print("Combined configs: ", combined_configs)
    combined_configs, removed_indices = remove_duplicates(combined_configs)
    # print("Combined configs after dup removed: ", combined_configs)

    # write combined dataset
    if not debug:
        write(out_fp, combined_configs, format=out_format)
        print(f"Saved {len(combined_configs)} filtered configs to {out_fp}...")
