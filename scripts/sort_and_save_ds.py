"""
Script to save a sample of an xyz dataset.
"""
import random
import sys

from ase.io import write

sys.path.append('../')
from react_ml_test.helpers.io import flexible_ase_read, get_name_from_path
from react_ml_test.helpers.settings import (
    ANTHRACENE_DA_DATA_DIR,
    CPD_MVK_DA_DATA_DIR,
    DATASETS_DIR,
    GUAJADIAL_DA_DATA_DIR,
)
from react_ml_test.helpers.sorting import (
    sort_configs_by_cv_dist,
    sort_configs_by_energy,
    sort_configs_by_soap_desc,
    sort_configs_by_structure,
    sort_dataset_by_size,
)

if __name__ == "__main__":

    sort_mode: str = ''
    out_fmt='extxyz'

    # ===== SPECIFY DATASET ===== 

    # -- CPD MVK DA ---
    # input_ds_fp = f"{CPD_MVK_DA_DATA_DIR}/endo_DA_train.xyz"
    # input_ds_fp = f"{CPD_MVK_DA_DATA_DIR}/endo_DA_train_soap_sim_MACE-OFF-23_reversed.xyz"
    # input_ds_fp = f"{CPD_MVK_DA_DATA_DIR}/endo_DA_train_soap_sim_MACE-MP0.xyz"
    # input_ds_fp = f"{CPD_MVK_DA_DATA_DIR}/endo_DA_valid.xyz"
    # input_ds_fp = f"{CPD_MVK_DA_DATA_DIR}/endo_DA_final_train_clean.xyz"
    # input_ds_fp = f"{CPD_MVK_DA_DATA_DIR}/test/endo_DA_PBE_combined_test.xyz"
    # input_ds_fp = f"{CPD_MVK_DA_DATA_DIR}/test/endo_DA_PBE_combined_test_rerun.xyz"
    input_ds_fp = f"{CPD_MVK_DA_DATA_DIR}/test/endo_DA_wB97M_combined_test.xyz"

    # reverse reactant trajectory
    # input_ds_fp = f"{ANTHRACENE_DA_DATA_DIR}/react_traj/anthracene_1_4_MMFF_vac_MACE-OFF23_46_reactants.xyz"

    # IRCs for anthracene to reverse
    # input_ds_fp = f"{ANTHRACENE_DA_DATA_DIR}/valid/anthracene_1_4_DA_TS_opt_wB97M_IRC_Full_trj.xyz"
    # input_ds_fp = f"{ANTHRACENE_DA_DATA_DIR}/valid/anthracene_DA_wB97M_TS_opt_IRC_Full_trj.xyz"

    # IRCs for guajadial to reverse
    # input_ds_fp = f"{GUAJADIAL_DA_DATA_DIR}/valid/guajadial_TS_opt_wB97M_IRC_F_trj.xyz"
    # input_ds_fp = f"{GUAJADIAL_DA_DATA_DIR}/valid/opt_F_trj.xyz"

    # load ds
    configs = flexible_ase_read(input_ds_fp)

    # ===== CHOOSE SORT MODE =====

    # -1. dummy reverse order
    # sort_mode = 'reversed'
    # sorted_configs = configs[::-1]

    # === 0. random === 
    # sort_mode = 'random'
    # random.seed(3)
    # random.shuffle(configs)
    # sorted_configs = configs

    # === 1. order based on distance of both pairs === 
    # sort_mode = 'cv_sorted'
    # cv = [(1, 12), (6, 11)]
    # sorted_configs = sort_configs_by_cv_dist(configs, cv)

    # === 2. sort based on energies (more stable energies first) === 
    # sort_mode = 'energy_sorted'
    # sorted_configs = sort_dataset_by_energy(configs)

    # sort_mode = 'energy_sorted_reversed'
    # sorted_configs = sort_dataset_by_energy(configs, reverse=True)

    # for config in sorted_configs:
    #     config.info.clear()  # weird ase error was happening with .info dict, check this in the future...

    # === 3a. soap similarity (more similar structures first) === 
    # specify pre-training configs file
    # sort_mode = 'soap_sim_MACE-OFF-23_reversed'
    # pretrain_ds_fp = f"{DATASETS_DIR}/spice/spice_train_sample_n=1000.xyz"

    # sort_mode, reverse = 'soap_sim_GO-MACE-23', False
    # sort_mode, reverse = 'soap_sim_GO-MACE-23_reversed', True
    # pretrain_ds_fp = f"{DATASETS_DIR}/go_mace/GO-MACE-23_train_sample_n=1000.xyz"

    # sort_mode, reverse = 'soap_sim_MACE-MP0', False
    # sort_mode, reverse = 'soap_sim_MACE-MP0_reversed', True
    # pretrain_ds_fp = f"{DATASETS_DIR}/mace_mp0/mace_mp0_mptrj_data_sample_n=1000.xyz"

    # pretrain_configs = flexible_ase_read(pretrain_ds_fp)
    # sorted_configs = sort_configs_by_soap_desc(configs, 
    #                                            pretrain_configs, 
    #                                            mode='sim', 
    #                                            reverse=reverse,
    #                                            compress_dim=512)

    # === 3b. soap distance === 
    # sort_mode = 'soap_'

    # === 4a. mace similarity === 


    # === 4b. mace distance === 

    # === 5. size ===
    # sort_mode = 'size_sorted'
    # sorted_configs = sort_dataset_by_size(configs)

    # === 6. just reverse ===
    # sort_mode = 'reversed'
    # configs.reverse()
    # sorted_configs = configs
    
    # === 7. sort by atomic position seed ===
    sort_mode = 'structure_sorted'
    sorted_configs = sort_configs_by_structure(configs)

    # ===========================================================

    # save to an XYZ file
    out_dir = '/'.join(input_ds_fp.split('/')[0:-1])
    dataset_name = get_name_from_path(input_ds_fp)
    out_fp = f"{out_dir}/{dataset_name}_{sort_mode}.xyz"
    write(out_fp, sorted_configs, format=out_fmt)
    print(f"Saved {sort_mode} dataset to {out_fp}...")
