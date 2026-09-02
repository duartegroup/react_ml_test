"""
Script to save a sample of an xyz dataset very efficiently, without opening the entire dataset.
"""
import time

from ase.io import write

import random

from react_ml_test_public.helpers.io import get_name_from_path, get_xyz_file_len
from react_ml_test_public.helpers.sampling import sample_dataset
from react_ml_test_public.helpers.settings import (
    DATASETS_DIR,
    GUAJADIAL_DA_DATA_DIR,
    REACTION_DATA_DIR,
    REPLAY_DATASET_DIR,
)

if __name__ == "__main__":

    # === SPECIFY DATASETS AND SAMPLES ===
    # datasets
    # dataset_fp = f"{DATASETS_DIR}/go_mace/GO-MACE-23_train.xyz"  # go-mace training set
    dataset_fp = f"{REPLAY_DATASET_DIR}/spice_train.xyz"  # spice training set
    # dataset_fp = f"{DATASETS_DIR}/mace_mp0/mace_mp0_mptrj_data.xyz"  # mace_mptrj
    # dataset_fp = f"{DATASETS_DIR}/ani/ANI-1xnr.xyz"  # ani
    # dataset_fp = f"{DATASETS_DIR}/ani/ANI-1ccx.xyz"  # ani
    # dataset_fp = f"{DATASETS_DIR}.xyz"  # ani
    # dataset_fp = f'{REACTION_DATA_DIR}/endo_DA/neb_endo_DA_vac_wB97M.xyz'
    # dataset_fp = f'{REACTION_DATA_DIR}/endo_DA/endo_DA_water_ace_wB97M_non_pbc.xyz'
    # dataset_fp = f'{REACTION_DATA_DIR}/endo_DA/endo_DA_vac_ace_wB97M.xyz'
    # dataset_fp = f"{REPLAY_DATASET_DIR}/mace_mp0_mptrj_data_OHC.xyz"  # mace_mptrj
    # dataset_fp = f"{REPLAY_DATASET_DIR}/GO-MACE-23_train.xyz"
    dataset_fp = f"{GUAJADIAL_DA_DATA_DIR}/guajadial_extra_traj_samples_dist_filtered_vac_only.xyz"
    # dataset_fp = f"{GUAJADIAL_DA_DATA_DIR}/guajadial_extra_traj_samples_dist_filtered_solv_only.xyz"
    
    # sample settings
    specific_indices = None
    n_samples = 40
    random_sample = True
    random_seed = 3
    element_set =  {'C', 'H', 'O'}
    # =====================================
    
    # get random samples of dataset, without opening it all
    dataset_name = get_name_from_path(dataset_fp)
    start_time = time.time()
    print(f"Sampling dataset {dataset_name}...")
    n_samples = len(specific_indices) if specific_indices is not None else n_samples
    configs_sample = sample_dataset(dataset_fp, 
                                    num_samples=n_samples, 
                                    specific_indices=specific_indices, 
                                    element_filter_set=element_set,
                                    random_sample=random_sample,
                                    random_seed=random_seed)
    print(f"Sampling configs took {time.time() - start_time: .4f}...")
    print(f"Taken {len(configs_sample)} samples...")

    # Save to an XYZ file
    out_dir = '/'.join(dataset_fp.split('/')[0:-1])
    str_elements = '_' + ''.join(element for element in element_set) if element_set is not None else ''
    str_n_samples = f"_sample_n={len(configs_sample)}" if n_samples is not None or specific_indices is not None else ''
    out_fp = f"{out_dir}/{dataset_name}{str_n_samples}{str_elements}.xyz"
    write(out_fp, configs_sample, format='extxyz')
    print(f"Saved dataset sample to {out_fp}...")
