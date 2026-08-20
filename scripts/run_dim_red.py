"""
Script for running PCA / UMAP on high dimensional data from an npz file, and saving it.
"""
import glob
import sys
from typing import Literal

import numpy as np

sys.path.append("../")
from react_ml_test.helpers.dim_red import run_pca, run_umap
from react_ml_test.helpers.io import (
    get_name_from_path,
    load_data_cache,
    load_hdf5_cache,
    save_npz_data_cache,
)
from react_ml_test.helpers.logging import logger
from react_ml_test.helpers.settings import DESCRIPTORS_DIR, NPZ_DIR
from react_ml_test.helpers.utils import natural_sort_key

dim_red_type: Literal['PCA', 'UMAP']

if __name__ == "__main__":

    input_name_stub = None
    hdf5_file = None

    # ===== REFERENCE DESCRIPTORS FILES =====
    # GO-MACE-23 train set
    # input_name_stub = f"{DESCRIPTORS_DIR}/GO-MACE-23_train_batch_*mace_desc.npz"  # go-mace test batches
    # ref_fps = sorted(glob.glob(input_name_stub), key=natural_sort_key)
    # assert len(ref_fps) > 0, f"No files detected with name stub {input_name_stub}..."
    # model_name = 'GO-MACE-23'

    # SPICE training set in batches
    # model_name = 'MACE-OFF23_medium'
    # input_name_stub = f"{DESCRIPTORS_DIR}/spice_train_batch_*mace_desc.npz"  
    # ref_fps = sorted(glob.glob(input_name_stub), key=natural_sort_key)
    # hdf5_file = load_hdf5_cache(f"{DESCRIPTORS_DIR}/spice_train_mace_desc") 

    # MACE-MP0 OHC only train set in batches
    model_name = 'MACE-MP-0b3_medium'
    input_name_stub = f"{DESCRIPTORS_DIR}/mace_mp0_mptrj_data_OHC_batch_*mace_desc.npz"  
    ref_fps = sorted(glob.glob(input_name_stub), key=natural_sort_key)
    # ref_fps = [ref_fps[0]]  # DEBUG!!
    
    # --- paper test fps ---
    test_fps = [
        f"{DESCRIPTORS_DIR}/endo_DA_full_dataset_force_filtered_vac_only_{model_name}_mace_desc.npz",   # full combined train, valid, test set + extra evaluation sets
        f"{DESCRIPTORS_DIR}/endo_DA_full_dataset_force_filtered_solv_only_{model_name}_mace_desc.npz",   # full combined train, valid, test set + extra evaluation sets
        f"{DESCRIPTORS_DIR}/cis_endo_TS_wB97M_{model_name}_mace_desc.npz",                     # opt TS
        f"{DESCRIPTORS_DIR}/endo_DA_water_TS_{model_name}_mace_desc.npz",                      # only want the reactant atoms
    ]
    test_names = ['Endo DA Vac', 'Endo DA Solv', 'Vac TS', 'Water TS']
    save_name = 'endo_DA'

    # input_fps = [f"{DESCRIPTORS_DIR}/endo_DA_train_random_GO-MACE-23_mace_desc.npz"]  # train set
    # input_fps = [f"{DESCRIPTORS_DIR}/endo_DA_train_random_{model_name}_mace_desc.npz",
    #              f"{DESCRIPTORS_DIR}/endo_DA_train_random_filtered_{model_name}_mace_desc.npz",        # vac, microsolvated, react in water
    #              f"{DESCRIPTORS_DIR}/endo_DA_train_random_microsolvated_{model_name}_mace_desc.npz",   # microsolvated only
    #              f"{DESCRIPTORS_DIR}/endo_DA_train_random_vacuum_{model_name}_mace_desc.npz",          # vacuum only
    #              f"{DESCRIPTORS_DIR}/endo_DA_train_random_react_in_water_{model_name}_mace_desc.npz"]  # only reactants in water

    # input_fp = f"{DATASETS_DIR}/mace_mp0/mace_mp0_mptrj_data_OHC.xyz"  # MACE-MP0 train set for OHC only
    # input_name_stub = f"{DATASETS_DIR}/mace_mp0/mace_mp0_mptrj_data_OHC_batch_*"  # MACE-MP0 OHC only train set in batches
    # input_fps = sorted(glob.glob(input_name_stub), key=natural_sort_key)
    
    # === DIMENSIONALITY REDUCTION PARAMETERS ===
    n_atom_env = 100000  # num atom environments to sample
    pca_dim = 32
    umap_dim = 2
    # ======================================
    
    ds_id_map = {name: i for i, name in enumerate([model_name] + test_names)}

    logger.info(f'Ref File Paths: {ref_fps}')
    logger.info(f'Test File Paths: {test_fps}')

    # load reference descriptors
    combined_ref_desc, combined_ref_elements, combined_ref_config_lens = [], [], []
    for i, ref_desc_fpath in enumerate(ref_fps):

        # load ref mace descs
        cache_data = load_data_cache(ref_desc_fpath, hdf5_file=hdf5_file)
        assert cache_data is not None, f'{ref_desc_fpath} not found...'
        ref_mace_desc, ref_elements, config_lens = cache_data['mace_desc'], cache_data['elements'], cache_data['config_lens']
        if np.issubdtype(ref_elements.dtype, np.bytes_):
            ref_elements = np.array([element.decode('utf-8') for element in ref_elements])
        combined_ref_desc.append(ref_mace_desc)
        combined_ref_elements.append(ref_elements)
        combined_ref_config_lens.append(config_lens)
        print(ref_mace_desc.shape)

    combined_ref_desc = np.concatenate(combined_ref_desc, axis=0)
    logger.info(f"Combined ref MACE desc shape: {combined_ref_desc.shape}")
    combined_ref_elements = np.concatenate(combined_ref_elements, axis=0)
    combined_ref_config_lens = np.concatenate(combined_ref_config_lens, axis=0)

    logger.info(f'Descriptor Datatype: {combined_ref_desc.dtype}')
    combined_ref_desc = combined_ref_desc.astype(np.float32)
    logger.info(f"Converted to {combined_ref_desc.dtype}")

    # random sample indices from 
    if n_atom_env is not None:
        rand_ind = np.random.choice(len(combined_ref_desc), n_atom_env, replace=False)
        combined_ref_desc = combined_ref_desc[rand_ind]
        combined_ref_elements = combined_ref_elements[rand_ind]
        combined_ref_config_lens = np.array([])
        logger.info(f"Sampled {len(combined_ref_desc)} atomic descriptors... (ref config lens are now meaningless)")

    # keep record of dataset ids
    ds_ids = [ds_id_map[model_name] for _ in range(len(combined_ref_elements))]

    # create concatenated array of all test features
    combined_test_desc, combined_test_elements, combined_test_config_lens = [], [], []
    for npz_fpath, test_name in zip(test_fps, test_names):
        logger.info(f"Processing: {npz_fpath}")
    
        # load mace descs
        cache_data = load_data_cache(npz_fpath, hdf5_file=hdf5_file)
        assert cache_data is not None, f'{npz_fpath} not found...'
        mace_desc, elements, config_lens = cache_data['mace_desc'], cache_data['elements'], cache_data['config_lens']
        logger.info(f"Loaded desc shape: {mace_desc.shape}")
        combined_test_desc.append(mace_desc)
        combined_test_elements.append(elements)
        combined_test_config_lens.append(config_lens)
        ds_ids.extend([ds_id_map[test_name] for _ in range(len(elements))])
   
    combined_test_desc = np.concatenate(combined_test_desc, axis=0)
    logger.info(f"Combined test desc shape: {combined_test_desc.shape}")
    combined_test_elements = np.concatenate(combined_test_elements, axis=0)
    combined_test_config_lens = np.concatenate(combined_test_config_lens, axis=0)

    # get final concatenated array of descriptors
    combined_desc = np.concatenate([combined_ref_desc, combined_test_desc], axis=0)
    combined_elements = np.concatenate([combined_ref_elements, combined_test_elements], axis=0)
    # combined_atom_nums = np.concatenate([combined_ref_atom_nums, combined_test_atom_nums], axis=0)
    combined_config_lens = np.concatenate([combined_ref_config_lens, combined_test_config_lens], axis=0)
    ds_ids = np.array(ds_ids)

    logger.info(f"Final Combined Desc Shape: {combined_desc.shape}")
    logger.info(f"Final Combined Elements Shape: {combined_elements.shape}")
    logger.info(f"Final Combined Config Lens Shape: {combined_config_lens.shape}")
    logger.info(f"DS IDs: {ds_ids.shape}")

    # run dim red on features, and save
    pca_str = f'_PCA_{pca_dim}' if pca_dim is not None else ''
    umap_str = f'_UMAP_{umap_dim}' if pca_dim is not None else ''
    output_pca_npz_fpath = f"{DESCRIPTORS_DIR}/{model_name}_{save_name}{pca_str}{umap_str}.npz"
    if pca_dim is not None:
        logger.info(f'Compressing descriptors for {model_name}_{save_name} with PCA to {pca_dim} dimensions...')
        combined_desc = run_pca(combined_desc, n_components=pca_dim)
    if umap_dim is not None:
        logger.info(f'Compressing descriptors for {model_name}_{save_name} with UMAP to {umap_dim} dimensions...')
        combined_desc = run_umap(combined_desc, n_components=umap_dim)
    logger.info(f"Compressed desc shape: {combined_desc.shape}")
    save_npz_data_cache(output_pca_npz_fpath, {'desc': combined_desc, 'elements': combined_elements, 
                                                'config_lens': combined_config_lens, 'ds_ids': ds_ids, 'ds_id_map': ds_id_map})
    