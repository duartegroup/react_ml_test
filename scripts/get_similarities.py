"""
Script for calculating cosine similarities between two sets of input descriptors (MACE of SOAP).
"""

import sys

import numpy as np
import torch
from ase.data import atomic_numbers

sys.path.append("../")
from react_ml_test.helpers.dim_red import calc_best_similarities
from react_ml_test.helpers.io import get_name_from_path, load_data_cache
from react_ml_test.helpers.logging import logger
from react_ml_test.helpers.settings import (
    CPD_MVK_DA_DATA_DIR,
    CSV_DIR,
    DATASETS_DIR,
    DESCRIPTORS_DIR,
    MODEL_DIR,
    NPZ_DIR,
    PLOT_DIR,
    SIMILARITIES_DIR,
)

# set CUDA device
CUDA_DEVICE_NUM = 0

CUDA_DEVICE = f"cuda:{CUDA_DEVICE_NUM}"
if torch.cuda.is_available():
    torch.cuda.set_device(CUDA_DEVICE_NUM)

if __name__ == "__main__":

    hdf5_file = None
    
    # ===== INPUT REFERENCE DESCRIPTORS  =====
    
    # GO-MACE!
    ref_name = "GO-MACE-23_train"
    ref_desc_fpaths = [f"{DESCRIPTORS_DIR}/{ref_name}_GO-MACE-23_mace_desc.npz"]
    model_name = "GO-MACE-23"
    
    # MACE-OFF23
    # ref_name = "spice_train"
    # model_name = "MACE-OFF23_medium"
    # # hdf5_file = load_hdf5_cache(f"{DESCRIPTORS_DIR}/{ref_name}_mace_desc.h5")
    # # ref_desc_fpaths = [f"{batch_name}.npz"for batch_name in hdf5_file.keys()]
    # ref_name_stub = f"{DESCRIPTORS_DIR}/{ref_name}_batch_*mace_desc.npz"  # MACE-MP0 OHC only train set in batches
    # ref_desc_fpaths = sorted(glob.glob(ref_name_stub), key=natural_sort_key)
    # assert len(ref_desc_fpaths) > 0, f"No files detected with name stub {ref_name_stub}..."
    # print('Ref Desc Fpaths: ', ref_desc_fpaths)

    # MACE-MP0
    # ref_name = "mace_mp0_mptrj_data"
    # ref_name_stub = f"{DESCRIPTORS_DIR}/{ref_name}_OHC_batch_*mace_desc.npz"  # MACE-MP0 OHC only train set in batches
    # ref_desc_fpaths = sorted(glob.glob(ref_name_stub), key=natural_sort_key)
    # assert len(ref_desc_fpaths) > 0, f"No files detected with name stub {ref_name_stub}..."
    # model_name = "MACE-MP-0b3_medium"

    # ===== INPUT TEST DESCRIPTORS  =====
    # test_desc_fpath = f'{DESCRIPTORS_DIR}/endo_DA_train_random_{model_name}_mace_desc.npz'
    # test_desc_fpath = f'{DESCRIPTORS_DIR}/endo_DA_final_train_clean_vacuum_{model_name}_mace_desc.npz'
    # test_desc_fpath = f'{DESCRIPTORS_DIR}/endo_DA_final_train_clean_react_in_water_{model_name}_mace_desc.npz'
    # test_desc_fpath = f'{DESCRIPTORS_DIR}/endo_DA_final_train_clean_water_{model_name}_mace_desc.npz'

    # --- specific samples + combined dataset for paper ---
    test_desc_fpaths = [
        # f"{DESCRIPTORS_DIR}/cis_endo_TS_wB97M_{model_name}_mace_desc.npz",        # opt TS
        # f"{DESCRIPTORS_DIR}/endo_DA_vac_reactants_{model_name}_mace_desc.npz",    # 
        # f"{DESCRIPTORS_DIR}/endo_DA_vac_products_{model_name}_mace_desc.npz",     # 
        # f"{DESCRIPTORS_DIR}/endo_DA_water_TS_{model_name}_mace_desc.npz",         # TODO: only want the reactant atoms
        # f"{DESCRIPTORS_DIR}/endo_DA_water_reactants_{model_name}_mace_desc.npz",  # only want the reactant atoms
        # f"{DESCRIPTORS_DIR}/endo_DA_water_products_{model_name}_mace_desc.npz",   # only want reactant atoms
        # f"{DESCRIPTORS_DIR}/endo_DA_water_{model_name}_mace_desc.npz",             
        # f"{DESCRIPTORS_DIR}/endo_DA_wB97M_combined_test_{model_name}_mace_desc.npz",   # full test set + extra evaluation sets
        # f"{DESCRIPTORS_DIR}/CPD_MVK_DA_test_{model_name}_mace_desc.npz",
        f"{DESCRIPTORS_DIR}/anthracene_DA_test_{model_name}_mace_desc.npz",
        f"{DESCRIPTORS_DIR}/guajadial_DA_test_{model_name}_mace_desc.npz",
    ]

    # test for correlation (endo DA example)
    # test_desc_fpaths = [
    #     f"{DESCRIPTORS_DIR}/endo_DA_wB97M_combined_test_{model_name}_mace_desc.npz"
    # ]

    # === CHOOSE SIMILARITY MEASURE ===
    sim_calc = 'cosine'
    # sim_calc = 'euclidean'
    element_stratify = True
    zeta = 8  # power term
    # ======================================

    # OTHER OPTIONS
    x1_batch_size = 100
    x2_batch_size = 2000
    override_cache = False  
    # ======

    combined_ref_desc, combined_ref_elements = [], []
    for i, ref_desc_fpath in enumerate(ref_desc_fpaths):

        # load ref mace descs
        cache_data = load_data_cache(ref_desc_fpath, hdf5_file=hdf5_file)
        assert cache_data is not None, f'{ref_desc_fpath} not found...'
        ref_mace_desc, ref_elements, config_lens = cache_data['mace_desc'], cache_data['elements'], cache_data['config_lens']
        if np.issubdtype(ref_elements.dtype, np.bytes_):
            ref_elements = np.array([element.decode('utf-8') for element in ref_elements])
        combined_ref_desc.append(ref_mace_desc)
        combined_ref_elements.append(ref_elements)
        # print(ref_mace_desc.shape)

    combined_ref_desc = np.concatenate(combined_ref_desc, axis=0)
    logger.info(f"Combined ref MACE desc shape: {combined_ref_desc.shape}")
    combined_ref_atom_nums = np.array([atomic_numbers[e] for e in ref_elements])

    for test_desc_fpath in test_desc_fpaths:

        test_name = get_name_from_path(test_desc_fpath)
        
        # load test mace descs
        cache_data = load_data_cache(test_desc_fpath)
        assert cache_data is not None, f'{test_desc_fpath} not found...'
        test_mace_desc, test_elements, config_lens = cache_data['mace_desc'], cache_data['elements'], cache_data['config_lens']
        if np.issubdtype(test_elements.dtype, np.bytes_):
            test_elements = np.array([element.decode('utf-8') for element in test_elements])
        test_atom_nums = np.array([atomic_numbers[e] for e in test_elements])
        logger.info(test_mace_desc.shape)

        # get best cosine similarities of each x1 -> x2 descriptor
        logger.info(f'Calculating best MACE {sim_calc} similarities between {test_name} and {ref_name}...')
        logger.info(f"Using batch size for test: {x1_batch_size}")
        logger.info(f"Using batch size for reference: {x2_batch_size}")
        element_stratify_str = "_per_element" if element_stratify else ""
        sim_cache_fpath = f"{SIMILARITIES_DIR}/{test_name}_{sim_calc}_sims{element_stratify_str}_z={zeta}.npz"

        config_mean_best_sims, best_sim_values, best_sim_indices = calc_best_similarities(x1=test_mace_desc, 
                                                                                        x2=combined_ref_desc,
                                                                                        zeta=zeta,
                                                                                        x1_atom_nums=test_atom_nums,
                                                                                        x2_atom_nums=combined_ref_atom_nums,
                                                                                        element_stratify_mean=element_stratify,
                                                                                        sim_calc=sim_calc,
                                                                                        x1_batch_size=x1_batch_size,
                                                                                        x2_batch_size=x2_batch_size,
                                                                                        config_lens=config_lens,
                                                                                        cache_fpath=sim_cache_fpath)  

        logger.info(f"Overall Mean Similarity: {np.mean(config_mean_best_sims)}")
        del(test_mace_desc)
        del(test_elements)
