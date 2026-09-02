"""
Script for calculating MACE descriptors for a given input dataset.
"""



import time

import torch


from react_ml_test_public.helpers.descriptors import get_mace_descriptors, get_soap_descriptors
from react_ml_test_public.helpers.io import (
    flexible_ase_read,
    get_name_from_path,
    get_xyz_file_len,
    load_model_calculator,
)
from react_ml_test_public.helpers.logging import logger
from react_ml_test_public.helpers.settings import (
    ANTHRACENE_DA_DATA_DIR,
    CPD_MVK_DA_DATA_DIR,
    CSV_DIR,
    DATASETS_DIR,
    GUAJADIAL_DA_DATA_DIR,
    MODEL_DIR,
    NPZ_DIR,
    PLOT_DIR,
)

# set CUDA device
CUDA_DEVICE_NUM = 0

CUDA_DEVICE = f"cuda:{CUDA_DEVICE_NUM}"
torch.cuda.set_device(CUDA_DEVICE_NUM)

if __name__ == "__main__":

    # ===== INPUT DATASET(S)  =====

    # input_fps = [f"{TRAIN_SUBSETS_DIR}/CPD_MVK_DA_final_train_clean.xyz"]  # random train set
    # input_fps = [f"{TRAIN_SUBSETS_DIR}/CPD_MVK_DA_final_train_clean_vacuum.xyz",
    #              f"{TRAIN_SUBSETS_DIR}/CPD_MVK_DA_final_train_clean_react_in_water.xyz",
    #              f"{TRAIN_SUBSETS_DIR}/CPD_MVK_DA_final_train_clean_water.xyz"
    #              ]

    # test set
    input_fps = [
        # f"{CPD_MVK_DA_DATA_DIR}/test/CPD_MVK_DA_test.xyz",
        f"{ANTHRACENE_DA_DATA_DIR}/test/anthracene_DA_test.xyz",
        f"{GUAJADIAL_DA_DATA_DIR}/test/guajadial_DA_test.xyz",
    ]

    # specific samples for close comparison
    # input_fps = [
    #     f"{CPD_MVK_DA_DATA_DIR}/descriptor_comp/cis_CPD_MVK_TS_wB97M.xyz",
    #     f"{CPD_MVK_DA_DATA_DIR}/descriptor_comp/CPD_MVK_DA_vac_reactants.xyz",
    #     f"{CPD_MVK_DA_DATA_DIR}/descriptor_comp/CPD_MVK_DA_vac_products.xyz",
    #     f"{CPD_MVK_DA_DATA_DIR}/descriptor_comp/CPD_MVK_DA_water_TS.xyz",         # only want the reactant atoms
    #     f"{CPD_MVK_DA_DATA_DIR}/descriptor_comp/CPD_MVK_DA_water_reactants.xyz",  # only want the reactant atoms
    #     f"{CPD_MVK_DA_DATA_DIR}/descriptor_comp/CPD_MVK_DA_water_products.xyz",   
    #     f"{CPD_MVK_DA_DATA_DIR}/descriptor_comp/CPD_MVK_DA_water.xyz",
    #     f"{CPD_MVK_DA_DATA_DIR}/descriptor_comp/CPD_MVK_DA_full_dataset_force_filtered.xyz"   # full combined train, valid, test set + extra evaluation sets
    # ]

    # --- pretraining datasets ---

    # OFF
    # input_name_stub = f"{DATASETS_DIR}/spice/spice_train_batch_*"  # spice training set in batches
    # input_fps = sorted(glob.glob(input_name_stub), key=natural_sort_key)
    # assert len(input_fps) > 0, f"No files detected with name stub {input_name_stub}..."
    # input_fps = input_fps[7:]
    # model_names = ["MACE-OFF23_medium"]

    # MP
    # input_fp = f"{DATASETS_DIR}/mace_mp0/mace_mp0_mptrj_data_OHC.xyz"  # MACE-MP0 train set for OHC only
    # input_name_stub = f"{DATASETS_DIR}/mace_mp0/mace_mp0_mptrj_data_OHC_batch_*"  # MACE-MP0 OHC only train set in batches
    # input_fps = sorted(glob.glob(input_name_stub), key=natural_sort_key)
    # assert len(input_fps) > 0, f"No files detected with name stub {input_name_stub}..."
    # model_names = ["MACE-MP-0b3_medium"]
    
    # GO
    # input_fps = [f"{DATASETS_DIR}/go_mace/GO-MACE-23_train.xyz"]  # go-mace training set
    # # input_name_stub = f"{DATASETS_DIR}/go_mace/GO-MACE-23_train_batch_*"  # go-mace test batches
    # # input_fps = sorted(glob.glob(input_name_stub), key=natural_sort_key)
    # assert len(input_fps) > 0, f"No files detected with name stub {input_name_stub}..."
    # model_names = ["GO-MACE-23"]

    # === SELECT MODEL ===

    # --- paper zero shot models ---
    model_names = [
        "GO-MACE-23",
        "MACE-OFF23_medium", 
        "MACE-MP-0b3_medium",
    ]

    # ======================================

    logger.info(f'Getting MACE descriptors for input datasets: {input_fps}')
    for input_fp in input_fps:

        # load input dataset
        input_name = get_name_from_path(input_fp)
        logger.info(f"Loading input dataset {input_fp}... (this may take some time!)")
        total_num_configs = get_xyz_file_len(input_fp)
        logger.info(f"Total num configs in input dataset: {total_num_configs}")
        start_time = time.time()
        configs = flexible_ase_read(input_fp)
        logger.info(f"Loaded configs in {time.time() - start_time: .4f}s")

        # get number of atoms in total sample
        num_atoms = len([atom for config in configs for atom in config ])
        logger.info(f"Total number of atoms: {num_atoms}")

        # get elements
        # elements = np.array([element for config in configs for element in config.get_chemical_symbols()])
        # print("Unique Elements: ", np.unique(elements))

        for model_name in model_names:
            # load model
            model_fpath = f"{MODEL_DIR}/{model_name}.model"
            calculator = load_model_calculator(model_name, 
                                            device=CUDA_DEVICE, 
                                            dtype="float32")

            # get mace descriptors for training dataset
            logger.info("Getting MACE descriptors...")
            mace_desc, elements = get_mace_descriptors(calculator, 
                                                    configs,
                                                    flatten_output=True,
                                                    cache_name=f'{input_name}_{model_name}')
            print('Elements shape: ', elements.shape)
            del(mace_desc)
            del(elements)
