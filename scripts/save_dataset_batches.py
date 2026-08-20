"""
Script to save a sample of an xyz dataset very efficiently, without opening the entire dataset.
"""
import sys
import time

sys.path.append('../')
from react_ml_test.helpers.io import save_dataset_batches
from react_ml_test.helpers.logging import logger
from react_ml_test.helpers.settings import DATASETS_DIR

if __name__ == "__main__":

    # === SPECIFY DATASETS AND SAMPLES ===
    # datasets
    dataset_fp = f"{DATASETS_DIR}/go_mace/GO-MACE-23_train.xyz"  # go-mace training set
    # dataset_fp = f"{DATASETS_DIR}/spice/spice_train.xyz"  # spice training set
    # dataset_fp = f"{DATASETS_DIR}/mace_mp0/mace_mp0_mptrj_data.xyz"  # mace_mptrj
    # dataset_fp = f"{DATASETS_DIR}/mace_mp0/mace_mp0_mptrj_data_OHC.xyz"  # mace_mptrj
    # dataset_fp = f"{DATASETS_DIR}/ani/ANI-1xnr.xyz"  # ani
    # dataset_fp = f"{DATASETS_DIR}/ani/ANI-1ccx.xyz"  # ani
    # dataset_fp = f"{DATASETS_DIR}.xyz"  # ani
    # dataset_fp = f'{REACTION_DATA_DIR}/endo_DA/neb_endo_DA_vac_wB97M.xyz'
    # dataset_fp = f'{REACTION_DATA_DIR}/endo_DA/endo_DA_water_ace_wB97M_non_pbc.xyz'
    # dataset_fp = f'{REACTION_DATA_DIR}/endo_DA/endo_DA_vac_ace_wB97M.xyz'
    
    # sample settings
    batch_size = 500
    # =====================================
    
    # get random samples of dataset, without opening it all
    start_time = time.time()
    logger.info(f"Saving dataset batches for {dataset_fp}...")
    save_dataset_batches(dataset_fp, batch_size)
    logger.info(f"Saving batches took {time.time() - start_time: .4f}...")
