"""
Remove duplicates from a dataset and associated cache files, if specified.
"""

import sys

import numpy as np
from ase.io import write

sys.path.append('../')

from react_ml_test.helpers.io import (
    flexible_ase_read,
    load_data_cache,
    save_npz_data_cache,
)
from react_ml_test.helpers.sampling import remove_duplicates
from react_ml_test.helpers.settings import (
    ANTHRACENE_DA_DATA_DIR,
    CPD_MVK_DA_DATA_DIR,
    DATASETS_DIR,
    GUAJADIAL_DA_DATA_DIR,
    SIMILARITIES_DIR,
)
from react_ml_test.helpers.utils import unflatten_array_by_lengths

if __name__ == "__main__":

    input_desc_fpaths = None

    # === SELECT DATA ===

    model_names = ['GO-MACE-23', 'MACE-OFF23_medium', 'MACE-MP-0b3_medium']

    # CPD + MVK
    # # input_index = None
    # input_name = "CPD_MVK_DA_test"
    # input_fpath = f"{CPD_MVK_DA_DATA_DIR}/test/{input_name}.xyz"
    # input_desc_fpaths = [f"{SIMILARITIES_DIR}/{input_name}_{model_name}_mace_desc_cosine_sims_per_element_z=8.npz" for model_name in model_names]
    
    # anthracene

    # guajadial
    input_index = 108    
    input_name = "guajadial_DA_test"
    input_fpath = f"{GUAJADIAL_DA_DATA_DIR}/test/guajadial_DA_test.xyz"
    
    # ===================

    input_desc_fpaths = [f"{SIMILARITIES_DIR}/{input_name}_{model_name}_mace_desc_cosine_sims_per_element_z=8.npz" for model_name in model_names]
    input_configs = flexible_ase_read(input_fpath)
    print("Number fo input configs: ", len(input_configs))
    input_configs, removed_test_indices = remove_duplicates(input_configs)
    print("Removed indices: ", removed_test_indices)

    if len(removed_test_indices) < 1:
        print("Nothing to do, no duplicates detected.")
        sys.exit(0)

    # WRITE MODE:
    write_outputs = True

    if input_desc_fpaths is not None:

        for input_desc_fpath in input_desc_fpaths:

            # load desc similarities
            cache_data = load_data_cache(input_desc_fpath)
            best_sim_values, best_sim_indices, config_lens = cache_data['best_sim_values'].flatten(), cache_data['best_sim_indices'], cache_data['config_lens']
            
            # test_config_lens = [test_config_lens[i] for i in test_sorted_indices]
            print(f"Best Sim Values shape: {best_sim_values.shape}")
            print(f"Best Sim Indices shape: {best_sim_indices.shape}")

            # per config sims and forces
            config_best_sims = unflatten_array_by_lengths(best_sim_values, config_lens)
            config_best_sim_indices = unflatten_array_by_lengths(best_sim_indices, config_lens)
            print("Len config sims: ", len(config_best_sims))

            config_best_sims = [x for i, x in enumerate(config_best_sims) if i not in removed_test_indices]
            config_best_sim_indices = [x for i, x in enumerate(config_best_sim_indices) if i not in removed_test_indices]
            config_lens = [x for i, x in enumerate(config_lens) if i not in removed_test_indices]
            print("Len filtered config sims: ", len(config_best_sims))

            # reconcatenate config-wise arrays and save cache
            best_sim_values = np.concatenate(config_best_sims)
            best_sim_indices = np.concatenate(config_best_sim_indices)
            print(f"Best Sim Values filtered shape: {best_sim_values.shape}")
            print(f"Best Sim Indices filtered shape: {best_sim_indices.shape}")

            # overwrite outputs
            if write_outputs:
                save_npz_data_cache(input_desc_fpath, {'best_sim_values': best_sim_values,
                                                       'best_sim_indices': best_sim_indices,
                                                       'config_lens': config_lens})

    if write_outputs:
        # save test set
        write(input_fpath, input_configs, format='extxyz')
        print("Overwriting input fpath to: ", input_fpath)