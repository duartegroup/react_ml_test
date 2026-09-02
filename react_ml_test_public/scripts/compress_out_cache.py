"""
Script to compress all npz files of a given file name
into a single hdf5 file for further efficiency in loading.
"""

from react_ml_test_public.helpers.io import save_combined_hdf5_cache
from react_ml_test_public.helpers.settings import DESCRIPTORS_DIR

if __name__ == "__main__":

    # INPUT DATASET NAME STUB
    input_dir = DESCRIPTORS_DIR
    # input_ds_name_stub = 'mace_mp0_mptrj_data_OHC'
    input_ds_name_stub = 'spice_train'
    # input_ds_name_stub = 'GO-MACE-23_train'
    data_type = 'mace_desc'
    # =======================

    # save hdf5 file
    out_cache_fpath = f'{input_dir}/{input_ds_name_stub}_{data_type}.h5'
    npz_fpath_stub = f"{input_dir}/{input_ds_name_stub}*_{data_type}"
    save_combined_hdf5_cache(out_cache_fpath, npz_fpath_stub)