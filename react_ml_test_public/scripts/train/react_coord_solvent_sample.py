"""
Script to run sampling of a reactive system dataset by distance from TS + fraction of solvation.
"""


from react_ml_test_public.helpers.sampling import react_coord_solvent_sampling
from react_ml_test_public.helpers.settings import (
    ANTHRACENE_DA_DATA_DIR,
    CPD_MVK_DA_DATA_DIR,
    GUAJADIAL_DA_DATA_DIR,
)

if __name__ == "__main__":

    # === CHANGE INPUTS HERE ===

    # --- CPD + MVK diels alder reaction ---
    # vacuum_ds_fpath = f"{CPD_MVK_DA_DATA_DIR}/train/endo_DA_final_train_clean_vacuum.xyz"
    # solvated_ds_fpath = f"{CPD_MVK_DA_DATA_DIR}/train/endo_DA_final_train_clean_react_in_water.xyz"
    # # vacuum_ds_fpath = f"{CPD_MVK_DA_DATA_DIR}/test/endo_DA_final_test_clean_vacuum.xyz"
    # # solvated_ds_fpath = f"{CPD_MVK_DA_DATA_DIR}/test/endo_DA_final_test_clean_react_in_water.xyz"
    # rs = [(1, 12), (6, 11)]
    # ts_ref_fpath = f"{CPD_MVK_DA_DATA_DIR}/TS/cis_endo_TS_wB97M.xyz"
    # out_dir = f"{CPD_MVK_DA_DATA_DIR}/train/subsets"
    # # r_dists = [0.4, 1.5, 3.0, 6.0]
    # # solv_fracs = [0.0, 0.25, 0.50, 0.75, 1.0]
    # r_dists = [1.5]
    # solv_fracs = [0.50]
    # sample_mode = 'diverse'
    # # sample_mode = 'local'
    # n_samples = 50 # [8, 16, 24]
    # save_name = "endo_DA"

    # --- anthracene DA ---
    # vacuum_ds_fpath = f"{ANTHRACENE_DA_DATA_DIR}/train/anthracene_DA_train_vac_only.xyz"
    # solvated_ds_fpath = f"{ANTHRACENE_DA_DATA_DIR}/train/anthracene_DA_train_solv_only.xyz"
    # rs = [(2, 20), (5, 22)]
    # ts_ref_fpath = f'{ANTHRACENE_DA_DATA_DIR}/TS/anthracene_DA_vac_TS_wB97M.xyz'
    # out_dir = f"{ANTHRACENE_DA_DATA_DIR}/train/subsets"
    # r_dists = [0.5, 1.5, 3.0, 6.0]
    # solv_fracs = [0.0, 0.25, 0.50, 0.75, 1.0]
    # sample_mode = 'diverse'
    # # sample_mode = 'local'
    # n_samples = 8 # [8, 16, 24]
    # save_name = "anthracene_DA"

    # --- guajadial DA ---
    vacuum_ds_fpath = f"{GUAJADIAL_DA_DATA_DIR}/train/guajadial_DA_train_vac_only.xyz"
    solvated_ds_fpath = f"{GUAJADIAL_DA_DATA_DIR}/train/guajadial_DA_train_solv_only.xyz"
    rs = [(13, 31), (6, 32)]
    ts_ref_fpath = f'{GUAJADIAL_DA_DATA_DIR}/TS/guajadial_DA_vac_TS_wB97M.xyz'
    out_dir = f"{GUAJADIAL_DA_DATA_DIR}/train/subsets"
    r_dists = [0.5, 1.5, 3.0, 6.0]
    solv_fracs = [0.0, 0.25, 0.50, 0.75, 1.0]
    sample_mode = 'diverse'
    n_samples = 8 # [8, 16, 24]
    save_name = "guajadial_DA"

    # choose rep
    # seed, rep_num = 3, 1
    # seed, rep_num = 5, 2
    seed, rep_num = 7, 3
    save = True

    react_coord_solvent_sampling(vacuum_ds_fpath=vacuum_ds_fpath,
                                 solvated_ds_fpath=solvated_ds_fpath,
                                 rs=rs,
                                 ts_ref_fpath=ts_ref_fpath,
                                 r_dists=r_dists,
                                 solv_fracs=solv_fracs,
                                 n_samples=n_samples,
                                 out_dir=out_dir,
                                 save_name=save_name,
                                 sample_mode=sample_mode,
                                 seed=seed,
                                 rep_num=rep_num,
                                 save=save)
