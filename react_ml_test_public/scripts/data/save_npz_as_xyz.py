"""
Simple script to use mlptrain to load and save an npz file as a .xyz file.
"""

import mlptrain as mlt

from react_ml_test_public.helpers.settings import (
    ANTHRACENE_DA_DATA_DIR,
    DATASETS_DIR,
    GUAJADIAL_DA_DATA_DIR,
    I_Cl_SN2_DATA_DIR,
)

if __name__ == "__main__":

    box = None

    # === DEFINE NPZ FPATH ===
    # npz_fp = f"{DATASETS_DIR}/da_paper_data/valid/endo_in_water_ace_wB97M_short_orca.npz"
    # xyz_save_fp = f"{DATASETS_DIR}/da_paper_data/valid/endo_in_water_ace_wB97M_short_orca_non_periodic.xyz"
    # box = None

    # npz_fp = f"{DATASETS_DIR}/da_paper_data/valid/endo_in_water_ace_wB97M_small_orca.npz"
    # xyz_save_fp = f"{DATASETS_DIR}/da_paper_data/valid/endo_in_water_ace_wB97M_small_orca.xyz"
    # box = None

    # npz_fp = f"{DATASETS_DIR}/da_paper_data/valid/endo_ace_wB97M_nonpca_orca.npz"
    # xyz_save_fp = f"{DATASETS_DIR}/da_paper_data/valid/endo_ace_wB97M_nonpca_orca.xyz"
    # box = None

    # --- anthracene systems ---
    # npz_fp = f"{ANTHRACENE_DA_DATA_DIR}/anthracene_gas_phase_al.npz"
    # xyz_save_fp = f"{ANTHRACENE_DA_DATA_DIR}/anthracene_gas_phase_al.xyz"
    # box = None

    # npz_fp = f"{ANTHRACENE_DA_DATA_DIR}/dataset_after_iter_35.npz"
    # xyz_save_fp = f"{ANTHRACENE_DA_DATA_DIR}/anthracene_DA_in_toluene_al_iter.xyz"
    # box = None

    # --- guajadial system ---
    # npz_fp = f"{GUAJADIAL_DA_DATA_DIR}/guajadial_vacuum_al.npz"
    # xyz_save_fp = f"{GUAJADIAL_DA_DATA_DIR}/guajadial_DA_vac_al_iter.xyz"
    # box = None

    npz_fp = f"{GUAJADIAL_DA_DATA_DIR}/guajadial_water_al.npz"
    xyz_save_fp = f"{GUAJADIAL_DA_DATA_DIR}/guajadial_DA_in_water_al_iter.xyz"
    box = None

    # --- I CL SN2 system ---
    # npz_fp = f"{I_Cl_SN2_DATA_DIR}/gasphase_dataset_after_iter_49.npz"
    # xyz_save_fp = f"{I_Cl_SN2_DATA_DIR}/I_Cl_SN2_vac_al_iter.xyz"
    # box = None

    # npz_fp = f"{I_Cl_SN2_DATA_DIR}/dataset_after_iter_11.npz"
    # xyz_save_fp = f"{I_Cl_SN2_DATA_DIR}/I_Cl_SN2_in_acetone_al_iter.xyz"
    # box = None

    config_set = mlt.ConfigurationSet(allow_duplicates=True)
    config_set._load_npz(npz_fp)
    if box is not None:
        for config in config_set:
            config.box = box
    config_set.save_xyz(xyz_save_fp, true=True, predicted=False)

    print(f"Converted {npz_fp} to {xyz_save_fp}...")