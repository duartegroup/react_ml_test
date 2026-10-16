"""
Script to perform a train / valid / test split of an input dataset (or group of datasets)
and save these as seperate files.
"""


from react_ml_test_public.helpers.settings import (
    ANTHRACENE_DA_DATA_DIR,
    GUAJADIAL_DA_DATA_DIR,
    I_Cl_SN2_DATA_DIR,
)
from react_ml_test_public.helpers.stats import round_sigfigs
from react_ml_test_public.helpers.train_utils import train_test_split

if __name__ == "__main__":

    # ===== specify sub-training sets file paths =====

    # --- dies alder paper datasets ---
    # out_dir = f"{TRAIN_DATA_DIR}/subsets"
    # combined_ds_name = "endo_DA_final"
    # ds_names = ["endo_ace_wB97M_data",
    #             "endo_in_water_random_data",
    #             "endo2_water_data",
    #             "watertol_ace_wB97M_27_data"]
    # ds_fps = [f"{TRAIN_DATA_DIR}/{train_set_name}.xyz" for train_set_name in ds_names]

    # --- anthracene DA dataset ---
    # out_dir = f"{ANTHRACENE_DA_DATA_DIR}"
    # combined_ds_name = "anthracene_DA"
    # ds_fps = [f"{ANTHRACENE_DA_DATA_DIR}/anthracene_DA_vac_al_iter_force_filtered.xyz", 
    #           f"{ANTHRACENE_DA_DATA_DIR}/anthracene_DA_in_toluene_al_iter_force_filtered.xyz",
    #           f"{ANTHRACENE_DA_DATA_DIR}/anthracene_DA_al_traj_samples_dist_filtered_orca_force_filtered.xyz"
    #           ]

    # --- guajadial DA dataset ---
    out_dir = f"{GUAJADIAL_DA_DATA_DIR}"
    combined_ds_name = "guajadial_DA"
    ds_fps = [f"{GUAJADIAL_DA_DATA_DIR}/guajadial_DA_vac_al_iter_force_filtered.xyz", 
              f"{GUAJADIAL_DA_DATA_DIR}/guajadial_DA_in_water_al_iter_force_filtered.xyz",
              f"{GUAJADIAL_DA_DATA_DIR}/guajadial_extra_traj_samples_dist_filtered_vac_force_filtered.xyz",
              f"{GUAJADIAL_DA_DATA_DIR}/guajadial_extra_traj_samples_dist_filtered_solv_force_filtered.xyz"
              ]

    # --- I Cl SN2 dataset ---
    # out_dir = I_Cl_SN2_TRAIN_DIR
    # combined_ds_name = "I_Cl_SN2"
    # ds_fps = [f"{I_Cl_SN2_DATA_DIR}/I_Cl_SN2_vac_al_iter_force_filtered.xyz", 
    #           f"{I_Cl_SN2_DATA_DIR}/I_Cl_SN2_in_acetone_al_iter_force_filtered.xyz"]

    # === SPECIFY SETTINGS FOR SPLIT ===
    valid_frac, test_frac = 0.15, 0.2
    mode = "random"
    seed = 3
    # =================================

    # split data into train, validation and test sets, we use the implicit
    train_frac = round_sigfigs(1.0 - valid_frac - test_frac, 2)
    train_test_split(ds_fps,
                     train_frac,
                     valid_frac,
                     test_frac,
                     mode,
                     seed=seed,
                     save_individual_subsets=False,
                     combined_save_name=combined_ds_name,
                     out_dir=out_dir
                     )
