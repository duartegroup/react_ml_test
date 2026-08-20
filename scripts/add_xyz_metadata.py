"""
Adds some xyz metadata to an .xyz file.= (i.e. config type labels / box dimensions)
"""
import sys

from ase.io import write

sys.path.append('../')
from react_ml_test.helpers.io import flexible_ase_read, get_name_from_path
from react_ml_test.helpers.settings import (
    ANTHRACENE_DA_DATA_DIR,
    CPD_MVK_DA_DATA_DIR,
    DATASETS_DIR,
    GUAJADIAL_DA_DATA_DIR,
    REACTION_DATA_DIR,
)

if __name__ == "__main__":

    config_types: dict[str, callable] = None
    box_dim: list[int] = None
    charge, spin = None, None

    # ==== SPECIFY INPUT FP HERE ==== 

    # --- CPD MVK system ---
    # input_fps = [f"{CPD_MVK_DA_DATA_DIR}/endo_DA_wB97M_combined_test.xyz"]
    # input_fps = [f"{CPD_MVK_DA_DATA_DIR}/train/CPD_MVK_DA_train.xyz", 
    #              f"{CPD_MVK_DA_DATA_DIR}/valid/CPD_MVK_DA_valid.xyz", 
    #              f"{CPD_MVK_DA_DATA_DIR}/test/CPD_MVK_DA_test.xyz"]
    # input_fps = [f"{CPD_MVK_DA_DATA_DIR}/test/CPD_MVK_DA_test_vac_only.xyz", 
    #              f"{CPD_MVK_DA_DATA_DIR}/test/CPD_MVK_DA_test_solv_only.xyz"]
    # input_fps = [f"{CPD_MVK_DA_DATA_DIR}/train/subsets/CPD_MVK_DA_train_42_config_strat_rep_{i}.xyz" for i in [1, 2, 3]]
    # # input_fps = [f"{CPD_MVK_DA_DATA_DIR}/test/CPD_MVK_DA_test_PBE_def2_TZVP.xyz"]
    # input_fps = [f"{CPD_MVK_DA_DATA_DIR}/test/CPD_MVK_DA_test_wB97M_TZVPPD.xyz"]

    # # select many
    # # pattern = f"{CPD_MVK_DA_DATA_DIR}/train/subsets/*.xyz"
    # # input_fps = [fp for fp in glob.glob(pattern)]

    # # config type spec
    # config_types = {
    #                 'vacuum': lambda x: x == 22, 
    #                 'solvated': lambda x: x > 80, 
    #                 }
    # box_dim = [100.0, 100.0, 100.0]
    # charge, spin = 0, 1

    # --- anthracene system ---
    # input_fps = [f"{ANTHRACENE_DA_DATA_DIR}/anthracene_DA_vac_al_iter.xyz",
    #              f"{ANTHRACENE_DA_DATA_DIR}/anthracene_DA_in_toluene_al_iter.xyz",
    #              f"{ANTHRACENE_DA_DATA_DIR}/anthracene_DA_vac_al_iter_force_filtered.xyz", 
    #              f"{ANTHRACENE_DA_DATA_DIR}/anthracene_DA_in_toluene_al_iter_force_filtered.xyz"]

    # input_fps = [f"{ANTHRACENE_DA_DATA_DIR}/full_al_traj_samples_dist_filtered_orca.xyz"]
    # input_fps = [f"{ANTHRACENE_DA_DATA_DIR}/train/anthracene_DA_train.xyz", 
    #              f"{ANTHRACENE_DA_DATA_DIR}/valid/anthracene_DA_valid.xyz", 
    #              f"{ANTHRACENE_DA_DATA_DIR}/test/anthracene_DA_test.xyz"]
    # input_fps = [
    #              f"{ANTHRACENE_DA_DATA_DIR}/test/anthracene_DA_test_vac_only.xyz",
    #              f"{ANTHRACENE_DA_DATA_DIR}/test/anthracene_DA_test_solv_only.xyz"
    #              ]
    # # input_fps = [f"{ANTHRACENE_DA_DATA_DIR}/valid/anthracene_1_4_MMFF_vac_MACE-OFF23_46_reaction.xyz"]
    # # input_fps = [f"{ANTHRACENE_DA_DATA_DIR}/test/anthracene_DA_test_PBE_def2_TZVP.xyz"]
    # input_fps = [f"{ANTHRACENE_DA_DATA_DIR}/test/anthracene_DA_test_wB97M_TZVPP.xyz",
    #              f"{ANTHRACENE_DA_DATA_DIR}/test/anthracene_DA_test_wB97M_TZVPPD.xyz"]

    # # # train subsets
    # # # pattern = f"{ANTHRACENE_DA_DATA_DIR}/train/subsets/*.xyz"
    # # # input_fps = [fp for fp in glob.glob(pattern)]

    # config_types = {
    #                 'vacuum': lambda x: x == 41, 
    #                 'solvated': lambda x: x > 42, 
    #                }
    # box_dim = [100.0, 100.0, 100.0]
    # charge, spin = 0, 1

    # --- guajadial DA system ---
    # input_fps = [f"{GUAJADIAL_DA_DATA_DIR}/guajadial_DA_vac_al_iter.xyz", f"{GUAJADIAL_DA_DATA_DIR}/guajadial_DA_in_water_al_iter.xyz"]
    # input_fps = [f"{GUAJADIAL_DA_DATA_DIR}/guajadial_extra_traj_samples_dist_filtered.xyz"]
    # input_fps = [f"{GUAJADIAL_DA_DATA_DIR}/guajadial_extra_traj_samples_dist_filtered_vac.xyz"]
    # input_fps = [f"{GUAJADIAL_DA_DATA_DIR}/guajadial_extra_traj_samples_dist_filtered_solv.xyz"]
    # input_fps = [f"{GUAJADIAL_DA_DATA_DIR}/test/guajadial_DA_test_PBE_def2_TZVP.xyz"]
    # input_fps = [f"{GUAJADIAL_DA_DATA_DIR}/train/guajadial_DA_train.xyz", 
    #              f"{GUAJADIAL_DA_DATA_DIR}/valid/guajadial_DA_valid.xyz", 
    #              f"{GUAJADIAL_DA_DATA_DIR}/test/guajadial_DA_test.xyz"]
    # input_fps = [f"{GUAJADIAL_DA_DATA_DIR}/test/guajadial_DA_test_vac_only.xyz", 
    #              f"{GUAJADIAL_DA_DATA_DIR}/test/guajadial_DA_test_solv_only.xyz"]
    input_fps = [f"{GUAJADIAL_DA_DATA_DIR}/test/guajadial_DA_test_wB97M_TZVPP.xyz",
                 f"{GUAJADIAL_DA_DATA_DIR}/test/guajadial_DA_test_wB97M_TZVPPD.xyz"]

    # # # train subsets
    # # # pattern = f"{GUAJADIAL_DA_DATA_DIR}/train/subsets/*.xyz"
    # # # input_fps = [fp for fp in glob.glob(pattern)]

    config_types = {
                    'vacuum': lambda x: x == 69, 
                    'solvated': lambda x: x > 70, 
                    }
    box_dim = [100.0, 100.0, 100.0]
    charge, spin = 0, 1

    # # clear config type labels
    # config_types = {
    #                 None: lambda x: True
    #                }

    # --- I CL SN2 System ---
    # input_fps = [
    #              f"{I_Cl_SN2_TRAIN_DIR}/I_Cl_SN2_train.xyz",
    #              f"{I_Cl_SN2_TRAIN_DIR}/I_Cl_SN2_valid.xyz",
    #              f"{I_Cl_SN2_TRAIN_DIR}/I_Cl_SN2_test.xyz",
    #              f"{I_Cl_SN2_TRAIN_DIR}/I_Cl_SN2_vac_train.xyz",
    #              f"{I_Cl_SN2_TRAIN_DIR}/I_Cl_SN2_vac_valid.xyz",
    #              f"{I_Cl_SN2_TRAIN_DIR}/I_Cl_SN2_vac_test.xyz",
    #              f"{I_Cl_SN2_TRAIN_DIR}/I_Cl_SN2_in_acetone_train.xyz",
    #              f"{I_Cl_SN2_TRAIN_DIR}/I_Cl_SN2_in_acetone_valid.xyz",
    #              f"{I_Cl_SN2_TRAIN_DIR}/I_Cl_SN2_in_acetone_test.xyz",
    #              ]

    # config_types = {
    #                 'vacuum': lambda x: x == 12, 
    #                 'solvated': lambda x: x > 100, 
    #                }

    # ==========================================

    DEBUG = False

    # ====

    for input_fp in input_fps:
        # input configs
        configs = flexible_ase_read(input_fp)
        print('Num configs: ', len(configs))

        # label config type (custom for diels alder reaction)
        if config_types is not None:
            for config in configs:
                x = len(config)
                for config_type, checking_func in config_types.items():
                    if checking_func(x):
                        if config_type is None:
                            config.info.pop('config_type', None)
                        else:
                            config.info['config_type'] = config_type

        if box_dim is not None:

            for config in configs:
                config.set_cell(box_dim)
                config.set_pbc([True, True, True])

        if charge is not None and spin is not None:
            for config in configs:
                config.info["total_charge"] = charge
                config.info["total_spin"] = spin

        # save to an XYZ file
        out_dir = '/'.join(input_fp.split('/')[0:-1])
        input_name = get_name_from_path(input_fp)
        out_fp = input_fp
        if DEBUG:
            out_fp = f"{out_dir}/{input_name}_annotated.xyz"
        write(out_fp, configs, format='extxyz')
        print(f"Saved annotated xyz to {out_fp}...")
