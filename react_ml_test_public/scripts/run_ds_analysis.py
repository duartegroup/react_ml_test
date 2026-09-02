from time import time

import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm

from react_ml_test_public.helpers.descriptors import get_soap_descriptors
from react_ml_test_public.helpers.dim_red import run_pca
from react_ml_test_public.helpers.io import (
    autode_props_parser,
    flexible_ase_read,
    get_ase_atoms_energy,
    get_name_from_path,
    load_data_cache,
    load_hdf5_cache,
    save_npz_data_cache,
)
from react_ml_test_public.helpers.logging import logger
from react_ml_test_public.helpers.plot import bar_plot, plot_2d
from react_ml_test_public.helpers.settings import (
    ANTHRACENE_DA_DATA_DIR,
    CPD_MVK_DA_DATA_DIR,
    CSV_DIR,
    DATASETS_DIR,
    DPI,
    GRAPH_PES_MODEL_LOADER,
    GUAJADIAL_DA_DATA_DIR,
    ISOLATED_ATOMS_DIR,
    KCAL_PER_MOL,
    MACE_MODEL_LOADER,
    MEV_PER_ATOM,
    MODEL_DIR,
    MODELS,
    NPZ_DIR,
    PLOT_DIR,
    SAMPLES_DIR,
    SAMPLES_OUT_DIR,
    SINGLE_COLUMN_FIG_SIZE,
    I_Cl_SN2_DATA_DIR,
    PlotSettings,
    eV_to_kcal_mol,
    eV_to_mEV,
)
from react_ml_test_public.helpers.utils import (
    eV_to_meV_per_atom,
    get_ase_props,
    load_model_calculator,
    natural_sort_key,
    unflatten_array_by_lengths,
)

N_CORES = 1


if __name__ == "__main__":
    input_ds_name: str = None
    include_energies = True
    include_forces = True
    custom_xyz_parsers = None


    # ===== SELECT DS NAME =====
    # input_ds_fps = [f"{TRAIN_SUBSETS_DIR}/endo_DA_final_train_clean_vacuum.xyz"]
    # input_ds_fps = [f"{TRAIN_SUBSETS_DIR}/endo_DA_final_train_clean_react_in_water.xyz"]
    # input_ds_fps = [f"{TRAIN_SUBSETS_DIR}/endo_DA_final_train_clean.xyz"]
    # input_ds_fps = [f"{TRAIN_SUBSETS_DIR}/endo_DA_final_valid_clean.xyz"]
    # input_ds_fps = [f"{TRAIN_SUBSETS_DIR}/endo_DA_final_test_clean.xyz"]

    # validation comparison
    # input_ds_fps = [
    #                 f"{TRAIN_SUBSETS_DIR}/endo_DA_final_valid_clean.xyz",
    #                 f"{TRAIN_SUBSETS_DIR}/endo_DA_final_valid_clean_force_filtered.xyz", 
    #                 # f"{TRAIN_SUBSETS_DIR}/endo_DA_final_train_clean_vac_solv_only.xyz",
    #                 # f"{TRAIN_SUBSETS_DIR}/endo_DA_valid.xyz"
    #                 # f"{TRAIN_SUBSETS_DIR}/endo_DA_valid_no_microsolv.xyz",
    #                 ]

    # --- unfiltered train, valid test ---
    # input_ds_fps = [
    #                 f"{TRAIN_SUBSETS_DIR}/endo_DA_final_train.xyz",
    #                 f"{TRAIN_SUBSETS_DIR}/endo_DA_final_valid.xyz",
    #                 f"{TRAIN_SUBSETS_DIR}/endo_DA_final_test_.xyz",
    #                ]
    # dataset_labels = ['Train', 'Valid', 'Test']

    # train, test, valid comparison
    # input_ds_fps = [
    #                 f"{TRAIN_SUBSETS_DIR}/endo_DA_final_train_clean_vac_solv_only_force_filtered.xyz",
    #                 f"{TRAIN_SUBSETS_DIR}/endo_DA_final_valid_clean_force_filtered.xyz",
    #                 f"{ENDO_DA_DATA_DIR}/endo_DA_final_test_combined_force_filtered.xyz",
    #                ]
    # dataset_labels = ['Train', 'Valid', 'Test']

    # --- level of theory comparison ---
    # input_ds_fps = [
    #                 # f"{TRAIN_SUBSETS_DIR}/endo_DA_final_test_clean.xyz",
    #                 # f"{TRAIN_SUBSETS_DIR}/endo_DA_final_test_clean_PBE_def2-SVP.xyz",
    #                 # f"{ENDO_DA_DATA_DIR}/endo_DA_neb_gas_wB97M.xyz",
    #                 # f"{ENDO_DA_DATA_DIR}/endo_DA_neb_gas_PBE_def2-SVP.xyz",
    #                 f"{ENDO_DA_DATA_DIR}/endo_DA_vac_ace_wB97M.xyz",
    #                 f"{ENDO_DA_DATA_DIR}/endo_DA_vac_ace_PBE_def2-SVP.xyz"
    #                 # f"{ENDO_DA_DATA_DIR}/endo_DA_water_ace_wB97M_non_pbc.xyz",
    #                 # f"{ENDO_DA_DATA_DIR}/endo_DA_water_ace_non_pbc_PBE_def2-SVP.xyz",  # TODO: SOME VALUES MISSING!
    #                ]
    # dataset_labels = ['ωB97M-D3BJ/def2-TZVP', 'PBE def2_SVP']
    # save_name = 'endo_DA_DFT_comparison'

    # --- CPD + MVK DA comparison ---

    # # ts reference
    # ts_ref_config = flexible_ase_read(f'{CPD_MVK_DA_DATA_DIR}/TS/endo_DA_vac_TS_wB97M.xyz')[0]
    # react_coords = [(1, 12), (6, 11)]
    # test_elements = {'C', 'H', 'O'}

    # # input_ds_fps = [
    # #     f"{ENDO_DA_DATA_DIR}/endo_ace_wB97M_data.xyz",
    # #     f"{ENDO_DA_DATA_DIR}/endo_in_water_random_data.xyz",
    # #     f"{ENDO_DA_DATA_DIR}/watertol_ace_wB97M_27_data.xyz",
    # # ]
    # # dataset_labels = ['endo_ace_wB97M_data', 'endo_in_water_random_data']

    # # filtered train, valid test
    # input_ds_fps = [
    #                 f"{CPD_MVK_DA_DATA_DIR}/train/CPD_MVK_DA_train.xyz",
    #                 f"{CPD_MVK_DA_DATA_DIR}/valid/CPD_MVK_DA_valid.xyz",
    #                 f"{CPD_MVK_DA_DATA_DIR}/test/CPD_MVK_DA_test.xyz",
    #                ]
    # # dataset_labels = ['Train']
    # dataset_labels = ['Train', 'Valid', 'Test']
    # save_name = 'CPD_MVK_DA'

    # old vs new test set comparison
    # input_ds_fps = [
    #                 # f"{CPD_MVK_DA_DATA_DIR}/test/CPD_MVK_DA_test_PBE.xyz",
    #                 # f"{CPD_MVK_DA_DATA_DIR}/test/CPD_MVK_DA_test.xyz",
    #                 # f"{CPD_MVK_DA_DATA_DIR}/test/endo_DA_PBE_combined_test.xyz",
    #                 f"{CPD_MVK_DA_DATA_DIR}/test/endo_DA_PBE_combined_test_structure_sorted.xyz",
    #                 # f"{CPD_MVK_DA_DATA_DIR}/test/endo_DA_PBE_combined_test_rerun.xyz",
    #                 f"{CPD_MVK_DA_DATA_DIR}/test/endo_DA_PBE_combined_test_rerun_structure_sorted.xyz",
    #                 # f"{CPD_MVK_DA_DATA_DIR}/test/endo_DA_wB97M_combined_test.xyz",
    #                ]
    # dataset_labels = [get_name_from_path(ds_fp) for ds_fp in input_ds_fps]
    # save_name = 'CPD_MVK_DA_pbe_wB97M_test_set_comp'

    # additional test sets
    # input_ds_fps = [
    #     # f"{ENDO_DA_DATA_DIR}/endo_ace_wB97M_nonpca_orca.xyz",
    #     # f"{ENDO_DA_DATA_DIR}/endo_DA_neb_gas_wB97M.xyz",
    #     f"{ENDO_DA_DATA_DIR}/endo_DA_neb_water_wB97M.xyz",
    #     # f"{ENDO_DA_DATA_DIR}/endo_DA_water_ace_wB97M_non_pbc.xyz",
    # ]
    # dataset_labels = [get_name_from_path(fp) for fp in input_ds_fps]
    # # custom_xyz_parsers = [None, None, autode_props_parser, None]
    # custom_xyz_parsers = [autode_props_parser]
    # include_forces = False
    # save_name = 'endo_DA_additional_tests'

    # selected ft model
    # rep_num, n = 1, 42
    # input_ds_fps = [
    #     # f"{CPD_MVK_DA_DATA_DIR}/train/subsets/CPD_MVK_DA_train_{n}_config_strat_rep_{rep_num}.xyz",
    #     # f"{CPD_MVK_DA_DATA_DIR}/train/subsets/CPD_MVK_DA_train_{n}_config_strat_rep_{rep_num}.xyz",
    #     # f"{CPD_MVK_DA_DATA_DIR}/train/subsets/CPD_MVK_DA_train_{n}_config_strat_rep_{rep_num}.xyz",
    #     f"{CPD_MVK_DA_DATA_DIR}/train/subsets/CPD_MVK_DA_train_{n}_config_strat_ts_gauss_rep_{rep_num}.xyz",
    #     # f"{CPD_MVK_DA_DATA_DIR}/train/subsets/CPD_MVK_DA_train_{n}_config_strat_ts_gauss_rep_{rep_num}.xyz",
    #     # f"{CPD_MVK_DA_DATA_DIR}/train/subsets/CPD_MVK_DA_train_{n}_config_strat_ts_gauss_rep_{rep_num}.xyz",
    # ]
    # dataset_labels = ['Rep 1', 'Rep 2', 'Rep 3']
    # save_name = 'CPD_MVK_DA_iter_train_subsets'

    # full test set 
    # f"{DATASETS_DIR}/da_paper_data/valid/endo_ace_wB97M_nonpca_orca.xyz",                       # vac no reaction
    # f"{DATASETS_DIR}/da_paper_data/neb/endo_DA_neb_gas_wB97M.xyz",                              # vac reaction
    # f"{DATASETS_DIR}/da_paper_data/neb/endo_DA_neb_water_wB97M.xyz",                              # vac reaction
    # f"{DATASETS_DIR}/da_paper_data/valid/endo_in_water_ace_wB97M_short_orca_non_periodic.xyz",  # water reaction
    # f"{TRAIN_SUBSETS_DIR}/endo_DA_valid_size_sorted.xyz",                                       # vac, water, reactants in water
    # custom_xyz_parsers = [None, None, None, None, autode_props_parser]

    # --- anthracene data ---
    # ts_ref_config = flexible_ase_read(f'{ANTHRACENE_DA_DATA_DIR}/TS/anthracene_DA_vac_TS_wB97M.xyz')[0]
    # react_coords = [(2, 20), (5, 22)]
    # test_elements = {'C', 'H', 'O'}
    # include_energies = True
    # include_forces = True

    # # raw complete ds
    # # input_ds_fps = [
    # #     f"{ANTHRACENE_DA_DATA_DIR}/anthracene_DA_vac_al_iter.xyz",
    # #     f"{ANTHRACENE_DA_DATA_DIR}/anthracene_DA_vac_al_iter_force_filtered.xyz",
    # #     f"{ANTHRACENE_DA_DATA_DIR}/anthracene_DA_in_toluene_al_iter.xyz",
    # #     f"{ANTHRACENE_DA_DATA_DIR}/anthracene_DA_in_toluene_al_iter_force_filtered.xyz",
    # #     f"{ANTHRACENE_DA_DATA_DIR}/anthracene_DA_al_traj_samples_dist_filtered_orca.xyz",
    # #     f"{ANTHRACENE_DA_DATA_DIR}/anthracene_DA_al_traj_samples_dist_filtered_orca_force_filtered.xyz"
    # # ]
    # # dataset_labels = ['Vac', 'Vac Force Filtered' 'Solvated', 'Solvated Force Filtered', 'Extra', 'Extra Force Filtered']
    # # save_name = 'anthracene_DA'

    # # training data
    # input_ds_fps = [
    #     f"{ANTHRACENE_DA_DATA_DIR}/train/anthracene_DA_train.xyz",
    #     f"{ANTHRACENE_DA_DATA_DIR}/valid/anthracene_DA_valid.xyz",
    #     f"{ANTHRACENE_DA_DATA_DIR}/test/anthracene_DA_test.xyz"
    # ]
    # dataset_labels = ['Train', 'Valid', 'Test']
    # save_name = 'anthracene_DA_train_subsets'

    # vac solv seperated
    # input_ds_fps = [
    #     f"{ANTHRACENE_DA_DATA_DIR}/train/anthracene_DA_train_vac_only.xyz",
    #     f"{ANTHRACENE_DA_DATA_DIR}/train/anthracene_DA_train_solv_only.xyz",
    #     f"{ANTHRACENE_DA_DATA_DIR}/valid/anthracene_DA_valid_vac_only.xyz",
    #     f"{ANTHRACENE_DA_DATA_DIR}/valid/anthracene_DA_valid_solv_only.xyz",
    #     f"{ANTHRACENE_DA_DATA_DIR}/test/anthracene_DA_test_vac_only.xyz",
    #     f"{ANTHRACENE_DA_DATA_DIR}/test/anthracene_DA_test_solv_only.xyz"
    # ]
    # dataset_labels = ['Train Vac', 'Train Solv', 'Valid Vac', 'Valid Solv', 'Test Vac', 'Test Solv']
    # save_name = 'anthracene_DA_train_subsets'

    # iter train subsets
    # rep_num, n = 2, 46

    # input_ds_fps = [
    #     # f"{ANTHRACENE_DA_DATA_DIR}/train/subsets/anthracene_DA_train_{n}_config_strat_rep_1.xyz",
    #     f"{ANTHRACENE_DA_DATA_DIR}/train/subsets/anthracene_DA_train_{n}_config_strat_rep_2.xyz",
    #     # f"{ANTHRACENE_DA_DATA_DIR}/train/subsets/anthracene_DA_train_{n}_config_strat_rep_3.xyz",
    #     # f"{ANTHRACENE_DA_DATA_DIR}/train/anthracene_DA_train_46_config_strat_rep_2_w_1_4_samples.xyz",
    #     # f"{ANTHRACENE_TRAIN_DIR}/anthracene_DA_train.xyz"
    # ]
    # # dataset_labels = [f'Train {n} Rep 1', f'Train {n} Rep 2', f'Train {n} Rep 3', 'Train Full DS']
    # dataset_labels = [f'Train {n} Rep 2', 'With 2 Additional 1_4 Samples']
    # save_name = 'anthracene_DA_iter_train_subsets'

    # additional sampling
    # input_ds_fps = [
    #     f"{SAMPLES_DIR}/anthracene_DA_vac_TS_wB97M_MACE-OFF23_medium_samples.xyz",
    #     f"{ANTHRACENE_TRAIN_DIR}/anthracene_DA_train.xyz"
    # ]
    # dataset_labels = ['Samples', 'Train']
    # save_name = 'anthracene_DA_constrained_sampling'

    # additional al trajectory sampling
    # input_ds_fps = [
    #     # f"{ANTHRACENE_DA_DATA_DIR}/full_al_traj_samples.xyz",
    #     f"{ANTHRACENE_DA_DATA_DIR}/full_al_traj_samples_dist_filtered_orca.xyz",
    #     f"{ANTHRACENE_DA_DATA_DIR}/train/anthracene_DA_train.xyz",
    # ]
    # # dataset_labels = ['Full AL Traj Samples', 'Dist. Filtered', 'Train']
    # dataset_labels = ['Dist. Filtered ORCA', 'Train']
    # save_name = 'anthracene_DA_al_traj_sampling'

    # --- guajadial DA data ---

    ts_ref_config = flexible_ase_read(f'{GUAJADIAL_DA_DATA_DIR}/TS/guajadial_DA_vac_TS_wB97M.xyz')[0]
    react_coords = [(13, 31), (6, 32)]
    test_elements = {'C', 'H', 'O'}
    include_energies = True
    include_forces = True

    # training data
    input_ds_fps = [
        f"{GUAJADIAL_DA_DATA_DIR}/train/guajadial_DA_train.xyz",
        f"{GUAJADIAL_DA_DATA_DIR}/valid/guajadial_DA_valid.xyz",
        f"{GUAJADIAL_DA_DATA_DIR}/test/guajadial_DA_test.xyz",
        # f"{GUAJADIAL_DA_DATA_DIR}/test/guajadial_DA_test_PBE_def2_TZVP.xyz",
    ]
    dataset_labels = ['Train', 'Valid', 'Test']
    # dataset_labels = ['wB97M', 'PBE']
    save_name = 'guajadial_DA_train_subsets'

    # vac solv seperated
    # input_ds_fps = [
    #     f"{GUAJADIAL_DA_DATA_DIR}/train/guajadial_DA_train_vac_only.xyz",
    #     f"{GUAJADIAL_DA_DATA_DIR}/train/guajadial_DA_train_solv_only.xyz",
    #     f"{GUAJADIAL_DA_DATA_DIR}/valid/guajadial_DA_valid_vac_only.xyz",
    #     f"{GUAJADIAL_DA_DATA_DIR}/valid/guajadial_DA_valid_solv_only.xyz",
    #     f"{GUAJADIAL_DA_DATA_DIR}/test/guajadial_DA_test_vac_only.xyz",
    #     f"{GUAJADIAL_DA_DATA_DIR}/test/guajadial_DA_test_solv_only.xyz"
    # ]
    # dataset_labels = ['Train Vac', 'Train Solv', 'Valid Vac', 'Valid Solv', 'Test Vac', 'Test Solv']
    # # dataset_labels = ['Train Solv', 'Valid Solv', 'Test Solv']
    # save_name = 'guajadial_DA_train_subsets'

    # additional al trajectory sampling
    # input_ds_fps = [
    #     # f"{GUAJADIAL_DA_DATA_DIR}/guajadial_extra_traj_samples.xyz",
    #     # f"{GUAJADIAL_DA_DATA_DIR}/guajadial_extra_traj_samples_dist_filtered.xyz",
    #     f"{GUAJADIAL_DA_DATA_DIR}/guajadial_extra_traj_samples_dist_filtered_vac.xyz",
    #     f"{GUAJADIAL_DA_DATA_DIR}/guajadial_extra_traj_samples_dist_filtered_solv.xyz",
    #     f"{GUAJADIAL_DA_DATA_DIR}/train/guajadial_DA_train.xyz",
    # ]
    # # dataset_labels = ['Full AL Traj Samples', 'Dist. Filtered', 'Train']
    # # dataset_labels = ['Dist. Filtered', 'Train']
    # dataset_labels = ['Dist. Filtered Vac Sample', 'Dist. Filtered Solv Sample', 'Train']
    # # dataset_labels = ['Dist. Filtered ORCA', 'Train']
    # save_name = 'guajadial_DA_al_traj_sampling'

    # iter train subsets
    # n = 50
    # sample_str = ""
    # # sample_str = "ts_gauss_"
    # input_ds_fps = [
    #     f"{GUAJADIAL_DA_DATA_DIR}/train/subsets/guajadial_DA_train_{n}_config_strat_{sample_str}rep_1.xyz",
    #     # f"{GUAJADIAL_DA_DATA_DIR}/train/subsets/guajadial_DA_train_{n}_config_strat_{sample_str}rep_2.xyz",
    #     # f"{GUAJADIAL_DA_DATA_DIR}/train/subsets/guajadial_DA_train_{n}_config_strat_{sample_str}rep_3.xyz",
    # ]
    # dataset_labels = ['Rep 1', 'Rep 2', 'Rep 3']
    # save_name = 'guajadial_DA_iter_train_subsets'

    # --- I Cl SN2 data ---
    # input_ds_fps = [
    #     f"{I_Cl_SN2_DATA_DIR}/I_Cl_SN2_vac_al_iter_force_filtered.xyz",
    #     f"{I_Cl_SN2_DATA_DIR}/I_Cl_SN2_in_acetone_al_iter_force_filtered.xyz"
    # ]
    # dataset_labels = ['Vacuum', 'In Acetone']
    # save_name = 'I_Cl_SN2'

    # input_ds_fps = [
    #     f"{I_Cl_SN2_TRAIN_DIR}/I_Cl_SN2_train.xyz",
    #     f"{I_Cl_SN2_TRAIN_DIR}/I_Cl_SN2_valid.xyz",
    #     f"{I_Cl_SN2_TRAIN_DIR}/I_Cl_SN2_test.xyz"
    # ]
    # dataset_labels = ['Train', 'Valid', 'Test']
    # save_name = 'I_Cl_SN2_train_subsets'

    # iter train subsets
    # input_ds_fps = [
    #     f"{I_Cl_SN2_TRAIN_DIR}/subsets/I_Cl_SN2_train_2_config_strat_rep_1.xyz",
    #     f"{I_Cl_SN2_TRAIN_DIR}/subsets/I_Cl_SN2_train_2_config_strat_rep_2.xyz",
    #     f"{I_Cl_SN2_TRAIN_DIR}/subsets/I_Cl_SN2_train_2_config_strat_rep_3.xyz",
    #     # f"{I_Cl_SN2_DATA_DIR}/anthracene_DA_train.xyz"
    # ]
    # dataset_labels = ['Train 42 Rep 1', 'Train 42 Rep 2', 'Train 42 Rep 3', 'Train Full DS']
    # save_name = 'I_Cl_SN2_iter_train_subsets'

    # ts_ref_config = flexible_ase_read(f'{I_Cl_SN2_DATA_DIR}/I_Cl_SN2_vac_TS_wB97M.xyz')[0]
    # react_coords = [(1, 2)]
    # test_elements = {'C', 'H', 'O', 'I', 'Cl'}

    # --------------------------

    # === OTHER SETTINIGS ===
    # energy_units = KCAL_PER_MOL
    energy_units = MEV_PER_ATOM
    plot_2d_xlim, plot_2d_ylim = None, None
    # plot_2d_xlim, plot_2d_ylim = (1.0, 7.5), (1.0, 7.5)

    # ================================

    input_ds_names = [get_name_from_path(input_ds_fp) for input_ds_fp in input_ds_fps]

    # load cache file
    hdf5_file = load_hdf5_cache()

    # load ts
    ts_ref_bond_distances = {ri: ts_ref_config.get_distance(*ri) for ri in react_coords}
    ts_ref_bond_dist_tuple = tuple([dist for dist in ts_ref_bond_distances.values()])
    logger.info(f'TS ref bond distances: {ts_ref_bond_distances}')

    from react_ml_test_public.helpers.io import get_ase_atoms_forces
    from react_ml_test_public.helpers.settings import EV
    from react_ml_test_public.helpers.utils import convert_units


    all_configs, all_soap_desc_2D, all_energies, all_energy_deltas, all_force_deltas, all_force_norms, all_force_comps, all_r_coord_dists = [], [], [], [], [], [], [], []
    for i, (input_ds_fp, input_ds_name) in enumerate(zip(input_ds_fps, input_ds_names)):

        logger.info(f"Dataset Name: {input_ds_name}")

        # load train configs
        configs = flexible_ase_read(input_ds_fp, cust_props_parser=custom_xyz_parsers[i] if custom_xyz_parsers is not None else None)
        # for config in configs:
        #     print(len(config))
        # === SELECT TRAIN CONFIG TYPES HERE ===
        filter_ids = [config_id for config_id, config in enumerate(configs)]                                                                                    # no filter
        # filter_ids = [config_id for config_id, config in enumerate(configs) if len(config) in {118, 22}]                                                       # size filtered
        # filter_ids = [config_id for config_id, config in enumerate(configs) if config.info['config_type'] == 'vacuum']                         # vacuum only
        # # train_filter_ids = [config_id for config_id, config in enumerate(train_configs) if config.info['config_type'] == 'microsolvated']                # microsolvated only
        # filter_ids = [config_id for config_id, config in enumerate(configs) if config.info['config_type'] in {'vacuum', 'reactants_in_water', 'water'}]      # vacuum + microsolvated only
        # # train_filter_ids = [config_id for config_id, config in enumerate(train_configs) if config.info['config_type'] == 'reactants_in_water']            # reactants in water only
        # # ==============================

        configs = [configs[config_id] for config_id in filter_ids]
        logger.info(f'Num Filtered Configs: {len(configs)}')
        all_configs.append(configs)

        config_types = [config.info.get('config_type') for config in configs]
        atom_nums = [len(config) for config in configs]
        atom_num_counts, type_counts = {}, {}
        for config_type in config_types:
            type_counts[config_type] = type_counts.get(config_type, 0) + 1
        for atom_num in atom_nums:
            atom_num_counts[atom_num] = atom_num_counts.get(atom_num, 0) + 1

        logger.info(f'Num Configs in Dataset: {len(configs)}')
        logger.info(f'Num Atoms in Dataset: {sum([len(config) for config in configs])}')
        logger.info(f"Atom Num Counts: {atom_num_counts}")
        logger.info(f'Config Types in Dataset: {set(config_types)}')
        logger.info(f"Config Type Counts: {type_counts}")

        # 1. get SOAP descriptors of training set
        start_time = time()
        average_soap = get_soap_descriptors(configs, 
                                            specific_elements=test_elements, 
                                            average='outer', 
                                            compress_dim=None, 
                                            flatten_output=True, 
                                            cache_name=input_ds_name)
        logger.info(f"Loaded / calculated {len(configs)} average train SOAP descriptors in {time() - start_time:.4f}s...")
        soap_desc_2D = run_pca(average_soap, n_components=2)
        all_soap_desc_2D.append(soap_desc_2D)

        if include_energies:

            # get absolute energies
            energies = np.array([get_ase_atoms_energy(config) for config in configs])
            # print('Energies: ', energies)
            all_energies.append(energies)

            # 2. get relative energies
            atom_num_groups = {}
            for config in configs:
                if atom_num_groups.get(len(config)) is None:
                    atom_num_groups[len(config)] = [config]
                else:
                    atom_num_groups[len(config)].append(config)

            atom_num_group_energy_mins = {num_atoms: np.min(get_ase_props(group_configs, prop='energy')) 
                                            for num_atoms, group_configs in atom_num_groups.items()}
            energy_deltas = np.array([abs(get_ase_atoms_energy(config) - atom_num_group_energy_mins[len(config)]) 
                            for config in configs])
            # print('Energy deltas: ', energy_deltas)
            all_energy_deltas.append(energy_deltas)

        if include_forces:

            atom_num_group_force_mins = {num_atoms: np.min([np.mean(np.linalg.norm(config_forces, axis=0) * eV_to_mEV) 
                                                            for config_forces in get_ase_props(group_configs, prop='forces')])
                                        for num_atoms, group_configs in atom_num_groups.items()}
            force_deltas = [abs(np.mean(np.linalg.norm(get_ase_props([config], prop='forces'), axis=0) * eV_to_mEV) - atom_num_group_force_mins[len(config)]) for config in configs]
            all_force_deltas.append(force_deltas)

            # 4. get force norms
            force_norms = np.array([np.linalg.norm(force_vec, axis=0) * eV_to_mEV for config in configs for force_vec in get_ase_atoms_forces(config)])
            all_force_norms.append(force_norms)

            # 5. get force components
            force_comps = np.concatenate([force_vec.flatten() * eV_to_mEV for config in configs for force_vec in get_ase_atoms_forces(config)])
            logger.info(f"Force Comps Shape: {force_comps.shape}")
            all_force_comps.append(force_comps)

        # 6. get distances from TS
        r_coord_dists = [[config.get_distance(*ri, mic=True) for ri in react_coords] for config in configs]
        all_r_coord_dists.append(r_coord_dists)

        logger.info("============================================")

        # get list of indices of ts distances to filter out configs based on element
        # for i, config in enumerate(train_configs):
        #     assert config[r1[0]].symbol == 'C' and config[r1[1]].symbol == 'C'
        #     assert config[r2[0]].symbol == 'C' and config[r2[1]].symbol == 'C'
        # r1_atoms, r2_atoms = [3, 14], [8, 13] # reaction coordinates in water

    # apply unit conversions for energy
    # if energy_units == KCAL_PER_MOL:
    #     # all_energies = [energies * eV_to_kcal_mol  for energies in all_energies]
    #     all_energy_deltas = [energies * eV_to_kcal_mol  for energies in all_energy_deltas]
    #     print('All energy deltas: ', all_energy_deltas)

    # apply unit conversions for energy
    if include_energies:
        if energy_units == KCAL_PER_MOL:
            all_energies = [energies * eV_to_kcal_mol  for energies in all_energies]
            all_energy_deltas = [energies * eV_to_kcal_mol  for energies in all_energy_deltas]
            print('Converted Energies: ', all_energies)
            print('Converted Energy deltas: ', all_energy_deltas)
        elif energy_units == MEV_PER_ATOM:
            all_energies = [convert_units(energies, EV, MEV_PER_ATOM, [len(config) for config in all_configs[i]])  for i, energies in enumerate(all_energies)]
            all_energy_deltas = [convert_units(energies, EV, MEV_PER_ATOM, [len(config) for config in all_configs[i]])  for i, energies in enumerate(all_energy_deltas)]

    from react_ml_test_public.helpers.settings import KCAL_PER_MOL

    dpi = 600
    fig_size = (3.5, 2.5)

    if not include_energies:
        all_energy_deltas = [None for _ in range(len(input_ds_names))]

    # 1. PCA SOAP plot
    for input_ds_name, soap_desc_2D, energy_deltas, configs in zip(input_ds_names, all_soap_desc_2D, all_energy_deltas, all_configs):

        plot_settings = PlotSettings(x_label='PC1', 
                                     y_label='PC2', 
                                     colour_bar_label=f'\u2206 Energy ({energy_units})', 
                                     title=input_ds_name, 
                                     dpi=dpi, 
                                     fig_size=fig_size,
                                     show_legend=False)

        # concat_pca_soap_descs = np.concatenate(train_config_pca_soap_descs, axis=0)
        soap_pca_1, soap_pca_2 = soap_desc_2D[:, 0], soap_desc_2D[:, 1]

        plot_2d([soap_desc_2D],
            f"{PLOT_DIR}/2D/{input_ds_name}_soap_PCA.png", 
            colour_maps=[energy_deltas] if energy_deltas is not None else None,
            plot_settings=plot_settings,
            )

        # if include_energies:
        #     properties = {"PCA-1": soap_pca_1, "PCA-2": soap_pca_2, "score": energy_deltas}
        #     widget = chemiscope.show(configs, properties=properties)
        #     widget.save(f"out/chemiscope/{input_ds_name}_soap_pca.json")

    # 1.2 combined PCA plot
    plot_settings = PlotSettings(x_label='PC1', 
                                 y_label='PC2', 
                                 title=input_ds_name, 
                                 dpi=dpi, 
                                 fig_size=fig_size,
                                 show_legend=False)

    plot_2d(all_soap_desc_2D,
            f"{PLOT_DIR}/ds_analysis/{save_name}_2D_soap_PCA.png", 
            plot_settings=plot_settings)

    # 2. plot energies
    plt.figure(figsize=fig_size, dpi=dpi)
    for input_ds_name, label, energies, configs in zip(input_ds_names, dataset_labels, all_energies, all_configs):

        indices = np.arange(0, len(configs))
        plt.plot(indices, energies, label=label, linewidth=0.6)

    plt.legend(loc='upper right')
    plt.xlabel('Index')
    plt.ylabel('Energy (meV / atom)')
    plt.tight_layout()

    # plot relative energies
    if include_energies:
        plt.figure(figsize=fig_size, dpi=dpi)
        for input_ds_name, label, energy_deltas, configs in zip(input_ds_names, dataset_labels, all_energy_deltas, all_configs):

            indices = np.arange(0, len(configs))
            # plt.scatter(indices, energy_deltas, s=1)
            plt.plot(indices, energy_deltas, label=label, linewidth=0.6)

            # save chemiscope energies
            # properties = {"Index": indices, f"Rel. Energy ({energy_units}))": energy_deltas}
            # chemiscope_out_fp = f"out/chemiscope/{input_ds_name}_rel_energies.json"
            # chemiscope.write_input(chemiscope_out_fp, 
            #                     configs,
            #                     properties=properties)
            # print(f'Saved chemiscope plot to {chemiscope_out_fp}...')

        plt.legend(loc='upper right')
        plt.xlabel('Index')
        plt.ylabel(f'Rel. Energy ({energy_units})')
        # plt.yscale("log")
        plt.tight_layout()
        plt.savefig(f"{PLOT_DIR}/ds_analysis/{save_name}_rel_energies.png", dpi=dpi)

    # 4. plot raw force components
    if include_forces:
        plt.figure(figsize=fig_size, dpi=dpi)
        for input_ds_name, label, force_comps in zip(input_ds_names, dataset_labels, all_force_comps):
            plt.plot(np.arange(0, len(force_comps)), force_comps, label=label, linewidth=0.6, alpha=0.6)
        # plt.legend(loc='upper right')
        plt.xlabel('Component Index')
        plt.ylabel(r'Force Component (meV / $\AA$)')
        # plt.yscale("log")
        plt.tight_layout()
        save_fp = f"{PLOT_DIR}/ds_analysis/{save_name}_force_comps.png"
        plt.savefig(save_fp, dpi=dpi)
        print(f"Saved {save_fp}...")

    print('React Coords Num', len(react_coords))
    if len(react_coords) == 2:
        # 5. TS dist plot
        for input_ds_name, energy_deltas, r_coord_dists, configs in zip(input_ds_names, all_energy_deltas, all_r_coord_dists, all_configs):

            points = np.array([np.array([x, y]) for x, y in r_coord_dists])
            plot_settings = PlotSettings(x_label=r'r1 / $\AA$', y_label=r'r2 / $\AA$', colour_bar_label='\u2206 Energy (meV / atom)', title=input_ds_name,
                                        dpi=dpi, fig_size=fig_size, show_legend=False, xlim=plot_2d_xlim, ylim=plot_2d_ylim)
            plot_2d([points],
                    f"{PLOT_DIR}/2D/{input_ds_name}_2D_react_coords.png", 
                    colour_maps=[energy_deltas] if energy_deltas is not None else None,
                    plot_settings=plot_settings,
                    ref_points=[ts_ref_bond_dist_tuple],
                    # ref_point_labels=['TS'],
                    )

            # properties = {r'$r_{1}$ / $\AA$': np.array(r1_dists), r'$r_{2}$ / $\AA$':np.array(r2_dists)}
            # widget = chemiscope.show(configs, properties=properties)
            # widget.save(f"out/chemiscope/{input_ds_name}_ts_dist.json")

        # combined TS dist plot
        all_points = [np.array([np.array([x, y]) for x, y in r_coord_dists]) for r_coord_dists in all_r_coord_dists]
        plot_settings = PlotSettings(x_label=r'r1 / $\AA$', y_label=r'r2 / $\AA$', title=input_ds_name, dpi=dpi, fig_size=fig_size, show_legend=False)
        plot_2d(all_points,
                f"{PLOT_DIR}/ds_analysis/{save_name}_2D_react_coords.png", 
                plot_settings=plot_settings,
                ref_points=[ts_ref_bond_dist_tuple],
                ref_point_labels=['TS'],
                )

    # train_ts_distances = np.array([abs(react_coord_dists[i] - ts_ref_bond_distances[r1]) for i in range(len(configs))])

    # plot histogram of TS dist on reaction coordinate distribution
    plt.figure(figsize=fig_size, dpi=dpi)
    colors = list(plt.get_cmap("tab10").colors)
    for i, (ds_label, input_ds_name, r_coord_dists) in enumerate(zip(dataset_labels, input_ds_names, all_r_coord_dists)):
        ts_dists = [sum([d - ts_ref_bond_dist_tuple[i] for i, d in enumerate(r_coord_dist)]) for r_coord_dist in r_coord_dists]
        # print(sorted(ts_dists))
        # ts_dists = [r1_dist - ts_ref_bond_dist_tuple[0] + r2_dist - ts_ref_bond_dist_tuple[1] for i, (r1_dist, r2_dist) in enumerate(zip(r1_dists, r2_dists))]
        plt.hist(ts_dists, bins=list(reversed([-0.2 * i for i in range(10)])) + [0.2 * i for i in range(50)], alpha=0.5, color=colors[i], label=ds_label)

        # plot kde
        ts_dists = np.array(ts_dists)
        bin_edges = np.histogram_bin_edges(ts_dists, bins=40)
        bin_width = bin_edges[1] - bin_edges[0]
        xs = np.linspace(ts_dists.min(), ts_dists.max(), 500)
        kde = gaussian_kde(ts_dists, bw_method="scott")
        kde_counts = kde(xs) * len(ts_dists) * bin_width
        plt.plot(xs, kde_counts, color=colors[i], linewidth=1)

    plt.xlabel(r'TS Dist. $\AA$', fontsize=10)
    plt.ylabel(r'Count', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.legend()
    plt.tight_layout()

    save_fp = f"{PLOT_DIR}/ds_analysis/{save_name}_data_hist.png"
    plt.savefig(save_fp, dpi=dpi)
    print(f"Saved {save_fp}...")
