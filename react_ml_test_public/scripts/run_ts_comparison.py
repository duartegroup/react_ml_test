import os
import re
from time import time
from typing import Union

import ase
import numpy as np

from react_ml_test_public.helpers.descriptors import get_soap_descriptors
from react_ml_test_public.helpers.dim_red import run_pca
from react_ml_test_public.helpers.io import (
    autode_props_parser,
    flexible_ase_read,
    get_ase_atoms_energy,
    get_name_from_path,
    load_data_cache,
    load_hdf5_cache,
    load_model_calculator,
    save_npz_data_cache,
)
from react_ml_test_public.helpers.logging import logger
from react_ml_test_public.helpers.plot import bar_plot, plot_2d
from react_ml_test_public.helpers.settings import (
    ANTHRACENE_DA_DATA_DIR,
    CPD_MVK_DA_DATA_DIR,
    CSV_DIR,
    DATASETS_DIR,
    GRAPH_PES_MODEL_LOADER,
    GUAJADIAL_DA_DATA_DIR,
    ISOLATED_ATOMS_DIR,
    MACE_MODEL_LOADER,
    MODEL_DIR,
    MODELS,
    NPZ_DIR,
    OPT_DIR,
    PLOT_DIR,
    SAMPLES_OUT_DIR,
    PlotSettings,
)
from react_ml_test_public.helpers.stats import calculate_rmse
from react_ml_test_public.helpers.utils import (
    eV_to_meV_per_atom,
    get_ase_props,
    natural_sort_key,
    unflatten_array_by_lengths,
)

if __name__ == "__main__":

    # === DEFINE SYSTEM ===

    # --- CPD_MVK DA ---

    # vac
    # system_name = "endo_DA_vac_TS"
    # mlip_ts_opt_save_name = 'CPD_MVK_DA_vac'

    # solv
    system_name = "CPD_MVK_TS_opt_water"
    mlip_ts_opt_save_name = 'CPD_MVK_DA_in_water'

    ts_ref_fp = f'{CPD_MVK_DA_DATA_DIR}/TS/{system_name}_wB97M.xyz'
    r1_atoms, r2_atoms = (1, 12), (6, 11) # reaction coordinates
    num_solute_atoms = 22

    # --- anthracene 9,10 ---

    # vac
    # system_name = "anthracene_DA_vac_TS"
    # mlip_ts_opt_save_name = 'anthracene_DA_vac'

    # solv
    # system_name = "anthracene_DA_TS_opt_SMD_toluene"
    # mlip_ts_opt_save_name = 'anthracene_DA_in_toluene'

    # ts_ref_fp = f'{ANTHRACENE_DA_DATA_DIR}/TS/{system_name}_wB97M.xyz'
    # r1_atoms, r2_atoms = (2, 20), (5, 22) # reaction coordinates
    # num_solute_atoms = 41

    # --- anthracene 1,4 ---

    # vac
    # system_name = "anthracene_1_4_DA_vac_TS"
    # mlip_ts_opt_save_name = 'anthracene_1_4_DA_vac'

    # # solv
    # system_name = "anthracene_1_4_DA_TS_opt_SMD_toluene"
    # mlip_ts_opt_save_name = 'anthracene_1_4_DA_in_toluene'

    # ts_ref_fp = f'{ANTHRACENE_DA_DATA_DIR}/TS/{system_name}_wB97M.xyz'
    # r1_atoms, r2_atoms = (16, 20), (15, 22) # reaction coordinates
    # num_solute_atoms = 41

    # --- guajadial DA ---

    # vac
    # system_name = "guajadial_DA_vac_TS"
    # mlip_ts_opt_save_name = 'guajadial_DA_vac'

    # # solv
    # # system_name = "guajadial_TS_opt_water"
    # # mlip_ts_opt_save_name = 'guajadial_DA_in_water'

    # ts_ref_fp = f'{GUAJADIAL_DA_DATA_DIR}/TS/{system_name}_wB97M.xyz'
    # r1_atoms, r2_atoms = (13, 31), (6, 32) # reaction coordinates
    # num_solute_atoms = 69

    # -------------------

    ts_ref_config = flexible_ase_read(ts_ref_fp)[0]
    ts_ref_bond_distances = {r1_atoms: ts_ref_config.get_distance(*r1_atoms), r2_atoms: ts_ref_config.get_distance(*r2_atoms)}
    ts_ref_point = np.array([float(dist) for dist in ts_ref_bond_distances.values()])
    logger.info(f'TS ref bond distance: {ts_ref_point}')

    # === SELECT MODELS ===

    # --- full zero-shot model comparison ---
    model_names = [
            'GO-MACE-23', 
            'MACE-OFF23_medium', 
            'MACE-MP-0b3_medium',
            'EGRET_1T', 
            'MACE-NEB',
            'ani-1xnr',
            'MACE-omol-0-extra-large-1024',
            'MACE-OFF23_medium_CPD_MVK_DA_train_42_config_strat_ts_gauss_fine_tuned_energy_rep_1'
    ]

    save_name = f"paper_zero_shot"

    # --- specific fine-tuned model comparison (+ zero shot) ---

    # # cpd mvk
    # # n, ft_system_name, ft_train_name, rep_num, omol_rep_num, omol_energy_str = 42, "CPD_MVK_DA", "CPD_MVK_DA_train", 1, 1, 'energy_'
    # # sample_str = 'ts_gauss_'

    # # anthracene
    # n, ft_system_name, ft_train_name, rep_num, omol_rep_num, omol_energy_str = 46, "anthracene_DA", "anthracene_DA_train", 2, 3, 'energy_'
    # sample_str = ''

    # # guajadial
    # # n, ft_system_name, ft_train_name, rep_num, omol_rep_num, omol_energy_str = 50, "guajadial_DA", "guajadial_DA_train", 1, 1, ''
    # # sample_str = ''

    # model_names = [
    #             # f"GO-MACE-23",
    #             # f"GO-MACE-23_{ft_train_name}_{n}_config_strat_fine_tuned_rep_{rep_num}",
    #             # f"GO-MACE-23_{ft_train_name}_{n}_config_strat_fine_tuned_energy_rep_{rep_num}",
    #             f"GO-MACE-23_{ft_train_name}_{n}_config_strat_{sample_str}fine_tuned_energy_rep_{rep_num}",
    #             # f"GO-MACE-23_{ft_train_name}_fine_tuned_rep_1",
    #             # f"GO-MACE-23_{ft_train_name}_fine_tuned_energy_rep_1",
    #             # f"GO-MACE-23_{ft_train_name}_from_scratch_rep_1",
    #             # f"GO-MACE-23_{ft_train_name}_from_scratch_energy_rep_1",
    #             # f"MACE-OFF23_medium", 
    #             # f"MACE-OFF23_medium_{ft_train_name}_{n}_config_strat_fine_tuned_rep_{rep_num}", 
    #             # f"MACE-OFF23_medium_{ft_train_name}_{n}_config_strat_fine_tuned_energy_rep_{rep_num}",
    #             f"MACE-OFF23_medium_{ft_train_name}_{n}_config_strat_{sample_str}fine_tuned_energy_rep_{rep_num}",
    #             # f"MACE-OFF23_medium_{ft_train_name}_fine_tuned_rep_1",
    #             # f"MACE-OFF23_medium_{ft_train_name}_fine_tuned_energy_rep_1",
    #             # f"MACE-OFF23_medium_{ft_train_name}_from_scratch_rep_1",
    #             # f"MACE-OFF23_medium_{ft_train_name}_from_scratch_energy_rep_1",
    #             # f"MACE-MP-0b3_medium",
    #             # f"MACE-MP-0b3_medium_{ft_train_name}_{n}_config_strat_fine_tuned_rep_{rep_num}",
    #             # f"MACE-MP-0b3_medium_{ft_train_name}_{n}_config_strat_fine_tuned_energy_rep_{rep_num}",
    #             f"MACE-MP-0b3_medium_{ft_train_name}_{n}_config_strat_{sample_str}fine_tuned_energy_rep_{rep_num}",
    #             # f"MACE-MP-0b3_medium_{ft_train_name}_fine_tuned_rep_1",
    #             # f"MACE-MP-0b3_medium_{ft_train_name}_fine_tuned_energy_rep_1",
    #             # f"MACE_MP0_{ft_train_name}_from_scratch_rep_1",
    #             # f"MACE_MP0_{ft_train_name}_from_scratch_energy_rep_1",
    #             'MACE-omol-0-extra-large-1024',
    #             # f"MACE-omol-0-extra-large-1024_{ft_train_name}_{n}_config_strat_fine_tuned_rep_{rep_num}",
    #             # f"MACE-omol-0-extra-large-1024_{ft_train_name}_{n}_config_strat_fine_tuned_rep_2",
    #             # f"MACE-omol-0-extra-large-1024_{ft_train_name}_{n}_config_strat_fine_tuned_rep_3",
    #             # f"MACE-omol-0-extra-large-1024_{ft_train_name}_{n}_config_strat_fine_tuned_energy_rep_{rep_num}",
    #             f"MACE-omol-0-extra-large-1024_{ft_train_name}_{n}_config_strat_{sample_str}fine_tuned_{omol_energy_str}rep_{omol_rep_num}",
    #             # f"MACE-omol-0-extra-large-1024_{ft_train_name}_{n}_config_strat_fine_tuned_energy_rep_2",
    #             # f"MACE-omol-0-extra-large-1024_{ft_train_name}_{n}_config_strat_fine_tuned_energy_rep_3",
    #             ]
    # save_name = f"paper_select_ft_models"

    # =============================================================

    for model_name in model_names:

        logger.info(f"Model: {model_name}")
        ts_model_opt_fp = f"{OPT_DIR}/{mlip_ts_opt_save_name}_{model_name}_opt_TS.xyz"
        ts_model_opt_config = flexible_ase_read(ts_model_opt_fp)[0]

        ts_model_opt_r1_dist = ts_model_opt_config.get_distance(r1_atoms[0], r1_atoms[1], mic=True)
        ts_model_opt_r2_dist = ts_model_opt_config.get_distance(r2_atoms[0], r2_atoms[1], mic=True)
        logger.info(f"Model TS Opt TS Distances: r1: {ts_model_opt_r1_dist}, r2: {ts_model_opt_r2_dist}")

        logger.info(f'TS Dist. Error: {sum(abs(np.array(ts_ref_point) - np.array([ts_model_opt_r1_dist, ts_model_opt_r2_dist]))):.2f}')
        struct_rmse = calculate_rmse(ts_model_opt_config.get_positions()[0:num_solute_atoms], ts_ref_config.get_positions()[0:num_solute_atoms])
        logger.info(f'MLIP OPT TS Reactants Structural RMSD: {struct_rmse:.3f}\n')
