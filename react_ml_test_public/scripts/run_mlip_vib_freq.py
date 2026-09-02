"""
Script to get vibrational frequency summary for an input geometry with a set of models.
"""
from ase import Atoms
from ase.calculators.calculator import Calculator

from react_ml_test_public.helpers.io import flexible_ase_read, get_name_from_path
from react_ml_test_public.helpers.logging import logger
from react_ml_test_public.helpers.settings import (
    ANTHRACENE_DA_DATA_DIR,
    CPD_MVK_DA_DATA_DIR,
    GUAJADIAL_DA_DATA_DIR,
    MODEL_DIR,
    OPT_DIR,
    PAPER_MODELS,
    VIB_DIR,
)
from react_ml_test_public.helpers.ts_opt import get_vib_freq
from react_ml_test_public.helpers.utils import load_model_calculator

if __name__ == "__main__":

    out_dir = VIB_DIR
    cuda_device = 'cuda:0'
    enable_cueq = False

    # === 1. SELECT MODEL ===

    # --- selected fine-tuned models ---

    # n, system_ft_name, rep_num = 42, "CPD_MVK_DA_train", 1
    # n, system_ft_name, rep_num = 46, "anthracene_DA_train", 2
    n, system_ft_name, rep_num = 50, "guajadial_DA_train", 1
    energy_corrected = True

    energy_str = 'energy_' if energy_corrected else ''
    model_names = [
                   f"GO-MACE-23_{system_ft_name}_{n}_config_strat_fine_tuned_{energy_str}rep_{rep_num}",
                   f"MACE-OFF23_medium_{system_ft_name}_{n}_config_strat_fine_tuned_{energy_str}rep_{rep_num}", 
                   f"MACE-MP-0b3_medium_{system_ft_name}_{n}_config_strat_fine_tuned_{energy_str}rep_{rep_num}",
                  ]

    # --- positive control models ---

    # system_ft_name = "CPD_MVK_DA_train"
    # # system_ft_name = "anthracene_DA_train"
    # # system_ft_name = "guajadial_DA_train"
    # energy_corrected = True

    # energy_str = 'energy_' if energy_corrected else ''
    # model_names = [
    #             f"GO-MACE-23_{system_ft_name}_fine_tuned_{energy_str}rep_1",
    #             f"MACE-OFF23_medium_{system_ft_name}_fine_tuned_{energy_str}rep_1",
    #             f"MACE-MP-0b3_medium_{system_ft_name}_fine_tuned_{energy_str}rep_1",
    #             ]

    # === 2. SELECT INPUT CONFIGS (OPT TS) ===

    # --- endo DA vac wB97M TS ---
    # init_config_set_fpaths = [f"{CPD_MVK_DA_DATA_DIR}/TS/endo_DA_vac_TS_wB97M.xyz" for _ in model_names]

    # --- model opt TS for input system ---
    # system_name = 'CPD_MVK_DA_vac'
    # system_name = 'anthracene_vac'
    # system_name = 'anthracene_1_4_MMFF_vac'
    system_name = 'guajadial_vac'
    # system_name = 'CPD_MVK_DA_in_water'
    init_config_set_fpaths = [f"{OPT_DIR}/{system_name}_{model_name}_opt_TS.xyz" for model_name in model_names]
    fmt_system_name =f"{system_name}_opt_TS"

    for model_name, input_config_fp in zip(model_names, init_config_set_fpaths):

        input_config_name = get_name_from_path(input_config_fp)
        logger.info(f'Calculating Vibrational Freq. on {input_config_name} with {model_name}...')
        calc = load_model_calculator(model_name, device=cuda_device, enable_cueq=enable_cueq)
        get_vib_freq(input_config_fp, calc, f"{fmt_system_name}_{model_name}", out_dir=out_dir)
