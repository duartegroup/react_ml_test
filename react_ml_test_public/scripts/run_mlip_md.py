"""
Script for evaluating the performance of foundation models on a specified test system.
"""


import os

# from ase.constraints.constraint import IndexedConstraint
from time import time

import torch
from ase.constraints import FixAtoms
from ase.io import read
from react_ml_test_public.helpers.optimisation import dimer_method

from react_ml_test_public.helpers.dynamics import get_trajectory_with_ase
from react_ml_test_public.helpers.io import get_name_from_path, load_model_calculator
from react_ml_test_public.helpers.logging import logger
from react_ml_test_public.helpers.settings import (
    CPD_MVK_DA_DATA_DIR,
    DA_PAPER_DATA_DIR,
    DATA_DIR,
    MD_INPUT_DIR,
    MODEL_DIR,
    MODELS,
    OPT_DIR,
    PS,
    TRAJ_DIR,
    MDSettings,
)

# set CUDA device
CUDA_DEVICE_NUM = 0
CUDA_DEVICE = f"cuda:{CUDA_DEVICE_NUM}"

if torch.cuda.is_available():
    torch.cuda.set_device(CUDA_DEVICE_NUM)

def run_md_eval(init_configs: list,
                md_config: MDSettings,
                md_runs_per_config: int,
                model_name: str,
                formatted_model_name: str,
                system_name: str,
                constraint: str = None,# IndexedConstraint = None,
                enable_cueq: bool = False,
                override_saved: bool = False):

    trajectories = []
    for i in range(len(init_configs)):

        init_config = init_configs[i].copy()

        # calc = load_model_calculator(model_name, device=CUDA_DEVICE, enable_cueq=enable_cueq, dtype='float32')

        for j in range(md_runs_per_config):
            logger.info(f"Running MD trajectory for init config {i + 1}, run number {j}...")

            start_time = time()
            traj_name = f'{system_name}_{formatted_model_name}_{md_config.temp}K_{md_config.fs / PS}ps_config_{i}_run_{j}'

            out_trj_name = f"{TRAJ_DIR}/{traj_name}_traj.xyz"
            if os.path.exists(out_trj_name) and not override_saved:
                logger.info(f"{out_trj_name} already exists, skipping...")
                continue

            trajectory = get_trajectory_with_ase(init_config,
                                                 model_name, 
                                                 md_config, 
                                                 traj_name,
                                                 constraint=constraint,
                                                 cuda_device=CUDA_DEVICE,
                                                 override_cache=override_saved)
            end_time = time()
            md_time = end_time - start_time
            steps = float(md_config.fs / md_config.dt)
            logger.info(f"Ran MD trajectory '{traj_name}' in {md_time:.4f}s, with an average step time of {float(md_time) / steps}...")
            trajectories.append(trajectory)

    return trajectories


if __name__ == "__main__":

    constraint = None

    # TODO: add this to a seperate function!!!
    # check if correct device is configured in pytorch
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA is available: {cuda_available}")

    # Check the default device
    if cuda_available:
        logger.info(f"Current device: {torch.cuda.current_device()}")
        logger.info(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        logger.info("CUDA is not available.")

    # Create a tensor and move it to the GPU if available
    device = torch.device(CUDA_DEVICE if cuda_available else "cpu")
    tensor = torch.tensor([1.0, 2.0, 3.0]).to(device)

    # Check if the tensor is on GPU
    logger.info(f"Tensor device: {tensor.device}")

    # ====== INPUT PARAMETERS HERE ======

    box_dim: list[float] = None

    # === SELECT MODELS ===
    # model_names = ["MACE-OFF23_medium"]
    # model_names = ["GO-MACE-23"]

    # model_names = ["MACE-MP0-128-L0"]
    # model_names = ["MACE-MP0-128-L1"]
    # model_names = ["MACE-MP-style_endo_DA_train"]
    # model_names = ["MACE_mlptrain_style_endo_DA_train"]

    # model_names = ["ani_1xnr"]
    # model_names = ["MACE-OFF23_medium_endo_DA_1.0_train_from_scratch"] 
    # model_names = ["MACE-OFF23_medium_endo_DA_1.0_train_fine_tuned"] 
    # save_name = model_names[0]

    # single good model test
    # model_names = [
    #                'MACE-OFF23_medium_endo_DA_train_random_20_config_fine_tuned_rep_1',
    #                'GO-MACE-23_endo_DA_train_random_20_config_fine_tuned_rep_1',
    #                ]

    # good models
    # with open(f'{DATA_DIR}/good_models.txt') as model_spec:
    #     model_names = [line.strip() for line in model_spec]
    # save_name = 'good_models'

    # --- paper zero-shot ---
    # model_names = [
    #                "GO-MACE-23",
    #                "MACE-OFF23_medium", 
    #                "MACE-MP-0b3_medium",
    #               ]
    # formatted_model_names = model_names

    # --- paper extended zero shot ---
    # model_names = [
    #     # 'GO-MACE-23', 
    #     # 'MACE-OFF23_medium', 
    #     # 'MACE-MP-0b3_medium',
    #     # 'EGRET_1T', 
    #     'ani-1xnr',
    #     'MACE-NEB',
    #     # 'MACE-ANI-CC',
    #     # "MACE-OFF23_medium_CPD_MVK_DA_train_42_config_strat_ts_gauss_fine_tuned_energy_rep_1",
    #     # 'MACE-omol-0-extra-large-1024',
    # ]
    # formatted_model_names = model_names

    # --- single shot ts ---
    # model_names = [
    #                "MACE-OFF23_medium_endo_DA_vac_TS_wB97M_fine_tuned_rep_1",
    #                "GO-MACE-23_endo_DA_vac_TS_wB97M_fine_tuned_rep_1",
    #                "MACE-MP0-128-L1_endo_DA_vac_TS_wB97M_fine_tuned_rep_1",
    #               ]

    # --- selected fine-tuned models (with best selected replicates) ---
    n, system_ft_name, rep_num, omol_rep_num, omol_energy_str = 42, "CPD_MVK_DA_train", 1, 1, 'energy_'
    # n, system_ft_name, rep_num, omol_rep_num, omol_energy_str = 46, "anthracene_DA_train", 2, 3, 'energy_'
    # n, system_ft_name, rep_num, omol_rep_num, omol_energy_str = 50, "guajadial_DA_train", 1, 1, ''
    energy_corrected = True
    # sample_str = ''
    sample_str = 'ts_gauss_'

    energy_str = 'energy_' if energy_corrected else ''
    model_names = [
                   f"GO-MACE-23_{system_ft_name}_{n}_config_strat_{sample_str}fine_tuned_{energy_str}rep_{rep_num}",
                #    f"MACE-OFF23_medium_{system_ft_name}_{n}_config_strat_{sample_str}fine_tuned_{energy_str}rep_{rep_num}", 
                #    f"MACE-MP-0b3_medium_{system_ft_name}_{n}_config_strat_{sample_str}fine_tuned_{energy_str}rep_{rep_num}",
                #    f"MACE-omol-0-extra-large-1024",
                #    f"MACE-omol-0-extra-large-1024_{system_ft_name}_{n}_config_strat_{sample_str}fine_tuned_{omol_energy_str}rep_{omol_rep_num}",
                  ]

    formatted_model_names = [
        f'GO-MACE-23_{system_ft_name}_{n}_config_{energy_str}{sample_str}ft', 
        # f'MACE-OFF23_{system_ft_name}_{n}_config_{energy_str}{sample_str}ft', 
        # f'MACE-MP-0b3_medium_{system_ft_name}_{n}_config_{energy_str}{sample_str}ft',
        # f"MACE-omol-0-extra-large-1024",
        # f'MACE-omol-0-extra-large-1024_{system_ft_name}_{n}_config_{omol_energy_str}{sample_str}ft'
    ]

    # extra sample models
    # model_names = [
    #              f"GO-MACE-23_{system_ft_name}_{n}_config_strat_w_1_4_vac_sample_fine_tuned_{energy_str}rep_{rep_num}",
    #             #  f"MACE-OFF23_medium_{system_ft_name}_{n}_config_strat_w_1_4_vac_sample_fine_tuned_{energy_str}rep_{rep_num}",
    #              # f"MACE-MP-0b3_medium_{system_ft_name}_{n}_config_strat_w_1_4_vac_sample_fine_tuned_{energy_str}rep_{rep_num}",
    #             ]
    
    # formatted_model_names = [
    #     f'GO-MACE-23_{system_ft_name}_{n}_config_w_1_4_vac_sample_{energy_str}ft', 
    #     # f'MACE-OFF23_{system_ft_name}_{n}_config_w_1_4_vac_sample_{energy_str}ft', 
    #     # f'MACE-MP-0b3_medium_{system_ft_name}_{n}_config_w_1_4_vac_sample_{energy_str}ft',
    # ]

    # --- fully fine-tuned (positive control) paper models ---
    # energy_corrected = True
    # system_ft_name = "CPD_MVK_DA_train"

    # energy_str = 'energy_' if energy_corrected else ''
    # model_names = [
    #             #   f"GO-MACE-23_{system_ft_name}_fine_tuned_{energy_str}rep_1",
    #             #   f"MACE-OFF23_medium_{system_ft_name}_fine_tuned_{energy_str}rep_1",
    #             #   f"MACE-MP0-128-L1_{system_ft_name}_fine_tuned_{energy_str}rep_1",
    #               f"MACE-MP-0b3_medium_{system_ft_name}_fine_tuned_{energy_str}rep_1",
    #               ]
    # energy_str = 'energy' if energy_corrected else ''
    # formatted_model_names = [
    #                         #  f'GO-MACE-23_{system_ft_name}_full_ft_{energy_str}', 
    #                         #  f'MACE-OFF23_{system_ft_name}_full_ft_{energy_str}', 
    #                         #  f'MACE-MP0-128-L1_{system_ft_name}_full_ft_{energy_str}',
    #                          f'MACE-MP-0b3_medium_{system_ft_name}_full_ft_{energy_str}'
    #                         ]

    # # full zero-shot model comparison
    # model_names = list(MODELS[5:])
    # model_names += ["MACE-OFF23_medium_endo_DA_1.0_train_from_scratch"]  # use medium mace off model architecture trained on full dataset as a from scratch reference
    # save_name = "zero_shot_models"

    # === SELECT INPUT GEOMETRY ===

    system_name, num_solute_atoms, solvent_name, fs = "CPD_MVK_DA", 22, 'water', 1000
    # system_name, num_solute_atoms, solvent_name, fs = "anthracene_DA", 41, 'toluene', 1000
    # system_name, num_solute_atoms, solvent_name, fs = "anthracene_1_4_DA", 41, 'toluene', 1000
    # system_name, num_solute_atoms, solvent_name, fs = "guajadial", 69, 'water', 2000 #4000  # 2000
    # system_name, num_solute_atoms, solvent_name, fs = "guajadial_DA", 69, 'water', 1000 # 4000 

    # --- vacuum traj ---
    # md_config = MDSettings(fs=fs, temp=300, dt=0.5, interval=20, md_method='ase')
    # num_init_configs = 1    # number of initial configurations to load from the starting config .xyz file
    # md_runs_per_config = 100
    # init_config_set_fpaths, formatted_system_names = [f"{OPT_DIR}/{system_name}_vac_{model_name}_opt_TS.xyz" for model_name in model_names], \
    #                                                  [f"{system_name}_vac_opt_TS" for _ in model_names]    # use model optimised vac TS

    # --- solvent traj ---
    md_config = MDSettings(fs=fs, temp=300, dt=0.5, interval=20, md_method='ase')
    num_init_configs = 1    # number of initial configurations to load from the starting config .xyz file
    md_runs_per_config = 100
    init_config_set_fpaths, formatted_system_names = [f"{OPT_DIR}/{system_name}_in_{solvent_name}_{model_name}_opt_TS.xyz" for model_name in model_names], \
        [f"{system_name}_{solvent_name}_opt_TS" for _ in model_names]

    # --- long solvent traj with fixed solute ---
    
    # set optimised TS in solvent
    # init_config_set_fpaths, formatted_system_names = [f"{OPT_DIR}/{system_name}_in_{solvent_name}_{model_name}_opt_TS.xyz" for model_name in model_names], \
    #     [f"{system_name}_{solvent_name}_opt_TS_fixed_solute" for _ in model_names]

    # solute_indices = set(range(num_solute_atoms))
    # num_atoms = len(read(init_config_set_fpaths[0], ':')[0])
    # constraint = FixAtoms(mask=[i in solute_indices for i in range(num_atoms)])
    # md_config = MDSettings(fs=100*PS, temp=300, dt=0.5, interval=100, md_method='ase')
    # num_init_configs = 1
    # md_runs_per_config = 1

    # === SPECIFY OTHER RUN OPTIONS ===
    enable_cueq = False
    override_saved = False

    # ====================================================

    # run model MD run
    for init_config_set_fpath, model_name, formatted_model_name, formatted_system_name in zip(init_config_set_fpaths, model_names, formatted_model_names, formatted_system_names):

        # load ase structure
        init_configs = read(init_config_set_fpath, f'0:{num_init_configs}')

        logger.info(f"Init Config Cell: {init_configs[0].cell}")
        logger.info(f"Init Config PBC: {init_configs[0].pbc}")
        logger.info(f"Loaded first {len(init_configs)} configurations from {init_config_set_fpath}...")
        logger.info(f"Evaluating model: {model_name}...")

        if box_dim is not None:
            logger.info(f"Manually overiding config cell to {box_dim} with PBC...")
            for init_config in init_configs:
                init_config.set_cell(box_dim)
                init_config.center()
                init_config.set_pbc([True, True, True])

        trajectories = run_md_eval(init_configs,
                                   md_config,
                                   md_runs_per_config,
                                   model_name,
                                   formatted_model_name,
                                   formatted_system_name,
                                   constraint=constraint,
                                   enable_cueq=enable_cueq,
                                   override_saved=override_saved)
        