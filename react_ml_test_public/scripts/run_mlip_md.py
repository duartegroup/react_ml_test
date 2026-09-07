"""
Script for evaluating the performance of foundation models on a specified test system.
"""


import argparse
import os

# from ase.constraints.constraint import IndexedConstraint
from time import time

import torch
from ase.constraints import FixAtoms
from ase.io import read

from react_ml_test_public.helpers import inputs as rmlt_inputs
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


def build_ft_model_names(base_model_names: list,
                         system_ft_name: str,
                         n: int,
                         sample_str: str,
                         energy_str: str,
                         rep_num: int,
                         omol_energy_str: str,
                         omol_rep_num: int):
    """Build the fine-tuned ``model_names`` and short ``formatted_model_names`` for a run.

    Each base model is assumed to have been fine-tuned on ``{system_ft_name}_{n}`` with the
    given sampling / energy / repetition naming, i.e.:

        model_names:          <base>_<system_ft_name>_<n>_config_strat_<sample_str>fine_tuned_<energy_str>rep_<rep_num>
        formatted_model_names: <label>_<system_ft_name>_<n>_config_<energy_str><sample_str>ft

    ``MACE-omol-0-extra-large-1024`` uses its own (``omol_*``) energy / repetition labels. The
    formatted labels are the compact names used in trajectory / output names (with
    ``MACE-OFF23_medium`` shortened to ``MACE-OFF23``).
    """
    model_names, formatted_model_names = [], []
    for base_model_name in base_model_names:
        is_omol = base_model_name == "MACE-omol-0-extra-large-1024"
        ft_energy_str = omol_energy_str if is_omol else energy_str
        ft_rep_num = omol_rep_num if is_omol else rep_num
        model_names.append(f"{base_model_name}_{system_ft_name}_{n}_config_strat_"
                           f"{sample_str}fine_tuned_{ft_energy_str}rep_{ft_rep_num}")
        formatted_stem = "MACE-OFF23" if base_model_name == "MACE-OFF23_medium" else base_model_name
        formatted_model_names.append(f"{formatted_stem}_{system_ft_name}_{n}_config_"
                                     f"{ft_energy_str}{sample_str}ft")
    return model_names, formatted_model_names


def run_md_eval(init_configs: list,
                md_config: MDSettings,
                md_runs_per_config: int,
                model_name: str,
                formatted_model_name: str,
                system_name: str,
                constraint: str = None,# IndexedConstraint = None,
                enable_cueq: bool = False,
                override_saved: bool = False,
                cuda_device: str = CUDA_DEVICE):

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
                                                 cuda_device=cuda_device,
                                                 override_cache=override_saved)
            end_time = time()
            md_time = end_time - start_time
            steps = float(md_config.fs / md_config.dt)
            logger.info(f"Ran MD trajectory '{traj_name}' in {md_time:.4f}s, with an average step time of {float(md_time) / steps}...")
            trajectories.append(trajectory)

    return trajectories


if __name__ == "__main__":

    """Run MLIP MD trajectories starting from each model's optimised TS.

    The path to the input YAML file is given as a single command-line argument:

        rmlt run-mlip-md inputs/md/CPD_MVK_DA_dynamics_vac.yaml

    (or equivalently: python -m react_ml_test_public.scripts.run_mlip_md
    inputs/md/CPD_MVK_DA_dynamics_vac.yaml)

    Input YAML format (see inputs/md/CPD_MVK_DA_dynamics_vac.yaml for a full example):

    Fields:
        system_name:          # system name; also prefixes the starting-config filenames.
        include_solvent:      # true: start from in-solvent-optimised TS, false: vacuum TS.
        solvent_name:         # solvent used in the starting-config filenames (include_solvent only).

        fs:                   # total simulation time (fs) of each MD run.
        temp:                 # MD temperature (K).
        dt:                   # integration time step (fs).
        interval:             # frame-saving interval (MD steps).
        md_method:            # MD engine, "ase" / "openmm".
        num_init_configs:     # number of starting configs loaded from the optimised-TS file.
        md_runs_per_config:   # number of MD runs per starting config.
        box_dim:              # optional [x, y, z] box override (default: keep the input cell).

        base_model_names:     # list of base model architectures to run (each fine-tuned).
        system_ft_name:       # fine-tuning training-set name.
        n:                    # fine-tuning set size.
        sample_str:           # sampling-strategy suffix used in fine-tuned names (e.g. "ts_gauss_").
        energy_str:           # energy label string used in fine-tuned names (e.g. "energy_").
        rep_num:              # fine-tuned model repetition number.
        omol_rep_num:         # OMOL model repetition number.
        omol_energy_str:      # OMOL energy suffix/prefix string.

        enable_cueq:          # use the cuEquivariance MACE speed-up (CUDA 12+).
        override_saved:       # re-run MD even if a saved trajectory already exists.
        cuda_device:          # GPU device for the MLIP calculators.

    model_names and formatted_model_names are built in the script from base_model_names and
    the fine-tuning components above (they are not set in the YAML).

    Any omitted field is set to None (or its documented default).
    """

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

    # input YAML config path from a single command-line argument
    parser = argparse.ArgumentParser(
        description="Run MLIP MD trajectories driven by an input YAML config file "
                    "(format described above).",
        epilog="Example: rmlt run-mlip-md inputs/md/CPD_MVK_DA_dynamics_vac.yaml",
    )
    parser.add_argument(
        "input_yaml",
        help="Path to the input YAML config file "
             "(e.g. inputs/md/CPD_MVK_DA_dynamics_vac.yaml).",
    )
    args = parser.parse_args()

    # load + process the input YAML (see react_ml_test_public/helpers/inputs.py)
    resolved = rmlt_inputs.resolve_mlip_md_yaml(args.input_yaml, parser)

    # === 1. SYSTEM / STARTING CONFIGS ===
    system_name = resolved["system_name"]
    include_solvent = resolved["include_solvent"]
    solvent_name = resolved["solvent_name"]

    # === 2. MD SETTINGS ===
    fs = resolved["fs"]
    temp = resolved["temp"]
    dt = resolved["dt"]
    interval = resolved["interval"]
    md_method = resolved["md_method"]
    num_init_configs = resolved["num_init_configs"]
    md_runs_per_config = resolved["md_runs_per_config"]
    box_dim = resolved["box_dim"]

    # === 3. MODELS (fine-tuned naming components -> model_names / formatted_model_names) ===
    base_model_names = resolved["base_model_names"]
    system_ft_name = resolved["system_ft_name"]
    n = resolved["n"]
    sample_str = resolved["sample_str"]
    energy_str = resolved["energy_str"]
    rep_num = resolved["rep_num"]
    omol_rep_num = resolved["omol_rep_num"]
    omol_energy_str = resolved["omol_energy_str"]

    # === 4. RUN OPTIONS ===
    enable_cueq = resolved["enable_cueq"]
    override_saved = resolved["override_saved"]
    cuda_device = resolved["cuda_device"]
    del resolved

    md_config = MDSettings(fs=fs, temp=temp, dt=dt, interval=interval, md_method=md_method)
    model_names, formatted_model_names = build_ft_model_names(base_model_names,
                                                              system_ft_name,
                                                              n,
                                                              sample_str,
                                                              energy_str,
                                                              rep_num,
                                                              omol_energy_str,
                                                              omol_rep_num)
    logger.info(f'Evaluating models: {model_names}')

    # === 5. INIT CONFIG SET (OPTIMISED TS PER MODEL) ===
    if include_solvent:
        init_config_set_fpaths = [f"{OPT_DIR}/{system_name}_in_{solvent_name}_{model_name}_opt_TS.xyz" for model_name in model_names]
        formatted_system_names = [f"{system_name}_{solvent_name}_opt_TS" for _ in model_names]
    else:
        init_config_set_fpaths = [f"{OPT_DIR}/{system_name}_vac_{model_name}_opt_TS.xyz" for model_name in model_names]
        formatted_system_names = [f"{system_name}_vac_opt_TS" for _ in model_names]

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
                                   override_saved=override_saved,
                                   cuda_device=cuda_device)
