"""
Script for evaluating the performance of foundation models on a specified test system.
"""


import argparse
import os

# from ase.constraints.constraint import IndexedConstraint
from time import time

from ase.constraints import FixAtoms
from ase.io import read

from react_ml_test_public.helpers import inputs as rmlt_inputs
from react_ml_test_public.helpers.dynamics import get_trajectory_with_ase, save_md_run_csv
from react_ml_test_public.helpers.io import (
    check_cuda_setup,
    compute_device_label,
    get_name_from_path,
    load_model_calculator,
)
from react_ml_test_public.helpers.logging import logger
from react_ml_test_public.helpers.settings import (
    CPD_MVK_DA_DATA_DIR,
    DA_PAPER_DATA_DIR,
    DATA_DIR,
    MD_INPUT_DIR,
    MODELS,
    OPT_DIR,
    PS,
    TRAJ_DIR,
    MDSettings,
)

# the MD engine the trajectories below are run with: get_trajectory_with_ase, i.e. ASE. Only
# ASE is wired up here, so this is also what the benchmark CSVs record as the calculator.
MD_CALCULATOR = 'ASE'


def run_md_eval(init_configs: list,
                md_config: MDSettings,
                md_runs_per_config: int,
                model_name: str,
                formatted_model_name: str,
                system_name: str,
                traj_dir: str = TRAJ_DIR,
                csv_dir: str = None,
                model_dir: str = None,
                constraint: str = None,# IndexedConstraint = None,
                enable_cueq: bool = False,
                override_saved: bool = False,
                cuda_device: str = "cuda:0",
                device_label: str = None,
                fair_chem_task: str = None,
                fair_chem_setting: str = None):

    """Run every MD trajectory for one model, and write a benchmark CSV for each of them.

    With ``csv_dir`` set, one CSV per run is saved under ``<csv_dir>/md`` (see
    ``save_md_run_csv``): the device the run used, the calculator, the wall time of every
    step, the mean step time, and the step count / simulated time / overall run time.
    ``device_label`` names that device in the CSV and defaults to the name of the resolved
    ``cuda_device`` (see ``compute_device_label``). ``enable_cueq``, ``model_dir`` and the
    FairChem pair ``fair_chem_task`` / ``fair_chem_setting`` are forwarded to every run's
    calculator.
    """

    # the CSV records the device by name (e.g. 'NVIDIA H100 80GB HBM3'), once for the run
    if csv_dir is not None and device_label is None:
        device_label = compute_device_label(cuda_device)

    trajectories = []
    for i in range(len(init_configs)):

        init_config = init_configs[i].copy()

        for j in range(md_runs_per_config):
            logger.info(f"Running MD trajectory for init config {i + 1}, run number {j}...")

            start_time = time()
            traj_name = f'{system_name}_{formatted_model_name}_{md_config.temp}K_{md_config.fs / PS}ps_config_{i}_run_{j}'

            out_trj_name = f"{traj_dir}/{traj_name}_traj.xyz"
            if os.path.exists(out_trj_name) and not override_saved:
                logger.info(f"{out_trj_name} already exists, skipping...")
                continue

            step_times = [] if csv_dir is not None else None

            trajectory = get_trajectory_with_ase(init_config,
                                                 model_name,
                                                 md_config,
                                                 traj_name,
                                                 out_dir=traj_dir,
                                                 constraint=constraint,
                                                 cuda_device=cuda_device,
                                                 override_cache=override_saved,
                                                 model_dir=model_dir,
                                                 enable_cueq=enable_cueq,
                                                 step_times=step_times,
                                                 fair_chem_task=fair_chem_task,
                                                 fair_chem_setting=fair_chem_setting)
            end_time = time()
            md_time = end_time - start_time
            logger.info(f"Ran MD trajectory '{traj_name}' in {md_time:.4f}s, with an average step time of {float(md_time) / steps}...")

            # the step times are only recorded for a run that actually ran, so an empty list
            # means the trajectory was reused and there is nothing to benchmark
            if step_times:
                save_md_run_csv(f"{csv_dir}/md/{traj_name}.csv",
                                step_times,
                                md_config,
                                md_time,
                                device=device_label,
                                calculator=MD_CALCULATOR)

            trajectories.append(trajectory)

    return trajectories


if __name__ == "__main__":

    """Run MLIP MD trajectories starting from each model's optimised TS.

    The path to the input YAML file is given as a single command-line argument:

        rmlt run-mlip-md inputs/md/CPD_MVK_DA_dynamics_vac.yaml

    (or equivalently: python -m react_ml_test_public.scripts.run_mlip_md
    inputs/md/CPD_MVK_DA_dynamics_vac.yaml)

    Input YAML format (see inputs/md/CPD_MVK_DA_dynamics_vac.yaml for a full example):

    Fields (path fields may reference environment variables, e.g. model_dir: "$DATA/models"):
        init_configs_fpath:   # optional starting-config file (e.g. a solvated config), read as
                              # relative to data_dir when it does not resolve from the cwd; when
                              # unset, each model's optimised TS in out/opt is used.
        data_dir:             # data root that a relative init_configs_fpath resolves against
                              # (defaults to DATA_DIR in settings).
        out_dir:              # output root; trajectories are written to <out_dir>/trj and one
                              # MD benchmark CSV per run to <out_dir>/csv/md (defaults to
                              # OUT_DIR in settings).
        model_dir:            # directory the model_names are loaded from, when they live
                              # outside the default MODEL_DIR (defaults to MODEL_DIR).
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

        model_names:          # list of model names to run (supports ${var} templating).
        system_ft_name:       # fine-tuning training-set name (for model-name templating).
        n:                    # fine-tuning set size.
        sample_str:           # sampling-strategy suffix used in fine-tuned names (e.g. "ts_gauss_").
        energy_str:           # energy label string used in fine-tuned names (e.g. "energy_").
        rep_num:              # fine-tuned model repetition number.
        omol_rep_num:         # OMOL model repetition number.
        omol_energy_str:      # OMOL energy suffix/prefix string.

        enable_cueq:          # use the cuEquivariance MACE speed-up (needs a CUDA GPU and
                              # mace-torch's cueq extras).
        fair_chem_task:       # FairChem only: the task a FAIRCHEM_MODELS entry runs as, e.g.
                              # "omol" for molecules (run this script from the fair_chem env).
        fair_chem_setting:    # FairChem only: e.g. "turbo", recommended for MD (fairchem's
                              # own default is "default").
        override_saved:       # re-run MD even if a saved trajectory already exists.
        cuda_device:          # GPU device for the MLIP calculators (default "cuda:0",
                              # falls back to "cpu" if that device is not found).

    Alongside each trajectory, a CSV of the run's performance is written to <out_dir>/csv/md
    (one per MD run, named after the trajectory): the compute device, the calculator, the wall
    time of every step, the mean step time, the number of steps run, the simulated time they
    amount to and the overall run time (see helpers.dynamics.save_md_run_csv).

    formatted_model_names are built in the script: fine-tuned entries of this training set get
    a compact "<label>_<system_ft_name>_<n>_config_<energy><sample_str>ft" label, other entries
    (e.g. zero-shot models) keep their model name.

    Any omitted field is set to None (or its documented default).
    """

    constraint = None

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
    init_configs_fpath = resolved["init_configs_fpath"]
    data_dir = resolved["data_dir"]
    out_dir = resolved["out_dir"]
    traj_dir = resolved["traj_dir"]
    csv_dir = resolved["csv_dir"]
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

    # === 3. MODELS (model_names list; fine-tuning naming components for ${...} templating) ===
    model_dir = resolved["model_dir"]
    model_names = resolved["model_names"]
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
    fair_chem_task = resolved["fair_chem_task"]
    fair_chem_setting = resolved["fair_chem_setting"]
    del resolved

    # resolve the requested device (i.e. fall back to 'cpu' if the GPU is not available)
    cuda_device = check_cuda_setup(cuda_device)

    # the MD benchmark CSVs go under <csv_dir>/md, one per run (see run_md_eval)
    md_csv_dir = f"{csv_dir}/md"

    md_config = MDSettings(fs=fs, temp=temp, dt=dt, interval=interval, md_method=md_method)

    # formatted labels for trajectory names: fine-tuned entries -> compact label, else model name
    ft_suffixes = [f"_{system_ft_name}_{n}_config_strat_{sample_str}fine_tuned_{en}rep_{rep}"
                   for en, rep in ((energy_str, rep_num), (omol_energy_str, omol_rep_num))]
    formatted_model_names = []
    for model_name in model_names:
        for ft_en, ft_suffix in zip((energy_str, omol_energy_str), ft_suffixes):
            if model_name.endswith(ft_suffix):
                base = model_name[:-len(ft_suffix)]
                stem = "MACE-OFF23" if base == "MACE-OFF23_medium" else base
                model_name = f"{stem}_{system_ft_name}_{n}_config_{ft_en}{sample_str}ft"
                break
        formatted_model_names.append(model_name)

    logger.info(f'Evaluating models: {model_names}')

    # === 5. INIT CONFIG SET (OPTIMISED TS PER MODEL) ===
    logger.info(f"Reading configs relative to data_dir: {data_dir}, "
                f"writing trajectories under out_dir: {out_dir} (i.e. {traj_dir})")
    os.makedirs(traj_dir, exist_ok=True)
    # one benchmark CSV per run is written here (see run_md_eval)
    os.makedirs(md_csv_dir, exist_ok=True)

    if init_configs_fpath is None:
        if include_solvent:
            init_configs_fpaths = [f"{OPT_DIR}/{system_name}_in_{solvent_name}_{model_name}_opt_TS.xyz" for model_name in model_names]
            formatted_system_names = [f"{system_name}_{solvent_name}_opt_TS" for _ in model_names]
        else:
            init_configs_fpaths = [f"{OPT_DIR}/{system_name}_vac_{model_name}_opt_TS.xyz" for model_name in model_names]
            formatted_system_names = [f"{system_name}_vac_opt_TS" for _ in model_names]
    else:
        init_configs_fpaths = [init_configs_fpath for _ in model_names]
        formatted_system_names = [f"{system_name}_{model_name}" for model_name in model_names]

    # ====================================================

    # run model MD run
    for init_configs_fpath, model_name, formatted_model_name, formatted_system_name in zip(init_configs_fpaths, model_names, formatted_model_names, formatted_system_names):

        # load ase structure
        init_configs = read(init_configs_fpath, f'0:{num_init_configs}')

        logger.info(f"Init Config Cell: {init_configs[0].cell}")
        logger.info(f"Init Config PBC: {init_configs[0].pbc}")
        logger.info(f"Loaded first {len(init_configs)} configurations from {init_configs_fpath}...")
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
                                   traj_dir=traj_dir,
                                   csv_dir=csv_dir,
                                   model_dir=model_dir,
                                   constraint=constraint,
                                   enable_cueq=enable_cueq,
                                   override_saved=override_saved,
                                   cuda_device=cuda_device,
                                   fair_chem_task=fair_chem_task,
                                   fair_chem_setting=fair_chem_setting)
