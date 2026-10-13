"""
Script for evaluating the performance of foundation models on a specified test system.
"""


import argparse
import os

from ase.io import read

from react_ml_test_public.helpers import inputs as rmlt_inputs
from react_ml_test_public.helpers.dynamics import run_md_eval
from react_ml_test_public.helpers.io import check_cuda_setup
from react_ml_test_public.helpers.logging import logger
from react_ml_test_public.helpers.settings import OPT_DIR, MDSettings

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
        save_md_checkpoints:   # write a resumable checkpoint YAML per run under
                              # <out_dir>/md_checkpoints, every checkpoint_every ps
                              # (default false).
        load_md_checkpoint:   # true: carry every run on from its own checkpoint, each
                              # taking the steps it has left to reach fs fs of simulated
                              # time; or the path of one checkpoint, to carry a single run
                              # on from it (default false).
        checkpoint_every:     # ps of simulated time between checkpoint writes (default 10).
        cuda_device:          # GPU device for the MLIP calculators (default "cuda:0",
                              # falls back to "cpu" if that device is not found).

    Alongside each trajectory, a CSV of the run's performance is written to <out_dir>/csv/md
    (one per MD run, named after the trajectory): the compute device, the calculator, the wall
    time of every step, the mean step time, the number of steps run, the simulated time they
    amount to and the overall run time (see helpers.dynamics.save_md_run_csv).

    With save_md_checkpoints, each run also writes a checkpoint YAML under
    <out_dir>/md_checkpoints, every checkpoint_every ps, holding the state a run is continued
    from (see helpers.dynamics.write_md_checkpoint). Running the same input again with
    load_md_checkpoint then carries each run on from its last checkpoint rather than starting
    it over - the intended use being a job that hit its time limit.

    A run carried on like that does not overwrite the CSVs of the stretches before it: the
    first stretch writes <traj_name>.csv and each later one <traj_name>_contd_<n>.csv, each
    naming the steps it covers, so the whole run's benchmark is the set of them (see
    helpers.dynamics._md_csv_fpath). Each is refreshed every checkpoint_every ps, so a stretch
    that is itself cut short keeps the timings it managed.

    This script is only the command line around the run: the runs themselves, their naming and
    their per-stretch outputs are helpers.dynamics.run_md_eval.

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
    save_md_checkpoints = resolved["save_md_checkpoints"]
    load_md_checkpoint = resolved["load_md_checkpoint"]
    checkpoint_every = resolved["checkpoint_every"]
    md_checkpoint_dir = resolved["md_checkpoint_dir"]
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
    # one benchmark CSV per run is written here (see helpers.dynamics.run_md_eval)
    os.makedirs(md_csv_dir, exist_ok=True)
    # checkpoints are only written when they are asked for, so their directory is only made
    # then (a checkpoint is still read from wherever load_md_checkpoint points)
    if save_md_checkpoints:
        os.makedirs(md_checkpoint_dir, exist_ok=True)

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
                                   md_csv_dir=md_csv_dir,
                                   model_dir=model_dir,
                                   constraint=constraint,
                                   enable_cueq=enable_cueq,
                                   override_saved=override_saved,
                                   cuda_device=cuda_device,
                                   fair_chem_task=fair_chem_task,
                                   fair_chem_setting=fair_chem_setting,
                                   save_md_checkpoints=save_md_checkpoints,
                                   load_md_checkpoint=load_md_checkpoint,
                                   checkpoint_every=checkpoint_every,
                                   md_checkpoint_dir=md_checkpoint_dir)
