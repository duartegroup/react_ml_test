"""
Helpers for running dynamics with MLIPs.
"""
import csv
import os
from time import time

from ase import units
from ase.md.langevin import Langevin
from ase.io import read
import ase
from react_ml_test_public.helpers.settings import MDSettings, PS, TRAJ_DIR
from react_ml_test_public.helpers.logging import logger
from react_ml_test_public.helpers.io import load_model_calculator


def get_trajectory_with_ase(init_config: ase.Atoms,
                            model_name: str,
                            md_config: MDSettings,
                            output_name: str,
                            out_dir: str = TRAJ_DIR,
                            constraint: str = None, # IndexedConstraint = None,
                            cuda_device: str = "cuda:0",  # falls back to 'cpu' if unavailable
                            override_cache: bool = False,
                            model_dir: str = None,
                            enable_cueq: bool = False,
                            step_times: list = None,
                            fair_chem_task: str = None,
                            fair_chem_setting: str = None):
    """

    :param init_config_fp:
    :param model_fp:
    :param output_name:
    :param enable_cueq: use the cuEquivariance MACE speed-up for the calculator (needs a CUDA
                        GPU and mace-torch's cueq extras).
    :param fair_chem_task: FairChem task to run a fairchem model as, e.g. 'omol' for molecules
                           (see load_model_calculator).
    :param fair_chem_setting: FairChem inference settings, e.g. 'turbo' (see
                              load_model_calculator).
    :param step_times: optional list to append the wall time (s) of every MD step to, for
                       benchmarking (see save_md_run_csv). A run that returns an existing
                       trajectory without taking any steps leaves it empty.
    :return:
    """
    traj_fp = f"{out_dir}/{output_name}_traj.xyz"

    # re-write trajectory file if it doesn't exist already
    if os.path.exists(traj_fp):
        if override_cache:
            logger.info(f"Overwriting {traj_fp}...")
            os.remove(traj_fp)
        else:
            logger.info(f"Found {traj_fp}, returning existing trj...")
            return read(traj_fp, index=":")

    # unpack MDConfig
    fs = md_config.fs
    temp = md_config.temp
    dt = md_config.dt
    interval = md_config.interval

    # load calculator (model_dir None -> MODEL_DIR, see resolve_model_dir; the last two
    # settings are FairChem's, and ignored by the other loaders)
    calc = load_model_calculator(model_name, device=cuda_device, dtype='float32',
                                 model_dir=model_dir, enable_cueq=enable_cueq,
                                 fair_chem_task=fair_chem_task,
                                 fair_chem_setting=fair_chem_setting)

    # make copy of init config
    init_config = init_config.copy()
    if constraint is not None:
        init_config.set_constraint(constraint)

    # 1. attach the calculator to the Atoms object
    init_config.set_calculator(calc)

    # 2. create the dynamics object
    dynamics = Langevin(
        init_config,
        timestep=dt * units.fs,
        temperature_K=temp,
        friction=0.02 #/ units.fs # currently using mlptrain standard, ,
    )

    # 3. time every step, for the benchmark CSV (attached before the writer below, so a
    # step's time is the step itself plus the trajectory write of the previous one)
    if step_times is not None:
        last_call = time()
        stepping = False

        def _record_step_time():
            nonlocal last_call, stepping
            call_time = time()
            # ASE calls its observers once before the first step (writing the initial
            # frame), so the first call only sets the baseline - every call after it
            # closes one step
            if stepping:
                step_times.append(call_time - last_call)
            stepping = True
            last_call = call_time

        dynamics.attach(_record_step_time, interval=1)

    # 4. attach a callback to periodically write the trajectory to a file
    dynamics.attach(
        lambda: init_config.write(traj_fp, format='extxyz', append=True),
        interval=interval,
    )

    # run the dynamics
    logger.info("Running dynamics trajectory...")
    dynamics.run(steps=fs)

    return read(traj_fp, index=":")


def save_md_run_csv(csv_fpath: str,
                    step_times: list,
                    dt: float,
                    run_time_s: float,
                    device: str,
                    calculator: str = "ASE"):
    """
    Save the benchmark results of one MD run to a CSV.

    The file holds the run's identity (device, calculator) first, then the wall time of
    every MD step in order, then the summary values::

        device,NVIDIA H100 80GB HBM3
        calculator,ASE
        step,step_time_s
        0,0.015400
        ...
        999,0.011000
        mean_step_time_s,0.011300
        total_steps,1000
        total_simulation_time_ps,0.5000
        total_run_time_s,12.400

    The times are microseconds-resolution seconds, so a fast step is not rounded away.

    Note that the simulated time comes from the steps actually taken and the time step, as
    ASE runs ``fs`` steps rather than ``fs / dt`` - it is therefore not the ``fs / PS``
    figure the trajectory name carries.

    :param csv_fpath: path of the CSV to write (its directory must exist).
    :param step_times: wall time (s) of every step taken, in order.
    :param dt: integration time step (fs).
    :param run_time_s: overall wall time (s) of the run, including loading the model.
    :param device: compute device the run used, e.g. 'CPU' (see compute_device_label).
    :param calculator: MD calculator/engine that was used.
    """
    total_steps = len(step_times)
    mean_step_time = sum(step_times) / total_steps if total_steps else 0.0

    with open(csv_fpath, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(['device', device])
        writer.writerow(['calculator', calculator])

        writer.writerow(['step', 'step_time_s'])
        for step, step_time in enumerate(step_times):
            writer.writerow([step, f"{step_time:.6f}"])

        writer.writerow(['mean_step_time_s', f"{mean_step_time:.6f}"])
        writer.writerow(['total_steps', total_steps])
        # the steps actually taken, at the time step used (fs -> ps)
        writer.writerow(['total_simulation_time_ps', f"{total_steps * dt / PS:.4f}"])
        writer.writerow(['total_run_time_s', f"{run_time_s:.3f}"])

    logger.info(f"Saved MD benchmark CSV to {csv_fpath}")


def get_trajectory_with_mlptrain(init_config,
                                 mlp_model: object,
                                 md_config: MDSettings,
                                 output_name: str):
    """

    :param system:
    :return:
    """
    import mlptrain as mlt

    # unpack MDConfig
    fs = md_config.fs
    temp = md_config.temp
    dt = md_config.dt
    interval = md_config.interval
    md_method = md_config.md_method

    if md_method == 'ase':

        # run ase dynamics
        trajectory = mlt.md.run_mlp_md(
            configuration=init_config,
            mlp=mlp_model,
            fs=fs,
            temp=temp,
            dt=dt,
            interval=interval,
        )

    elif md_method == 'openmm':

        # run openmm dynamics
        trajectory = mlt.md_openmm.run_mlp_md_openmm(
            configuration=init_config,
            mlp=mlp_model,
            fs=fs,
            temp=temp,
            dt=dt,
            interval=interval,
        )

    else:
        logger.info("Please select one md_method from ('ase', 'openmm')...")
        return

    # save and return trajectory
    trajectory.save(filename=f'{TRAJ_DIR}/{output_name}_traj.xyz')
    return trajectory
