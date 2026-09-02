"""
Helpers for running dynamics with MLIPs.
"""
import os
from ase import units
from ase.md.langevin import Langevin
from ase.io import read
import ase
from react_ml_test_public.helpers.settings import MDSettings, TRAJ_DIR
from react_ml_test_public.helpers.logging import logger
from react_ml_test_public.helpers.utils import load_model_calculator


def get_trajectory_with_ase(init_config: ase.Atoms,
                            model_name: str,
                            md_config: MDSettings,
                            output_name: str,
                            out_dir: str = TRAJ_DIR,
                            constraint: str = None, # IndexedConstraint = None,
                            cuda_device: str = "cuda:0",
                            override_cache: bool = False):
    """

    :param init_config_fp:
    :param model_fp:
    :param output_name:
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

    # load calculator
    calc = load_model_calculator(model_name, device=cuda_device, dtype='float32')

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

    # 3. attach a callback to periodically write the trajectory to a file
    dynamics.attach(
        lambda: init_config.write(traj_fp, format='extxyz', append=True),
        interval=interval,
    )

    # run the dynamics
    logger.info("Running dynamics trajectory...")
    dynamics.run(steps=fs)

    return read(traj_fp, index=":")


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
