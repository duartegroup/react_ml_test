import os
import shutil

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import FixAtoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate
from ase.vibrations import Vibrations

from react_ml_test_public.helpers.io import flexible_ase_read, load_model_calculator
from react_ml_test_public.helpers.logging import logger
from react_ml_test_public.helpers.settings import OPT_DIR, VIB_DIR
from react_ml_test_public.helpers.solvation import get_solv_shell
from react_ml_test_public.helpers.stats import calculate_rmse


def mlip_dimer_method(input_config_fp: str, 
                      model_name: str,
                      react_coords: list[tuple],
                      save_name: str,
                      out_dir: str = OPT_DIR,
                      ref_config_fp: str = None,
                      num_solute_atoms: int = None,
                      atoms_per_solv_mol: int = None,
                      solv_shell_cutoff: float = 3.5,
                      displace_mag: float = 0.05,
                      reverse_displace: bool = False,
                      fmax: float = 0.001,
                      cuda_device: str = 'cuda:0',
                      enable_cueq: bool = False,
                      box_dim: list[int, int, int] = None,
                      override_cache: bool = False,
                      model_dir: str = None
                      ) -> Atoms:
    """
    Runs dimer TS Optimisation method with an MLIP.

    Parameters:
        input_config_fp (str): Path to input configuration file
        model_name (str): Name of the MLIP model to use (in models dir)
        react_coords (list[tuple]): List of reaction coordinate tuples [(a1 -> a2), (a3 -> a4)].
        save_name (str): Name for saving output files.
        out_dir (str, optional): Output directory. Defaults to OPT_DIR.
        displace_mag (float, optional): Magnitude of initial atomic displacement along reaction coordinate for dimer. Defaults to 0.05.
        fmax (float, optional): Optimisation convergence threshold. Defaults to 0.001.
        cuda_device: which gpu device to use, default 'cuda:0' (falls back to 'cpu' if the device is not available)
        enable_cueq: enable cuequivariance speedup for MACE models (requires CUDA 12.0)
        box_dim: PBC box to override in input configuration for optimisation
        override_cache: whether to override previous runs of the same name

    Returms:
        ase.Atoms: optimised TS configuration.
    """

    # check for existing opt ts with this save name, and if not overriding cache, skip
    opt_ts_save_fp = f'{out_dir}/{save_name}_{model_name}_opt_TS.xyz'
    if not override_cache and os.path.exists(opt_ts_save_fp):
        logger.info(f'Detected OPT TS file, {opt_ts_save_fp} skipping...')
        ts_config = flexible_ase_read(opt_ts_save_fp)[0]
        return ts_config 

    # get input config and count solute atoms if not specified
    input_config = flexible_ase_read(input_config_fp)[0]
    num_solute_atoms = len(input_config) if num_solute_atoms is None else num_solute_atoms

    # get MLIP
    if box_dim is not None:
        input_config.set_cell(box_dim)
        input_config.set_pbc([True, True, True])
        logger.info(f"Set PBC box dimensions to {box_dim}")
    else:
        logger.info(f"Using input box dimensions of {input_config.cell}")
    input_config.calc = load_model_calculator(model_name, device=cuda_device,
                                              enable_cueq=enable_cueq, model_dir=model_dir)
    input_config.get_potential_energy()
    input_xyz = input_config.get_positions()
    ref_xyz = input_xyz
    ref_r_dists = [input_config.get_distance(*r_atoms, mic=True) for r_atoms in react_coords]

    # get solvation shell for solvent freezing mask
    if atoms_per_solv_mol is not None and solv_shell_cutoff is not None:
        # assumes solute atoms are at the start of the .xyz file, and that solvent atoms 
        # are grouped sequentially by molecule
        solute_indices = list(range(num_solute_atoms))
        solv_indices = list(range(num_solute_atoms, len(input_config)))
        solv_shell_mols = get_solv_shell(input_config, solute_indices, solv_indices, 
                                         atoms_per_solv_mol, cutoff=solv_shell_cutoff)
        logger.info(f"Solvent molecules in solvent shell: {len(solv_shell_mols)}")
        solv_shell_indices = [i for mol in solv_shell_mols for i in mol]
        
        # freeze solvent atoms
        solv_mask = [i not in set(solv_shell_indices).union(solute_indices) for i in range(len(input_config))]
        input_config.set_constraint(FixAtoms(mask=solv_mask))
        # print("Solv mask: ", solv_mask)

    # Set up the dimer
    react_coord_ids = set([a_id for r_coord in react_coords for a_id in r_coord])
    logger.info(f'React Coord IDs: {react_coord_ids}')
    with DimerControl(initial_eigenmode_method='displacement',
                      displacement_method='vector',
                      maximum_translation=0.1,
                      logfile=None,
                      mask=[i in react_coord_ids for i in range(len(input_config))]) as d_control:
        
        # 'f_rot_min': 0.1,
        # 'f_rot_max': 1.00,
        # 'max_num_rot': 1,
        d_atoms = MinModeAtoms(input_config, d_control)

        # displace the atoms along reaction coordinate
        rev_fac = 1.0 if reverse_displace else -1.0
        displacement_vector = [[0.0] * 3 for _ in range(len(input_config))]
 
        for i, j in react_coords:
            di = input_xyz[j] - input_xyz[i]
            dj = input_xyz[i] - input_xyz[j]
            di = di / np.linalg.norm(di)
            dj = dj / np.linalg.norm(dj)
            displacement_vector[i] = list(displace_mag * rev_fac * di)
            displacement_vector[j] = list(displace_mag * rev_fac * dj)
            # logger.info(f"Input XYZ i {input_xyz[i]}")
            # logger.info(f"Input XYZ j {input_xyz[j]}")
            # logger.info(f"di: {di}")
            # logger.info(f"dj: {dj}")
            # logger.info(f"Displace Vec i: {displacement_vector[i]}")
            # logger.info(f"Displace Vec j: {displacement_vector[j]}")
        d_atoms.displace(displacement_vector=displacement_vector)

        # converge to a saddle point
        traj_save_fp = f'{out_dir}/{save_name}_{model_name}_dimer_traj.traj'
        with MinModeTranslate(d_atoms, 
                              trajectory=traj_save_fp,
                              logfile=f"{out_dir}/{save_name}_{model_name}_dimer_opt_log.txt") as dim_rlx:

            logger.info(f'Running dimer optimisation on {input_config_fp} with {model_name} (box_dim={box_dim}, fmax={fmax})...')
            dim_rlx.run(fmax=fmax)

        traj_configs = ase_read(traj_save_fp, index=':')
    
        # save full trajectory
        old_traj_fp = traj_save_fp
        ase_write(traj_save_fp.replace('.traj', '.xyz'), traj_configs, format='extxyz')
        os.remove(old_traj_fp)
        logger.info(f"Saved optimisation trajectory to: {traj_save_fp.replace('.traj', '.xyz')}")

        # save optimised TS (last frame)
        opt_ts_config = traj_configs[-1]
        opt_ts_config.set_constraint(None)
        ase_write(opt_ts_save_fp, opt_ts_config, format='extxyz')
        logger.info(f'Saved Opt TS to: {opt_ts_save_fp}')

        # print important stats
        ref_config = flexible_ase_read(ref_config_fp)[0] if ref_config_fp is not None else None 
        ref_xyz = ref_config.get_positions() if ref_config is not None else ref_xyz
        ref_r_dists = [ref_config.get_distance(*r_atoms, mic=True) for r_atoms in react_coords] if ref_config is not None else ref_r_dists
        ref_log_str = input_config_fp if ref_config_fp is None else ref_config_fp
        logger.info(f"Using {ref_log_str} as reference TS geometry:")

        struct_rmse = calculate_rmse(opt_ts_config.get_positions()[0:num_solute_atoms], ref_xyz[0:num_solute_atoms])
        logger.info(f'MLIP OPT TS Reactants Structural RMSD: {struct_rmse:.3f}')
        opt_TS_r_dists = [opt_ts_config.get_distance(*r_atoms, mic=True) for r_atoms in react_coords]
        logger.info("Ref TS Distances:" + ' '.join([f" r{i} ({r_coord}): {ref_r_dists[i]:.3f}" for i, r_coord in enumerate(react_coords)]))
        logger.info("MLIP Opt TS Distances:" + ' '.join([f" r{i} ({r_coord}): {opt_TS_r_dists[i]:.3f}" for i, r_coord in enumerate(react_coords)]))    
        react_coord_dist_error = sum([abs(ref_r_dists[i] - opt_TS_r_dists[i]) for i, _ in enumerate(react_coords)])
        logger.info(f"React Coord Dist. Error: {react_coord_dist_error:.3f}")

    # return optimised TS
    return opt_ts_config


def get_vib_freq(input_config_fp: str,
                 calc: Calculator,
                 save_name: str,
                 out_dir: str = VIB_DIR,
                 ):

    # run vib analysis
    input_config = flexible_ase_read(input_config_fp)[0]
    input_config.calc = calc
    vib = Vibrations(input_config, name=f"{save_name}")
    vib.run()

    # log summary
    out_fp = f'{out_dir}/{save_name}_vib.txt'
    if os.path.exists(out_fp):
        os.remove(out_fp)
    vib.summary(log=out_fp)

    # save expected imaginary mode if exists
    n = 0
    vib.write_mode(n)
    vib_mode_fp = f"{save_name}.{n}.traj"
    vib_mode_configs = ase_read(vib_mode_fp, index=':')
    old_vib_mode_fp = vib_mode_fp
    vib_mode_fp = f"{out_dir}/{save_name}_vib_mode_{n}.xyz" 
    ase_write(vib_mode_fp, vib_mode_configs, format='xyz')
    os.remove(old_vib_mode_fp)
    logger.info(f"Saved vib mode {n} to: {vib_mode_fp}")

    # remove cache directory
    vib.clean()
    shutil.rmtree(f"{save_name}")
