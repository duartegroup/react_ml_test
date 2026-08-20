"""
Script to use Dimer method to find optimal TS using MLIP.
"""
import glob
import os
import sys
import time

import mlptrain as mlt
import pytest
from mlptrain.box import Box

sys.path.append('../')
from ase.constraints import FixAtoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from react_ml_test.helpers.dynamics import get_trajectory_with_ase
from react_ml_test.helpers.io import get_ase_atoms_energy
from react_ml_test.helpers.logging import logger
from react_ml_test.helpers.plot import index_plot
from react_ml_test.helpers.settings import (
    ANTHRACENE_DA_DATA_DIR,
    CPD_MVK_DA_DATA_DIR,
    GUAJADIAL_DA_DATA_DIR,
    MODEL_DIR,
    OPT_DIR,
    PAPER_MODELS,
    PS,
    SOLVENTS_DIR,
    MDSettings,
)
from react_ml_test.helpers.ts_opt import get_vib_freq, mlip_dimer_method
from react_ml_test.helpers.utils import load_model_calculator

if __name__ == "__main__":

    atoms_per_solv_mol = None
    solv_shell_cutoff = 0.0
    solvent_xyz_fpath = None
    reverse_displace = False
    out_dir = OPT_DIR
    solvation_box_size = None

    # === 1. SELECT INPUT CONFIG (TS GUESS) ===

    # --- endo DA vac wB97M TS ---
    system_ft_name = "CPD_MVK_DA_train"
    save_name = 'CPD_MVK_DA'
    react_coords = [(1, 12), (6, 11)]
    num_solute_atoms = 22
    input_config_fp = f"{CPD_MVK_DA_DATA_DIR}/TS/CPD_MVK_DA_TS_MMFF.xyz"  # MMFF94 constrained opt
    ref_config_fp = f"{CPD_MVK_DA_DATA_DIR}/TS/endo_DA_vac_TS_wB97M.xyz"  # DFT opt TS

    # water (uncomment this bit to add the solvent)
    solvent_xyz_fpath, solvent_name, solvent_density = f'{SOLVENTS_DIR}/h2o.xyz', 'water', 0.997
    atoms_per_solv_mol = 3
    solvation_box_size = 18.5

    # --- anthracene vac wB97M TS ---
    # input_config_fp = f"{ANTHRACENE_DA_DATA_DIR}/TS/anthracene_DA_TS_MMFF_opt.xyz"
    # ref_config_fp = f"{ANTHRACENE_DA_DATA_DIR}/TS/anthracene_DA_vac_TS_wB97M.xyz"
    # save_name = 'anthracene_DA'
    # react_coords = [(2, 20), (5, 22)]
    # num_solute_atoms = 41

    # anthracene vac 1,4 MMFF94 TS
    # input_config_fp = f"{ANTHRACENE_DA_DATA_DIR}/TS/anthracene_DA_MMFF_1_4_TS_opt.xyz"
    # ref_config_fp = f"{ANTHRACENE_DA_DATA_DIR}/TS/anthracene_1_4_DA_vac_TS_wB97M.xyz"
    # save_name = 'anthracene_1_4_DA'
    # react_coords = [(16, 20), (15, 22)]
    # num_solute_atoms = 41

    # # toluene
    # solvent_xyz_fpath, solvent_name, solvent_density = f'{SOLVENTS_DIR}/toluene.xyz', 'toluene', 0.8603
    # atoms_per_solv_mol = 15
    # solvation_box_size = 21.0

    # --- guajadial vac wB97M TS ---
    # input_config_fp = f"{GUAJADIAL_DA_DATA_DIR}/TS/guajadial_DA_vac_TS_MMFF.xyz"
    # ref_config_fp = f"{GUAJADIAL_DA_DATA_DIR}/TS/guajadial_DA_vac_TS_wB97M.xyz"
    # save_name = 'guajadial_DA'
    # react_coords = [(13, 31), (6, 32)]
    # num_solute_atoms = 69
    # reverse_displace = True 

    # # water
    # solvent_xyz_fpath, solvent_name, solvent_density = f'{SOLVENTS_DIR}/h2o.xyz', 'water', 0.997
    # atoms_per_solv_mol = 3
    # solvation_box_size = 22.0

    # === 2. SELECT MODEL ===
    
    # --- full zero-shot paper models ---
    model_names = [
        # 'GO-MACE-23', 
        # 'MACE-OFF23_medium', 
        # 'MACE-MP-0b3_medium',
        'EGRET_1T', 
        # 'MACE-NEB',
        # 'ani-1xnr',
        # 'MACE-ANI-CC',
        # 'MACE-omol-0-extra-large-1024'
    ]

    # --- selected fine-tuned models ---

    # n, system_ft_name, rep_num, omol_rep_num, omol_energy_str = 42, "CPD_MVK_DA_train", 1, 1, 'energy_'
    # n, system_ft_name, rep_num, omol_rep_num, omol_energy_str = 46, "anthracene_DA_train", 2, 3, 'energy_'
    # # n, system_ft_name, rep_num, omol_rep_num, omol_energy_str = 50, "guajadial_DA_train", 1, 1, ''
    # energy_corrected = True
    # sample_str = ""
    # # sample_str = "ts_gauss_"

    # energy_str = 'energy_' if energy_corrected else ''
    # model_names = [
    #                #    f"GO-MACE-23_{system_ft_name}_{n}_config_strat_{sample_str}fine_tuned_{energy_str}rep_{rep_num}",
    #                #  f"MACE-OFF23_medium_{system_ft_name}_{n}_config_strat_{sample_str}fine_tuned_{energy_str}rep_{rep_num}", 
    #                #    f"MACE-MP-0b3_medium_{system_ft_name}_{n}_config_strat_{sample_str}fine_tuned_{energy_str}rep_{rep_num}",
    #                #    f"MACE-omol-0-extra-large-1024_{system_ft_name}_{n}_config_strat_{sample_str}fine_tuned_rep_{rep_num}",  # FOR guajadial - energy rep is worse!
    #                f"MACE-omol-0-extra-large-1024_{system_ft_name}_{n}_config_strat_{sample_str}fine_tuned_{omol_energy_str}rep_{omol_rep_num}",
    #               ]
    
    # # special anthracene extra config
    # model_names = [
    #             f"GO-MACE-23_{system_ft_name}_{n}_config_strat_w_1_4_vac_sample_fine_tuned_{energy_str}rep_{rep_num}", 
    #             f"MACE-OFF23_medium_{system_ft_name}_{n}_config_strat_w_1_4_vac_sample_fine_tuned_{energy_str}rep_{rep_num}", 
    #             # f"MACE-MP-0b3_medium_{system_ft_name}_{n}_config_strat_w_1_4_vac_sample_fine_tuned_{energy_str}rep_{rep_num}", 
    #             ]

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

    # === 4. SELECT OPTIMISER SETTINGS ===
    fmax = 0.001
    displace_mag = 0.05 # 0.1 # 0.05
    enable_cueq = False
    override_cache = False
    cuda_device = "cuda:0"

    logger.info(f'Testing models: {model_names}')
    # ====================================

    # run dimer method with each model
    for model_name in model_names:
        start_time = time.time()
        fmt_save_name = save_name + '_vac'
        box_dim = [100.0] * 3

        # 1. run mlip ts opt with dimer method in vacuum
        logger.info(f'Running {model_name} dimer method in vacuum...')
        opt_ts_config = mlip_dimer_method(input_config_fp, 
                                          model_name,
                                          react_coords,
                                          fmt_save_name,
                                          ref_config_fp=ref_config_fp,
                                          displace_mag=displace_mag,
                                          reverse_displace=reverse_displace,
                                          fmax=fmax,
                                          enable_cueq=enable_cueq,
                                          box_dim=box_dim,
                                          override_cache=override_cache)
        opt_ts_config_fp = f'{out_dir}/{fmt_save_name}_{model_name}_opt_TS.xyz'
        logger.info(f'Ran {model_name} dimer method in {time.time() - start_time:.4f}s...')

        # confirm MLIP Vibrational frequencies, only in vacuum
        logger.info(f'Calculating Vibrational Freq. on OPT TS with {model_name}...')
        calc = load_model_calculator(model_name, device=cuda_device, enable_cueq=enable_cueq)
        get_vib_freq(opt_ts_config_fp, 
                     calc, 
                     f"{fmt_save_name}_{model_name}", 
                     out_dir=out_dir)

        # 2. if solvent is specified: solvate, equilibrate with MD and then optimise with fixed solvent
        if solvent_xyz_fpath is not None:

            box_dim = [solvation_box_size] * 3

            # load input config with mlptrain + solvent mol
            input_set = mlt.ConfigurationSet()
            input_set.load_xyz(filename=opt_ts_config_fp, charge=0, mult=1)
            input_config = input_set[0]
            input_config.box = Box(box_dim)

            solvent_mol = mlt.Molecule(solvent_xyz_fpath)

            # solvate config with PBC
            logger.info(f'Solvating optimised vacuum TS with {solvent_name}...')
            input_config.solvate(
                box_size=solvation_box_size,
                solvent_molecule=solvent_mol,
                solvent_density=solvent_density,
            )
            solvated_config_ase = input_config.ase_atoms

            # run solvent equilbration with frozen solute
            start_time = time.time()
            eq_time = 10*PS
            md_config = MDSettings(fs=eq_time, temp=300, dt=0.5, interval=100, md_method='ase')
            solute_indices = set(range(num_solute_atoms))
            solvated_config_ase.set_constraint(FixAtoms(mask=[i in solute_indices for i in range(len(solvated_config_ase))]))
            solv_eq_trj = get_trajectory_with_ase(solvated_config_ase, model_name, md_config, f"{save_name}_{model_name}_solv_eq", out_dir=OPT_DIR)
            logger.info(f'Ran {model_name} {eq_time} solvent equilibration in {time.time() - start_time:.4f}s...')

            # save energy plot to confirm equilibration
            solv_eq_trj_energies = [get_ase_atoms_energy(config) for config in solv_eq_trj]
            solv_eq_energy_plot_fp = f"{OPT_DIR}/{save_name}_{model_name}_solv_eq_energies.png"
            index_plot([solv_eq_trj_energies], save_fp=solv_eq_energy_plot_fp)
            logger.info(f"Saved index plot of solvent equilibration energies to {solv_eq_energy_plot_fp}")

            # save final equilibrated config (and make sure to remove fixed atoms constraint)
            solv_eq_config = solv_eq_trj[-1]
            solv_eq_config.set_constraint(None)
            solv_eq_config_fp = f"{OPT_DIR}/{save_name}_{model_name}_solv_eq_config.xyz"
            ase_write(solv_eq_config_fp, solv_eq_config, format='extxyz')

            # run mlip ts opt with dimer method on solute with fixed equilibrated solvent
            start_time = time.time()
            logger.info(f'Running {model_name} dimer method in {solvent_name}...')
            fmt_save_name = save_name + f"_in_{solvent_name}"
            solv_opt_ts_config = mlip_dimer_method(solv_eq_config_fp, 
                                                   model_name,
                                                   react_coords,
                                                   fmt_save_name,
                                                   ref_config_fp=ref_config_fp,
                                                   num_solute_atoms=num_solute_atoms,
                                                   atoms_per_solv_mol=atoms_per_solv_mol,
                                                   solv_shell_cutoff=0.0,
                                                   displace_mag=displace_mag,
                                                   reverse_displace=reverse_displace,
                                                   fmax=fmax,
                                                   enable_cueq=enable_cueq,
                                                   box_dim=box_dim,
                                                   override_cache=True)
            logger.info(f'Ran {model_name} solvated dimer method in {time.time() - start_time:.4f}s...')

            