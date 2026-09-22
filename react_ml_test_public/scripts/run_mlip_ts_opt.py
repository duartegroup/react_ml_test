"""
Script to use Dimer method to find optimal TS using MLIP.
"""
import argparse
import glob
import os
import time

import mlptrain as mlt
import pytest
from ase.constraints import FixAtoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from mlptrain.box import Box

from react_ml_test_public.helpers import inputs as rmlt_inputs
from react_ml_test_public.helpers.dynamics import get_trajectory_with_ase
from react_ml_test_public.helpers.io import (
    get_ase_atoms_energy,
    load_model_calculator,
    resolve_cuda_device,
)
from react_ml_test_public.helpers.logging import logger
from react_ml_test_public.helpers.plot import index_plot
from react_ml_test_public.helpers.settings import (
    ANTHRACENE_DA_DATA_DIR,
    CPD_MVK_DA_DATA_DIR,
    GUAJADIAL_DA_DATA_DIR,
    OPT_DIR,
    PAPER_MODELS,
    PS,
    SOLVENTS_DIR,
    MDSettings,
)
from react_ml_test_public.helpers.ts_opt import get_vib_freq, mlip_dimer_method

if __name__ == "__main__":

    """Run Dimer-method TS optimisations with MLIPs, optionally in solvent.

    The path to the input YAML file is given as a single command-line argument:

        rmlt run-mlip-ts-opt inputs/ts/CPD_MVK_DA_ts_opt_vac.yaml

    (or equivalently: python -m react_ml_test_public.scripts.run_mlip_ts_opt
    inputs/ts/CPD_MVK_DA_ts_opt_vac.yaml)

    Input YAML format (see inputs/ts/CPD_MVK_DA_ts_opt_vac.yaml for a full example):

    Fields (path fields may reference environment variables, e.g. model_dir: "$DATA/models"):
        save_name:              # base name used in all output files.
        react_coords:           # reactive atom pairs [(i, j), ...] defining the reaction coordinate.
        num_solute_atoms:       # number of solute atoms (leading atoms of each config).
        input_config_fp:        # path to the MLIP TS-guess config.
        ref_config_fp:          # path to the DFT reference TS (optional).
        reverse_displace:       # displace the TS guess along -reaction coordinate.

        model_names:            # list of model names to run (supports ${var} templating).
        system_ft_name:         # fine-tuning training-set name (for model-name templating).
        n:                      # fine-tuning set size.
        sample_str:             # sampling-strategy suffix in fine-tuned names (e.g. "ts_gauss_").
        energy_str:             # energy label string in fine-tuned names (e.g. "energy_").
        rep_num:                # fine-tuned model replicate number.
        omol_rep_num:           # OMOL model replicate number.
        omol_energy_str:        # OMOL energy suffix/prefix string.

        include_solvent:        # whether to run the solvated (equilibrated) TS optimisation.
        solvent_xyz_fpath:      # solvent molecule .xyz (only used when include_solvent: true).
        solvent_name:           # solvent name used in output names (only when include_solvent).
        solvent_density:        # solvent density / g cm^-3 (only when include_solvent).
        atoms_per_solv_mol:     # atoms per solvent molecule (only when include_solvent).
        solvation_box_size:     # cubic solvation box side length / A (only when include_solvent).
        solv_shell_cutoff:      # solvent-shell cutoff for the frozen-solvent TS optimisation.

        out_dir:                # output directory (defaults to OPT_DIR in settings).
        model_dir:              # directory the model_names are loaded from, when they live
                                # outside the default MODEL_DIR (defaults to MODEL_DIR).
        fmax:                   # optimisation convergence threshold / eV A^-1.
        displace_mag:           # initial displacement along the reaction coordinate / A.
        enable_cueq:            # use the cuEquivariance MACE speed-up (CUDA 12+).
        override_cache:         # re-run even if an optimised TS already exists.
        cuda_device:            # GPU device for the MLIP calculators (default "cuda:0",
                                # falls back to "cpu" if that device is not found).

    Any omitted field is set to None (or its documented default).
    """

    # input YAML config path from a single command-line argument
    parser = argparse.ArgumentParser(
        description="Run Dimer-method TS optimisations with MLIPs driven by an input "
                    "YAML config file (format described above).",
        epilog="Example: rmlt run-mlip-ts-opt inputs/ts/CPD_MVK_DA_ts_opt_vac.yaml",
    )
    parser.add_argument(
        "input_yaml",
        help="Path to the input YAML config file "
             "(e.g. inputs/ts/CPD_MVK_DA_ts_opt_vac.yaml).",
    )
    args = parser.parse_args()

    # load + process the input YAML (see react_ml_test_public/helpers/inputs.py)
    resolved = rmlt_inputs.resolve_mlip_ts_opt_yaml(args.input_yaml, parser)

    # === 1. TS GUESS (INPUT CONFIG) ===
    save_name = resolved["save_name"]
    react_coords = resolved["react_coords"]
    num_solute_atoms = resolved["num_solute_atoms"]
    input_config_fp = resolved["input_config_fp"]
    ref_config_fp = resolved["ref_config_fp"]
    reverse_displace = resolved["reverse_displace"]

    # === 2. MODEL SELECTION ===
    model_dir = resolved["model_dir"]
    model_names = resolved["model_names"]
    system_ft_name = resolved["system_ft_name"]
    n = resolved["n"]
    sample_str = resolved["sample_str"]
    energy_str = resolved["energy_str"]
    rep_num = resolved["rep_num"]
    omol_rep_num = resolved["omol_rep_num"]
    omol_energy_str = resolved["omol_energy_str"]

    # === 3. SOLVATION (optional) ===
    include_solvent = resolved["include_solvent"]
    solvent_xyz_fpath = resolved["solvent_xyz_fpath"]
    solvent_name = resolved["solvent_name"]
    solvent_density = resolved["solvent_density"]
    atoms_per_solv_mol = resolved["atoms_per_solv_mol"]
    solvation_box_size = resolved["solvation_box_size"]
    solv_shell_cutoff = resolved["solv_shell_cutoff"]

    # === 4. OPTIMISER / RUN SETTINGS ===
    out_dir = resolved["out_dir"]
    fmax = resolved["fmax"]
    displace_mag = resolved["displace_mag"]
    enable_cueq = resolved["enable_cueq"]
    override_cache = resolved["override_cache"]
    cuda_device = resolved["cuda_device"]
    del resolved

    # resolve the requested device (i.e. fall back to 'cpu' if the GPU is not available)
    cuda_device = resolve_cuda_device(cuda_device)

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
                                          cuda_device=cuda_device,
                                          enable_cueq=enable_cueq,
                                          box_dim=box_dim,
                                          override_cache=override_cache,
                                          model_dir=model_dir)
        opt_ts_config_fp = f'{out_dir}/{fmt_save_name}_{model_name}_opt_TS.xyz'
        logger.info(f'Ran {model_name} dimer method in {time.time() - start_time:.4f}s...')

        # confirm MLIP Vibrational frequencies, only in vacuum
        logger.info(f'Calculating Vibrational Freq. on OPT TS with {model_name}...')
        calc = load_model_calculator(model_name, device=cuda_device, enable_cueq=enable_cueq,
                                     model_dir=model_dir)
        get_vib_freq(opt_ts_config_fp,
                     calc,
                     f"{fmt_save_name}_{model_name}",
                     out_dir=out_dir)

        # 2. if solvation is enabled: solvate, equilibrate with MD and then optimise with fixed solvent
        if include_solvent:

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
            solv_eq_trj = get_trajectory_with_ase(solvated_config_ase, model_name, md_config, f"{save_name}_{model_name}_solv_eq", out_dir=out_dir, cuda_device=cuda_device, model_dir=model_dir)
            logger.info(f'Ran {model_name} {eq_time} solvent equilibration in {time.time() - start_time:.4f}s...')

            # save energy plot to confirm equilibration
            solv_eq_trj_energies = [get_ase_atoms_energy(config) for config in solv_eq_trj]
            solv_eq_energy_plot_fp = f"{out_dir}/{save_name}_{model_name}_solv_eq_energies.png"
            index_plot([solv_eq_trj_energies], save_fp=solv_eq_energy_plot_fp)
            logger.info(f"Saved index plot of solvent equilibration energies to {solv_eq_energy_plot_fp}")

            # save final equilibrated config (and make sure to remove fixed atoms constraint)
            solv_eq_config = solv_eq_trj[-1]
            solv_eq_config.set_constraint(None)
            solv_eq_config_fp = f"{out_dir}/{save_name}_{model_name}_solv_eq_config.xyz"
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
                                                   solv_shell_cutoff=solv_shell_cutoff,
                                                   displace_mag=displace_mag,
                                                   reverse_displace=reverse_displace,
                                                   fmax=fmax,
                                                   cuda_device=cuda_device,
                                                   enable_cueq=enable_cueq,
                                                   box_dim=box_dim,
                                                   override_cache=True,
                                                   model_dir=model_dir)
            logger.info(f'Ran {model_name} solvated dimer method in {time.time() - start_time:.4f}s...')

    logger.info('Finished running all models...')
