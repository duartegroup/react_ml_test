"""
Generate DFT (ORCA / VASP) single-point input files from an .xyz configuration file.
"""
import argparse
import os

import numpy as np
from ase.io import read, write
from autode.calculations import Calculation
from autode.config import Config as autodeConfig
from autode.species import Species
from mlptrain.config import Config
from mlptrain.configurations import ConfigurationSet
from mlptrain.configurations.calculate import _method_and_keywords

from react_ml_test_public.helpers import inputs as rmlt_inputs
from react_ml_test_public.helpers.io import (
    format_orca_basis_block,
    get_name_from_path,
    insert_orca_input_block,
)
from react_ml_test_public.helpers.logging import logger

# from mlptrain.configurations.configuration_set import

METHOD_NAME = "orca"


def save_orca_inputs_from_xyz(input_fp: str,
                              charge: int,
                              mult: int,
                              out_dir: str,
                              extra_name_tag: str = None,
                              basis: list = None):
    """
    Generate an ORCA input file for each config in an extxyz file.

    Parameters:
        input_fp (str): Path to input .xyz file
        charge (int): Charge of every config in the file
        mult (int): Multiplicity of every config in the file
        out_dir (str): Directory the per-config input directories are created under
        extra_name_tag (str): Optional tag appended to the input name
        basis (list): Optional list of (key, value) pairs written into each input as an
                      ORCA %basis block, e.g. [("AuxJ", "aug-cc-pVTZ/JK")]
    """

    input_name = get_name_from_path(input_fp)
    input_name = input_name if extra_name_tag is None else input_name + f'_{extra_name_tag}'
    config_out_dir = f"{out_dir}/{input_name}"

    # create output dir if doesn't exist already
    if not os.path.exists(config_out_dir):
        os.makedirs(config_out_dir, exist_ok=True)
        logger.info(f"Created out dir: {config_out_dir}")

    # open config set here
    method, kwds = _method_and_keywords(method_name=METHOD_NAME.lower())

    # loop through configs and generate input for each
    config_set = ConfigurationSet(allow_duplicates=True)
    config_set.load_xyz(input_fp, charge, mult)

    basis_block = format_orca_basis_block(basis) if basis is not None else None

    for i, config in enumerate(config_set):
        calc = Calculation(
            name=f'{config_out_dir}/{input_name}_{i}',
            molecule=Species(
                name=f'{input_name}_{i}',
                atoms=config.atoms,
                charge=config.charge,
                mult=config.mult,
            ),
            method=method,
            keywords=Config.orca_keywords,
            n_cores=Config.n_cores,
        )

        # create input file
        calc._executor.generate_input()

        # autode writes no %basis block (its ORCA keywords are simple input only), so the
        # basis the YAML sets is inserted into the file autode has just generated
        if basis_block is not None:
            insert_orca_input_block(calc._executor.input.filename, basis_block)


def save_vasp_inputs_from_xyz(input_fp: str,
                              box_size: float,
                              incar_settings: dict,
                              out_dir: str):
    """
    Generate INCAR and POSCAR for each config in an extxyz file.

    Parameters:
        input_fp (str): Path to input .xyz file
        box_size (float): Cubic box size in Å
        incar_settings (dict): Dictionary of INCAR parameters
        out_dir (str): Directory the per-config input directories are created under
    """

    input_name = get_name_from_path(input_fp)
    configs_out_dir = f"{out_dir}/{input_name}_VASP"

    # create output dir if doesn't exist already
    if not os.path.exists(configs_out_dir):
        os.makedirs(configs_out_dir, exist_ok=True)
        logger.info(f"Created out dir: {configs_out_dir}")

    # Read structure
    configs = read(input_fp, ':')
    cell = np.eye(3) * box_size

    for i, config in enumerate(configs):

        # Set cubic cell and center atoms
        config.set_cell(cell)
        config.center()

        # create config specific output dir if doesn't exist already
        config_out_dir = f"{configs_out_dir}/{input_name}_{i}"
        if not os.path.exists(config_out_dir):
            os.makedirs(config_out_dir, exist_ok=True)
            logger.info(f"Created out dir: {config_out_dir}")

        # --- Write POSCAR ---
        write(f"{config_out_dir}/POSCAR", config, format="vasp")

        # --- Write INCAR ---
        with open(f"{config_out_dir}/INCAR", "w") as f:

            default_settings = {"SYSTEM": f"{input_name}_{i}",
                                "PREC": "Accurate",
                                "ISMEAR": 0,
                                "ALGO": "Normal",
                                "NELM": 100
                                }

            for key, value in default_settings.items():
                f.write(f"{key} = {value}\n")

            for key, value in incar_settings.items():
                f.write(f"{key} = {value}\n")


if __name__ == "__main__":

    """Generate DFT input files from an .xyz configuration file.

    The path to the input YAML file is given as a single command-line argument:

        rmlt save-dft-inputs-from-xyz inputs/dft/guajadial_DA_test.yaml

    (or equivalently: python -m react_ml_test_public.scripts.save_dft_inputs_from_xyz
    inputs/dft/guajadial_DA_test.yaml)

    Input YAML format (see inputs/dft/guajadial_DA_test.yaml for a full example):

    Fields (path fields may reference environment variables, e.g. "$DATA/reaction_data"):
        input_fpath:          # .xyz file holding the configs to generate inputs for. A
                              # relative path is read from the cwd when it resolves there,
                              # else from data_dir.
        data_dir:             # data root that a relative input_fpath resolves against
                              # (defaults to DATA_DIR in settings).
        method:               # level of theory, "orca" / "vasp".

        charge:               # charge of every config (method: orca).
        mult:                 # multiplicity of every config (method: orca).

        out_dir:              # directory the per-config input directories are created
                              # under (defaults to DFT_DIR, i.e. out/dft).
        extra_name_tag:       # optional tag appended to the input name (method: orca).
        n_cores:              # cores per ORCA calculation, applied to mlptrain's Config
                              # (method: orca; defaults to Config's own value).
        orca_keywords:        # ORCA keywords, applied to mlptrain's Config as a list of
                              # strings, e.g. ["WB97M-D3BJ", "def2-TZVPPD"] (method: orca).
        basis:                # ORCA block input, written into each .inp as a %basis block
                              # (method: orca; optional). This is for the basis settings the
                              # ! keyword line cannot express, i.e. the auxiliary basis
                              # slots, e.g. basis: AuxJ "aug-cc-pVTZ/JK" - a block key and
                              # its value, a mapping of keys to values, or a list of either
                              # (a bare basis set name sets the orbital "Basis").
        max_core:             # autode's maximum number of cores, applied to its Config
                              # (method: orca; optional).

        box_size:             # cubic box size in Å, for the VASP POSCAR (method: vasp).
        incar_settings:       # mapping of INCAR parameters written into each INCAR, on top
                              # of the built-in defaults (method: vasp; optional).

    ORCA inputs are written to <out_dir>/<input name>[_<extra_name_tag>]/ and VASP inputs
    to <out_dir>/<input name>_VASP/<input name>_<i>/, one subdirectory per config.
    """

    # input YAML config path from a single command-line argument
    parser = argparse.ArgumentParser(
        description="Generate DFT (ORCA / VASP) input files from an .xyz configuration "
                    "file, driven by an input YAML config file (format described above).",
        epilog="Example: rmlt save-dft-inputs-from-xyz inputs/dft/guajadial_DA_test.yaml",
    )
    parser.add_argument(
        "input_yaml",
        help="Path to the input YAML config file "
             "(e.g. inputs/dft/guajadial_DA_test.yaml).",
    )
    args = parser.parse_args()

    # load + process the input YAML (see react_ml_test_public/helpers/inputs.py)
    resolved = rmlt_inputs.resolve_save_dft_inputs_yaml(args.input_yaml, parser)

    # === 1. INPUT CONFIG / METHOD ===
    input_fp = resolved["input_fpath"]
    method = resolved["method"]
    out_dir = resolved["out_dir"]

    # === 2. ORCA SETTINGS ===
    charge = resolved["charge"]
    mult = resolved["mult"]
    n_cores = resolved["n_cores"]
    orca_keywords = resolved["orca_keywords"]
    basis = resolved["basis"]
    max_core = resolved["max_core"]
    extra_name_tag = resolved["extra_name_tag"]

    # === 3. VASP SETTINGS ===
    box_size = resolved["box_size"]
    incar_settings = resolved["incar_settings"]
    del resolved

    # the ORCA settings are read from the mlptrain / autode configs when the input files
    # are generated, so apply the ones the YAML sets and leave the rest at their defaults
    if n_cores is not None:
        Config.n_cores = n_cores
    if orca_keywords is not None:
        Config.orca_keywords = orca_keywords
    if max_core is not None:
        autodeConfig.max_core = max_core

    logger.info(f"Generating {method} input files from {input_fp} in {out_dir}...")

    if method == 'orca':
        save_orca_inputs_from_xyz(input_fp, charge, mult, out_dir,
                                  extra_name_tag=extra_name_tag, basis=basis)
    elif method == 'vasp':
        save_vasp_inputs_from_xyz(input_fp, box_size, incar_settings, out_dir)
