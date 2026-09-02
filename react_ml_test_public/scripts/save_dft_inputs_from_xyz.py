"""
"""
import os

import numpy as np
from ase.io import read, write
from autode import Calculation
from autode.calculations import Calculation
from autode.config import Config as autodeConfig
from autode.species import Species
from mlptrain.config import Config
from mlptrain.configurations import ConfigurationSet
from mlptrain.configurations.calculate import _method_and_keywords
from mlptrain.log import logger

from react_ml_test_public.helpers.settings import (
    ANTHRACENE_DA_DATA_DIR,
    CPD_MVK_DA_DATA_DIR,
    GUAJADIAL_DA_DATA_DIR,
)
from react_ml_test_public.helpers.utils import get_name_from_path

# from mlptrain.configurations.configuration_set import 

METHOD_NAME = "orca"


def save_orca_inputs_from_xyz(input_fp: str,
                              charge: int, 
                              mult: int,
                              work_dir: str,
                              extra_name_tag: str = None):

    input_name = get_name_from_path(input_fp)
    input_name = input_name if extra_name_tag is None else input_name + f'_{extra_name_tag}'
    out_dir = f"{work_dir}/{input_name}"

    # create output dir if doesn't exist already
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        logger.info(f"Created out dir: {out_dir}")

    # open config set here
    method, kwds = _method_and_keywords(method_name=METHOD_NAME.lower())

    # loop through configs and generate input for each
    config_set = ConfigurationSet(allow_duplicates=True)
    config_set.load_xyz(input_fp, charge, mult)

    for i, config in enumerate(config_set):
        calc = Calculation(
            name=f'{out_dir}/{input_name}_{i}',
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


def save_vasp_inputs_from_xyz(input_fp: str, 
                              box_size: float, 
                              incar_settings: dict,
                              work_dir: str):
    """
    Generate INCAR and POSCAR from an extxyz file.

    Parameters:
        extxyz_file (str): Path to input .extxyz file
        box_size (float): Cubic box size in Å
        incar_settings (dict): Dictionary of INCAR parameters
    """

    input_name = get_name_from_path(input_fp)
    out_dir = f"{work_dir}/{input_name}_VASP"

    # create output dir if doesn't exist already
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        logger.info(f"Created out dir: {out_dir}")

    # Read structure
    configs = read(input_fp, ':')
    cell = np.eye(3) * box_size

    for i, config in enumerate(configs):
        
        # Set cubic cell and center atoms
        config.set_cell(cell)
        config.center()

        # create config specific output dir if doesn't exist already
        config_out_dir = f"{out_dir}/{input_name}_{i}"
        if not os.path.exists(config_out_dir):
            os.mkdir(config_out_dir)
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

    box_size = None
    incar_settings = None

    # === SPECIFY INPUT PARAMS ===

    # --- CPD + MVK DA test ---
    # work_dir = "scripts/dft/sp"
    # input_fp = f"{CPD_MVK_DA_DATA_DIR}/test/CPD_MVK_DA_test.xyz"
    # charge, mult = 0, 1
    # method = 'orca'

    # # PBE def2-SVP
    # # Config.n_cores = 8
    # # Config.orca_keywords = ["PBE", "def2-TZVP", "TightSCF", "EnGrad"]
    # # extra_name_tag = 'PBE_def2_TZVP'

    # # MACE-OFF LoT
    # Config.n_cores = 16
    # Config.orca_keywords = ["WB97M-D3BJ", "def2-TZVPPD", "TightSCF", "EnGrad"]
    # extra_name_tag = 'wB97M_TZVPPD'

    # --- anthracene full al trajectory samples ---
    # work_dir = "scripts/dft/sp"
    # # input_fp = "data/reaction_data/anthracene_DA/full_al_traj_samples_dist_filtered.xyz"
    # # input_fp = f"{ANTHRACENE_DA_DATA_DIR}/react_traj/anthracene_1_4_MMFF_vac_MACE-OFF23_46_reaction.xyz"
    # # input_fp = f"{ANTHRACENE_DA_DATA_DIR}/TS/anthracene_1_4_DA_vac_OFF46_opt_TS.xyz"
    # input_fp = f"{ANTHRACENE_DA_DATA_DIR}/TS/anthracene_1_4_DA_in_toluene_OFF46_opt_TS_cutoff=5_2.xyz"
    # charge, mult = 0, 1
    # Config.n_cores = 8
    # Config.orca_keywords = ["WB97M-D3BJ", "def2-TZVP", "TightSCF", "EnGrad"]
    # method = 'orca'

    # --- anthracene test ---
    # work_dir = "scripts/dft/sp"
    # input_fp = f"{ANTHRACENE_DA_DATA_DIR}/test/anthracene_DA_test.xyz"
    # charge, mult = 0, 1
    # method = 'orca'

    # # # # PBE def2-SVP
    # # # # Config.n_cores = 8
    # # # # Config.orca_keywords = ["PBE", "def2-TZVP", "TightSCF", "EnGrad"]
    # # # # extra_name_tag = 'PBE_def2_TZVP'

    # # MACE-OFF LoT
    # Config.n_cores = 16
    # Config.orca_keywords = ["WB97M-D3BJ", "def2-TZVPP", "TightSCF", "EnGrad"]
    # autodeConfig.max_core = 6000
    # extra_name_tag = 'wB97M_TZVPP'

    # --- guajadial full al trajectory samples ---
    # work_dir = "scripts/dft/sp"
    # # input_fp = f"{GUAJADIAL_DA_DATA_DIR}/guajadial_extra_traj_samples_dist_filtered_vac.xyz"
    # # input_fp = f"{GUAJADIAL_DA_DATA_DIR}/guajadial_extra_traj_samples_dist_filtered_solv.xyz"
    # charge, mult = 0, 1
    # Config.n_cores = 16
    # Config.orca_keywords = ["WB97M-D3BJ", "def2-TZVP", "TightSCF", "EnGrad"]
    # method = 'orca'

    # --- guajadial test ---
    work_dir = "scripts/dft/sp"
    # input_fp = f"{GUAJADIAL_DA_DATA_DIR}/test/guajadial_DA_test.xyz"
    input_fp = f"{GUAJADIAL_DA_DATA_DIR}/test/guajadial_DA_test_vac_only.xyz"
    charge, mult = 0, 1
    method = 'orca'

    # # PBE def2-SVP
    # # Config.n_cores = 8
    # # Config.orca_keywords = ["PBE", "def2-TZVP", "TightSCF", "EnGrad"]
    # # extra_name_tag = 'PBE_def2_TZVP'

    # MACE-OFF LoT
    Config.n_cores = 16
    Config.orca_keywords = ["WB97M-D3BJ", "def2-TZVPPD", "TightSCF", "EnGrad"]
    autodeConfig.max_core = 4000 #6000
    # extra_name_tag = 'wB97M_TZVPP'
    extra_name_tag = 'wB97M_TZVPPD'

    # --- CPD + MVK Test Set VASP for plane-wave DFT ---
    # work_dir = "scripts/dft/sp"
    # input_fp = f"{CPD_MVK_DA_DATA_DIR}/test/CPD_MVK_DA_test.xyz"
    # method = 'vasp'
    # incar_settings = {
    #     "ENCUT": 550,
    #     "EDIFF": 1E-5,
    #     "SIGMA": 0.1,
    #     "ISPIN": 1
    # }
    # box_size = 50.0  # affects plane-wave performance (make as small as possible)

    # ============================

    if method == 'orca':
        save_orca_inputs_from_xyz(input_fp, charge, mult, work_dir, extra_name_tag=extra_name_tag)
    elif method == 'vasp':
        save_vasp_inputs_from_xyz(input_fp, box_size, incar_settings, work_dir)
