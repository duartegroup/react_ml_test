"""
Script to update metadata with ase for an .xyz file. 
WARNING THIS WILL OVERRIDE FILES IN THE SPECIFIED DIRECTORY!!!
"""
import os
import sys
import glob
sys.path.append('../')
from ase.io import read, write
from react_ml_test.helpers.settings import OPT_DIR, CPD_MVK_DA_DATA_DIR
from react_ml_test.helpers.logging import logger
from react_ml_test.helpers.io import flexible_ase_read

box_dim = None
constraint = None
charge, spin = None, None

# === PUT INPUT DIR ETC IN HERE ===

# --- removal of trajectory extra metadata ---
# directory = OPT_DIR
# out_format = 'extxyz'
# remove_momenta = True
# remove_constraints = True

# # CPD water
# pattern = f"{directory}/CPD_MVK_DA_in_water*opt_TS.xyz"
# box_dim = [18.5, 18.5 , 18.5]

# anthracene toluene
# pattern = f"{directory}/anthracene_*_in_toluene*opt_TS.xyz"
# box_dim = [21.0, 21.0, 21.0]

# guajadial water
# pattern = f"{directory}/guajadial_in_water*opt_TS.xyz"
# box_dim = [22.0, 22.0 , 22.0]

# --- addition of charge and spin data ---
directory = f"{CPD_MVK_DA_DATA_DIR}/train/subsets"
out_format = 'extxyz'
remove_momenta = False
remove_constraints = False
charge, spin = 0, 1

# CPD water
pattern = f"{directory}/CPD_MVK_DA*.xyz"

# anthracene toluene
# pattern = f"{directory}/anthracene_*_in_toluene*opt_TS.xyz"
# box_dim = [21.0, 21.0, 21.0]

# guajadial water
# pattern = f"{directory}/guajadial_in_water*opt_TS.xyz"
# box_dim = [22.0, 22.0 , 22.0]

# =================================

for filepath in glob.glob(pattern):
    print('Filepath: ', filepath)
    configs = flexible_ase_read(filepath)

    updated_configs = []
    for atoms in configs:
        if remove_momenta:
            if "momenta" in atoms.arrays:
                del atoms.arrays["momenta"]
                logger.info("Removed momenta...")

        if remove_constraints:
            atoms.set_constraint(None)
            logger.info("Removed constraints...")

        if box_dim is not None:
            atoms.set_cell(box_dim)
            atoms.set_pbc([True, True, True])
            logger.info(f"Set PBC box dimensions to {box_dim}")

        if charge is not None and spin is not None:
            for config in atoms:
                config.info["total_charge"] = charge
                config.info["total_spin"] = spin

        updated_configs.append(atoms)

    write(filepath, updated_configs, format="extxyz")
    logger.info(f"Updated: {filepath}")
