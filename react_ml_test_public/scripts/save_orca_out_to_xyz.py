"""
Script to save all ORCA outputs in a given directory with a specified input name into a single .npz + extended .xyz file.
"""
import argparse
import glob
import os

import ase.io

from mlptrain.utils import orca_output_to_npz

from react_ml_test_public.helpers import inputs as rmlt_inputs
from react_ml_test_public.helpers.logging import logger

if __name__ == "__main__":

    """Save a set of ORCA outputs as one .npz + extended .xyz file.

    The path to the input YAML file is given as a single command-line argument:

        rmlt save-orca-out-to-xyz inputs/dft/save_orca_out_to_xyz_guajadial_eg.yaml

    (or equivalently: python -m react_ml_test_public.scripts.save_orca_out_to_xyz
    inputs/dft/save_orca_out_to_xyz_guajadial_eg.yaml)

    Input YAML format (see inputs/dft/save_orca_out_to_xyz_guajadial_eg.yaml for an example):

    Fields (path fields may reference environment variables, e.g. "$DATA/out/dft/sp"):
        work_dir:             # directory searched, recursively, for ORCA output files.
        input_name:           # stem those files are named after: every
                              # <work_dir>/**/*<input_name>_*.out is saved.

    Both fields are required. The outputs are written as a single .npz and extended .xyz by
    mlptrain's orca_output_to_npz, which is what reads them.
    """

    # input YAML config path from a single command-line argument
    parser = argparse.ArgumentParser(
        description="Save the ORCA outputs under a directory, by input name, as one .npz + "
                    "extended .xyz file, driven by an input YAML config file (format "
                    "described above).",
        epilog="Example: rmlt save-orca-out-to-xyz inputs/dft/save_orca_out_to_xyz_guajadial_eg.yaml",
    )
    parser.add_argument(
        "input_yaml",
        help="Path to the input YAML config file "
             "(e.g. inputs/dft/save_orca_out_to_xyz_guajadial_eg.yaml).",
    )
    args = parser.parse_args()

    # load + process the input YAML (see react_ml_test_public/helpers/inputs.py)
    resolved = rmlt_inputs.resolve_save_orca_out_to_xyz_yaml(args.input_yaml, parser)

    # === 1. ORCA OUTPUT SET TO SAVE ===
    work_dir = resolved["work_dir"]
    input_name = resolved["input_name"]
    del resolved

    # get out_fpaths
    out_fpaths = glob.glob(os.path.join(work_dir, f"**/*{input_name}_*.out"), recursive=True)

    logger.info(f"Saving the {input_name} ORCA outputs under {work_dir}...")
    orca_output_to_npz(out_fpaths, input_name, work_dir)

    # rewrite the .xyz as one extended xyz with a 100 A cell and REF_ energies/forces
    xyz_fpath = f"{work_dir}/{input_name}.xyz"

    if os.path.exists(xyz_fpath):
        frames = ase.io.read(xyz_fpath, ":")

        for atoms in frames:
            energy, forces = atoms.get_potential_energy(), atoms.get_forces()
            atoms.calc = None
            atoms.set_cell([100, 100, 100])
            atoms.pbc = True
            atoms.info["REF_energy"] = energy
            atoms.arrays["REF_forces"] = forces

        ase.io.write(xyz_fpath, frames, format="extxyz")

    os.remove(f"{work_dir}/{input_name}.npz")
