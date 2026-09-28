"""
Script for checking the ORCA outputs of a set of DFT calculations.
"""

import argparse

from react_ml_test_public.helpers import inputs as rmlt_inputs
from react_ml_test_public.helpers.io import check_orca_outputs
from react_ml_test_public.helpers.logging import logger

if __name__ == "__main__":

    """Report on a set of ORCA outputs, run by run.

    The path to the input YAML file is given as a single command-line argument:

        rmlt check-orca-output inputs/dft/check_orca_output.yaml

    (or equivalently: python -m react_ml_test_public.scripts.check_orca_output
    inputs/dft/check_orca_output.yaml)

    Input YAML format (see inputs/dft/check_orca_output.yaml for an example):

    Fields (path fields may reference environment variables, e.g. "$DATA/out/dft/sp"):
        work_dir:             # directory searched, recursively, for ORCA output files.
        input_name:           # stem those files are named after: every
                              # <work_dir>/**/*<input_name>_*.out is checked.

    Both fields are required. Each output found is read and reported by whether it says ORCA
    terminated normally - the number of runs, which of them succeeded and failed, and the
    number of atoms in each (see helpers.io.check_orca_outputs).
    """

    # input YAML config path from a single command-line argument
    parser = argparse.ArgumentParser(
        description="Check the ORCA outputs under a directory for runs that did not "
                    "terminate normally, driven by an input YAML config file (format "
                    "described above).",
        epilog="Example: rmlt check-orca-output inputs/dft/check_orca_output.yaml",
    )
    parser.add_argument(
        "input_yaml",
        help="Path to the input YAML config file "
             "(e.g. inputs/dft/check_orca_output.yaml).",
    )
    args = parser.parse_args()

    # load + process the input YAML (see react_ml_test_public/helpers/inputs.py)
    resolved = rmlt_inputs.resolve_check_orca_output_yaml(args.input_yaml, parser)

    # === 1. ORCA OUTPUT SET TO CHECK ===
    work_dir = resolved["work_dir"]
    input_name = resolved["input_name"]
    del resolved

    logger.info(f"Checking the {input_name} ORCA outputs under {work_dir}...")
    check_orca_outputs(work_dir, input_name)
