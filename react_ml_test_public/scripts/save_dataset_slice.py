"""
Script to save a sample of an xyz dataset very efficiently, without opening the entire dataset.
"""
import argparse
import time

from react_ml_test_public.helpers import inputs as rmlt_inputs
from react_ml_test_public.helpers.io import save_dataset_slice
from react_ml_test_public.helpers.logging import logger

if __name__ == "__main__":

    """Save a sliced sample of an .xyz dataset, without opening the entire dataset.

    The path to the input YAML file is given as a single command-line argument:

        rmlt save-dataset-slice inputs/datasets/save_dataset_slice_go_mace_eg.yaml

    (or equivalently: python -m react_ml_test_public.scripts.save_dataset_slice
    inputs/datasets/save_dataset_slice_go_mace_eg.yaml)

    Input YAML format (see inputs/datasets/save_dataset_slice_go_mace_eg.yaml for an example):

    Fields:
        dataset_fp:           # path to the .xyz dataset to slice, taken to be relative to
                              # data_dir when it does not resolve from the cwd.
        index:                # slice to save, as for any list indexing, e.g.:
                              #   0:      first configuration
                              #   -2:     second to last
                              #   ':':    all
                              #   '-3:':  three last
                              #   '::2':  even
                              #   '1::2': odd
        data_dir:             # data root directory for a relative dataset_fp (default
                              # DATA_DIR from settings.py).

    dataset_fp and index are required. The slice is written to
    <dataset_fp without extension>_sliced.xyz.
    """

    # input YAML config path from a single command-line argument
    parser = argparse.ArgumentParser(
        description="Save a sliced sample of an .xyz dataset, driven by an input YAML config "
                    "file (format described above).",
        epilog="Example: rmlt save-dataset-slice "
               "inputs/datasets/save_dataset_slice_go_mace_eg.yaml",
    )
    parser.add_argument(
        "input_yaml",
        help="Path to the input YAML config file "
             "(e.g. inputs/datasets/save_dataset_slice_go_mace_eg.yaml).",
    )
    args = parser.parse_args()

    # load + process the input YAML (see react_ml_test_public/helpers/inputs.py)
    resolved = rmlt_inputs.resolve_save_dataset_slice_yaml(args.input_yaml, parser)

    # === SPECIFY DATASET AND SLICE ===
    data_dir = resolved["data_dir"]
    dataset_fp = resolved["dataset_fp"]
    index = resolved["index"]
    del resolved

    # get the slice of the dataset, without opening it all
    start_time = time.time()
    logger.info(f"Using data dir: {data_dir}...")
    logger.info(f"Saving dataset slice for {dataset_fp}...")
    save_dataset_slice(dataset_fp, index)
    logger.info(f"Saving slice took {time.time() - start_time: .4f}...")
