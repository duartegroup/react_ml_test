"""
Script to save a sample of an xyz dataset very efficiently, without opening the entire dataset.
"""
import argparse
import time

from react_ml_test_public.helpers import inputs as rmlt_inputs
from react_ml_test_public.helpers.io import save_dataset_batches
from react_ml_test_public.helpers.logging import logger

if __name__ == "__main__":

    """Split an .xyz dataset into batches, without opening the entire dataset.

    The path to the input YAML file is given as a single command-line argument:

        rmlt save-dataset-batches inputs/datasets/save_dataset_batches_go_mace_eg.yaml

    (or equivalently: python -m react_ml_test_public.scripts.save_dataset_batches
    inputs/datasets/save_dataset_batches_go_mace_eg.yaml)

    Input YAML format (see inputs/datasets/save_dataset_batches_go_mace_eg.yaml for an example):

    Fields:
        dataset_fp:           # path to the .xyz dataset to split into batches, taken to be
                              # relative to data_dir when it does not resolve from the cwd.
        batch_size:           # number of configurations per output batch.
        data_dir:             # data root directory for a relative dataset_fp (default
                              # DATA_DIR from settings.py).

    dataset_fp and batch_size are required. Batches are written as
    <dataset_fp without extension>_batch_<i>.xyz.
    """

    # input YAML config path from a single command-line argument
    parser = argparse.ArgumentParser(
        description="Split an .xyz dataset into batches, driven by an input YAML config "
                    "file (format described above).",
        epilog="Example: rmlt save-dataset-batches "
               "inputs/datasets/save_dataset_batches_go_mace_eg.yaml",
    )
    parser.add_argument(
        "input_yaml",
        help="Path to the input YAML config file "
             "(e.g. inputs/datasets/save_dataset_batches_go_mace_eg.yaml).",
    )
    args = parser.parse_args()

    # load + process the input YAML (see react_ml_test_public/helpers/inputs.py)
    resolved = rmlt_inputs.resolve_save_dataset_batches_yaml(args.input_yaml, parser)

    # === SPECIFY DATASET AND BATCH SIZE ===
    data_dir = resolved["data_dir"]
    dataset_fp = resolved["dataset_fp"]
    batch_size = resolved["batch_size"]
    del resolved

    # get batches of the dataset, without opening it all
    start_time = time.time()
    logger.info(f"Using data dir: {data_dir}...")
    logger.info(f"Saving dataset batches for {dataset_fp}...")
    save_dataset_batches(dataset_fp, batch_size)
    logger.info(f"Saving batches took {time.time() - start_time: .4f}...")
