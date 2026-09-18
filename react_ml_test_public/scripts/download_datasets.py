"""
Use load_atoms (and other tools) to save datasets as .xyz files.
"""

from ase.io import write
from load_atoms import load_dataset

from react_ml_test_public.helpers.settings import DATASETS_DIR


if __name__ == "__main__":

    # ds = load_dataset('ANI-1x')
    ds = load_dataset('ANI-1xnr')
    structures = list(ds)

    print(ds)

    # save xyz files
    write(f'{DATASETS_DIR}/ani_1xnr.xyz', structures)
