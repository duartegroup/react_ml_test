"""
Use load_atoms (and other tools) to save datasets as .xyz files.
"""
import sys

from ase import Atoms
from ase.data import atomic_numbers as element_to_atomic_num_dict
from ase.io import write

sys.path.append('../')
import h5py
from load_atoms import load_dataset

from react_ml_test.helpers.settings import DATASETS_DIR

Ha_to_eV = 27.2114079527


def print_h5_structure(group, out_file, indent=0):
    """
    Recursively print the structure of a HDF5 file.
    """

    for i, key in enumerate(group.keys()):
        item = group[key]
        print("  " * indent + f"- {key} ({'Group' if isinstance(item, h5py.Group) else 'Dataset'})", file=out_file)
        if isinstance(item, h5py.Group):
            print_h5_structure(item, out_file, indent + 1)


def load_ani1xnr_subset_structures(iter_group: h5py.Group):

    for data in iter_group.values():

        # iterate over each structure in the dataset:
        elements = data["species"][()]
        coords = data["coordinates"][()]
        dft_energy = data["energy"][()]
        dft_forces = data["forces"][()]
        cell = data["cell"][()][0]

        Zs = [element_to_atomic_num_dict[element.decode()] for element in elements]

        for i in range(coords.shape[0]):
            structure = Atoms(positions=coords[i], numbers=Zs)
            
            # energy is in hartree, convert to eV
            structure.set_cell(cell)
            structure.info["energy"] = dft_energy[i] * Ha_to_eV

            # forces are in hartree/angstrom, convert to eV/angstrom
            structure.arrays["forces"] = dft_forces[i] * Ha_to_eV
            yield structure


if __name__ == "__main__":

    # ds = load_dataset('ANI-1x')
    ds = load_dataset('ANI-1xnr')
    structures = list(ds)

    print(ds)

    # Path to the ANI-1x data set
    # path_to_h5file = f'{DATASETS_DIR}/ani1xnr.h5'
    
    # with h5py.File(path_to_h5file, "r") as f:

    #     # handle first iterations data
    #     for iter_name, iter_group in f.items():

    #         if iter_name != "nano-data-it1-6.h5":
    #             structures.extend(list(load_ani1xnr_subset_structures(iter_group)))

    #     # handle nano reactor data
    #     for iter_group in f["nano-data-it1-6.h5"].values():
    #         structures.extend(list(load_ani1xnr_subset_structures(iter_group)))

    # save xyz files
    write(f'{DATASETS_DIR}/ani_1xnr.xyz', structures)
