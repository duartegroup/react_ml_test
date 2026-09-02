from time import time

import h5py

from react_ml_test_public.helpers.io import load_hdf5_cache


def print_structure(name, obj):
    indent = '  ' * (name.count('/') - 1)
    if isinstance(obj, h5py.Group):
        print(f"{indent}[Group] {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}[Dataset] {name} shape={obj.shape}, dtype={obj.dtype}")


if __name__ == "__main__":

    start_time = time()
    h5_file = load_hdf5_cache()
    h5_file.visititems(print_structure)
    end_time = time()
    run_time = end_time - start_time
    print(f'Loaded hdf5 cache in {run_time:.4f}s...')
