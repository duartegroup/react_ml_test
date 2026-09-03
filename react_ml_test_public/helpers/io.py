"""
IO file helpers.
"""
import os
import re
from typing import Callable, Dict, List, Literal, Union
import ase
from ase.io import read
from typing import TextIO
from ase.io.extxyz import read_extxyz
import numpy as np
import glob
import h5py
import hdf5plugin
from react_ml_test_public.helpers.settings import (
    ANI_1XNR_MODEL_DIR,
    ANI_MODEL_LOADER,
    ANI_MODELS,
    GRAPH_PES_MODEL_LOADER,
    MACE_MODEL_LOADER,
    MACE_MODELS,
    MODEL_DIR,
    OUT_CACHE_FPATH,
)
from react_ml_test_public.helpers.logging import logger


def convert_npz_to_xyz():
    """
    ['R', 'E_true', 'E_predicted', 'F_true', 'F_predicted', 'Z', 'L', 'C', 'M', 'allow_pickle']
    (101, 190, 3)

    """

    data = np.load("../data/datasets/da_paper_data/valid/endo_in_water_ace_wB97M_short_orca.npz")
    print(data.files)
    print(data['R'].shape)


def get_xyz_config_offsets(xyz_fpath: str, 
                           return_elements: bool = False) -> Union[list[int], tuple[list[int], list[list[str]]]]:
    """
    Get offsets of where all configurations begin and end in a .xyz file without explcitily loading Atoms objects through ase.
    This is useful for sampling individual configs from very large datasets.
    """

    offsets, elements = [], []
    with open(xyz_fpath, "r") as f:
        while True:

            # start of configuration
            start_offset = f.tell()
            first_line = f.readline()

            # EOF check
            if not first_line:
                break

            # number of atoms
            num_atoms = int(first_line.strip())
            _ = f.readline()  # Skip comment

            # skip atom lines
            config_elements = []
            for _ in range(num_atoms):
                atom_line = f.readline()
                config_elements.append(atom_line.split(' ')[0])

            end_offset = f.tell()  # End of configuration
            offsets.append((start_offset, end_offset))
            elements.append(config_elements)

    # add ability to load elements
    if return_elements:
        return offsets, elements
    else:
        return offsets


def save_dataset_batches(dataset_fp: str,
                         batch_size: int):
    """
    Save an input .xyz file into batches of batch_size, for seperate processing to save
    memory requirements of loading.
    """
    
    ds_name = get_name_from_path(dataset_fp)
    out_dir = '/'.join(dataset_fp.split('/')[0:-1])

    # get config offsets in file
    logger.info(f"Counting input configurations...")
    offsets = get_xyz_config_offsets(dataset_fp)
    n = len(offsets)
    logger.info(f"Total num configs in input dataset: {n}")

    # determine number of batches as given by batch size
    num_batches = int(n / batch_size)
    remainder = n % batch_size
    batch_lengths = [batch_size for _ in range(num_batches)]
    batch_lengths = batch_lengths + [remainder] if remainder != 0 else batch_lengths
    num_batches = len(batch_lengths)

    batched_offsets = [(offsets[sum(batch_lengths[:i])][0], 
                        offsets[sum(batch_lengths[:i+1]) - 1][1]) 
                        for i in range(num_batches)]

    with open(dataset_fp, "r") as f:

        for i, (start_offset, end_offset) in enumerate(batched_offsets):

            f.seek(start_offset)  # Jump to the configuration start
            frame_data = f.read(end_offset - start_offset)  # Read only this frame

            # write batch to file
            batch_fp = f"{out_dir}/{ds_name}_batch_{i}.xyz"
            with open(batch_fp, mode="w") as batch_file:
                batch_file.write(frame_data)
                logger.info(f"Saved dataset batch to to {batch_fp}...")

    logger.info(f"Successfully saved {ds_name} into {num_batches} batches...")


def flexible_ase_read(fpath: str,
                      cust_props_parser: Callable = None,
                      ) -> List[ase.Atoms]:
    """
    FLexible reader for ase .xyz or .extxyz files which handles different energy / force labels.
    """

    if cust_props_parser is not None:

        configs = read(fpath, ':')
        with open(fpath, 'r') as extxyz_file:
            custom_configs = read_extxyz(extxyz_file, index=slice(None), properties_parser=cust_props_parser)
            for config, custom_config in zip(configs, custom_configs):
                config.info['energy'] = custom_config.get_potential_energy()
    else:
        configs = read(fpath, ':')

    return configs


def get_xyz_file_len(fpath: str) -> int:
    """
    Counts the number of structures in an .xyz or .extxyz file
    """
    count = 0
    with open(fpath, 'r') as f:

        while True:
            num_atoms_line = f.readline()
            if not num_atoms_line:
                break  # EOF reached

            num_atoms = int(num_atoms_line.strip())  # Read number of atoms
            f.readline()  # Skip the comment line

            for _ in range(num_atoms):
                if not f.readline():
                    break  # Prevent reading past EOF

            count += 1

    return count


def orca_props_parser(prop_line: str) -> dict:
    match = re.search(r"E\s*(-?\d+\.\d+)", prop_line)
    if match:
        return {"energy": float(match.group(1)) * ase.units.Hartree}
    else:
        raise ValueError("No E <float> found in Props Line for orca xyz props parser...")


def qm_energy_label_props_parser(prop_line: str) -> dict:
    match = re.search(r"QM_energy\s*=\s*(-?\d+\.\d+)", prop_line)
    if match:
        return {"energy": float(match.group(1))}
    else:
        raise ValueError("No QM_energy = <float> found in Props Line for QM_energy label xyz props parser...")


def autode_props_parser(prop_line: str) -> dict:
    match = re.search(r"E\s*=\s*(-?\d+\.\d+)", prop_line)
    if match:
        return {"energy": float(match.group(1)) * ase.units.Hartree}
    else:
        raise ValueError("No E = <float> found in Props Line for autoDE xyz props parser...")


def get_name_from_path(fpath: str):
    """
    Get the name of a file (i.e. dataset / model) from a path string.
    """
    fname = fpath.split('/')[-1]
    if '.xyz' in fname:
        return fname.split('.xyz')[0]
    elif '.model' in fname:
        return fname.split('.model')[0]
    elif '.info' in fname:
        return fname.split('.info')[0]
    elif '.csv' in fname:
        return fname.split('.csv')[0]
    elif '.npz' in fname:
        return fname.split('.npz')[0]
    elif '.h5' in fname:
        return fname.split('.h5')[0]
    elif '.out' in fname:
        return fname.split('.out')[0]
    else:
        raise ValueError('File extension not handled by path splitter.')


def get_dir_from_path(fpath: str):
    """
    Get the directory path of a file from a path string.
    """

    return fpath.rsplit('/', 1)[0]


def load_data_cache(cache_path: str, hdf5_file: TextIO = None):
    """
    Loads a .npz file, including pickled metadata.
    """

    # if hdf5 file exists, load from it
    if hdf5_file is not None:
        data = {}

        try:
            group = hdf5_file[get_name_from_path(cache_path)]
            # logger.info('Group Name: ', get_name_from_path(cache_path))
            # logger.info('Group Keys: ', group.keys())
            for data_label in group.keys():
                data[data_label] = group[data_label][()]  # read dataset into numpy array
            logger.info(f"Loaded npz data cache from h5py file...")

        except KeyError:
            logger.info(f"Could not load hdf5 key {cache_path}...")
            if os.path.exists(cache_path):
                data = np.load(cache_path, allow_pickle=True)
                logger.info(f"Loaded npz data cache from {cache_path}...")
            else:
                return None
    else:
        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            logger.info(f"Loaded npz data cache from {cache_path}...")
        else:
            return None
    return data


def save_npz_data_cache(npz_cache_path: str, 
                        array_dict: Dict[str, np.ndarray], 
                        meta: Dict = None):
    """
    Saves numpy arrays and pickled metadata as a .npz file.    
    """

    # np.savez(npz_cache_path, **array_dict, meta=np.array(meta, dtype=object))
    np.savez(npz_cache_path, **array_dict)
    logger.info(f"Saved npz data cache to {npz_cache_path}...")


def save_combined_hdf5_cache(out_cache_fpath: str,
                             npz_fpath_stub: str):
    """
    Combines several npz files into a single hdf5 cache file with compression
    """

    # Set your directory and output path
    npz_files = glob.glob(f"{npz_fpath_stub}*.npz")
    assert len(npz_files) > 0, f"No .npz files detected with fpath stub {npz_fpath_stub}"

    logger.info(f"Compressing {len(npz_files)} .npz files to {out_cache_fpath}...")

    with h5py.File(out_cache_fpath, "w") as h5f:

        for i, path in enumerate(npz_files):
            logger.info(f'Processing {path}...')
            name = os.path.splitext(os.path.basename(path))[0]  # filename without .npz
            npz_data = np.load(path, allow_pickle=True)

            # create a group for this npz file
            group = h5f.create_group(name)

            # store each array inside the group, ignoring meta object arrays
            for key in npz_data.files:
                data = npz_data[key]
                if data.dtype == object:  
                    continue
                print('Dtype: ', data.dtype)
                if '<U' in str(data.dtype):
                    data = data.astype('S2')
                group.create_dataset(key, data=data, **hdf5plugin.Zstd(clevel=22))  # use ztsd compression
                # group.create_dataset(key, data=data, compression="lzw")  # use lzw

            if i % 10 == 0:
                logger.info(f"Processed {i} / {len(npz_files)} npz files...")

    logger.info(f"Converted {len(npz_files)} files to {out_cache_fpath}...")


def load_hdf5_cache(out_cache_fpath: str = OUT_CACHE_FPATH) -> Union[h5py.File, None]:

    if os.path.exists(out_cache_fpath):
        return h5py.File(out_cache_fpath, 'r')
    else:
        return None


def load_isolated_atom_energies(fpath: str):

    configs = flexible_ase_read(fpath)
    return {int(config.get_atomic_numbers()[0]): str(get_ase_atoms_energy(config)) for config in configs}


def get_ase_atoms_energy(config: ase.Atoms):
    """
    Load energy accounting for flexibly loaded xyz / extxyz files through ase.
    """ 
    try:
        energy = config.info['energy'] if config.info.get('energy') is not None else config.get_potential_energy()
        
    except RuntimeError:

        for energy_key in {'energy', 'QM_energy', 'REF_energy', 'E'}:
            energy = config.info.get(energy_key)
            if energy is not None:
                break
        if energy is None:
            raise RuntimeError("Energy label not found...")
        # config.info['energy'] = energy

    return energy


def get_ase_atoms_forces(config: ase.Atoms):
    """
    Load forces accounting for flexibly loaded xyz / extxyz files through ase.
    """
    try:
        forces = config.get_forces()
        # config.arrays['forces'] = forces
    except RuntimeError:

        for forces_key in {'forces', 'QM_forces', 'REF_forces'}:
            forces = config.arrays.get(forces_key)
            if forces is not None:
                break
        if forces is None:
            raise RuntimeError("Forces label not found...")
        # config.arrays['forces'] = forces

    return forces


def extract_num_atoms(text: str):
    patterns = [
        r"Number of atoms\s*[:=]\s*(\d+)",
        r"Total number of atoms\s*[:=]\s*(\d+)",
        r"n\s*atoms\s*[:=]\s*(\d+)",
        r"(\d+)\s+atoms",
    ]

    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return int(m.group(1))

    # Fallback: count atoms in the Cartesian coordinate table
    # This matches lines like:
    # C    0.000000   0.000000   0.000000
    coord_block = re.search(
        r"CARTESIAN COORDINATES.*?\n(?:[-=]+\n)?(.*?)(?:\n\s*\n|CARTESIAN COORDINATES|THE OPTIMIZATION|ORCA TERMINATED|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if coord_block:
        lines = coord_block.group(1).splitlines()
        atom_lines = [
            line for line in lines
            if re.match(r"^\s*[A-Z][a-z]?\s+[-+]?\d", line)
        ]
        if atom_lines:
            return len(atom_lines)

    return None


def check_orca_outputs(work_dir: str, input_name: str):
    succ, fail = [], []
    succ_num_atoms = {}
    fail_num_atoms = {}

    out_fpaths = glob.glob(os.path.join(work_dir, f"**/*{input_name}_*.out"), recursive=True)

    for out_fpath in out_fpaths:
        fname = get_name_from_path(out_fpath)
        run_id = int(re.search(r"(\d+)(?!.*\d)", fname).group()) + 1

        try:
            with open(out_fpath) as f:
                text = f.read()

            num_atoms = extract_num_atoms(text)

            if re.search(r"ORCA\s+TERMINATED\s+NORMALLY", text, re.IGNORECASE):
                succ.append(run_id)
                succ_num_atoms[run_id] = num_atoms
            else:
                fail.append(run_id)
                fail_num_atoms[run_id] = num_atoms

        except FileNotFoundError:
            print(f"{out_fpath} not found...")
            fail.append(run_id)
            fail_num_atoms[run_id] = None

    print("=== ORCA RUN STATS ===")
    print("Total Runs: ", len(out_fpaths))
    print("Success: ", len(succ))
    print("Success Jobs: ", sorted(succ))
    print("Success Num Atoms: ", {k: succ_num_atoms[k] for k in sorted(succ_num_atoms)})
    print("Fail: ", len(fail))
    print("Fail Jobs: ", sorted(fail))
    print("Fail Num Atoms: ", {k: fail_num_atoms[k] for k in sorted(fail_num_atoms)})
    print("=========================")


def load_model_calculator(model_name: str, 
                          device='cuda:0', 
                          model_loader: Literal['MACE', 'ani', 'graph-pes'] = None,
                          enable_cueq: bool = False,
                          dtype: Literal["", "float32", "float64"] = "",
                          head: str = "Default"):
    """
    Flexible model loader for different types of model based on model_name (i.e. ACE models/ MACE models / ANI models / transformer models etc.)
    """

    try:

        # attempt to load model_loader from model name
        if model_loader is None:

            if model_name in MACE_MODELS:
                model_loader = MACE_MODEL_LOADER
            elif model_name in ANI_MODELS:
                model_loader = ANI_MODEL_LOADER
            elif any(sub_model_name in model_name for sub_model_name in MACE_MODELS + ['MACE', 'EGRET']):
                model_loader = MACE_MODEL_LOADER
            elif any(sub_model_name in model_name for sub_model_name in ANI_MODELS):
                model_loader = ANI_MODELS
            else:
                model_loader = MACE_MODEL_LOADER  # default to MACE model loader!

        if model_loader == MACE_MODEL_LOADER:

            # mace model loader (imported lazily to keep io.py importable without MACE)
            from mace.calculators import MACECalculator

            return MACECalculator(model_paths=f"{MODEL_DIR}/{model_name}.model",
                                device=device,
                                enable_cueq=enable_cueq,
                                default_dtype=dtype,
                                head=head
                                )

        elif model_loader == ANI_MODEL_LOADER:

            # ani model loader (torchani imported lazily)
            import torchani

            # read constants from constant file and construct AEV computer.
            const_fpath = f'{ANI_1XNR_MODEL_DIR}/{model_name}/rHCNO-5.2R_32-3.5A_a8-4.params'  # noqa: E501
            consts = torchani.neurochem.Constants(const_fpath)
            aev_computer = torchani.AEVComputer(**consts)

            # read self energies and construct energy shifter.
            sae_fpath = f'{ANI_1XNR_MODEL_DIR}/{model_name}/sae_linfit.dat'  # noqa: E501
            energy_shifter = torchani.neurochem.load_sae(sae_fpath)

            # single model
            model_dir = f'{ANI_1XNR_MODEL_DIR}/{model_name}/train0/networks'  # noqa: E501
            model = torchani.neurochem.load_model(consts.species, model_dir)

            # create pipeline for computing energies
            nnp = torchani.nn.Sequential(aev_computer, model, energy_shifter)

            # get torchani ASE calculator (calculates in Hartrees)
            return torchani.ase.Calculator(consts.species, nnp)

        elif model_loader == GRAPH_PES_MODEL_LOADER:

            from graph_pes.models import load_model
            import torch

            model = load_model(f"{MODEL_DIR}/{model_name}.model")
            model.to(dtype=torch.float64)
            return model.ase_calculator()
        else:
            raise NameError(f"Model loader: {model_loader} is not valid.")
    
    except Exception as e:
        logger.warning(f"{model_name} was unable to be loaded because of error: \n{e} \n... Returning None for calc instead...")
        return None


def substitute_yaml_variables(value: dict, variables: dict):
    """
    Little recursive function to fill in YAML variables based on its own defined variable names.

    USAGE: yaml_config = substitute_yaml_variables(yaml_config, yaml_config)
    """
    if isinstance(value, str):
        return re.sub(
            r"\$\{(\w+)\}",
            lambda match: str(variables.get(match.group(1), match.group(0))),
            value,
        )
    elif isinstance(value, list):
        return [substitute_yaml_variables(item, variables) for item in value]
    elif isinstance(value, dict):
        return {
            key: substitute_yaml_variables(item, variables)
            for key, item in value.items()
        }
    return value