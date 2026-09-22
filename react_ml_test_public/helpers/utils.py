"""
Random helper functions.
"""
import copy
import hashlib
import math
import os
import re
from typing import Callable, Dict, List, Literal, Union

import ase
import numpy as np
import pandas as pd
import torch
from mace.calculators import MACECalculator
from mace.tools.utils import AtomicNumberTable
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from react_ml_test_public.helpers.io import (
    flexible_ase_read,
    get_ase_atoms_energy,
    get_ase_atoms_forces,
    get_name_from_path,
    load_data_cache,
    save_npz_data_cache,
)
from react_ml_test_public.helpers.settings import (
    EV,
    EV_PER_AMSTRONG,
    FORMATTED_MODEL_NAMES,
    KCAL_PER_MOL,
    KJ_PER_MOL,
    MEV,
    MEV_PER_AMSTRONG,
    MEV_PER_ATOM,
    NPZ_DIR,
    eV_to_kcal_mol,
    eV_to_kJ_mol,
    eV_to_mEV,
)
from react_ml_test_public.helpers.stats import (
    calculate_comp_rmse,
    calculate_mae,
    calculate_mag_rmse,
    calculate_rmse,
    round_sigfigs,
)


def format_value_with_uncertainty(value: float, 
                                  z: float):
    """
    Format value and uncertainty rounded to the significant figure of the uncertainty.
    Both value and uncertainty are rounded to the same order of magnitude as the 
    most significant digit of the uncertainty.
    """
    if z == 0 or np.isnan(z):
        return f"{value:.1f}", "0.0"
   
    # Find the order of magnitude of the uncertainty's most significant digit
    z_order = math.floor(math.log10(abs(z)))
    
    # Calculate the rounding factor (power of 10)
    rounding_factor = 10 ** z_order
    
    # Round both value and uncertainty to this order of magnitude
    rounded_value = round(value / rounding_factor) * rounding_factor
    rounded_z = round(z / rounding_factor) * rounding_factor
    
    # Format as integers if they are whole numbers, otherwise as floats
    if rounded_value == int(rounded_value):
        value_str = f"{int(rounded_value)}"
    else:
        # Determine decimal places needed based on the rounding factor
        decimal_places = max(0, -z_order)
        value_str = f"{rounded_value:.{decimal_places}f}"
    
    if rounded_z == int(rounded_z):
        z_str = f"{int(rounded_z)}"
    else:
        decimal_places = max(0, -z_order)
        z_str = f"{rounded_z:.{decimal_places}f}"
    
    return value_str, z_str


def convert_units(x: np.ndarray,
                  in_unit: Literal[EV, MEV, EV_PER_AMSTRONG, MEV_PER_AMSTRONG],
                  out_unit: Literal[MEV, MEV_PER_ATOM, MEV_PER_AMSTRONG, KJ_PER_MOL, KCAL_PER_MOL],
                  n_atoms_list: list[int] = None):
    """
    Unit conversion function.
    """

    if in_unit == EV:
        # apply unit conversions
        if out_unit == MEV:
            x = np.array(x) * eV_to_mEV
        elif out_unit == MEV_PER_ATOM:
            x = np.array([eV_to_meV_per_atom(val, n_atoms) for val, n_atoms in zip(x, n_atoms_list)])
        elif out_unit == KJ_PER_MOL:
            x = np.array(x) * eV_to_kJ_mol
        elif out_unit == KCAL_PER_MOL:
            x = np.array(x) * eV_to_kcal_mol
        else:
            print('No out unit specified...')

    elif in_unit == EV_PER_AMSTRONG:
        if out_unit == MEV_PER_AMSTRONG:
            x = x * eV_to_mEV
            # print('Converted eV / A to meV / A')

    elif in_unit == MEV:
        raise NotImplementedError

    return x


def detach_ase_calculator(configs: list[ase.Atoms]):
    """
    Simply detach the ase calculator from a set of ase.Atoms objects once you are done with them.
    Prevents read/writing errors.
    """
    for config in configs:
        config.calc = None
    print('WARNING THIS MEANS YOU HAVE TO RELY ON .info and .arrays objects for all energies and force labels!!!')


def get_relative_energies(configs: list[ase.Atoms]) -> np.ndarray:
    """
    Return the energies relative to the minum within each set of system sizes (number of atoms)
    for a given set of configs.

    Parameters:
        configs (List[ase.Atoms]): list of input ase.Atoms configurations
    Returns:
        List[float]: list of config energies relative to the minimum for each system size, in the units 
                     originally loaded from the .xyz file.
    """
    
    atom_num_groups = {}
    for config in configs:
        if atom_num_groups.get(len(config)) is None:
            atom_num_groups[len(config)] = [config]
        else:
            atom_num_groups[len(config)].append(config)

    atom_num_group_energy_mins = {num_atoms: np.min(get_ase_props(group_configs, prop='energy')) 
                                    for num_atoms, group_configs in atom_num_groups.items()}
    return np.array([abs(get_ase_atoms_energy(config) - atom_num_group_energy_mins[len(config)]) for config in configs])
    

def unflatten_array_by_lengths(arr: np.ndarray, lengths: list[int]) -> List[np.ndarray]:
    """

    Convert a flat (1D) np.array into a list of sliced np.arrays as indexed by the provided sub-array lengths (i.e. config lengths).
    Useful for converting from a single flattened array of all forces for a training set to a list of per-config forces for e.g.

    Arguments:
        arr (np.ndarray): a 1D np array to be converted to a list of sub-arrays (e.g. forces)
        lengths (list[int]): a list of lengths to partition the np.array
    """
    return [arr[sum(lengths[:i]):sum(lengths[:i+1])] for i in range(len(lengths))]


def get_molecules(config: ase.Atoms):
    """
    Partitions the atoms in a cell into molecules based on a distance function and graph connection.
    """  

    return


def calc_energy_shifts(configs: List[ase.Atoms],
                       per_element_energy_offsets: Dict[int, float]) -> np.ndarray:
    """
    Calculate a per-config energy offset for a given set of configs based on per-element mean energies calculated on an input dataset.

    The calculation simply multiplies per-element mean energies by the number of each element in each input config.
    """

    # get per config shifts (in eV)
    config_energy_shifts = np.array([sum([per_element_energy_offsets[atom.number] for atom in config]) 
                                     for config in configs])
    
    return config_energy_shifts


def guess_per_element_mean_energies(
    configs_fpath: str,
    calc = None,
    calc_name: str = None,
    custom_xyz_parser: Callable = None,
    dtype = np.float64,
    pbc_box: List[str] = None,
) -> dict[int, float]:
    r"""

    ADAPTED FROM GRAPH-PES

    Guess the per-element mean (:math:`\mu_Z`) and variance (:math:`\sigma_Z^2`)
    of a per-structure quantity using ridge regression under the following assumptions:

    1. the per-structure property, :math:`P`, is a summation over local
       properties of its components atoms: :math:`P = \sum_{i=1}^{N} p_{Z_i}`.
    2. the per-atom properties, :math:`p_{Z_i}`, are independent and identically
       distributed (i.i.d.) for each atom of type :math:`Z_i` according to a
       normal distribution: :math:`p_{Z_i} \sim \mathcal{N}(\mu_{Z_i}, \sigma_{Z_i}^2)`.

    Parameters
    ----------
    per_structure_quantity
        The per-structure quantity to guess the per-element mean and variance of.
    batch
        The batch of graphs to use for guessing the per-element mean and variance.

    Returns
    -------
    means
        A dictionary mapping atomic numbers to per-element means.
    variances
        A dictionary mapping atomic numbers to per-element variances.
    """  # noqa: E501

    configs = flexible_ase_read(configs_fpath, cust_props_parser=custom_xyz_parser)

    if calc_name is None:
        energies = np.array([get_ase_atoms_energy(config) for config in configs], dtype=np.float64)
    else:

        npz_cache_path = f"{NPZ_DIR}/{calc_name}_{get_name_from_path(configs_fpath)}_energies.npz"        
        # load cached npz file of energies / forces if it exists
        cache_data = load_data_cache(npz_cache_path)
        if cache_data is not None:
            energies = cache_data['energies']
        else:
            if calc is None:
                raise OSError(f"NO MODEL CALCULATOR, AND NO NPZ CACHE FOUND for {model_name}. Add {model_name} to ./models and try again.")
            print('Calculating energies...')
            # calculate energies
            energies = []
            for config in configs:
                if pbc_box is not None:
                    config.pbc = True
                    config.set_cell(pbc_box)
                    
                config.calc = calc
                energies.append(config.get_potential_energy())
            energies = np.array(energies, dtype=np.float64)
            save_npz_data_cache(npz_cache_path, {'energies': energies})

    # extract the atomic numbers to tensor N such that:
    # N[structure, Z] is the number of atoms of atomic number Z in structure
    batch_Z = np.concatenate([np.array(config.numbers, dtype=np.int64) for config in configs])
    unique_Zs = np.unique(batch_Z)  # (n_Z,)
    N = np.zeros((len(configs), len(unique_Zs)), dtype=dtype)  # (batch, n_Z)

    # Compute the count matrix
    for i, config in enumerate(configs):
        for j, Z in enumerate(unique_Zs):
            N[i, j] = np.sum((np.array(config.numbers) == Z).astype(dtype))  # Count occurrences of Z in config.numbers

    # calculate the per-element mean
    # use Ridge rather than LinearRegression to avoid singular matrices
    # when e.g. only one structure contains an atom of a given type...
    ridge = Ridge(fit_intercept=False, alpha=0.001, solver='svd')
    ridge.fit(N, torch.tensor(energies))
    mu_Z = torch.tensor(ridge.coef_)
    means = {int(Z): float(mu) for Z, mu in zip(unique_Zs, mu_Z)}

    return means
    

def get_formatted_model_name(model_name: str):
    return FORMATTED_MODEL_NAMES.get(model_name) if model_name in FORMATTED_MODEL_NAMES.keys() else model_name


def add_to_mace_z_table(calc: MACECalculator, 
                        atom_nums: list[int]):
    """
    Add new atomic numbers to a mace calculator internal z table 
    (for predictions outside of trained elements).
    """
    calc.z_table = AtomicNumberTable(list(set(calc.z_table.zs + atom_nums)))

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def min_max_scale(xs):
    return (xs - np.min(xs)) / (np.max(xs) - np.min(xs))

def eV_to_meV_per_atom(eV: float, n_atoms: int) -> Union[float, np.ndarray]:
    return (eV * 1000) / float(n_atoms)


def get_ase_props(configs: List[ase.Atoms], 
                  prop: Literal["energy", "local_energies", "forces"],
                  calc: Callable = None,
                  pbc_cell: List[int] = None,
                  flatten_force_array: bool = True) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Get a specifed property over a list of ase Atoms objects.

    This can be used for any ase Calculator (graph-based or non-graph-based).

    Arguments:
        configs (List[ase.Atoms]): List of ASE Atoms objects representing atomic configurations.
        prop (Literal["energy", "local_energies", "forces"]): The property to extract from each configuration.
        calc (Callable): If making a prediction, the ase calculator. Optional.
        pbc_cell (List[int]): Periodic boundary condition flags for each cell dimension (e.g., [1, 1, 1]). Optional.
        flatten_force_array (bool): Whether to flatten the forces for all configs to a single array of shape (n_atoms, 3) 
                                    or instead return a list of config force vectors [(n_config_atoms, 3), ...]. Defaults to True.

    Returns:
        Union[np.ndarray, List[np.ndarray]]: Extracted property values. Returns a single array if `prop` is "energy" or
        `flatten_force_array` is True; otherwise returns a list of arrays (e.g., one per structure).
    """

    if prop == "energy":

        energies = []
        for config in configs: 

            if calc is None:
                # get energies saved in xyz file
                energy = get_ase_atoms_energy(config)
            else:

                # apply pbc box
                if pbc_cell is not None:
                    config = copy.deepcopy(config)
                    config.pbc = True
                    config.set_cell(pbc_cell)

                # predict energies with calculator
                energy = calc.get_potential_energy(config)

            energies.append(energy)
        return np.array(energies)

    elif prop == "local_energies":
        assert calc is not None, "A calculator must be attached to calculate local atomic energies, as these are not a real quantity."
        local_energies = []
        for config in config:

            # apply pbc box
            if pbc_cell is not None:
                config = copy.deepcopy(config)
                config.pbc = True
                config.set_cell(pbc_cell)

            config_local_energies = calc.get_property('energy', config, type='contributions')
            local_energies += (config_local_energies).tolist()

        return np.array(local_energies)

    elif prop == "forces":

        forces = []
        for config in configs:

            if calc is None:
                config_forces = get_ase_atoms_forces(config)
            else:

                # apply pbc box
                if pbc_cell is not None:
                    config = copy.deepcopy(config)
                    config.pbc = True
                    config.set_cell(pbc_cell)

                config_forces = calc.get_forces(config)
            
            forces.append(config_forces)
        
        if all(config_forces is None for config_forces in forces):
            return None
        
        if flatten_force_array:
            return np.array([force for config_forces in forces for force in config_forces])
        else:
            return forces
    
    else:
        raise ValueError("prop must be one of: [energy, local_energies, forces]")
    

def generate_error_table(predictions_dict: dict,
                         test_sets: list,
                         test_set_names: list,
                         error_type: Literal['rmse', 'rmse_mag', 'rmse_comp', 'mae', 'R2'],
                         title: str = "RMSE Table",
                         use_relative_values: bool = False
                         ):
    """
    Generate a table of RMSE values for n predictions and m test sets.

    Parameters:
        predictions_dict (dict): Dictionary where keys are prediction labels and values are lists of np.ndarray,
                                 each corresponding to predictions for the respective test set.
        test_sets (list of np.ndarray): List of test sets (actual values).
        test_set_names
        error_type
        title

    Returns:
        pd.DataFrame: Table of RMSE values.
    """
    if not predictions_dict or len(test_sets) == 0:
        raise ValueError("Predictions and test sets must not be empty.")

    rmse_values = []
    prediction_labels = list(predictions_dict.keys())
    for label in prediction_labels:
        row = []
        predictions_list = predictions_dict[label]
        assert len(predictions_list) == len(test_sets), f"The number of prediction arrays for '{label}' must match the number of test sets, are of lengths: predictions: {len(predictions_list)} len tests: {len(test_sets)}"

        for predictions, test_set in zip(predictions_list, test_sets):

            if test_set is None:
                row.append(None)
                continue
            assert len(predictions) == len(test_set), f"Predictions for '{label}' and the corresponding Test Set must have the same length."

            # use values relative to min (e.g. for energies)
            if use_relative_values:
                test_set -= np.min(test_set)
                predictions -= np.min(predictions)

            if error_type == 'rmse':
                error = round_sigfigs(calculate_rmse(predictions, test_set), 3)
            elif error_type == 'rmse_mag':
                error = round_sigfigs(calculate_mag_rmse(predictions, test_set), 3)
            elif error_type == 'rmse_comp':
                # error = round_sigfigs(calculate_comp_rmse(predictions, test_set), 3)
                error = calculate_comp_rmse(predictions, test_set)
            elif error_type == 'mae':
                error = round_sigfigs(calculate_mae(predictions, test_set), 3)
            elif error_type == 'R2':
                error = round_sigfigs(r2_score(test_set, predictions), 3)
            else:
                raise ValueError("error_type must be one of rmse or mae...")

            row.append(error)
        rmse_values.append(row)

    # Create a DataFrame for nicely formatted output
    df = pd.DataFrame(
        rmse_values,
        columns=[f"{test_set_names[j]}" for j in range(len(test_sets))],
        # index=[get_formatted_model_name(label) for label in prediction_labels]
        index=[label for label in prediction_labels]
    )

    print(f"\n{title}")
    print(df)
    return df


def atoms_hash(config: ase.Atoms, include_cell=False):
    """
    Returns a unique hash of an ase.Atoms object (atomic numbers, positions, pbc) for easy use
    as dictionary keys or unique orderings of configurations.
    """

    # atomic numbers
    numbers = config.get_atomic_numbers()

    # positions (rounded for stability)
    pos = np.round(config.get_positions(), decimals=6)

    parts = [numbers.tobytes(), pos.tobytes()]

    if include_cell:
        cell = np.round(config.get_cell().array, decimals=2)
        parts.append(cell.tobytes())

    # include PBC if relevant
    parts.append(np.array(config.get_pbc(), dtype=int).tobytes())

    data = b"".join(parts)
    return hashlib.sha256(data).hexdigest()