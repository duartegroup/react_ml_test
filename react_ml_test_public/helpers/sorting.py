"""
Different sort methods for datasets.
"""
import ase
from typing import Dict, List, Literal, Tuple
from react_ml_test_public.helpers.descriptors import get_soap_descriptors
from react_ml_test_public.helpers.dim_red import calc_best_similarities, run_pca
from react_ml_test_public.helpers.io import get_ase_atoms_energy, get_ase_atoms_forces
from react_ml_test_public.helpers.logging import logger
from react_ml_test_public.helpers.utils import get_ase_props, atoms_hash
import numpy as np


def sort_configs_by_ts_dist(configs: List[ase.Atoms],
                            rs: List[Tuple],
                            ts_ref_rs_dist: Dict[tuple, float],
                            reverse: bool = False) -> List[ase.Atoms]:
    """
    Return a sorted copy of a dataset, sorted by the distances of a defined collective variable.

    Parameters:
        configs: List[ase.Atoms] the list of configurations to sort by the collective variable
        cv: List[Tuple] a list of pairs of atom IDs (atom_1, atom_2) which we want to measure the distance 
                        between, which define the collective variable
        cv: List[ase.Atoms] 
    Return:
        List[ase.Atoms]: a copied, sorted list of the input configs
    """
    return sorted(configs, key=lambda config: sum(abs(config.get_distance(*ri, mic=True) - ts_ref_rs_dist[ri]) for ri in rs), reverse=reverse)


def sort_configs_by_cv_dist(configs: List[ase.Atoms],
                            cv: List[Tuple],
                            reverse: bool = False) -> List[ase.Atoms]:
    """
    Return a sorted copy of a dataset, sorted by the distances of a defined collective variable.

    Parameters:
        configs: List[ase.Atoms] the list of configurations to sort by the collective variable
        cv: List[Tuple] a list of pairs of atom IDs (atom_1, atom_2) which we want to measure the distance 
                        between, which define the collective variable
        cv: List[ase.Atoms] 
    Return:
        List[ase.Atoms]: a copied, sorted list of the input configs
    """
    return sorted(configs, key=lambda config: sum([config.get_distance(i, j) for i, j in cv]), reverse=reverse)


def sort_configs_by_energy(configs: List[ase.Atoms],
                           reverse: bool = False) -> List[ase.Atoms]:
    """
    Sort a dataset based on energies, currently from low to high by each number of atoms.

    Do this based on a per system-size basis. I.e. get the means per system size (to control for large energy differences due to 
    number of atoms).
    """

    # group configs by number of atoms
    atom_num_groups = {}
    for config in configs:
        if atom_num_groups.get(len(config)) is None:
            atom_num_groups[len(config)] = [config]
        else:
            atom_num_groups[len(config)].append(config)

    atom_num_group_mins = {num_atoms: np.min(get_ase_props(group_configs, prop='energy'))
                            for num_atoms, group_configs in atom_num_groups.items()}
    return sorted(configs, key=lambda config: abs(get_ase_atoms_energy(config) - atom_num_group_mins[len(config)]), reverse=reverse)


def sort_configs_by_forces(configs: list[ase.Atoms],
                           reverse: bool = False) -> list[ase.Atoms]:
    """
    Return a sorted copy of a dataset, sorted by the sum of the atomic force-vector norms.

    Forces must be specified for every config, either as a calculator result or as the 'forces' /
    'QM_forces' / 'REF_forces' arrays of an extxyz file (see get_ase_atoms_forces); a RuntimeError
    is raised if any config has none. Sorted low to high by default (good for detecting
    equilibrium structures in a dataset).

    Parameters:
        configs: List[ase.Atoms] the list of configurations to sort by their summed atomic force norm
        reverse: bool sort from highest to lowest summed force norm instead
    Return:
        List[ase.Atoms]: a copied, sorted list of the input configs
    """

    # check forces are specified up front, so a config missing them is reported clearly rather
    # than failing part-way through the sort
    config_force_norms = []
    for i, config in enumerate(configs):

        try:
            forces = np.asarray(get_ase_atoms_forces(config))
        except RuntimeError as e:
            raise RuntimeError(f"Forces are not specified for config {i} of {len(configs)}, and "
                               f"sort_configs_by_forces sorts by the atomic force norms...") from e

        # summed 3D norm of each atomic force vector
        config_force_norms.append(float(np.linalg.norm(forces, axis=1).sum()))

    sorted_indices = sorted(range(len(configs)), key=lambda i: config_force_norms[i], reverse=reverse)
    return [configs[i] for i in sorted_indices]


def multi_sort(configs: List[ase.Atoms]):

    return

def sort_configs_by_soap_desc(configs: List[ase.Atoms],
                              ref_configs: List[ase.Atoms],
                              mode: Literal['sim', 'dist'],
                              reverse: bool = False,
                              compress_dim: int = 8) -> List[ase.Atoms]:
    """

    By default orders by highest SOAP similarity to lowest SOAP similarity.
    """

    # # set maximum compression dimension to minimum no. atoms in each config
    # min_config_size = min([len(config) for config in configs])
    # if compress_dim > min_config_size:
    #     print('Setting compression dimension to minimum no. of atoms: ', compress_dim)
    #     compress_dim = min_config_size

    # get elements unique to both sets (intersection)
    input_elements = set([element for config in configs for element in config.get_chemical_symbols()])
    ref_elements = set([element for config in ref_configs for element in config.get_chemical_symbols()])
    unique_elements = input_elements & ref_elements

    # get soap descriptors
    input_soap_mat = get_soap_descriptors(configs, specific_elements=unique_elements, average='off', flatten_output=True)

    # get pretraining domain reference soap matrix, reduced
    ref_soap_mat = get_soap_descriptors(ref_configs, specific_elements=unique_elements, average='off', flatten_output=True)

    logger.info("Input SOAP Mat Shape: ", input_soap_mat.shape)
    logger.info('Ref SOAP Mat Shape: ', ref_soap_mat.shape)

    if compress_dim is not None:
        input_soap_mat = run_pca(input_soap_mat, n_components=compress_dim)
        ref_soap_mat = run_pca(ref_soap_mat, n_components=compress_dim)
        logger.info("Input SOAP Mat Shape after PCA: ", input_soap_mat.shape)
        logger.info("Ref SOAP Mat Shape after PCA: ", ref_soap_mat.shape)

    # re-construct list of config soap features after PCA
    input_config_soap_vecs, start = [], 0
    for config in configs:
        input_config_soap_vecs.append(input_soap_mat[start:start + len(config)])
        start += len(config)

    if mode == 'sim':

        config_mean_best_sims = []
        for config_soap_vec in input_config_soap_vecs:

            # get best cosine similarities of each x1 -> x2 descriptor
            best_sim_tuples, best_sim_mean, _ = calc_best_similarities(config_soap_vec, ref_soap_mat)
            # logger.info("Best Sim Tuples: ", best_sim_tuples)
            config_mean_best_sims.append(best_sim_mean)

    # logger.info("Config Mean Best Sims: ", config_mean_best_sims)
    sorted_index_train_configs = sorted(enumerate(configs), key=lambda x: config_mean_best_sims[x[0]], reverse=not reverse)
    return [config for _, config in sorted_index_train_configs]


def sort_configs_by_structure(configs: List[ase.Atoms]):
    """
    Sorts dataset by a unique atoms object hash, meaning that two datasets containing equivalent structures
    should be returned in the exact same order.
    """
    return sorted(configs, key=atoms_hash)


def sort_dataset_by_size(configs: List[ase.Atoms], reverse=False):
    """
    Quite obvious what this does.
    """

    return sorted(configs, key=lambda config: len(config), reverse=reverse)


def sort_dataset_by_element_count(configs: List[ase.Atoms], element: str, reverse=False):
    """
    Sorts the input list of ASE.atoms configs based on the count of the specified element.
    """

    config_element_counts = [config.get_chemical_symbols().count(element) for config in configs][0:100]
    print('Element counts: ', config_element_counts)
    sorted_configs = sorted(configs, key=lambda config: config.get_chemical_symbols().count(element), reverse=reverse)
    config_element_counts = [config.get_chemical_symbols().count(element) for config in sorted_configs][0:100]
    print('Element counts after sorting: ', config_element_counts)
    return sorted_configs
