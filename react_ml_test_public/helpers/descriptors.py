"""
Helper functions for getting model and dataset descriptors.
"""
import os
from typing import Callable, List, Literal, Union
import ase
import numpy as np
from dscribe.descriptors import SOAP
import random

from react_ml_test_public.helpers.dim_red import run_pca
from react_ml_test_public.helpers.io import load_data_cache, save_npz_data_cache
from react_ml_test_public.helpers.settings import DESCRIPTORS_DIR, NPZ_DIR
from react_ml_test_public.helpers.utils import unflatten_array_by_lengths
from react_ml_test_public.helpers.logging import logger


def get_mace_descriptors(calc: Callable,
                         configs: List[ase.Atoms],
                         invariants_only: bool = True,
                         compress_dim: int = None,
                         flatten_output: bool = False,
                         cache_name: str = None,
                         override_cached: bool = False) -> Union[tuple[np.ndarray], tuple[List[np.ndarray]]]:
    """
    Get atomic MACE descriptors from a trained MACE model for a list of configurations.
    :return:
    """

    # check npz cache
    npz_cache_fpath = f"{DESCRIPTORS_DIR}/{cache_name}_mace_desc.npz" if cache_name is not None else ''
    if os.path.exists(npz_cache_fpath) and not override_cached:
        cache_data = load_data_cache(npz_cache_fpath)
        mace_desc_flat, elements_flat, config_lens = cache_data['mace_desc'], cache_data['elements'], cache_data['config_lens']
        config_mace_descs = unflatten_array_by_lengths(mace_desc_flat, config_lens)
        config_elements = unflatten_array_by_lengths(elements_flat, config_lens)
        logger.info(f'Loaded MACE desc for {len(config_lens)} configs from npz cache file: {npz_cache_fpath}')
    else:
        # calc atomic mace descriptors
        config_mace_descs, config_elements = [], []
        num_configs = len(configs)
        for i, config in enumerate(configs):

            if i % 100 == 0:
                logger.info(f"Calculated mace descriptors for {i} / {num_configs} configs...")
            
            # get atomic mace descriptors
            config_mace_descs.append(calc.get_descriptors(config, invariants_only=invariants_only))
            config_elements.append(np.array(config.get_chemical_symbols()))

        mace_desc_flat = np.concatenate(config_mace_descs, axis=0)
        elements_flat = np.concatenate(config_elements, axis=0).astype('S4')

        # save npz cache
        save_npz_data_cache(npz_cache_fpath, {'mace_desc': mace_desc_flat,
                                              'elements': elements_flat,
                                              'config_lens': np.array([len(mace_desc) for mace_desc in config_mace_descs]).astype(np.uint16),
                                              })
        logger.info(f'Saved MACE desc to npz cache file: {npz_cache_fpath}')

    # PCA compression on features
    if compress_dim is not None:
        logger.info(f"Compressing MACE desc to {compress_dim} dim using PCA...")
        mace_desc_flat = run_pca(mace_desc_flat, n_components=compress_dim)
        config_mace_descs = unflatten_array_by_lengths(mace_desc_flat, config_lens)

    logger.info(f"MACE Desc Shape: {mace_desc_flat.shape}")

    # return mace descriptors and respective elements
    if flatten_output:
        return mace_desc_flat, elements_flat
    else:
        return config_mace_descs, config_elements


def get_soap_descriptors(configs: List[ase.Atoms],
                         specific_elements: set[str] = None,
                         specific_atom_ids: list[int] = None,
                         atoms_per_config: int = None,
                         average: Literal['off', 'outer', 'inner'] = 'off',
                         dtype: str = "float32",
                         compress_dim: int = None,
                         flatten_output: bool = False,
                         n_jobs: int = 1,
                         cache_name: str = None) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Calculate and return the atomic SOAP descriptors for a given set of configs.

    Parameters:
        configs (list[ase.Atoms]): a list of ase.Atoms objects
        specific_elements (Optional[set[str]]): optionally specified set of specific elements to only return their SOAP descriptors 
        specific_atom_ids (Optional[list[int]]): optionally specified set of atom ids to only return their SOAP descriptors per config 
        atoms_per_config (Optional[int]]): optionally specify the number of atoms to randomly sample per configuration
        average (Literal['off', 'outer', 'inner']): average mode for SOAP descriptors per config
        dtype (str): dtype for SOAP descriptors
        compress_dim (Optional[int]): dimension to reduce each SOAP vector using PCA if specified.
        flatten_output (bool): if true, returns one flattended array of all SOAP features for a dataset, if false, returns a list
                                of per-config SOAP features
        n_jobs (int): number of CPU cores / processes to work on calculating SOAP descriptors
        cache_name (str): file name to save / load data from a cache, if None caching does not occur
    Returns:
        Union[np.ndarray, List[np.ndarray]]: A numpy array of atomic SOAP descriptors if flatten_output is true, else a list of per-config SOAP vectors
    """

    specific_elements_txt = f'Only returning descriptors for elements: {specific_elements}.' if specific_elements is not None else ''
    atoms_per_config_txt = f'Sampling {atoms_per_config} atoms per config.' if atoms_per_config is not None else ''
    averaging_txt = f'Using {average} averaging.' if average != 'off' else ''
    logger.info(f"Getting SOAP Descriptors for {len(configs)} configs. {specific_elements_txt} {atoms_per_config_txt} {averaging_txt}")

    # npz cache save path
    npz_cache_fpath = f"{NPZ_DIR}/{cache_name}_soap_desc_average_{average}_configs_{len(configs)}" + ".npz"
    if os.path.exists(npz_cache_fpath):
        cache_data = load_data_cache(npz_cache_fpath)
        soap_mat, config_lens, species = cache_data['soap'], cache_data['config_lens'], cache_data['species']
        config_soap_vecs = unflatten_array_by_lengths(soap_mat, config_lens)
        logger.info(f'Loaded soap desc for {len(config_lens)} configs from npz cache file: {npz_cache_fpath}')
    else:
        # for sampling atoms
        random.seed(3)

        # get species
        species = [element for config in configs for element in config.get_chemical_symbols()]
        unique_species = list(set(species))
        logger.info(f"Creating SOAP for elements: {unique_species}")

        soap = SOAP(species=unique_species,
                    periodic=False,
                    r_cut=5.5,
                    n_max=12,
                    l_max=8,
                    sigma=0.5,
                    average=average,
                    compression={"mode": "mu1nu1"},
                    dtype=dtype
                    )

        # get config soap vectors
        config_soap_vecs = soap.create(configs, n_jobs=n_jobs)
        logger.info(f"Config SOAP vecs len: {len(config_soap_vecs)}")

        # save npz cache
        save_npz_data_cache(npz_cache_fpath, {'soap': np.concatenate(config_soap_vecs, axis=0),
                                              'config_lens': np.array([len(soap_vec) for soap_vec in config_soap_vecs]),
                                              'species': np.array(species)})
        logger.info(f'Saved soap desc to npz cache file: {npz_cache_fpath}')

    # reshape 
    n_feats = 0
    n_feats_element_filtered = 0
    if average == 'off' and specific_elements is not None:

        for i, (feats, config) in enumerate(zip(config_soap_vecs, configs)):
            
            feats = feats.reshape(-1, 1) if feats.ndim == 1 else feats
            n_feats += len(feats)

            # TODO: before concatenating into an np array, get the soap vectors of specific_atom_ids only if specified
            if average == 'off' and specific_atom_ids is not None:
                ...
                pass

            # filter based on element
            config_species = [element for element in config.get_chemical_symbols()]
            soap_with_elements = np.column_stack((feats, np.array(config_species)))
            feats = feats[np.isin(soap_with_elements[:, -1], list(specific_elements))]
            n_feats_element_filtered += len(feats)
            config_soap_vecs[i] = feats
        
        logger.info(f'No. Feats: {n_feats}')
        logger.info(f'No. Feats After Element Filtering: {n_feats_element_filtered}')

    if compress_dim is not None:

        logger.info(f"Compressing to {compress_dim} dimensions with PCA...")
        soap_mat = np.concatenate(config_soap_vecs, axis=0)
        soap_mat = run_pca(soap_mat, n_components=compress_dim)
        logger.info(f"SOAP Mat Shape after PCA: {soap_mat.shape}")

        # re-construct list of config soap features after PCA
        config_soap_vecs, start = [], 0
        for config in configs:
            config_soap_vecs.append(soap_mat[start:start + len(config)])
            start += len(config)

    if average == 'off':
        return np.concatenate(config_soap_vecs, axis=0) if flatten_output else config_soap_vecs
    else:
        return np.array(config_soap_vecs) if flatten_output else config_soap_vecs