"""
Helpers for eval of models.
"""
from typing import Callable, Literal, Tuple, List, TextIO
import numpy as np
from react_ml_test_public.helpers.io import flexible_ase_read, get_name_from_path, load_isolated_atom_energies, load_data_cache, load_model_calculator, save_npz_data_cache
from react_ml_test_public.helpers.settings import EV, EV_PER_AMSTRONG, MEV, MEV_PER_AMSTRONG, MEV_PER_ATOM, NPZ_DIR
from react_ml_test_public.helpers.utils import convert_units, get_ase_props
from react_ml_test_public.helpers.utils import calc_energy_shifts, guess_per_element_mean_energies
from react_ml_test_public.helpers.logging import logger


def get_model_predictions(model_name: str,
                          test_set_fpaths: List[str],
                          include_forces: bool = True,
                          apply_energy_offsets: bool = True,
                          offset_ref_fpath: str = None,
                          pred_isolated_atom_fpath: str = None,
                          custom_xyz_parsers: List[Callable] = None,
                          cuda_device: str = 'cuda:0',
                          pbc_box: List[float] = None,
                          energy_units: Literal[EV, MEV, MEV_PER_ATOM] = MEV_PER_ATOM,
                          force_units: Literal[EV_PER_AMSTRONG, MEV_PER_AMSTRONG] = MEV_PER_AMSTRONG,
                          model_loader: str = None,
                          enable_cueq: bool = False,
                          hdf5_file: TextIO = None) -> list[np.ndarray]:
    """
    Generates predictions from one or more trained models on a given test dataset.

    Arguments:
        model_name (List[str]): The model name used for prediction.
        test_set_fpaths (List[str]): A list of file paths to the test set XYZ files.
        include_forces (bool): Whether to include force predictions. Defaults to True.
        apply_energy_offsets (bool): Whether to apply energy offsets for energy predictions. Defaults to True.
                                     (set to False for relative energy comparisons).
        offset_ref_fpath (str): Path to reference offset file for energy adjustment. Optional.
        pred_isolated_atom_fpath (str): Path to isolated atom predictions for reference. Optional.
        custom_xyz_parsers (List[Callable]): Custom parser functions for reading XYZ files. Optional.
        cuda_device (str): CUDA device identifier, e.g., 'cuda:0'. Defaults to 'cuda:0'.
        pbc_box (List[float]): Dimensions of the periodic boundary condition box, if applicable. Optional.
        energy_units (Literal['meV', 'meV / atom']): Desired output units for energy. Defaults to 'meV / atom'.
        force_units (Literal['meV', 'meV / A']): Desired output units for forces. Defaults to 'meV / A'.
        model_loader (str): Custom model loader (i.e. MACE, ACE, ANI...). Optional.
        enable_cueq (bool): Whether to enable cuequivariance acceleration. Defaults to False.
        hdf5_file (TextIO): An open HDF5 file handle for storing predictions. Optional.

    Returns:
        list[np.ndarray]: a list of predictions by the model for each specified test_set file
    """
    
    input_energy_units = EV
    input_force_units = EV_PER_AMSTRONG

    calculator = load_model_calculator(model_name, 
                                       device=cuda_device, 
                                       model_loader=model_loader, 
                                       enable_cueq=enable_cueq)

    model_pred_energies, model_pred_forces = [], []
    for j, test_set_fpath in enumerate(test_set_fpaths):

        custom_xyz_parser = custom_xyz_parsers[j] if custom_xyz_parsers is not None else None
        configs = flexible_ase_read(test_set_fpath, custom_xyz_parser)
        # test_set_atom_nums = list(set([int(z) for config in configs for z in config.numbers]))

        # get model offsets
        pred_energy_offsets = None
        if apply_energy_offsets:
            if pred_isolated_atom_fpath is not None:
                logger.info(f"Getting prediction offsets from isolated atoms using: {pred_isolated_atom_fpath}...")
                pred_energy_offsets = load_isolated_atom_energies(pred_isolated_atom_fpath)
            elif offset_ref_fpath is not None:
                logger.info(f"Getting model prediction per-element average offsets from {offset_ref_fpath}...")
                pred_energy_offsets = guess_per_element_mean_energies(offset_ref_fpath, 
                                                                    calc=calculator, 
                                                                    calc_name=model_name, 
                                                                    dtype=np.float32, 
                                                                    pbc_box=pbc_box,
                                                                    hdf5_file=hdf5_file
                                                                    )
            else:
                logger.info(f"Getting model prediction per-element average offsets based on test set: {test_set_fpath}...")
                pred_energy_offsets = guess_per_element_mean_energies(test_set_fpath, 
                                                                    calc=calculator, 
                                                                    calc_name=model_name, 
                                                                    dtype=np.float32, 
                                                                    pbc_box=pbc_box,
                                                                    hdf5_file=hdf5_file
                                                                    )
        logger.info(f"Model: {model_name} Per-element means: {pred_energy_offsets}")

        # load cached npz file of energies / forces if it exists
        offset_str = '_no_offset' if not apply_energy_offsets else ''
        npz_cache_path = f"{NPZ_DIR}/{model_name}_{get_name_from_path(test_set_fpath)}_energies_forces{offset_str}.npz"
        cache_data = load_data_cache(npz_cache_path, hdf5_file=hdf5_file)
        if cache_data is not None:
            pred_energies, pred_forces = cache_data['energies'], cache_data['forces']

            # resave if forces are missing
            if pred_forces is None and include_forces:
                pred_forces = get_ase_props(configs, 'forces', calc=calculator, pbc_cell=pbc_box)
                save_npz_data_cache(npz_cache_path, {'energies': pred_energies, 'forces': pred_forces, 
                                                     'energy_unit': EV, 'force_unit': EV_PER_AMSTRONG})
        else:
            logger.info("Calculating model energies and force predictions...")
            pred_energies = get_ase_props(configs, 'energy', calc=calculator, pbc_cell=pbc_box)
            pred_forces = get_ase_props(configs, 'forces', calc=calculator, pbc_cell=pbc_box) if include_forces else None       

            # TODO: refactor this all to allow for un-flattened force arrays         

            # save predictions to npz cache
            save_npz_data_cache(npz_cache_path, {'energies': pred_energies, 'forces': pred_forces, 
                                                 'energy_unit': EV, 'force_unit': EV_PER_AMSTRONG})

        # apply offset if specified
        if pred_energy_offsets is not None:
            config_shifts = calc_energy_shifts(configs, pred_energy_offsets)
            pred_energies = pred_energies - config_shifts

        # apply unit conversions      
        pred_energies = convert_units(pred_energies, input_energy_units, energy_units, n_atoms_list=[len(config) for config in configs])

        if include_forces:
            pred_forces = convert_units(pred_forces, input_force_units, force_units)

        model_pred_energies.append(pred_energies)
        model_pred_forces.append(pred_forces)

    return model_pred_energies, model_pred_forces


def get_test_and_pred_data(model_names: List[str],
                           test_set_fpaths: List[str],
                           include_forces: bool = True,
                           apply_energy_offsets: bool = True,
                           offset_ref_fpath: str = None,
                           test_isolated_atom_fpath: str = None,
                           pred_isolated_atom_fpath: str = None,
                           custom_xyz_parsers: List[Callable] = None,
                           cuda_device: str = 'cuda:0',
                           pbc_box: List[float] = None,
                           energy_units: Literal[MEV, MEV_PER_ATOM] = MEV_PER_ATOM,
                           force_units: Literal[EV_PER_AMSTRONG, MEV_PER_AMSTRONG] = MEV_PER_AMSTRONG,
                           model_loaders: List[str] = None,
                           enable_cueq: bool = False,
                           hdf5_file: TextIO = None,
                           config_type_filter: str = None) -> Tuple[list, list, dict, dict]:
    """
    Get the test and predicted energies / forces for a set of test sets with a set of models.

    Arguments:
        model_names (List[str]): The list of model names to use for prediction.
        test_set_fpaths (List[str]): A list of file paths to the test set XYZ files.
        include_forces (bool): Whether to include force predictions. Defaults to True.
        offset_ref_fpath (str): Path to reference offset file for energy adjustment. Optional.
        pred_isolated_atom_fpath (str): Path to isolated atom predictions for reference. Optional.
        custom_xyz_parsers (List[Callable]): Custom parser functions for reading XYZ files. Optional.
        cuda_device (str): CUDA device identifier, e.g., 'cuda:0'. Defaults to 'cuda:0'.
        pbc_box (List[float]): Dimensions of the periodic boundary condition box, if applicable. Optional.
        energy_units (Literal['meV', 'meV / atom']): Output units for energy. Defaults to 'meV / atom'.
        force_units (Literal['meV', 'meV / A']): Output units for forces. Defaults to 'meV / A'.
        model_loaders (List[str]): Custom model loaders. Optional.
        enable_cueq (bool): Whether to enable cuequivariance acceleration. Defaults to False.
        hdf5_file (TextIO): An open HDF5 file handle for storing predictions. Optional.

    Returns:
        Tuple[list, list, dict, dict]: two lists of test energies / forces and two dicts of per-model pred energies / forces.
    """

    all_test_energies = []
    all_test_forces = []
    all_pred_energies_dict = {}
    all_pred_forces_dict = {}

    # get test set ground truth energies and forces
    for i, test_set_fpath in enumerate(test_set_fpaths):
        custom_xyz_parser = custom_xyz_parsers[i] if custom_xyz_parsers is not None else None

        # get offsets for test set
        test_energy_offsets = None
        if apply_energy_offsets:
            if test_isolated_atom_fpath is not None:
                logger.info(f"Getting test offsets from isolated atoms using: {test_isolated_atom_fpath}...")
                test_energy_offsets = load_isolated_atom_energies(test_isolated_atom_fpath)
            elif offset_ref_fpath is not None:
                logger.info(f"Calculating per-element average ground truth offsets from {offset_ref_fpath}...")
                test_energy_offsets = guess_per_element_mean_energies(offset_ref_fpath, dtype=np.float32, pbc_box=pbc_box, custom_xyz_parser=custom_xyz_parser)
            else:
                logger.info(f"Calculating per-element average ground truth offsets based on test set: {test_set_fpath}...")
                test_energy_offsets = guess_per_element_mean_energies(test_set_fpath, dtype=np.float32, pbc_box=pbc_box, custom_xyz_parser=custom_xyz_parser)

        logger.info(f"True Per-element means for {test_set_fpath}: {test_energy_offsets}")

        configs = flexible_ase_read(test_set_fpath, custom_xyz_parser)

        test_energies = get_ase_props(configs, 'energy')
        test_forces = get_ase_props(configs, 'forces') if include_forces else None

        if test_energy_offsets is not None:
            config_shifts = calc_energy_shifts(configs, test_energy_offsets)
            test_energies = test_energies - config_shifts

        test_energies = convert_units(test_energies, EV, energy_units, n_atoms_list=[len(config) for config in configs])

        if include_forces:
            test_forces = convert_units(test_forces, EV_PER_AMSTRONG, force_units)

        all_test_energies.append(test_energies)
        all_test_forces.append(test_forces)

    # run inference on test sets:
    for i, model_name in enumerate(model_names):

        model_loader = model_loaders[i] if model_loaders is not None else None

        model_pred_energies, model_pred_forces = get_model_predictions(model_name,
                                                                       test_set_fpaths,
                                                                       include_forces=include_forces,
                                                                       apply_energy_offsets=apply_energy_offsets,
                                                                       offset_ref_fpath=offset_ref_fpath,
                                                                       pred_isolated_atom_fpath=pred_isolated_atom_fpath,
                                                                       custom_xyz_parsers=custom_xyz_parsers,
                                                                       cuda_device=cuda_device,
                                                                       pbc_box=pbc_box,
                                                                       energy_units=energy_units,
                                                                       force_units=force_units,
                                                                       model_loader=model_loader,
                                                                       enable_cueq=enable_cueq,
                                                                       hdf5_file=hdf5_file)

        # add predictions to dictionary for each model
        all_pred_energies_dict[model_name] = model_pred_energies
        all_pred_forces_dict[model_name] = model_pred_forces

    return all_test_energies, all_test_forces, all_pred_energies_dict, all_pred_forces_dict
