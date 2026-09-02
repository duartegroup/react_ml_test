"""
Train utils for training analysis, callbacks etc.
"""

import math
import random
from typing import Literal

import ase
import numpy as np
import torch

from react_ml_test_public.helpers.io import flexible_ase_read, get_name_from_path
from react_ml_test_public.helpers.utils import unflatten_array_by_lengths


def WeightedEnergyForceMSELoss(configs: list[ase.Atoms],
                               ref_energies: np.ndarray, 
                               pred_energies: np.ndarray, 
                               ref_forces: np.ndarray, 
                               pred_forces: np.ndarray,
                               energy_weight: float = 1.0, 
                               forces_weight: float = 10.0,
                               return_per_config: bool = False):
    num_atoms = np.array([len(config) for config in configs])
    if return_per_config:
        loss_energy = np.square((ref_energies - pred_energies) / num_atoms) 
        print("Loss Energy Shape: ", loss_energy.shape)
        loss_forces_per_config = unflatten_array_by_lengths(np.square(ref_forces - pred_forces), [len(config) for config in configs])
        loss_forces = np.array([np.mean(config_loss_forces) for config_loss_forces in loss_forces_per_config])
        print("Loss Forces Shape: ", loss_forces.shape)
    else:
        loss_energy = np.mean(np.square((ref_energies - pred_energies) / num_atoms))    
        loss_forces = np.mean(np.square(ref_forces - pred_forces))   
    return energy_weight * loss_energy + forces_weight * loss_forces


def train_test_split(input_ds_fps: list[str],
                     train_frac: float,
                     valid_frac: float,
                     test_frac: float,
                     sample_mode: Literal["ordered", "random"],
                     seed: int = None,
                     save_individual_subsets: bool = True,
                     combined_save_name: str = None,
                     out_dir: str = '.',
                     ):
    """
    Create a train:validation:test split of the input dataset. Allows the combining of multiple sub-data sets
    if they are specified, to be combined into one training, validation and test dataset of the combined subsets.
    Otherwise save train / validation / test sets as seperate files for each input sub-set.

    The different input datasets correspond to 'stratified' subsets of different types of input data.

    :param input_ds_fps: list of input sub-training xyz files (or one xyz file if the data isn't grouped)
    :param train_frac: train set fraction for each input ds
    :param valid_frac: validation set fraction for each input ds
    :param test_frac: test set fraction for each input ds
    :param sample_mode: either 'ordered' for splitting the input ds from beginning to end in order, or
            'random' for shuffling the dataset with the given seed before sampling
    :param seed: random seed for sampling
    :param save_individual_subsets: bool, if true we create separate training / valid / test sets for
            each subset
    """

    # check sample fractions do not exceed dataset
    assert train_frac + valid_frac + test_frac <= 1.0, f"Train:valid:test fractions must sum to 1.0, but have values: {train_frac}:{valid_frac}:{test_frac} with sum: {sum([train_frac, valid_frac, test_frac])}"

    random.seed(seed)

    all_train_configs, all_valid_configs, all_test_configs = [], [], []
    for input_ds_fp in input_ds_fps:

        # create config set and load configs
        total_config_set = flexible_ase_read(input_ds_fp)
        print(f"{input_ds_fp} has {len(total_config_set)} configs...")

        if sample_mode == "random":
            random.shuffle(total_config_set)

        elif sample_mode == "ordered":
            pass
        else:
            print("Please select one of 'ordered' or 'random' for the mode parameter.")
            return None

        # get subset fractions
        n_samples = len(total_config_set)
        train_end = int(math.ceil(n_samples * train_frac))
        valid_end = train_end + int(math.ceil(n_samples * valid_frac))

        # split config set into subsets and add to total lists
        train_configs = total_config_set[:train_end]
        valid_configs = total_config_set[train_end:valid_end] if valid_frac > 0.0 else []
        test_configs = total_config_set[valid_end:] if test_frac > 0.0 else []

        # save config sets to new fps
        if save_individual_subsets:
            subsets = [train_configs, valid_configs, test_configs]
            subset_names = ['train', 'valid', 'test']
            input_ds_name = get_name_from_path(input_ds_fp)
            for subset, subset_name in zip(subsets, subset_names):
                if combined_save_name is not None and len(subset) > 0:
                    save_fp = f"{out_dir}/{input_ds_name}_{subset_name}.xyz"
                    print(f"Saving {save_fp}...")
                    ase.io.write(save_fp, subset, format='extxyz')

            print(f"Num Train Configs for {input_ds_fp}: ", len(train_configs))
            print(f"Num Valid Configs for {input_ds_fp}: ", len(valid_configs))
            print(f"Num Test Configs for {input_ds_fp}: ", len(test_configs))

        all_train_configs.extend(train_configs)
        all_valid_configs.extend(valid_configs)
        all_test_configs.extend(test_configs)

    print("Num Train Configs: ", len(all_train_configs))
    print("Num Valid Configs: ", len(all_valid_configs))
    print("Num Test Configs: ", len(all_test_configs))

    # shuffle each final combined set of configs
    random.shuffle(all_train_configs)
    random.shuffle(all_valid_configs)
    random.shuffle(all_test_configs)

    # save config sets to new fps
    subsets = [all_train_configs, all_valid_configs, all_test_configs]
    subset_names = ['train', 'valid', 'test']
    for subset, subset_name in zip(subsets, subset_names):
        if combined_save_name is not None and len(subset) > 0:
            save_fp = f"{out_dir}/{subset_name}/{combined_save_name}_{subset_name}.xyz"
            print(f"Saving {save_fp}...")
            ase.io.write(save_fp, subset, format='extxyz')
