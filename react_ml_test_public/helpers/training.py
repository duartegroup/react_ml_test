"""
Helpers for managing training data for fine-tuning and other learning experiments.
"""
import math
import os
import random
import shutil
import subprocess
from typing import Callable, Union
from react_ml_test_public.helpers.io import flexible_ase_read, get_name_from_path, get_dir_from_path
from react_ml_test_public.helpers.model_eval import get_test_and_pred_data
from react_ml_test_public.helpers.settings import MODEL_DIR, MACEArchitectSettings, MACETrainSettings
from react_ml_test_public.helpers.logging import logger
import ase
from react_ml_test_public.helpers.train_utils import WeightedEnergyForceMSELoss

def train_mace(train_fpath: str,
               train_settings: MACETrainSettings,
               model_save_name: str,
               valid_fpath: str = None,
               pt_train_fpath: str = None,
               pt_valid_fpath: str = None,
               model_architect_settings: MACEArchitectSettings = None,
               parent_model_fpath: str = None,
               dtype: str = 'float32'
               ) -> str:
    """
    Runs training of a MACE model on a given training dataset.

    :param train_fpath: fpath of the training data .xyz file
    :param train_settings: settings for training
    :param model_save_name: name of the final saved model
    :param pt_train_fpath: fpath of the training replay dataset
    :param pt_valid_fpath: fpath of the validation replay dataset
    :param valid_fpath: validation .xyz file fpath
    :param model_architect_settings: settings for the model architecture
    :param parent_model_fpath: fpath of the foudation model

    :return: name of the final saved model
    """

    pt_train_str = f" with multihead training on {pt_train_fpath} " if pt_train_fpath is not None else ""
    print(f"Training {model_save_name} model on {train_fpath}{pt_train_str}...")

    # run mace training script as sub-process
    mace_train_script = f"""
    mace_run_train \
    --name="{model_save_name}" \
    --train_file={train_fpath} \
    --loss="weighted" \
    --E0s="{train_settings.E0s}" \
    --seed={train_settings.seed} \
    --energy_weight={train_settings.energy_weight} \
    --forces_weight={train_settings.forces_weight} \
    --lr={train_settings.lr} \
    --scaling="rms_forces_scaling" \
    --batch_size={train_settings.batch_size} \
    --max_num_epochs={train_settings.max_num_epochs} \
    --amsgrad \
    --energy_key="{train_settings.energy_key}" \
    --forces_key="{train_settings.forces_key}" \
    --device={train_settings.device} \
    --error_table="PerAtomRMSE" \
    --wandb \
    --wandb_project={train_settings.wandb_project} \
    --wandb_name="{model_save_name}" \
    --enable_cueq={train_settings.enable_cueq} \
    --default_dtype={train_settings.dtype} \
    """

    if parent_model_fpath is None:
        # train from scratch
        mace_train_script += f"""--model={model_architect_settings.model_type} \
        --num_interactions={model_architect_settings.num_interactions} \
        --num_channels={model_architect_settings.num_channels} \
        --correlation={model_architect_settings.correlation} \
        --max_L={model_architect_settings.max_L} \
        --r_max={model_architect_settings.r_max} \
        """
    else:
        # add parent model for fine-tuning
        mace_train_script += f"--foundation_model='{parent_model_fpath}'    "

    # add multihead fine-tuning parameters and replay dataset fpath
    if train_settings.multiheads_finetuning:
        mace_train_script += f"--multiheads_finetuning=True    "
        if train_settings.heads != "":
            mace_train_script += f"--heads='{train_settings.heads}'  "    #
            mace_train_script += f"--atomic_numbers={train_settings.atomic_numbers}   "

        else:
            mace_train_script += f"--num_samples_pt={train_settings.num_samples_pt}   "
            mace_train_script += f"--subselect_pt={train_settings.subselect_pt}   "
            mace_train_script += f"--weight_pt={train_settings.weight_pt}   "
            mace_train_script += f"--atomic_numbers={train_settings.atomic_numbers}   "
            mace_train_script += f"--force_mh_ft_lr={train_settings.force_mh_ft_lr}   "
            mace_train_script += f"--pt_train_file={pt_train_fpath}   "
            mace_train_script += f"--E0s='{train_settings.E0s}' "
    else:
        mace_train_script += f"--multiheads_finetuning=False    "

    # add lora params
    if train_settings.lora:
        mace_train_script += f"--lora=True    "
        mace_train_script += f"--lora_rank={train_settings.lora_rank}   "
        mace_train_script += f"--lora_alpha={train_settings.lora_alpha}   "

    # add fpath parameters
    if valid_fpath is not None:
        if not os.path.exists(valid_fpath):
            logger.warning(f"Validation file {valid_fpath} not found, reverting to default 10% by MACE...")
        else:
            mace_train_script += f"--valid_file={valid_fpath} "

    # add optional parameters
    if train_settings.ema:
        mace_train_script += f"--ema    "
        mace_train_script += f"--ema_decay={train_settings.ema_decay}   "

    if train_settings.swa:
        mace_train_script += f"--swa    "
        mace_train_script += f"--start_swa={train_settings.start_swa}   "
    
    if train_settings.patience is not None:
        mace_train_script += f"--patience={train_settings.patience} "

        if train_settings.patience_warmup > 0:
            mace_train_script += f"--patience_warmup={train_settings.patience_warmup} "

    if train_settings.valid_batch_size is not None:
        mace_train_script += f"--valid_batch_size={train_settings.valid_batch_size} "

    if train_settings.scheduler_patience is not None:
        mace_train_script += f"--scheduler_patience={train_settings.scheduler_patience} "

    if train_settings.restart_latest:
        mace_train_script += f"--restart_latest "

    print("MACE Train Script: ", mace_train_script)

    # run training as sub-process
    process = subprocess.Popen(['bash', '-c', mace_train_script],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                encoding='utf-8',
                                errors='replace')

    # read and print output line by line
    for line in process.stdout:
        print(line, end="")  # Print each line as it comes in, avoiding extra newlines

    # wait for the process to complete and check for errors
    process.stdout.close()
    return_code = process.wait()

    # check return status
    if return_code != 0:
        print("Error occurred:")
        for error_line in process.stderr:
            print(error_line, end="")
        process.stderr.close()


def run_iterative_train(train_fpath: str,
                        valid_fpath: str,
                        mace_train_settings: MACETrainSettings,
                        splits: Union[list[float], list[int]] = None,
                        single_sample_indices: Union[list[int]] = None,
                        specified_subset_fpaths: list[str] = None,
                        rep_nums: list = None,
                        model_architect_settings: MACEArchitectSettings = None,
                        parent_model_fpath: str = None,
                        model_save_name: str = None,
                        batch_size_scheduler: Callable = None,
                        stratify_splits: bool = False,
                        rep_seeds: list[int] = None):
    """
    Runs independent training experiments over increasing portions of an input dataset with a given order.
    Models trained on each portion of the dataset are trained independently.

    Parameters:
        train_fpath:
        valid_fpath:
        splits: List[float] or List[int], a list of dataset fractions (floats) or a list of sample sizes (ints)
                for the subsampling of the input training set.
    """

    assert parent_model_fpath is not None or model_save_name is not None, "Please specify either a model_save_name for training or a parent_model_fpath for fine-tuning..."
    assert (splits is not None) ^ \
            (single_sample_indices is not None) ^ \
            (specified_subset_fpaths is not None), "Please specify either splits, single sample indicies, or specified_subset_fpaths, but not both..."

    if rep_nums is not None and len(rep_nums) > 1:
        logger.warning('WARNING: The seed will be overridden for each rep...')
    seeds = random.sample(range(1, 100 + 1), len(rep_nums) if rep_nums is not None else 1) if rep_seeds is not None else rep_seeds
    logger.info(f'Using Rep Seeds: {seeds}')

    subset_fpaths = []
    subset_sizes = []

    if specified_subset_fpaths is not None:
        logger.info(f'Overriding train_fpath splitting / sampling and using {len(specified_subset_fpaths)} specified sub_fpaths:')
        for subset_fpath in specified_subset_fpaths:
            subset_fpaths.append(subset_fpath)
            subset_sizes.append(len(flexible_ase_read(subset_fpath)))

    else:
        # load training set
        train_configs = flexible_ase_read(train_fpath)
        train_name = get_name_from_path(train_fpath)
        train_dir = get_dir_from_path(train_fpath)
        logger.info(f'Length of training set: {len(train_configs)}')

        # splits / slices of dataset
        if splits is not None:

            # get configs by config type
            if stratify_splits:
                config_types = set(config.info.get('config_type') for config in train_configs)
                assert len(config_types) > 0, 'No config_type information found in ase file, cannot stratify splits...'
                strat_configs = {config_type: [] for config_type in config_types}
                for config in train_configs:
                    strat_configs[config.info.get('config_type')].append(config)
                
            for split in splits:
                subset_fpath = f"{train_dir}/subsets/{train_name}_{split}_config_strat.xyz"
                n_samples = int(math.ceil(len(train_configs) * split)) if isinstance(split, float) else split
                subset_sizes.append(n_samples)

                if stratify_splits:
                    samples_per_config_type = int(split / float(len(config_types)))
                    print('Samples Per Config Type: ', samples_per_config_type)
                    subset_configs = [config for configs in strat_configs.values() for config in configs[0:samples_per_config_type]]
                    print('Subset Size: ', len(subset_configs))
                else:
                    subset_configs = train_configs[:n_samples]
                ase.io.write(subset_fpath, subset_configs, format='extxyz')
                subset_fpaths.append(subset_fpath)
            print("Subset Sizes: ", subset_sizes)

        # single samples from dataset
        if single_sample_indices is not None:
            for index in single_sample_indices:
                subset_fpath = f"{train_dir}/subsets/{train_name}_single_config_{index}.xyz"
                subset_sizes.append(1)
                ase.io.write(subset_fpath, train_configs[index], format='extxyz')
                subset_fpaths.append(subset_fpath)
            print("Single sample indices: ", single_sample_indices)

    all_model_names = [] if parent_model_fpath is None else [get_name_from_path(parent_model_fpath)]
    

    for subset_fpath, subset_size in zip(subset_fpaths, subset_sizes):
        
        if splits is not None:
            subset_log_txt = f"first {subset_size} configurations"
        elif single_sample_indices is not None:
            subset_log_txt = f"index {subset_fpath.split('_')[-1].split('.')[0]}"
        else:
            subset_log_txt = f"subset fpath: {subset_fpath}"

        if parent_model_fpath is None:
            print(f"Training {model_save_name} on {subset_log_txt} of training set...")
        else:
            print(f"Fine-tuning {parent_model_fpath} on {subset_log_txt} of training set...")

        # update batch size based on dataset if specified
        batch_size = batch_size_scheduler(subset_size) if batch_size_scheduler is not None else mace_train_settings.batch_size
        batch_size = subset_size if batch_size >= subset_size else batch_size 
        modified_mace_train_settings = mace_train_settings._replace(batch_size=batch_size)
        print("Using batch size: ", mace_train_settings.batch_size)

        model_save_name = model_save_name if model_save_name is not None else get_name_from_path(parent_model_fpath)
        for i, rep in enumerate(rep_nums if rep_nums is not None else [1]):

            # update seed (unless we have only one rep, then just use the specified seed)
            print("Running rep: ", rep)
            if rep_nums is not None and len(rep_nums) > 1:
                modified_mace_train_settings = modified_mace_train_settings._replace(seed=seeds[i])
            print("Using seed: ", modified_mace_train_settings.seed)

            fmt_model_save_name = model_save_name + f"_rep_{rep}" if rep_nums is not None else model_save_name

            # add train name to model
            train_fname = get_name_from_path(subset_fpath)
            fmt_model_save_name = f"{fmt_model_save_name}_{train_fname}"
            if parent_model_fpath is not None:
                fmt_model_save_name += '_fine_tuned'

            # run training
            train_mace(subset_fpath,
                       modified_mace_train_settings,
                       fmt_model_save_name,
                       valid_fpath=valid_fpath,
                       model_architect_settings=model_architect_settings,
                       parent_model_fpath=parent_model_fpath)

            # move output model file to models dir, and rename with replicate number at end (lazy I know)
            trained_model_name_no_rep = fmt_model_save_name.replace(f"_rep_{rep}", "") if rep_nums is not None else fmt_model_save_name
            rep_str = f"_rep_{rep}" if rep_nums is not None else ""
            shutil.move(f"{fmt_model_save_name}.model", os.path.join(MODEL_DIR, os.path.basename(f"{trained_model_name_no_rep}{rep_str}.model")))
            all_model_names.append(fmt_model_save_name)

            # remove accompanying compiled model file
            try:
                os.remove(f"{fmt_model_save_name}_compiled.model")
            except FileNotFoundError:
                print(f"No need to remove {fmt_model_save_name}_compiled.model, as it doesn't exist")
