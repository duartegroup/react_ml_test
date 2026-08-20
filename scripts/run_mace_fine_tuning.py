"""
Script to fine-tune a pretrained MACE MLIP.
"""
import os
import re
import shutil
import sys

sys.path.append("../")
from react_ml_test.helpers.io import (
    get_dir_from_path,
    get_name_from_path,
    get_xyz_file_len,
    load_isolated_atom_energies,
)
from react_ml_test.helpers.settings import (
    ANTHRACENE_DA_DATA_DIR,
    CPD_MVK_DA_DATA_DIR,
    GUAJADIAL_DA_DATA_DIR,
    MODEL_DIR,
    REACTION_DATA_DIR,
    MACEArchitectSettings,
    MACETrainSettings,
)
from react_ml_test.helpers.training import train_mace
from react_ml_test.scripts.run_test_set_comparison import (
    run_model_comparison_on_test_set,
)

if __name__ == "__main__":

    model_save_names = None
    parent_model_fpaths = None
    model_architect_settings = None
    lora_rank, lora_alpha = None, None
    use_lora = False
    log_lora = False
   
    # ====== SELECT DATA ======

    # --- CPD + MVK DA in water ---

    # train
    train_fpath = f"{CPD_MVK_DA_DATA_DIR}/train/CPD_MVK_DA_train.xyz"

    # valid
    valid_fpath = f"{CPD_MVK_DA_DATA_DIR}/valid/CPD_MVK_DA_valid.xyz" 

    # --- anthracene DA in toluene ---

    # train
    # train_fpath = f"{ANTHRACENE_DA_DATA_DIR}/train/anthracene_DA_train.xyz"    # paper datasets
    # train_fpath = f"{ANTHRACENE_DA_DATA_DIR}/train/anthracene_DA_train_46_config_strat_rep_2_w_1_4_vac_sample.xyz"     # paper 46 config dataset + 1 1,4 configs
    # # # train_fpath = f"{ANTHRACENE_DA_DATA_DIR}/train/anthracene_DA_train_46_config_strat_rep_2_w_1_4_samples.xyz"     # paper 46 config dataset + 2 1,4 configs

    # valid
    # valid_fpath = f"{ANTHRACENE_DA_DATA_DIR}/valid/anthracene_DA_valid.xyz"

    # --- guajadial DA in water ---

    # # train
    # train_fpath = f"{GUAJADIAL_DA_DATA_DIR}/train/guajadial_DA_train.xyz"

    # # valid
    # valid_fpath = f"{GUAJADIAL_DA_DATA_DIR}/valid/guajadial_DA_valid.xyz"

    # --------------------------------

    train_fpaths = [train_fpath]
    system_ft_name = re.sub(r"_rep_\d+", "", get_name_from_path(train_fpath))

    # === SPECIFY PARENT MODEL FPATH ====

    # --- base models for full dataset fine-tuning ---
    # train_type = "standard"

    # # parent_model_fpath, dtypes = f"{MODEL_DIR}/GO-MACE-23.model", ['float32']  # WARNING: requires float32 precision, not float64
    # # parent_model_fpath, dtypes = f"{MODEL_DIR}/MACE-OFF23_medium.model", ['float64']
    # # parent_model_fpath, dtypes = f"{MODEL_DIR}/MACE-MP-0b3_medium.model", ['float64']
    # parent_model_fpath, dtypes = f"{MODEL_DIR}/MACE-omol-0-extra-large-1024.model", ['float64']

    # # add train name to model
    # model_save_name = get_name_from_path(parent_model_fpath) + f"_{system_ft_name}"
    # if parent_model_fpath is not None:
    #     model_save_name += '_fine_tuned'

    # train_fpaths = [train_fpath]
    # model_save_names = [model_save_name]
    # parent_model_fpaths = [parent_model_fpath]

    # rep_nums, seeds = [1], [3]
    # # rep_nums, seeds = [2], [5]
    # # rep_nums, seeds = [3], [7]

    # wandb_project = f"{system_ft_name}_full_ft"

    # --- specific models from iter training (subsets) ---

    train_type = "standard"

    # parent_model_fpath, dtypes = f"{MODEL_DIR}/GO-MACE-23.model", ['float32'] * 3  # WARNING: requires float32 precision, not float64
    # parent_model_fpath, dtypes = f"{MODEL_DIR}/MACE-OFF23_medium.model", ['float64'] * 3
    # parent_model_fpath, dtypes = f"{MODEL_DIR}/MACE-MP-0b3_medium.model", ['float64'] * 3
    parent_model_fpath, dtypes = f"{MODEL_DIR}/MACE-omol-0-extra-large-1024.model", ['float64'] * 3

    # system_dir, ft_train_name, sample_str, wandb_project = CPD_MVK_DA_DATA_DIR, "CPD_MVK_DA_train", "", "CPD_MVK_DA_iter_train"
    system_dir, ft_train_name, sample_str, wandb_project = CPD_MVK_DA_DATA_DIR, "CPD_MVK_DA_train", "ts_gauss_", "CPD_MVK_DA_iter_train"  # ts gaussian sampling
    # system_dir, ft_train_name, sample_str, wandb_project = ANTHRACENE_DA_DATA_DIR, "anthracene_DA_train", "", "anthracene_DA_iter_train_2"
    # system_dir, ft_train_name, sample_str, wandb_project = GUAJADIAL_DA_DATA_DIR, "guajadial_DA_train", "", "guajadial_DA_iter_train"
    # system_dir, ft_train_name, sample_str, wandb_project = GUAJADIAL_DA_DATA_DIR, "guajadial_DA_train", "ts_gauss_", "guajadial_DA_iter_train"

    # choose n samples + replicates
    n, rep_nums, seeds = 42, [1, 2, 3], [3, 5, 7]
    # n, rep_nums, seeds = 46, [1, 2, 3], [3, 5, 7]
    # n, rep_nums, seeds = 46, [2], [5]
    # n, rep_nums, seeds = 46, [3], [7]
    # n, rep_nums, seeds = 50, [2], [5]
    # n, rep_nums, seeds = 50, [3], [7]
    # n, rep_nums, seeds = 50, [1, 2, 3], [3, 5, 7]
    # n, rep_nums, seeds = 42, [2, 3], [5, 7]

    # LORA options
    # use_lora = False
    use_lora, log_lora = True, True
    lora_rank, lora_alpha = 1, 0.25
    # lora_rank, lora_alpha = 2, 0.5
    # lora_rank, lora_alpha = 4, 1.0

    # get all relevant paths / models
    lora_str = '_lora' if use_lora else ''
    train_fpaths = [f"{system_dir}/train/subsets/{ft_train_name}_{n}_config_strat_{sample_str}rep_{rep_num}.xyz" for rep_num in rep_nums] 
    system_ft_names = [re.sub(r"_rep_\d+", "", get_name_from_path(train_fpath)) for train_fpath in train_fpaths]
    model_save_names = [get_name_from_path(parent_model_fpath) + f"_{ft_name}" + '_fine_tuned' + lora_str for ft_name in system_ft_names]
    parent_model_fpaths = [parent_model_fpath] * len(rep_nums)

    # --- models for second stage energy training ---
    # train_type = "energy"

    # # n, rep_nums, seeds, wandb_project = 42, [1], [3], "CPD_MVK_DA_energy_train"
    # # n, rep_nums, seeds, wandb_project = 42, [2], [5], "CPD_MVK_DA_energy_train"
    # # n, rep_nums, seeds, wandb_project = 42, [3], [7], "CPD_MVK_DA_energy_train"
    # # n, rep_nums, seeds, wandb_project = 42, [1, 2, 3], [3, 5, 7], "CPD_MVK_DA_energy_train"
    # # n, rep_nums, seeds, wandb_project = 42, [2, 3], [5, 7], "CPD_MVK_DA_energy_train"

    # # n, rep_nums, seeds, wandb_project = 46, [1], [3], "anthracene_DA_energy_train"
    # # n, rep_nums, seeds, wandb_project = 46, [2], [5], "anthracene_DA_energy_train"
    # # n, rep_nums, seeds, wandb_project = 46, [1, 2, 3], [3, 5, 7], "anthracene_DA_energy_train"
    # # n, rep_nums, seeds, wandb_project = 46, [3], [7], "anthracene_DA_energy_train"

    # n,  rep_nums, seeds, wandb_project = 50, [1, 2, 3], [3, 5, 7], "guajadial_DA_energy_train"
    # # n,  rep_nums, seeds, wandb_project = 50, [2, 3], [5, 7], "guajadial_DA_energy_train"
    # # n,  rep_nums, seeds, wandb_project = 50, [3], [7], "guajadial_DA_energy_train"

    # # base_model, from_scratch_name, dtypes = "GO-MACE-23", "GO-MACE-23", ['float32'] * 3
    # # base_model, from_scratch_name, dtypes = "MACE-OFF23_medium", "MACE-OFF23_medium", ['float64'] * 3
    # # base_model, from_scratch_name, dtypes = "MACE-MP-0b3_medium", "MACE_MP0", ['float64'] * 3
    # base_model, dtypes = "MACE-omol-0-extra-large-1024", ['float64'] * 3

    # specific num configs fine-tuned
    # sample_str = ""
    # sample_str = "ts_gauss_"    
    # parent_model_fpaths = [f"{MODEL_DIR}/{base_model}_{system_ft_name}_{n}_config_strat_{sample_str}fine_tuned_rep_{i}.model" for i in rep_nums]
    # train_fpaths = [f"{get_dir_from_path(train_fpath)}/subsets/{system_ft_name}_{n}_config_strat_{sample_str}rep_{i}.xyz" for i in rep_nums]
    
    # or specific num configs from scratch
    # sample_str = ""
    # # sample_str = "ts_gauss_"    
    # parent_model_fpaths = [f"{MODEL_DIR}/{from_scratch_name}_from_scratch_{system_ft_name}_{n}_config_strat_{sample_str}rep_{i}.model" for i in rep_nums]
    # train_fpaths = [f"{get_dir_from_path(train_fpath)}/subsets/{system_ft_name}_{n}_config_strat_{sample_str}rep_{i}.xyz" for i in rep_nums]

    # or fully fine-tuned
    # rep_nums, seeds = [1], [3]
    # parent_model_fpaths = [f"{MODEL_DIR}/{base_model}_{system_ft_name}_fine_tuned_rep_{rep_nums[0]}.model"]
    # train_fpaths = [train_fpath]

    # or anthracene extra config fine-tuning energy training
    # parent_model_fpaths = [
    #                     #    f"{MODEL_DIR}/anthracene_DA_GO46_anthracene_DA_train_46_config_strat_w_1_4_vac_sample_fine_tuned_rep_2.model",
    #                        f"{MODEL_DIR}/anthracene_DA_OFF46_anthracene_DA_train_46_config_strat_w_1_4_vac_sample_fine_tuned_rep_2.model",
    #                        f"{MODEL_DIR}/anthracene_DA_MP46_anthracene_DA_train_46_config_strat_w_1_4_vac_sample_fine_tuned_rep_2.model"]
    # # dtypes = ['float32', 'float64', 'float64']
    # dtypes = ['float64', 'float64']
    # rep_nums, seeds = [2] * 3, [5] * 3
    # train_fpaths = [f"{ANTHRACENE_DA_DATA_DIR}/train/anthracene_DA_train_46_config_strat_rep_2_w_1_4_vac_sample.xyz"] * 3
    # system_ft_name = re.sub(r"_rep_\d+", "", get_name_from_path(train_fpaths[0]))
    # wandb_project = f"{system_ft_name}_full_ft"

    # or fully trained from scratch
    # base_model, dtypes = "GO-MACE-23", ['float32']
    # # base_model, dtypes = "MACE-OFF23_medium", ['float64']
    # # base_model, dtypes = "MACE_MP0", ['float64']
    # rep_nums, seeds = [1], [3]
    # parent_model_fpaths = [f"{MODEL_DIR}/{base_model}_{system_ft_name}_from_scratch_rep_{rep_nums[0]}.model"]
    # train_fpaths = [train_fpath]

    # model_save_names = [re.sub(r"_rep_\d+", "", get_name_from_path(fpath)) for fpath in parent_model_fpaths]

    # --- system transfer ---
    # train_type = "standard"

    # # select model fine-tuned on another system

    # # parent_model_fpaths, transfer_name, dtypes = [f"{MODEL_DIR}/GO-MACE-23_CPD_MVK_DA_train_42_config_strat_fine_tuned_energy_rep_1.model"], "CPD_MVK_DA_GO42", ['float64']
    # # parent_model_fpaths, transfer_name, dtypes = [f"{MODEL_DIR}/MACE-OFF23_medium_CPD_MVK_DA_train_42_config_strat_fine_tuned_energy_rep_1.model"], "CPD_MVK_DA_OFF42", ['float64']
    # # parent_model_fpaths, transfer_name, dtypes = [f"{MODEL_DIR}/MACE-MP-0b3_medium_CPD_MVK_DA_train_42_config_strat_fine_tuned_energy_rep_1.model"], "CPD_MVK_DA_MP42", ['float64']

    # # anthracene + 1 TS
    # # parent_model_fpaths, transfer_name, dtypes = [f"{MODEL_DIR}/GO-MACE-23_anthracene_DA_train_46_config_strat_fine_tuned_energy_rep_2.model"], "anthracene_DA_GO46", ['float64']
    # # parent_model_fpaths, transfer_name, dtypes = [f"{MODEL_DIR}/MACE-OFF23_medium_anthracene_DA_train_46_config_strat_fine_tuned_energy_rep_2.model"], "anthracene_DA_OFF46", ['float64']
    # # parent_model_fpaths, transfer_name, dtypes = [f"{MODEL_DIR}/MACE-MP-0b3_medium_anthracene_DA_train_46_config_strat_fine_tuned_energy_rep_2.model"], "anthracene_DA_MP46", ['float64']

    # # select target and specific subset in target dataset
    # n, rep_nums, seeds, wandb_project = 46, [2], [5], "anthracene_DA_iter_train"
    # train_fpaths = [f"{ANTHRACENE_DA_DATA_DIR}/train/anthracene_DA_train_46_config_strat_rep_2_w_1_4_vac_sample.xyz"]

    # # n, rep_nums, seeds, wandb_project = 50, [1], [3], "guajadial_DA_iter_train"
    # # train_fpaths = [f"{get_dir_from_path(train_fpath)}/subsets/{system_ft_name}_{n}_config_strat_rep_{rep_nums[0]}.xyz"]

    # system_ft_name = re.sub(r"_rep_\d+", "", get_name_from_path(train_fpaths[0]))
    # wandb_project = f"{system_ft_name}_full_ft"
    # model_save_names = [transfer_name + f"_{system_ft_name}" + '_fine_tuned']
    # print("Model save name: ", model_save_names[0])

    # ===== OR: SPECIFY MODEL ARCHITECTURE FOR FROM SCRATCH TRAINING ======
    # train_type = "standard"

    # # # rep_nums, seeds = [1, 2, 3], [3, 5, 7]
    # rep_nums, seeds = [1], [3]
    # # # rep_nums, seeds = [2], [5]
    # # # rep_nums, seeds = [3], [7]

    # wandb_project = f"{system_ft_name}_full_ft"

    # # --- GO-MACE-23 from scratch ---
    # # model_save_names, dtypes = [f"GO-MACE-23_{system_ft_name}_from_scratch"], ['float32']
    # # model_architect_settings = MACEArchitectSettings(model_type="ScaleShiftMACE",
    # #                                                  num_interactions=2,
    # #                                                  num_channels=128,        
    # #                                                  correlation=3,
    # #                                                  max_L=1,            
    # #                                                  r_max=3.7,
    # #                                                  )
                                                     
    # # --- MACE-OFF23 from scratch ---
    # model_save_names, dtypes = [f"MACE-OFF23_medium_{system_ft_name}_from_scratch"], ['float64']
    # model_architect_settings = MACEArchitectSettings(model_type="ScaleShiftMACE",
    #                                                  num_interactions=2,
    #                                                  num_channels=128,        
    #                                                  correlation=3,
    #                                                  max_L=1,                
    #                                                  r_max=5.0,
    #                                                 )

    # # --- MACE-MP-0 from scratch ---
    # # model_save_names, dtypes = [f"MACE_MP0_{system_ft_name}_from_scratch"], ['float64']
    # # model_architect_settings = MACEArchitectSettings(model_type="ScaleShiftMACE",
    # #                                                  num_interactions=2,
    # #                                                  num_channels=128,        
    # #                                                  correlation=3,
    # #                                                  max_L=1,                
    # #                                                  r_max=6.0,
    # #                                                  )

    # ===== SPECIFY E0s (ELSE AVERAGE) =====
    E0s_fpath = "data/isolated_atoms/wB97M-D3BJ_def2-TZVP_isolated_atom_energies_CHO.xyz"
    E0s = load_isolated_atom_energies(E0s_fpath)

    # modifiable parameters 
    lr = 0.001  # default
    batch_size = 2 # 2
    valid_batch_size = 2
    log_lr = False
    log_batch_size = True

    # === SPECIFY FINE-TUNING PARAMETERS HERE ===

    if train_type == "standard":
        # default fine-tuning params
        train_settings = MACETrainSettings(energy_weight=1.0,
                                           forces_weight=10.0,
                                           E0s=E0s if E0s else "average",
                                           lr=lr,
                                           batch_size=min(max([get_xyz_file_len(train_fpath) for train_fpath in train_fpaths]), batch_size),
                                           max_num_epochs=500,  # 500 default
                                           energy_key="REF_energy",
                                           forces_key="REF_forces",
                                           ema=True,
                                           ema_decay=0.99,
                                           scheduler_patience=None,     # None
                                           patience=20,                 # 20
                                           patience_warmup=20,          # default: 20
                                           valid_batch_size=valid_batch_size,
                                           restart_latest=False,
                                           device="cuda",
                                           enable_cueq=False,
                                           wandb_project=wandb_project,
                                           lora=use_lora,
                                           lora_rank=lora_rank,
                                           lora_alpha=lora_alpha)
    elif train_type == "energy":
        # --- second round energy training ---
        energy_weight = 10.0
        model_save_names = [model_save_name + "_energy" for model_save_name in model_save_names]
        train_settings = MACETrainSettings(energy_weight=energy_weight,
                                           forces_weight=1.0,
                                           E0s=E0s if E0s else "average",
                                           lr=lr,
                                           batch_size=min(get_xyz_file_len(train_fpath), batch_size),
                                           max_num_epochs=500,          # default: 500
                                           energy_key="REF_energy",
                                           forces_key="REF_forces",
                                           ema=True,
                                           ema_decay=0.99,
                                           scheduler_patience=None,       # default: None
                                           patience=20, # 20,                 # default: 20
                                           patience_warmup=40, # 40,          # default: 20
                                           valid_batch_size=valid_batch_size,
                                           restart_latest=False,
                                           device="cuda",
                                           enable_cueq=False,
                                           wandb_project=wandb_project,
                                           lora=use_lora,
                                           lora_rank=lora_rank,
                                           lora_alpha=lora_alpha)
    else:
        raise NotImplementedError
    
    valid_fname = get_name_from_path(valid_fpath)
    lr_str = str(lr).replace('.', '_')
    for i, rep in enumerate(rep_nums):

        # update seed (unless we have only one rep, then just use the specified seed)
        print("Running rep: ", rep)
        modified_mace_train_settings = train_settings._replace(seed=seeds[i], dtype=dtypes[i] if dtypes is not None else 'float64')
        print("Using seed: ", modified_mace_train_settings.seed)

        model_save_name = model_save_names[i] + f"_rank_{lora_rank}" if log_lora else model_save_names[i]
        model_save_name = model_save_name + f"_lr_{lr_str}" if log_lr else model_save_name
        model_save_name = model_save_name + f"_batch_{batch_size}" if log_batch_size else model_save_name
        model_save_name += f"_rep_{rep}"

        # run training
        train_mace(train_fpaths[i],
                   modified_mace_train_settings,
                   model_save_name,
                   valid_fpath=valid_fpath,
                   model_architect_settings=model_architect_settings,
                   parent_model_fpath=parent_model_fpaths[i] if parent_model_fpaths is not None else None)

        # move output model file to models dir, and rename with replicate number at end (lazy I know)
        trained_model_name_no_rep = model_save_name.replace(f"_rep_{rep}", "")
        shutil.move(f"{model_save_name}.model", os.path.join(MODEL_DIR, os.path.basename(f"{trained_model_name_no_rep}_rep_{rep}.model")))

        # remove accompanying compiled model file
        try:
            os.remove(f"{model_save_name}_compiled.model")
        except FileNotFoundError:
            print(f"No need to remove {model_save_name}_compiled.model, as it doesn't exist")
        
        # remove all checkpoints from checkpoints directory
        # for filepath in glob.glob(os.path.join("checkpoints", f"*{model_save_name}*")):
        #     if os.path.isfile(filepath):
        #         os.remove(filepath)
