"""
Script to fine-tune a reactive MLP using existing model weights.
"""
import json
import os
import random
import shutil
import sys

import ase

sys.path.append("../")
import sys

from ase.io import write
from react_ml_test.helpers.training_justine import train_mace
from sklearn.model_selection import train_test_split

from react_ml_test.helpers.io import (
    get_dir_from_path,
    get_name_from_path,
    get_xyz_file_len,
    load_isolated_atom_energies,
)
from react_ml_test.helpers.sampling import sample_dataset
from react_ml_test.helpers.settings import (
    ANTHRACENE_DA_DATA_DIR,
    CPD_MVK_DA_DATA_DIR,
    DATASETS_DIR,
    MODEL_DIR,
    REACTION_DATA_DIR,
    REPLAY_DATASET_DIR,
    MACETrainSettings,
)
from react_ml_test.helpers.train_utils import train_test_split
from react_ml_test.scripts.run_test_set_comparison import (
    run_model_comparison_on_test_set,
)

# ==== function that will sample the replay dataset to create the training set ====

def create_datasets_frac(dataset_fpath, train_fpath, valid_fpath, train_frac, valid_frac, seed):
    total_config_set = flexible_ase_read(input_ds_fp)
    n_samples = len(total_config_set)
    n_train = int(math.ceil(n_samples * train_frac))
    n_valid = int(math.ceil(n_samples * valid_frac))
    structures = []
    with open(dataset_fpath, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line=line.strip()
            if line == "":
                continue
            try :
                natoms = int(line)
            except ValueError:
                raise ValueError(f"Invalid line for natoms: {line}")
            header = f.readline()
            atoms = [f.readline() for _ in range(natoms)]
            block = [line + "\n", header] + atoms
            structures.append(block)
    total = len(structures)

    if n_train + n_valid > total:
        raise ValueError(f"Dataset trop petit ({total}) pour n_train + n_val = {n_train + n_valid}")

    random.seed(seed)
    random.shuffle(structures)
    train = structures[:n_train]
    val = structures[n_train:n_train+n_valid]

    with open(train_fpath, "w") as f:
        for s in train:
            f.writelines(s)

    with open(valid_fpath, "w") as f:
        for s in val:
            f.writelines(s)
    print(f"Total structures: {total}")
    print(f"Train: {len(train)}")
    print(f"Validation: {len(val)}")

def create_datasets(dataset_fpath, train_fpath, valid_fpath, n_train, n_valid, allowed_species, seed):
    structures = []
    with open(dataset_fpath, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line=line.strip()
            if line == "":
                continue
            try :
                natoms = int(line)
            except ValueError:
                raise ValueError(f"Invalid line for natoms: {line}")
            header = f.readline()
            atoms = [f.readline() for _ in range(natoms)]

            species_in_config = set()
            for atom_line in atoms:
                species = atom_line.split()[0]
                species_in_config.add(species)
            if species_in_config.issubset(allowed_species):
                block = [line + "\n", header] + atoms
                structures.append(block)
    total = len(structures)

    if n_train + n_valid > total:
        raise ValueError(f"Dataset trop petit ({total}) pour n_train + n_val = {n_train + n_valid}")

    random.seed(seed)
    random.shuffle(structures)
    train = structures[:n_train]
    val = structures[n_train:n_train+n_valid]

    with open(train_fpath, "w") as f:
        for s in train:
            f.writelines(s)

    with open(valid_fpath, "w") as f:
        for s in val:
            f.writelines(s)
    print(f"Total structures: {total}")
    print(f"Train: {len(train)}")
    print(f"Validation: {len(val)}")


if __name__ == "__main__":

    model_save_name = None

    # ====== SELECT DATA ======
    # --- endo DA in water ---
    # -- train -- 
    # train_fpath = f"{CPD_MVK_DA_DATA_DIR}/train/CPD_MVK_DA_train.xyz"  # vac / solv only
    train_fpath = f"{CPD_MVK_DA_DATA_DIR}/train/CPD_MVK_DA_train_42_config_strat_rep_2.xyz"
    # train_fpath, wandb_project = f"{REACTION_DATA_DIR}/endo_DA/endo_DA_vac_TS_wB97M.xyz", 'mace_ts_fine_tune'                                # vacuum TS
    # train_fpath, wandb_project = f"{REACTION_DATA_DIR}/endo_DA/endo_DA_water_TS_wB97M.xyz", 'mace_ts_fine_tune'                          # single TS in water
    # -- valid -- 
    valid_fpath = f"{CPD_MVK_DA_DATA_DIR}/valid/CPD_MVK_DA_valid.xyz" 

    # --- anthracene DA in toluene ---
    # -- train --
    # train_fpath = f"{ANTHRACENE_DA_DATA_DIR}/train/anthracene_DA_train.xyz"     # paper datasets
    # -- valid -- 
    # valid_fpath = f"{ANTHRACENE_DA_DATA_DIR}/valid/anthracene_DA_valid.xyz"

    # --- guajadial DA in water ---
    # ...
    # --------------------------------


    # === SPECIFY PARENT MODEL FPATH ====
    # # ---- for mhft ----
    parent_model_fpath, dtype =  f"{MODEL_DIR}/GO-MACE-23.model", "float32"
    # # parent_model_fpath, dtype = f"{MODEL_DIR}/MACE-OFF23_medium.model", 'float64'
    # parent_model_fpath, dtype = f"{MODEL_DIR}/MACE-MP-0b3_medium.model", 'float64'
    system_ft_name = get_name_from_path(train_fpath)
    # # add train name to model
    model_save_name = get_name_from_path(parent_model_fpath) + f"_{system_ft_name}"
    if parent_model_fpath is not None:
        model_save_name += "_mhft_pt=10_42_CPD_MVK_DA_rep_2_seed_5"
    wandb_project = "mhft_GO-MACE_42_CPD_MVK_DA"
    # # wandb_project = "mhft_MACE-OFF-23-medium_42_CPD_MVK_DA"
    # wandb_project = "mhft_MACE-MP-0b3-medium_42_CPD_MVK_DA"

    # ---- for second round energy training ----
    # n, rep_nums, seeds, wandb_project = 42, [1], [3], "GO-MACE_42_CPD_MVK_DA_energy_train"
    # parent_model_fpath, dtype = f"{MODEL_DIR}/GO-MACE-23_CPD_MVK_DA_train_42_config_strat_mhft_pt10_42_CPD_MVK_DA_rep_1.model", "float32"
    # model_save_name = "GO-MACE-23_CPD_MVK_DA_train_42_config_strat_mhft_pt10_42_CPD_MVK_DA_rep_1"# get_name_from_path(parent_model_fpath)
    # system_ft_name = get_name_from_path(train_fpath)

    # modifiable parameters 
    lr = 0.001  # default
    # lr = 0.01
    # lr = 0.02
    log_lr = False


    # ===== SPECIFY NUMBER OF REPS AS A LIST ======
    # rep_nums, seeds = [1, 2, 3], [3, 5, 7]
    rep_nums, seeds = [1], [3]
    # rep_nums, seeds = [2], [5]
    # rep_nums, seeds = [3], [7]


    # === SPECIFY FINE-TUNING PARAMETERS HERE ===
    # default fine-tuning params
    # see settings.py for more infos
    # =============================================
    

    valid_fname = get_name_from_path(valid_fpath)
    lr_str = str(lr).replace('.', '_')
    for i, rep in enumerate(rep_nums):


        # ===== SPECIFY E0s (ELSE AVERAGE) =====
        # ---- for replay head ----
        pt_GO_E0s_fpath = "/home/jjourde/react_ml_test/htc_data/data/isolated_atoms/GO_isolated_atom_energies.xyz"
        pt_GO_E0s = load_isolated_atom_energies(pt_GO_E0s_fpath)
        pt_MP_E0s_fpath = "/home/jjourde/react_ml_test/htc_data/data/isolated_atoms/MP_isolated_atom_energies.xyz"
        pt_MP_E0s = load_isolated_atom_energies(pt_MP_E0s_fpath)

        # ---- for target head ----
        E0s_fpath = "/data/chem-cat/jjourde/data/isolated_atoms/wB97M-D3BJ_def2-TZVP_isolated_atom_energies_CHO.xyz"
        E0s = load_isolated_atom_energies(E0s_fpath)

         # ---- prepare for heads dictionnary ----
        target_E0_clean = {str(k): float(v) for k, v in E0s.items()}
        pt_GO_E0_clean = {str(k): float(v) for k, v in pt_GO_E0s.items()}
        pt_MP_E0_clean = {str(k): float(v) for k, v in pt_MP_E0s.items()}
        with open("target_E0s.json", "w") as f:
            json.dump(target_E0_clean, f)
        with open("pt_GO_E0s.json", "w") as f:
            json.dump(pt_GO_E0_clean, f)
        with open("pt_MP_E0s.json", "w") as f:
            json.dump(pt_MP_E0_clean, f)


        # === CREATION OF REPLAY TRAINING AND VALIDATION DATASETS ====
        # ---- for GO-MACE ----
        # create_datasets(dataset_fpath=f"{REPLAY_DATASET_DIR}/GO-MACE-23_train.xyz", train_fpath=f"{REPLAY_DATASET_DIR}/GO-MACE_train=100_seed={seeds[i]}.xyz", valid_fpath=f"{REPLAY_DATASET_DIR}/GO-MACE_test=300_seed={seeds[i]}_test100.xyz", n_train = 100, n_valid=300, seed=seeds[i])
        pt_train_fpath, pt_valid_fpath = f"{REPLAY_DATASET_DIR}/GO-MACE_train=10_seed=5.xyz", f"{REPLAY_DATASET_DIR}/GO-MACE_test=300_seed=5_train10.xyz"  # WARNING: requires float32 precision, not float64
        heads_dict = {"target_head":{"train_file": train_fpath, "valid_file": valid_fpath, "E0s": "target_E0s.json",}, "pt_head":{"train_file": pt_train_fpath, "valid_file": pt_valid_fpath, "E0s": "pt_GO_E0s.json",},}
        heads_dict = json.dumps(heads_dict)
        heads_dict = str(heads_dict)
    
        # # ---- for MACE-OFF23 ----
        # create_datasets(dataset_fpath=f"{REPLAY_DATASET_DIR}/spice_train.xyz", train_fpath=f"{REPLAY_DATASET_DIR}/spice_train=5_seed={seeds[i]}.xyz", valid_fpath=f"{REPLAY_DATASET_DIR}/spice_test=300_seed={seeds[i]}_train5.xyz", n_train = 15, n_valid=300, allowed_species={"H", "C", "O"}, seed=seeds[i])
        # pt_train_fpath, pt_valid_fpath = f"{REPLAY_DATASET_DIR}/spice_train=100_seed=7.xyz", f"{REPLAY_DATASET_DIR}/spice_test=300_seed=7_train100.xyz"
        # heads_dict = {"target_head":{"train_file": train_fpath, "valid_file": valid_fpath, "E0s": "target_E0s.json",}, "pt_head":{"train_file": pt_train_fpath, "valid_file": pt_valid_fpath, "E0s": "target_E0s.json",},}
        # heads_dict = json.dumps(heads_dict)
        # heads_dict = str(heads_dict)

        # # ---- for MACE-MP-0b3 ----
        # create_datasets(dataset_fpath=f"{REPLAY_DATASET_DIR}/mace_mp0_mptrj_data.xyz", train_fpath=f"{REPLAY_DATASET_DIR}/MACE-MP0b3_train=1000_seed=5.xyz", valid_fpath=f"{REPLAY_DATASET_DIR}/MACE-MP0b3_test=300_seed=5_train1000.xyz", n_train = 1000, n_valid=300,  allowed_species={"H", "C", "O"},seed=5)
        # pt_train_fpath, pt_valid_fpath = f"{REPLAY_DATASET_DIR}/MACE-MP0b3_train=10_seed=5.xyz", f"{REPLAY_DATASET_DIR}/MACE-MP0b3_test=300_seed=5_train10.xyz"
        # heads_dict = {"target_head":{"train_file": train_fpath, "valid_file": valid_fpath, "E0s": "target_E0s.json",}, "pt_head":{"train_file": pt_train_fpath, "valid_file": pt_valid_fpath, "E0s": "pt_MP_E0s.json",},}
        # heads_dict = json.dumps(heads_dict)
        # heads_dict = str(heads_dict)


        # ==== SPECIFY MACE SETTINGS ====
        # ---- for mhft ----
        train_settings = MACETrainSettings(energy_weight=1.0,
                                       forces_weight=10.0,
                                       #E0s=E0s if E0s else "average",  # not necessary in mhft with heads_dict
                                       lr=lr,
                                       batch_size=2, #min(get_xyz_file_len(train_fpath), 2),
                                       max_num_epochs=500,
                                       energy_key="REF_energy",
                                       forces_key="REF_forces",
                                       ema=True,
                                       ema_decay=0.99,
                                       scheduler_patience=None,     # None
                                       patience=20,                 # 20
                                       patience_warmup=0,          # default: 20
                                       restart_latest=False,
                                       device="cuda",
                                       enable_cueq=False,
                                       dtype=dtype,
                                       wandb_project=wandb_project, 
                                       multiheads_finetuning = True,
                                       heads = heads_dict,     # relevant if using heads dict
                                       num_samples_pt = 1000,        # relevant if not using heads dict
#                                       filter_type_pt = "combinations", # mlptrain version not compatible?
                                       subselect_pt = "random",     # relevant if not using heads dict
                                       weight_pt = 1.0,             # relevant if not using heads dict
                                       atomic_numbers = "[1,6,8]",  
                                       force_mh_ft_lr = False)      # relevant if not using heads dict


    # ---- for second round energy training ----
    #     energy_weight = 10.0
    #     model_save_name += f"_mhft_energy"
    #     train_settings = MACETrainSettings(energy_weight=energy_weight,
    #                                     forces_weight=1.0,
    #                                     #    E0s=E0s if E0s else "average",
    #                                     lr=lr,
    #                                     batch_size=2,
    #                                     max_num_epochs=500,          # default: 500
    #                                     energy_key="REF_energy",
    #                                     forces_key="REF_forces",
    #                                     ema=True,
    #                                     ema_decay=0.99,
    #                                     scheduler_patience=None,       # default: None
    #                                     patience=20,                 # default: 20
    #                                     patience_warmup=0,          # default: 20
    #                                     restart_latest=False,
    #                                     device="cuda",
    #                                     enable_cueq=False,
    #                                     dtype=dtype,
    #                                     wandb_project=wandb_project,
    #                                     multiheads_finetuning = True,
    #                                     heads = heads_dict,     # relevant if using heads dict
    #                                     num_samples_pt = 1000,        # relevant if not using heads dict
    # #                                       filter_type_pt = "combinations", # mlptrain version not compatible?
    #                                     subselect_pt = "random",     # relevant if not using heads dict
    #                                     weight_pt = 1.0,             # relevant if not using heads dict
    #                                     atomic_numbers = "[1,6,8]",  
    #                                     force_mh_ft_lr = False)

        

        # update seed (unless we have only one rep, then just use the specified seed)
        print("Running rep: ", rep)
        modified_mace_train_settings = train_settings._replace(seed=seeds[i])
        print("Using seed: ", modified_mace_train_settings.seed)

        model_save_name = model_save_name + f"_lr_{lr_str}" if log_lr else model_save_name
        model_save_name += f"_rep_{rep}"

        # run training
        train_mace(train_fpath,
                   modified_mace_train_settings,
                   model_save_name,
                   pt_train_fpath = pt_train_fpath,
                   pt_valid_fpath = pt_valid_fpath,
                   valid_fpath=valid_fpath,
                   model_architect_settings=None,
                   parent_model_fpath=parent_model_fpath)

        # move output model file to models dir, and rename with replicate number at end (lazy I know)
        trained_model_name_no_rep = model_save_name.replace(f"_rep_{rep}", "")

        shutil.move(f"{model_save_name}.model", os.path.join(MODEL_DIR, os.path.basename(f"{trained_model_name_no_rep}_rep_{rep}.model")))
#         os.remove(f"{model_save_name}_compiled.model")
