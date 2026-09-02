"""
Script to train a MACE model (or models) from scratch.
"""
import glob



from react_ml_test_public.helpers.io import (
    get_name_from_path,
    get_xyz_file_len,
    load_isolated_atom_energies,
)
from react_ml_test_public.helpers.settings import (
    ANTHRACENE_DA_DATA_DIR,
    CPD_MVK_DA_DATA_DIR,
    GUAJADIAL_DA_DATA_DIR,
    MODEL_DIR,
    MACEArchitectSettings,
    MACETrainSettings,
)
from react_ml_test_public.helpers.training import run_iterative_train
from react_ml_test_public.helpers.utils import natural_sort_key

if __name__ == "__main__":

    E0s, parent_model_fpath, model_save_name = None, None, None
    specified_subset_fpaths = None
    model_architect_settings = None
    dtype = 'float32'
    wandb_project = "da_endo_fine_tune"
    seed = 25
    
    # ====== SELECT TRAIN, VALID SETS =====

    # === SELECT VALID SET ===

    # --- endo DA ---
    # valid_fpath = f"{CPD_MVK_DA_DATA_DIR}/valid/CPD_MVK_DA_valid.xyz"

    # --- anthracene DA ---
    # valid_fpath = f"{ANTHRACENE_DA_DATA_DIR}/valid/anthracene_DA_valid.xyz"
    # valid_fpath = f"{ANTHRACENE_DA_DATA_DIR}/valid/anthracene_DA_valid_vac_only.xyz"

    # --- guajadial DA ---
    valid_fpath = f"{GUAJADIAL_DA_DATA_DIR}/valid/guajadial_DA_valid.xyz"

    # --- I CL SN2 ---
    # valid_fpath = f"{I_Cl_SN2_TRAIN_DIR}/I_Cl_SN2_valid.xyz"

    # === SELECT TRAIN DATASET ===

    # --- grid based solvation vs reactive samples ---
    # train_fpath = ""
    # system_dir, wandb_project = CPD_MVK_DA_DATA_DIR, "CPD_MVK_DA_react_solv"
    # # system_dir, wandb_project = ANTHRACENE_DA_DATA_DIR, "anthracene_DA_react_solv"
    # # system_dir, wandb_project = GUAJADIAL_DA_DATA_DIR, "guajadial_DA_react_solv"

    # # ds_name_stub = f"{system_dir}/train/subsets/*_n=8_r_dist*_diverse*.xyz"
    # # ds_name_stub, seed = f"{system_dir}/train/subsets/*_n=8_r_dist*_diverse*_rep_1.xyz", 3
    # # ds_name_stub, seed = f"{system_dir}/train/subsets/*_n=8_r_dist*_diverse*_rep_2.xyz", 5
    # # ds_name_stub, seed = f"{system_dir}/train/subsets/*_n=8_r_dist*_diverse*_rep_3.xyz", 7
    # rep_nums = None

    # # specific runs for incomplete jobs
    # # ds_name_stub, seed = f"{system_dir}/train/subsets/*_n=8_r_dist_0_5_diverse_solv__rep_3.xyz", 7
    # # ds_name_stub, seed = f"{system_dir}/train/subsets/*_n=8_r_dist_1_5_diverse*_rep_3.xyz", 7
    # # ds_name_stub, seed = f"{system_dir}/train/subsets/*_n=8_r_dist_3_0_diverse*_rep_3.xyz", 7
    # ds_name_stub, rep_nums, seed = f"{system_dir}/train/subsets/*_n=8_r_dist_6_0_diverse*_rep_3.xyz", None, 7

    # # ds_name_stub, rep_nums, seed = f"{system_dir}/train/subsets/*_n=8_r_dist_1_5_diverse_solv_50_rep_3.xyz", None, 7
    # # ds_name_stub, rep_nums, seed = f"{system_dir}/train/subsets/*_n=8_r_dist_1_5_diverse_solv_75_rep_3.xyz", None, 7
    # # ds_name_stub, rep_nums, seed = f"{system_dir}/train/subsets/*_n=8_r_dist_3_0_diverse_solv_100_rep_3.xyz", None, 7
    # # ds_name_stub, rep_nums, seed = f"{system_dir}/train/subsets/*_n=8_r_dist_6_0_diverse*.xyz", None, 7

    # ds_name_stub, seed = f"{system_dir}/train/subsets/*_n=8_r_dist_1_5_diverse_solv_100_rep_3.xyz", [3], 7

    # uncomment this to find all files with name stub
    # specified_subset_fpaths = sorted(glob.glob(ds_name_stub), key=natural_sort_key)
    # assert len(specified_subset_fpaths) > 0, f"No files detected in directory {CPD_MVK_DA_DATA_DIR} with name stub {ds_name_stub}..."

    # --- RAND ITER TRAINING WITH REPLICATES ---
    train_fpath = ""

    # choose system:
    # system_dir, ft_train_name, sample_str, wandb_project = CPD_MVK_DA_DATA_DIR, "CPD_MVK_DA_train", "", "CPD_MVK_DA_iter_train"
    # system_dir, ft_train_name, sample_str, wandb_project = CPD_MVK_DA_DATA_DIR, "CPD_MVK_DA_train", "ts_gauss_", "CPD_MVK_DA_iter_train"  # ts gaussian sampling
    # system_dir, ft_train_name, wandb_project = ANTHRACENE_DA_DATA_DIR, "anthracene_DA_train", "anthracene_DA_iter_train_2"
    system_dir, ft_train_name, sample_str, wandb_project = GUAJADIAL_DA_DATA_DIR, "guajadial_DA_train", "", "guajadial_DA_iter_train"
    # system_dir, ft_train_name, sample_str, wandb_project = GUAJADIAL_DA_DATA_DIR, "guajadial_DA_train", "ts_gauss_", "guajadial_DA_iter_train"
    # system_dir, ft_train_name, wandb_project = I_Cl_SN2_TRAIN_DIR, "I_Cl_SN2_train", "I_Cl_SN2_iter_train"

    # choose replicate
    # ds_name_stub, seed, rep_nums = f"{system_dir}/train/subsets/{ft_train_name}_*_config_strat_{sample_str}rep_1.xyz", 3, [1]
    # # ds_name_stub, seed, rep_nums = f"{system_dir}/train/subsets/{ft_train_name}_*_config_strat_{sample_str}rep_2.xyz", 5, [2]
    # # ds_name_stub, seed, rep_nums = f"{system_dir}/train/subsets/{ft_train_name}_*_config_strat_{sample_str}rep_3.xyz", 7, [3]

    # specified_subset_fpaths = sorted(glob.glob(ds_name_stub), key=natural_sort_key)

    # or instead extra runs if any failures
    specified_subset_fpaths, seed, rep_nums = [f"{system_dir}/train/subsets/{ft_train_name}_{n}_config_strat_{sample_str}rep_1.xyz" for n in [42]], 3, [1] # extra runs if failed 
    # specified_subset_fpaths, seed, rep_nums = [f"{system_dir}/train/subsets/{ft_train_name}_{n}_config_strat_{sample_str}rep_2.xyz" for n in [46, 50]], 5, [2] # extra runs if failed 
    # specified_subset_fpaths, seed, rep_nums = [f"{system_dir}/train/subsets/{ft_train_name}_{n}_config_strat_{sample_str}rep_3.xyz" for n in [46, 50]], 7, [3] # extra runs if failed 
    # ----------------

    print('Subset Fpaths: ', specified_subset_fpaths)
    assert len(specified_subset_fpaths) > 0, f"No files detected in directory with name stub {ds_name_stub}..."

    # ===== SPECIFY E0s (ELSE AVERAGE) =====
    E0s_fpath = "data/isolated_atoms/wB97M-D3BJ_def2-TZVP_isolated_atom_energies_CHO.xyz"
    E0s = load_isolated_atom_energies(E0s_fpath)
    print('Loaded isolated atom energies: ', E0s)

    # ===== SPECIFY FINE-TUNING PARENT MODEL FPATH =====

    # --- foundation models ---
    # parent_model_fpath, dtype = f"{MODEL_DIR}/GO-MACE-23.model", 'float32'  # WARNING: requires float32 precision, not float64
    # parent_model_fpath, dtype = f"{MODEL_DIR}/MACE-OFF23_medium.model", 'float64'
    # parent_model_fpath, dtype = f"{MODEL_DIR}/MACE-MP-0b3_medium.model", 'float64'
    # parent_model_fpath, dtype = f"{MODEL_DIR}/MACE-omol-0-extra-large-1024.model", 'float64'

    # --- system transfer ---
    # parent_model_fpath, dtype = f"{MODEL_DIR}/GO-MACE-23_CPD_MVK_DA_train_42_config_strat_fine_tuned_energy_rep_1.model", 'float64'
    # parent_model_fpath, dtype = f"{MODEL_DIR}/MACE-OFF23_medium_CPD_MVK_DA_train_42_config_strat_fine_tuned_energy_rep_1.model", 'float64'
    # parent_model_fpath, dtype = f"{MODEL_DIR}/MACE-MP-0b3_medium_CPD_MVK_DA_train_42_config_strat_fine_tuned_energy_rep_1.model", 'float64'
    
    # === SPECIFY FINE-TUNING SETTINGS ===

    # fine-tuning specific defaults
    patience_warmup: int = 20  # 0
    scheduler_patience: int = None
    patience: int = 20
    patience_delta: float = None
    max_epochs = 500
    valid_batch_size: int = 2
    
    # ===== OR: SPECIFY MODEL ARCHITECTURE FOR FROM SCRATCH TRAINING ======

    # === GO-MACE-23 ===
    # model_save_name, dtype = "GO-MACE-23_from_scratch", 'float32'
    # model_architect_settings = MACEArchitectSettings(model_type="ScaleShiftMACE",
    #                                                  num_interactions=2,
    #                                                  num_channels=128,        
    #                                                  correlation=3,
    #                                                  max_L=1,            
    #                                                  r_max=3.7,
    #                                                  )
                                                     
    # === MACE-OFF23 medium architecture ===
    model_save_name, dtype = "MACE-OFF23_medium_from_scratch", 'float64'
    model_architect_settings = MACEArchitectSettings(model_type="ScaleShiftMACE",
                                                     num_interactions=2,
                                                     num_channels=128,        
                                                     correlation=3,
                                                     max_L=1,                
                                                     r_max=5.0,
                                                    )

    # === MACE-MP-0 ===
    # model_save_name, dtype = "MACE_MP0_from_scratch", 'float64'
    # model_architect_settings = MACEArchitectSettings(model_type="ScaleShiftMACE",
    #                                                  num_interactions=2,
    #                                                  num_channels=128,        
    #                                                  correlation=3,
    #                                                  max_L=1,                
    #                                                  r_max=6.0,
    #                                                  )
    
    # === mlptrain ===
    # model_save_name = "MACE_mlptrain_style"  
    # model_architect_settings = MACEArchitectSettings(model_type="ScaleShiftMACE",
    #                                                 num_interactions=2,
    #                                                 num_channels=128,        
    #                                                 correlation=3,
    #                                                 l_max=1,                
    #                                                 r_max=5.0,
    #                                                 E0s=E0s if E0s else "average",
    #                                                 )  

    # from-scratch training specific params
    # scheduler_patience: int = None
    # patience: int = 20

    # === TRAIN SETTINGS ===

    # --- train settings for small dataset iter training / fine-tuning ---
    train_settings = MACETrainSettings(energy_weight=1.0,
                                       forces_weight=10.0,
                                       E0s=E0s if E0s else "average",
                                       lr=0.001,
                                       batch_size=2,
                                       max_num_epochs=max_epochs,
                                       ema=True,
                                       ema_decay=0.99,
                                       scheduler_patience=scheduler_patience,
                                       patience=patience,
                                       patience_warmup=patience_warmup,
                                       valid_batch_size=valid_batch_size,
                                       seed=seed,
                                       restart_latest=False,
                                       device="cuda",
                                       enable_cueq=False,
                                       dtype=dtype,
                                       wandb_project=wandb_project)

    #
    run_iterative_train(train_fpath,
                        valid_fpath,
                        train_settings,
                        specified_subset_fpaths=specified_subset_fpaths,
                        rep_nums=rep_nums,
                        model_save_name=model_save_name,
                        model_architect_settings=model_architect_settings,
                        parent_model_fpath=parent_model_fpath,
                        batch_size_scheduler=None,
                        )
