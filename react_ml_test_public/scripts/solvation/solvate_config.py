"""Script to use mlptrains solvation code to solvate an input configuration (i.e. reactants in vacuum)."""

from react_ml_test_public.helpers.settings import (
    ANTHRACENE_DA_DATA_DIR,
    CPD_MVK_DA_DATA_DIR,
    GUAJADIAL_DA_DATA_DIR,
    OPT_DIR,
    SOLVENTS_DIR,
    I_Cl_SN2_DATA_DIR,
)
from react_ml_test_public.helpers.solvation import solvate_config

if __name__ == '__main__':

    solvent_name = None
    solvent_xyz_fpath = None
    solvent_mol = None
    solvent_density = None
    solvation_box_size = None
    solv_shell_cutoff = None
    pbc = True

    # select model
    # model_name = 'GO-MACE-23'
    # model_name = 'MACE-MP0-128-L1'
    # model_name = 'MACE-OFF23_medium'
    # model_name = 'MACE-OFF23_medium_endo_DA_train_fine_tuned'

    # === SELECT INPUT OPTIONS ===

    # --- CPD MVK endo DA TS ---
    # input_dir = f"{CPD_MVK_DA_DATA_DIR}/TS"
    # input_xyz_fpath = f'{input_dir}/cis_endo_TS_wB97M.xyz'
    # system_name = 'CPD_MVK_TS_wB97M'
    # num_solute_atoms = 22
    # solv_shell_cutoff = None
    # # solvation_box_size = 18.5
    # solvation_box_size = 10.0

    # --- CPD MVK endo DA NEB ---
    # input_dir = f"{CPD_MVK_DA_DATA_DIR}"
    # input_xyz_fpath = f'{input_dir}/CPD_MVK_DA_neb_vac_wB97M.xyz'
    # system_name = 'CPD_MVK_DA_neb_vac_wB97M'
    # num_solute_atoms = 22
    # solv_shell_cutoff = 5.0  # TODO: issue right now is solvation shell is moving with the reactants! could define from central point instead???
    # solvation_box_size = 18.5

    # --- anthracene DA TS ---
    # input_dir = f"{ANTHRACENE_DA_DATA_DIR}/TS"
    # num_solute_atoms = 41
    # solvation_box_size = 21.0

    # # 
    # # input_xyz_fpath = f'{input_dir}/anthracene_DA_vac_TS_wB97M.xyz'
    # # system_name = 'anthracene_DA_TS'

    # # anthracene 1,4 DA MMFF TS
    # input_dir = f"{ANTHRACENE_DA_DATA_DIR}/TS"
    # num_solute_atoms = 41
    # solvation_box_size = 21.0
    # # input_xyz_fpath, system_name = f'{input_dir}/anthracene_DA_MMFF_1_4_TS_opt.xyz', 'anthracene_1_4_DA_MMFF_TS'
    # input_xyz_fpath, system_name = f'{input_dir}/anthracene_1_4_DA_vac_OFF46_opt_TS.xyz', 'anthracene_1_4_DA_OFF46_TS'

    # --- guajadial DA ---
    input_dir = f"{GUAJADIAL_DA_DATA_DIR}/TS"
    input_xyz_fpath = f'{input_dir}/guajadial_DA_vac_TS_wB97M.xyz'
    system_name = 'guajadial_DA_TS'
    num_solute_atoms = 69
    # solvation_box_size = 22.0
    solvation_box_size = 12.0

    # --- I Cl SN2 ---
    # input_dir = I_Cl_SN2_DATA_DIR
    # input_xyz_fpath = f'{input_dir}/I_Cl_SN2_vac_TS_wB97M.xyz'
    # system_name = 'I_Cl_SN2_TS'

    # === SELECT SOLVENT PARAMS ===
    # water
    solvent_xyz_fpath, solvent_density = f'{SOLVENTS_DIR}/h2o.xyz', 0.997
    atoms_per_solv_mol = 3

    # toluene
    # solvent_xyz_fpath, solvent_density = f'{SOLVENTS_DIR}/toluene.xyz', 0.8603
    # atoms_per_solv_mol = 15

    # ====
    solvate_config(
        input_xyz_fpath=input_xyz_fpath,
        input_dir=input_dir,
        system_name=system_name,
        solvation_box_size=solvation_box_size,
        solvent_xyz_fpath=solvent_xyz_fpath,
        solvent_density=solvent_density,
        solvent_name=solvent_name,
        pbc=pbc,
        solv_shell_cutoff=solv_shell_cutoff,
        num_solute_atoms=num_solute_atoms,
        atoms_per_solv_mol=atoms_per_solv_mol,
    )
