"""Script to cut a solvation shell out of a reactant in a solvent box."""
from react_ml_test_public.helpers.settings import ANTHRACENE_DA_DATA_DIR, CPD_MVK_DA_DATA_DIR, OPT_DIR, I_Cl_SN2_DATA_DIR, \
    GUAJADIAL_DA_DATA_DIR, SOLVENTS_DIR
import ase
from react_ml_test_public.helpers.io import get_name_from_path, flexible_ase_read
from react_ml_test_public.helpers.solvation import get_solv_shell
from react_ml_test_public.helpers.logging import logger


if __name__ == '__main__':


    # === SELECT INPUT xyz ===

    # --- anthracene DA TS ---
    input_dir = f"{ANTHRACENE_DA_DATA_DIR}/TS"
    num_solute_atoms = 41
    solv_shell_cutoff = 5.2
    atoms_per_solv_mol = 15

    # anthracene 1,4 DA OFF46 Opt TS
    input_xyz_fpath = f'{input_dir}/anthracene_1_4_DA_in_toluene_OFF46_opt_TS.xyz'

    # =======================

    input_config_ase = flexible_ase_read(input_xyz_fpath)[0]

    logger.info(f"Cutting sphere around solute with cutoff of {solv_shell_cutoff}")
    solute_indices = list(range(num_solute_atoms))
    solv_indices = list(range(num_solute_atoms, len(input_config_ase)))
    solv_shell_mols = get_solv_shell(input_config_ase, solute_indices, solv_indices, 
                                        atoms_per_solv_mol, cutoff=solv_shell_cutoff)
    print("Num solv shell mols: ", len(solv_shell_mols))
    # logger.info(f"Solvent molecules in solvent shell: {len(solv_shell_mols)}")
    solv_shell_indices = [i for mol in solv_shell_mols for i in mol]
    
    # freeze solvent atoms
    solv_shell_mask = [i in set(solv_shell_indices).union(solute_indices) for i in range(len(input_config_ase))]
    # print("Solv Shell mask: ", solv_shell_mask)
    input_config_ase = input_config_ase[solv_shell_mask]
    print("Num atoms in solvated config: ", len(input_config_ase))

    box_dim = [100.0, 100.0, 100.0]
    input_config_ase.set_cell(box_dim)
    input_config_ase.set_pbc([False, False, False])

    cutoff_str = str(solv_shell_cutoff).replace('.', '_')
    out_fp = f'{input_dir}/{get_name_from_path(input_xyz_fpath)}_cutoff={cutoff_str}.xyz'
    ase.io.write(out_fp, input_config_ase, format='extxyz')
    logger.info(f'Saved solvated molecule to {out_fp}...')