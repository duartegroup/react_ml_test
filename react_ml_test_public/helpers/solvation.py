import ase
import ase.io
from react_ml_test_public.helpers.logging import logger
import numpy as np

def get_solv_shell(config: ase.Atoms,
                   solute_indices: list,
                   solv_indices: list,
                   atoms_per_solv_mol: int,
                   cutoff: float = 3.0):
    """
    Returns a list of solute and solvent molecule atoms that are within `cutoff` Å of the solute. 
    Uses ASE minimum-image distances. It is recommended that solute atoms are at the start of the .xyz file, 
    and solvent atoms must be grouped sequentially by molecule.

    - config: ASE Atoms object
    - solute_indices: a list of indices in the configuration for the solute atoms.
    - solv_indices: a list of indices in the configuration for the solvent atoms.
    - atoms_per_solv_mol: the number of atoms in each solvent molecules, i.e. 3 for h2o
    - cutoff: distance threshold in Å for solvent molecules ot include in the solvation shell
    """

    # get solvent molecules
    if len(solv_indices) % atoms_per_solv_mol != 0:
        raise ValueError("solv_indices length not divisible by atoms_per_molecule")
    n_mol = len(solv_indices) // atoms_per_solv_mol
    molecules = [solv_indices[i*atoms_per_solv_mol:(i+1)*atoms_per_solv_mol] for i in range(n_mol)]

    # get minimum distances between solute atoms and all solvent molecule atoms, 
    # include in shell if at least one atom in a solvent molecule is within the cutoff
    first_shell = []
    for mol in molecules:
        d = np.array([np.min(config.get_distances(solute_id, np.array(mol), mic=True)) for solute_id in solute_indices])
        if np.min(d) <= cutoff:
            first_shell.append(mol)

    return first_shell


def solvate_config(input_xyz_fpath: str, 
                   input_dir: str, 
                   system_name: str, 
                   solvation_box_size: float,
                   solvent_xyz_fpath=None, 
                   solvent_density=None, 
                   solvent_name=None,
                   pbc: bool = True, 
                   solv_shell_cutoff: float = None,
                   num_solute_atoms=None,
                   atoms_per_solv_mol=None):
    """Solvate input configurations and write the solvated structures to an XYZ file."""
    import mlptrain as mlt
    from mlptrain.box import Box

    if pbc:
        box_dim = [solvation_box_size] * 3
        pbc_str = "_PBC"
    else:
        box_dim = [100.0, 100.0, 100.0]
        pbc_str = ""

    # load input config + solvent mol
    input_set = mlt.ConfigurationSet()
    input_set.load_xyz(filename=input_xyz_fpath, charge=0, mult=1)

    solvent_mol = mlt.Molecule(solvent_xyz_fpath) if solvent_xyz_fpath is not None else None
    if solvent_xyz_fpath is not None:
        solvent_str = solvent_xyz_fpath.split('/')[-1].split('.')[0]
    elif solvent_name is not None:
        solvent_str = solvent_name
    else:
        solvent_str = ""
    solvation_box_size_str = str(solvation_box_size).replace('.', '_')

    solv_configs = []
    for input_config in input_set:
        input_config.solvate(
            box_size=solvation_box_size,
            solvent_molecule=solvent_mol,
            solvent_density=solvent_density,
            solvent_name=solvent_name
        )

        if solv_shell_cutoff is not None:
            box_dim = [100.0, 100.0, 100.0]
            pbc_str = "_sphere"

        input_config.box = Box(box_dim)
        input_config_ase = input_config.ase_atoms

        # cut sphere out of cube for cluster configs
        if solv_shell_cutoff is not None:
            logger.info(f"Cutting sphere around solute with cutoff of {solv_shell_cutoff}")
            solute_indices = list(range(num_solute_atoms))
            solv_indices = list(range(num_solute_atoms, len(input_config_ase)))
            solv_shell_mols = get_solv_shell(
                input_config_ase, solute_indices, solv_indices,
                atoms_per_solv_mol, cutoff=solv_shell_cutoff
            )
            print("Num Solv shell mols: ", len(solv_shell_mols))
            solv_shell_indices = [i for mol in solv_shell_mols for i in mol]

            solv_shell_mask = [
                i in set(solv_shell_indices).union(solute_indices)
                for i in range(len(input_config_ase))
            ]
            input_config_ase = input_config_ase[solv_shell_mask]
            print("Num atoms in solvated config: ", len(input_config_ase))

        solv_configs.append(input_config_ase)

    out_fp = (
        f'{input_dir}/{system_name}_in_{solvent_str}_'
        f'{solvation_box_size_str}{pbc_str}.xyz'
    )
    ase.io.write(out_fp, solv_configs, format='extxyz')
    logger.info(f'Saved solvated molecule to {out_fp}...')

    return out_fp

