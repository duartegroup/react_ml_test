"""
Helpers for evaluating specific trajectories.
"""
from typing import Dict, List
import matplotlib.pyplot as plt
from react_ml_test_public.helpers.settings import ConfigDef
from ase.io import read
import ase


def eval_products(config: ase.Atoms,
                  config_defs: List[ConfigDef]
                 ) -> Dict[str, bool]:
    """
    Determine the formation of certain products for a config, as defined by a list of product definitions.

    :param configs: the configs
    :param config_defs: a list of config definitions (see settings.py for details)
    returns: Dict[str, typ] dictionary of {prod_name: True/False, ...} for all products in prod defs
    """

    prod_formation_dict = {}

    for config_def in config_defs:
        name = config_def.name
        bond_defs = config_def.bond_defs

        # check all bond definitions in the product definition
        prod_formed = all([(config.get_distance(a1, a2, mic=True) <= threshold) == bonded 
                            for a1, a2, bonded, threshold in bond_defs])
        
        prod_formation_dict[name] = prod_formed

    return prod_formation_dict


def get_config_def_indices(configs: List[ase.Atoms],
                           config_defs: List[ConfigDef]) -> Dict[str, int]:
    """
    Identify configs in a list which form specific defined configurations, as defined by a list of config definitions.

    Parameters:
        configs (List[ase.Atoms]): a list of configurations as ase.Atoms objects
        config_defs (List[ConfigDef]): a list of ConfigDefs which define whether a certain type of configuration is present.
    Returns:
        Dict(int): a dict of {ConfigDef_name: set(indices)} which maps the name of each type of config def to the
        indices of configs in the input list which correspond to that config def. 
    """

    prod_formation_indices = {config_def.name: set() for config_def in config_defs}
    prod_formation_indices.update({'other': set()})
    
    # print("Checking product formation for: \n", prod_defs)
    for i, config in enumerate(configs):
        prod_formation_dict = eval_products(config, config_defs)
        prod_detected = False
        for prod_name, formed in prod_formation_dict.items():
            if formed:
                prod_formation_indices[prod_name].add(i)
                prod_detected = True
        if not prod_detected:
            prod_formation_indices['other'].add(i)

    return prod_formation_indices


def rdf(tag: str):
    """
    Calculate the radial distribution function of a trajectory for a given pair of atoms
    specified by the tag.

    :param: tag: one of 'HH_intra', 'HC_intra', 'HO_intra', 'CC_intra', 'CO_intra', 'OO_intra'
    :return:
    """
    from aseMolec import anaAtoms as aa

    for f in ['xtb_md', 'mace03_md']:
        traj = read('moldyn/'+f+'.xyz', '50:') #ignore first 50 frames
        for at in traj:
            at.pbc = True #create a fake box for rdf compatibility
            at.cell = [100,100,100]
        rdf = aa.compute_rdfs_traj_avg(traj, rmax=5, nbins=70) #aseMolec provides functionality to compute RDFs
        plt.plot(rdf[1], rdf[0][tag], '.-', label=f, alpha=0.7, linewidth=3)

    plt.legend()
    plt.yticks([])
    plt.xlabel(r'R ($\rm \AA$)')
    plt.ylabel('RDF '+tag)


def print_atom_distance_table(atoms: ase.Atoms, threshold: float = 2.0, elements: set = None):

    # Print header
    print(f"{'Atom 1':<10} {'Atom 2':<10} {'Element 1':<10} {'Element 2':<10} {'Distance (Å)':<15}")

    # Loop over all unique atom pairs
    for i in range(len(atoms)):

        # check elements if specified
        if elements is not None and atoms[i].symbol not in elements:
            continue
        
        for j in range(i + 1, len(atoms)):  # Avoid double-counting

            # check elements if specified
            if elements is not None and atoms[j].symbol not in elements:
                continue

            # get distance
            distance = atoms.get_distance(i, j)
            if distance <= threshold:
                print(f"{i:<10} {j:<10} {atoms[i].symbol:<10} {atoms[j].symbol:<10} {distance:<15.3f}")
