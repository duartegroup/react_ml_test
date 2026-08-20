from typing import List, Literal
import ase
from ase.io import read, write
import random
import tempfile
from random import shuffle
import numpy as np
from react_ml_test.helpers.io import flexible_ase_read, get_xyz_config_offsets
from react_ml_test.helpers.logging import logger
from react_ml_test.helpers.settings import DPI
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def config_equal_tol(c1: ase.Atoms, c2: ase.Atoms, tol=1e-6):
    """
    Check equivalence of two ase.Atoms objects.
    """
    if not np.array_equal(c1.numbers, c2.numbers):
        return False
    return np.allclose(c1.positions, c2.positions, atol=tol, rtol=0)

def remove_duplicates(configs: list[ase.Atoms]):
    """
    Check for duplicates in a list of ase.Atoms objects and remove them.
    """
    remove_indices = set()
    for i in range(len(configs)):
        for j in range(i+1, len(configs)):
            c1, c2 = configs[i], configs[j]
            if config_equal_tol(c1, c2):
                remove_indices.add(j)
    configs = [configs[i] for i in range(len(configs)) if i not in remove_indices]
    print(f'Returning {len(configs)} configs, filtered out {len(remove_indices)} duplicates...')
    return configs, remove_indices

def outside_in(items):
    return [items[i] if i % 2 == 0 else items[-(i//2 + 1)] for i in range(len(items))]

def sample_dataset(fpath: str,
                   num_samples: int = None,
                   element_filter_set: set[str] = None,
                   specific_indices: list[int] = None,
                   random_sample: bool = False,
                   random_seed: int = 3) -> list[ase.Atoms]:
    """
    Load configurations from an xyz file chosen by the input indices (avoids loading the entire dataset).

    Parameters:
        fpath (str): file path of the input dataset to be sampled (.xyz file)
        num_samples (int): number of samples to be taken from the dataset
        must_contain_elements set[str]: a set of elements for which there must be one of in a configuration to be considered for sampling
        random_seed (int): random seed for random sampling methods
        specific_indices (list[int]): a list of specific indices to sample, overrides all other parameters 
    Returns:
        list[ase.Atoms]: a list of atoms objects which are a sampled subset of the input dataset
    """

    # get config offsets in file
    logger.info('Calculating dataset length...')
    offsets = get_xyz_config_offsets(fpath)
    logger.info(f"Total num configs in input dataset: {len(offsets)}")

    n = len(offsets) if num_samples is None else num_samples

    assert not(specific_indices is True and random_sample is True), "Choose one of random sample, or specific indices. Do not specify both."

    # uniformly shuffle config offsets
    if random_sample is True:
        random.seed(random_seed)
        random.shuffle(offsets)
        logger.info(f"Sampling {n} samples from dataset...")
    if specific_indices is not None:
        n = len(specific_indices)
        offsets = [offsets[index] for index in specific_indices]
        logger.info(f"Sampling {n} samples from dataset at indices: {specific_indices}...")

    # extract and read only selected configurations using temporary files
    sampled, configs = 0, []
    with open(fpath, "r") as f:
        for i, (start_offset, end_offset) in enumerate(offsets):

            # we have finished sampling
            if n is not None and sampled == n:
                break

            f.seek(start_offset)  # Jump to the configuration start
            frame_data = f.read(end_offset - start_offset)  # Read only this frame

            # write config to a temp file
            with tempfile.NamedTemporaryFile(mode="w+", delete=True) as tmpfile:
                tmpfile.write(frame_data)
                tmpfile.flush()

                frame_data_list = frame_data.split('\n')

                # skip atom lines
                config_element_set = set()
                for atom_line in frame_data_list[2:-1]:
                    split_atom_line = atom_line.split(' ')[0].strip()
                    config_element_set.add(split_atom_line)

                # element filtering
                if element_filter_set is not None:
                    if len(element_filter_set.intersection(config_element_set)) == 0:
                        # print(f'Not including config with elements {config_element_set} as intersection is: {element_filter_set.intersection(config_element_set)}')
                        if i % 10000 == 0:
                            logger.info(f'Processed {i} / {n} configurations...')
                        continue

                # read only this frame using ASE
                config = read(tmpfile.name)
                configs.append(config)
                sampled += 1

            if i % 10000 == 0:
                logger.info(f'Processed {i} / {n} configurations...')

    if num_samples is not None and sampled < num_samples:
        logger.info(f'WARNING: Could not sample {num_samples} samples, instead returning {sampled} samples...')

    return configs


def react_coord_solvent_sampling(vacuum_ds_fpath: str,
                                 solvated_ds_fpath: str,
                                 rs: List[tuple],
                                 ts_ref_fpath: str,
                                 r_dists: List[float],
                                 solv_fracs: List[float],
                                 n_samples: int,
                                 out_dir: str,
                                 save_name: str,
                                 seed: int = 3,
                                 sample_mode: Literal['diverse', 'local'] = 'local',
                                 rep_num: int = None,
                                 save: bool = True
                                ):
    """
    Function to take in a reactive dataset of 1) vacuum and 2) solvated configurations and
    a sequence of 1) distances from the TS (r_dists) and 2) fraction of solvated samples (solv_fracs),
    and create a grid of split datasets of size n_samples.

    This is to breakdown a dataset into portions which can test the diversity required of both
    reactive configurations and solvated configurations for a given system in training MLIP models.

    Parameters:
        vacuum_ds_fpath (str):
        solvated_ds_fpath (str):
        rs (List[tuple]): 
        ts_ref_fpath (str):
        r_dists (List[float]): a list of max distances away from the TS to sample from, calculated as the sum of distances from the TS 
                                for each reaction coordinate in rs. Sampling occurs uniformly across each specified maximum range. I.e. [0.5, 1.0]
                                with n_samples=4 for will produce 1) 4 samples between (0.0, 0.5) and 2) 2 samples in each range (0.0, 0.5), (0.5, 1.0). This is to allow a uniform increasing of sampling
                                across the given distances.
        solv_fracs (List[float]): a list of solvent fractions to balance against vacuum samples, from values of 0.0 to 1.0.
        n_samples (int): for best uniform results n_samples should be divisible by len(solv_fracs) (so even splits can be made between vacuum and solvent)
        out_dir (str):
        save_name (str):
        seed: int = 3,
        sample_mode: str = ,

    Returns:


    """

    print('Using random seed: ', seed)
    random.seed(seed)

    # read input configs
    vac_configs = flexible_ase_read(vacuum_ds_fpath)
    solvated_configs = flexible_ase_read(solvated_ds_fpath)

    # get distances from TS
    ts_ref_config = flexible_ase_read(ts_ref_fpath)[0]
    ts_ref_bond_distances = {ri: ts_ref_config.get_distance(*ri) for ri in rs}
    logger.info(f'TS ref bond distances: {ts_ref_bond_distances}')
    vac_ts_dists = [sum([(config.get_distance(*ri, mic=True) - ts_ref_bond_distances[ri]) ** 2 for ri in rs]) 
                                 for config in vac_configs]
    solvated_ts_dists = [sum([(config.get_distance(*ri, mic=True) - ts_ref_bond_distances[ri]) ** 2 for ri in rs]) 
                                 for config in solvated_configs]

    # get list of indices of ts distances to filter out configs based on element
    # for i, config in enumerate(train_configs):
    #     assert config[r1[0]].symbol == 'C' and config[r1[1]].symbol == 'C'
    #     assert config[r2[0]].symbol == 'C' and config[r2[1]].symbol == 'C'
    # r1_atoms, r2_atoms = [3, 14], [8, 13] # reaction coordinates in water

    # construct dictionaries of vac and solvated configs within 
    # each range of distances from TS (for uniform sampling)
    r_dist_ranges = [(0.0, r_dists[0])] + [(r_dists[i], r_dists[i+1]) for i in range(len(r_dists) - 1)]
    print("R Dist Ranges: ", r_dist_ranges)
    vac_in_r_dist_ranges = {r_dist_range: [] for r_dist_range in r_dist_ranges}
    solvated_in_r_dist_ranges = {r_dist_range: [] for r_dist_range in r_dist_ranges}
    for r_dist_range in r_dist_ranges:
        print(r_dist_range)
        for i, config in enumerate(vac_configs):
            if r_dist_range[0] ** 2 <= vac_ts_dists[i] < r_dist_range[1] ** 2: 
                vac_in_r_dist_ranges[r_dist_range].append(config)
        
        print("Vac R Dist No. Samples: ", len(vac_in_r_dist_ranges[r_dist_range]))
        # vac_in_r_dist_ranges[r_dist_range] = sort_configs_by_ts_dist(vac_in_r_dist_ranges[r_dist_range], rs, ts_ref_bond_distances)
        shuffle(vac_in_r_dist_ranges[r_dist_range])
        # vac_in_r_dist_ranges[r_dist_range] = sort_configs_by_ts_dist(vac_in_r_dist_ranges[r_dist_range][0:n_samples], rs, ts_ref_bond_distances)
        # ordered_vac_sub_dists = [sum([(config.get_distance(*ri, mic=True) - ts_ref_bond_distances[ri]) ** 2 for ri in rs]) 
        #                          for config in vac_in_r_dist_ranges[r_dist_range]]
        # print('ordered_vac_sub_dists: ', ordered_vac_sub_dists)

        for i, config in enumerate(solvated_configs):
            if r_dist_range[0] ** 2 <= solvated_ts_dists[i] < r_dist_range[1] ** 2: 
                solvated_in_r_dist_ranges[r_dist_range].append(config)
        print("Solvated R Dist No. Samples: ", len(solvated_in_r_dist_ranges[r_dist_range]))
        # solvated_in_r_dist_ranges[r_dist_range] = sort_configs_by_ts_dist(solvated_in_r_dist_ranges[r_dist_range], rs, ts_ref_bond_distances)
        shuffle(solvated_in_r_dist_ranges[r_dist_range])
        # solvated_in_r_dist_ranges[r_dist_range] = sort_configs_by_ts_dist(solvated_in_r_dist_ranges[r_dist_range][0:n_samples], rs, ts_ref_bond_distances)
        # ordered_solvated_sub_dists = [sum([(config.get_distance(*ri, mic=True) - ts_ref_bond_distances[ri]) ** 2 for ri in rs]) 
        #                     for config in solvated_in_r_dist_ranges[r_dist_range]]
        # print('ordered_solvated_sub_dists: ', ordered_solvated_sub_dists)

    print('R Dist Ranges: ', r_dist_ranges)
    for i, curr_r_dist_range in enumerate(r_dist_ranges):
        max_r_dist = r_dist_ranges[i][1]

        if sample_mode == 'diverse':
            # we sample evenly across ranges up until the specified max range (to uniformly sample configs outward from the TS, 'Downhill')
            print(f'Max R Dist < {max_r_dist}')
            r_dist_ranges_in_threshold = r_dist_ranges[0:i+1]
            print('i + 1: ', i + 1)
            n_samples_per_range = int(n_samples / (i + 1))
            remainder = n_samples % (i + 1)
            print('R Dist Ranges in Threshold: ', r_dist_ranges_in_threshold)
            print('N Samples Per Range: ', n_samples_per_range)
            print('N Samples Remainder: ', remainder)

            # currently taking the closest to the TS for each sub-sample in each range
            vac_in_r_dist = [config for r_dist_range in r_dist_ranges_in_threshold for config in vac_in_r_dist_ranges[r_dist_range][0:n_samples_per_range]]
            solvated_in_r_dist = [config for r_dist_range in r_dist_ranges_in_threshold for config in solvated_in_r_dist_ranges[r_dist_range][0:n_samples_per_range]]

            # add remainding configurations from latest range 
            if remainder != 0:
                vac_in_r_dist += vac_in_r_dist_ranges[curr_r_dist_range][n_samples_per_range: n_samples_per_range + remainder]
                solvated_in_r_dist += solvated_in_r_dist_ranges[curr_r_dist_range][n_samples_per_range: n_samples_per_range + remainder]
            print(f"No. vac configs in r_dist: {len(vac_in_r_dist)}")
            print(f"No. solvated configs in r_dist: {len(solvated_in_r_dist)}")

            vac_in_r_dist_dists = [sum([(config.get_distance(*ri, mic=True) - ts_ref_bond_distances[ri]) ** 2 for ri in rs]) for config in vac_in_r_dist]
            solv_in_r_dist_dists = [sum([(config.get_distance(*ri, mic=True) - ts_ref_bond_distances[ri]) ** 2 for ri in rs]) for config in solvated_in_r_dist]
            print('Vac in R Dist: ', vac_in_r_dist_dists)
            print('Solvated in R Dist: ', solv_in_r_dist_dists)
        elif sample_mode == 'local':
            vac_in_r_dist = [config for config in vac_in_r_dist_ranges[curr_r_dist_range][0:n_samples]]
            solvated_in_r_dist = [config for config in solvated_in_r_dist_ranges[curr_r_dist_range][0:n_samples]]

        # select solvation fractions
        for frac in solv_fracs:
            print('Solvated Fraction: ', frac)
            vac_n_samples = int(n_samples * (1 - frac))
            solvated_n_samples = int(n_samples * (frac))

            # take sub-samples based on vac:solvated fraction
            vac_sample = outside_in(vac_in_r_dist)[0:vac_n_samples] if vac_n_samples > 0 else []
            solvated_sample = outside_in(solvated_in_r_dist)[0:solvated_n_samples:] if solvated_n_samples > 0 else []

            print(f'Num Vac Samples: {len(vac_sample)}')
            print(f'Num Solvated Samples: {len(solvated_sample)}')

            if save:
                # combine dataset and save
                combined = vac_sample + solvated_sample
                shuffle(combined)
                max_r_dist_str = str(max_r_dist).replace('.', '_')
                rep_num_str = f"_rep_{rep_num}" if rep_num is not None else ''
                out_fp = f"{out_dir}/{save_name}_n={n_samples}_r_dist_{max_r_dist_str}_{sample_mode}_solv_{int(frac * 100)}{rep_num_str}.xyz"
                write(out_fp, combined, format='extxyz')
                logger.info(f"Saved sample to {out_fp}...")


def _reaction_coordinate(atoms, ref_ts_config, react_coords, mic=True):
    """
    Construct the 1D reaction coordinate from the difference of two bond-length
    changes relative to a reference transition-state configuration.

    The coordinate is defined as:

        x = (d1(atoms) - d1(ref_ts_config)) - (d2(atoms) - d2(ref_ts_config))

    where d1 and d2 are the two bond lengths defined by `react_coords`.

    Parameters
    ----------
    atoms : ase.Atoms
        Configuration to evaluate.
    ref_ts_config : ase.Atoms
        Reference transition-state configuration.
    react_coords : tuple[tuple[int, int], tuple[int, int]]
        Two atom-index pairs defining the two bond lengths.
        Example: ((0, 1), (2, 3)).
    mic : bool, default=True
        Whether to use the minimum-image convention.

    Returns
    -------
    float
        Reaction coordinate value for `atoms`.
    """
    pair1, pair2 = react_coords
    d1 = atoms.get_distance(*pair1, mic=mic)
    d2 = atoms.get_distance(*pair2, mic=mic)
    d1_ref = ref_ts_config.get_distance(*pair1, mic=mic)
    d2_ref = ref_ts_config.get_distance(*pair2, mic=mic)
    return (d1 - d1_ref) + (d2 - d2_ref)


def _weighted_permutation_from_seed(weights, seed):
    """
    Generate a weighted random permutation using the Gumbel-top-k trick.

    This returns a full ordering of the input items such that larger weights
    are more likely to appear earlier in the permutation.

    Parameters
    ----------
    weights : array-like, shape (N,)
        Nonnegative sampling weights. They should sum to 1, but do not need to.
    seed : int or None
        Seed used to initialize the random number generator for this permutation.

    Returns
    -------
    np.ndarray, shape (N,)
        Permutation of indices from 0 to N-1, ordered from highest-ranked
        weighted draw to lowest-ranked weighted draw.
    """
    weights = np.asarray(weights, dtype=float)
    if np.any(weights < 0):
        raise ValueError("weights must be nonnegative.")
    if np.all(weights == 0):
        raise ValueError("At least one weight must be positive.")

    rng = np.random.default_rng(seed)
    g = rng.gumbel(size=len(weights))
    keys = np.log(weights) + g
    order = np.argsort(keys)[::-1]
    return order


def ts_gaussian_importance_sample(
    configs: List[ase.Atoms],
    ref_ts_config: ase.Atoms,
    react_coords: List[tuple],
    u: float = 0.0,
    sigma: float = 1.5,
    num_reps: int = 3,
    seeds: list = None,
    alpha=1.0,
    beta=0.0,
    bw_method="scott",
    eps=1e-8,
    plot=True,
    bins=40,
    mic=True,
    save_fp: str = None
):
    """
    Reweight a set of ASE configurations toward a Gaussian target over a 1D
    reaction coordinate, then generate one weighted permutation per replicate
    using a supplied list of seeds.

    For each replicate:
    1. Compute the importance weights.
    2. Generate a Gumbel-weighted permutation of the full dataset.
    3. Return the full permutation.

    The plotted "after" distribution shows the actual selected reaction-coordinate
    values for the first 50 points from each replicate.

    Parameters
    ----------
    configs : list[ase.Atoms]
        Source configurations.
    ref_ts_config : ase.Atoms
        Reference transition-state configuration used to define the coordinate.
    react_coords : tuple[tuple[int, int], tuple[int, int]]
        Two atom-index pairs defining the bond lengths used in the coordinate.
    u : float, default=0.0
        Mean of the target Gaussian distribution in reaction-coordinate space.
    sigma : float, default=1.5
        Standard deviation of the target Gaussian distribution.
    num_reps : int, default=3
        Number of independent replicates to generate.
    seeds : list[int] or tuple[int] or None
        List of seeds, one per replicate. Must have length equal to `num_reps`.
        If None, a ValueError is raised.
    alpha : float, default=0.5
        Tempering factor in (0, 1]. Smaller values reduce extreme weights.
    bw_method : str or float, default="scott"
        Bandwidth method passed to scipy.stats.gaussian_kde.
    eps : float, default=1e-8
        Small positive constant used to stabilize density ratios.
    plot : bool, default=True
        If True, plot the original distribution and overlay the selected samples
        from each replicate.
    bins : int, default=40
        Number of histogram bins.
    mic : bool, default=True
        Whether to use the minimum-image convention when measuring distances.

    Returns
    -------
    permutations : list[np.ndarray]
        A list of length `num_reps`. Each entry is a full weighted permutation
        of indices for the dataset.
    """
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1].")

    if len(configs) == 0:
        raise ValueError("configs must not be empty.")

    if seeds is None:
        raise ValueError("seeds must be provided as a list or tuple with one seed per replicate.")

    if len(seeds) != num_reps:
        raise ValueError("Length of seeds must equal num_reps.")

    # Build the 1D coordinate for all configurations.
    x1 = np.array(
        [
            _reaction_coordinate(config, ref_ts_config, react_coords, mic=mic)
            for config in configs
        ],
        dtype=float,
    )

    # estimate source density.
    kde = gaussian_kde(x1, bw_method=bw_method)
    f_source = np.maximum(kde(x1), eps)

    # target gaussian distribution 
    log_q = -0.5 * ((x1 - u) / sigma) ** 2 + beta * (x1 - u)

    # normalize for numerical stability
    log_q = log_q - np.max(log_q)

    f_target = np.exp(log_q)
    f_target = np.maximum(f_target, eps)

    # tempered importance weights.
    weights = (f_target / f_source) ** alpha
    weights = np.maximum(weights, eps)
    weights /= weights.sum()

    sampled_configs = []
    sampled_x1 = []
    permutations = []

    eg_sample_size = 50
    for i in range(num_reps):
        seed = seeds[i]
        order = _weighted_permutation_from_seed(weights, seed)
        permutations.append([configs[idx] for idx in order])

        idx = order[:eg_sample_size]
        sampled_configs.append([configs[i] for i in idx])
        sampled_x1.append(x1[idx])

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        # Common bins for both panels
        bin_edges = np.histogram_bin_edges(x1, bins=bins)
        bin_width = bin_edges[1] - bin_edges[0]
        xs = np.linspace(x1.min(), x1.max(), 500)

        cmap = plt.get_cmap('tab10')

        # --- Before ---
        counts_before, _, _ = axes[0].hist(
            x1,
            bins=bin_edges,
            alpha=0.7,
            color=cmap(0),
            label="Before"
        )

        kde_before = gaussian_kde(x1, bw_method=bw_method)
        kde_before_counts = kde_before(xs) * len(x1) * bin_width
        axes[0].plot(xs, kde_before_counts, linewidth=2, color=cmap(0), label="KDE approx")

        axes[0].set_title("Before")
        axes[0].set_xlabel("Reaction coordinate")
        axes[0].set_ylabel("Count")
        axes[0].legend()

        # --- After (replicates overlay) ---
        max_after = 0
        for i, xr in enumerate(sampled_x1, start=1):
            counts, _, _ = axes[1].hist(
                xr,
                bins=bin_edges,
                alpha=0.3,
                color=cmap(i),
                label=f"Replicate {i}"
            )
            max_after = max(max_after, counts.max())

            kde_rep = gaussian_kde(xr, bw_method=bw_method)
            kde_rep_counts = kde_rep(xs) * len(xr) * bin_width
            axes[1].plot(xs, kde_rep_counts, color=cmap(i), linewidth=2)

        axes[1].set_title(f"First {eg_sample_size} points of each replicate (sigma={sigma}, beta={beta})")
        axes[1].set_xlabel("Reaction coordinate")
        axes[1].legend()

        # Match y-limits
        ymax = max(counts_before.max(), max_after)
        axes[0].set_ylim(0, ymax * 1.05)
        axes[1].set_ylim(0, ymax * 1.05)

        plt.tight_layout()

        if save_fp is not None:
            fig.savefig(save_fp, dpi=DPI)
            print(f"Saved e.g. sample plot to {save_fp}")

    return permutations
