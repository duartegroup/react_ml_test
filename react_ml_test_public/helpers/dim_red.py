"""
Helpers for running dimensionality reduction algorithms and analysis.
"""
from typing import Literal
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from react_ml_test_public.helpers.io import load_data_cache, save_npz_data_cache
from react_ml_test_public.helpers.stats import calc_cos_sim, calc_euclid_dist
from react_ml_test_public.helpers.logging import logger
from react_ml_test_public.helpers.utils import unflatten_array_by_lengths


def run_pca(descriptors, n_components: int = 2):
    """
    Perform PCA on the dataset and return the reduced data.
    :param descriptors: Input data (n_samples x n_features)
    :param n_components: Number of components for PCA (default is 2 for 2D plot)
    :return: PCA-transformed data
    """
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(descriptors)
    return pca_result


def run_tsne(descriptors, n_components=2, perplexity=30.0, random_state=42):
    """
    Perform t-SNE on the dataset and return the reduced data.
    :param descriptors: Input data (n_samples x n_features)
    :param n_components: Number of components for t-SNE (default is 2 for 2D plot)
    :param perplexity: Perplexity parameter for t-SNE (default is 30.0)
    :param random_state: Random state for reproducibility (default is 42)
    :return: t-SNE-transformed data
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    tsne_result = tsne.fit_transform(descriptors)
    return tsne_result


def run_umap(descriptors, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42, n_jobs: int = 1):
    """
    Perform UMAP on the dataset and return the reduced data.
    :param descriptors: Input data (n_samples x n_features)
    :param n_components: Number of components for UMAP (default is 2 for 2D plot)
    :param n_neighbors: The number of neighbors to use in the UMAP algorithm (default is 15)
    :param min_dist: Minimum distance between points (default is 0.1)
    :param random_state: Random state for reproducibility (default is 42)
    :return: UMAP-transformed data
    """
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    return reducer.fit_transform(descriptors)


def calc_best_similarities(x1: np.ndarray,
                           x2: np.ndarray,
                           zeta: float = 1.0,
                           x1_atom_nums: np.ndarray = None,
                           x2_atom_nums: np.ndarray = None,
                           element_stratify_mean: bool = False,
                           sim_calc: Literal['cosine', 'euclidean'] = 'cosine',
                           x1_batch_size: int = 100,
                           x2_batch_size: int = 1000,
                           config_lens: list[int] = None,
                           cache_fpath: str = None):
    """
    The best similarity for each atom desc i in x1, is the cosine similarity closest to 1.0 in
    the cosine similarity matrix row: cos_sim[i, j] for all atom desc j in x2.

    (As in the MACE-MP-0 paper supplementary).

    Parameters:
        x1 (np.ndarray): Test descriptors array. An array of shape (n, l).
        x2 (np.ndarray): Reference descriptors array (normally much bigger). An array of shape (m, l).
        x1_atom_nums (np.ndarray): atom numbers corresponding to atomic descriptor vectors in x1 
        x2_atom_nums (np.ndarray): atom numbers corresponding to atomic descriptor vectors in x2 
        zeta (float): parameter for sharpening or softening the similarity comparison, by default
                      zeta=1.0 is standard cos sim, but if zeta > 1, you sharpen contrast, and zeta <1.0,
                      softens contrast.
        element_stratify_mean (bool): true if the output mean is to be stratified by element,
                                      i.e. the mean of element-wise means, false if we simply
                                      take the overall mean
        sim_calc Literal['cosine', 'euclidean']: whether to calculate similarity based on cosine similarity or euclidean distance

    Returns:
        best_sim_tuples (List[Tuple]): A list of Tuples of: [
        (x1_id, x1_element, best_matching_x2_id, x2_element, best_matching_cos_sim_value), ...]
        best_sim_element_means (dict): element-wise means of best cosine similarity
        best_sim_mean (float): the mean of best cosine similarities 
                               (element-wise if element_stratify_mean is True).
    """

    # check cache for existing comparison
    cache_data = load_data_cache(cache_fpath)
    if cache_data is not None:
        best_sim_values, best_sim_indices, config_lens = cache_data['best_sim_values'], cache_data['best_sim_indices'], cache_data['config_lens']
    else:
    
        assert x1.shape[1] == x2.shape[1], f'Shape of second dim on x1 and x2 must be the same... but are of size {x1.shape[1]} and {x2.shape[1]}...'

        # do block-wise similarity calculations for memory efficiency
        n1, n2 = x1.shape[0], x2.shape[0]    
        best_sim_values = np.empty((n1, 1), dtype=np.float32)
        best_sim_indices = np.empty((n1, 1), dtype=np.float32)

        # for each batch in x1
        for i in range(0, n1, x1_batch_size):
            i_end = min(i + x1_batch_size, n1)

            actual_x1_batch_size = i_end - i
            x1_batch_sim_vec = np.empty((actual_x1_batch_size, n2), dtype=np.float32)

            if i == 0:
                logger.info(f"Allocated array shape: {x1_batch_sim_vec.shape}")
                logger.info(f"Allocated calculation array size: {x1_batch_sim_vec.nbytes / (1024 ** 2):.2f}MB")

            for j in range(0, n2, x2_batch_size):
                j_end = min(j + x2_batch_size, n2)
                
                # compute pairwise similarities for block
                if sim_calc == 'cosine':
                    x1_batch, x2_batch = x1[i:i_end], x2[j:j_end]
                    # sim_block = torch.mm(x1_batch, x2_batch.T)  # [batch_A, batch_B]
                    # row_results.append(sim_block.cpu()) 
                    x1_batch_sim_vec[0:actual_x1_batch_size, j:j_end] = calc_cos_sim(x1_batch, x2_batch, zeta)
                elif sim_calc == 'euclidean':
                    x1_batch_sim_vec[0:actual_x1_batch_size, j:j_end] = calc_euclid_dist(x1[i:i_end], x2[j:j_end], zeta)
            
            x1_batch_best_sim_values = np.max(x1_batch_sim_vec, axis=1)
            x1_batch_best_sim_indices = np.argmax(x1_batch_sim_vec, axis=1)
            best_sim_values[i:i_end, 0] = x1_batch_best_sim_values
            best_sim_indices[i:i_end, 0] = x1_batch_best_sim_indices

            if i % 10 == 0:
                logger.info(f"Calculated similarities for {i_end} / {n1} x1 atoms...")

        # get best similarities of each x1 -> x2 descriptor
        # best_sim_values = np.max(sim_vec, axis=1)
        # best_sim_indices = np.argmax(sim_vec, axis=1)
        logger.info(f"Best Sim Values Shape: {best_sim_values.shape}")
        logger.info(f"Best Sim Indices Shape: {best_sim_indices.shape}")

        save_npz_data_cache(cache_fpath, {'best_sim_values': best_sim_values,
                                        'best_sim_indices': best_sim_indices,
                                        'config_lens': config_lens})

    if config_lens is not None:
        config_mean_best_sims = [np.mean(config_best_sim_values) for config_best_sim_values in unflatten_array_by_lengths(best_sim_values, config_lens)]
        
        # if element_stratify_mean:
        #     if x1_atom_nums is not None and x2_atom_nums is not None:
        #         # element-wise averaging of best values over each element
        #         best_cos_sims = np.column_stack((best_sim_indices, best_sim_values, x1_atom_nums))

        #         # Compute the per-group mean
        #         unique_atom_nums = np.unique(best_cos_sims[:, -1])
        #         # logger.info(f"Element stratifying mean for atom nums: {unique_atom_nums}")
        #         best_sim_element_means = {atom_num: best_cos_sims[best_cos_sims[:, -1] == atom_num][:, 1].mean(axis=0) 
        #                         for atom_num in unique_atom_nums}

        #         # get overall stratified per-element mean
        #         best_sim_mean = np.mean(np.array(list(best_sim_element_means.values())))
        #     else:
        #         raise ValueError("For element stratified mean you must specify atomic numbers for x1 dn x2, " \
        #         "corresponding to atomic descriptor vectors.")
        return config_mean_best_sims, best_sim_values, best_sim_indices

    else:
        return best_sim_values, best_sim_indices
    
    # create output list of tuples
    # best_sim_tuples = None
    # if x1_atom_nums is not None and x2_atom_nums is not None:
    #     best_sim_tuples = [(i, chemical_symbols[x1_atom_nums[i]], 
    #                         best_sim_indices[i], chemical_symbols[x2_atom_nums[best_sim_indices[i]]], 
    #                         best_sim_values[i]) 
    #                         for i in range(best_sim_indices.shape[0])]
    # else:
    #     best_sim_tuples = [(i, best_sim_indices[i], best_sim_values[i]) for i in range(best_sim_indices.shape[0])]