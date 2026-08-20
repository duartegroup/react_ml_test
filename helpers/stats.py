"""
Helpers for statistical analyses.
"""

import math
from typing import Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from ase import Atoms
from ase.build.rotate import minimize_rotation_and_translation

def max_norm(x: np.ndarray):
    return x / np.max(x)

def bound_norm(x: np.ndarray, min_val: float, max_val: float):
    return (x - min_val) / (np.max(x) - min_val)

def min_max_norm(x: np.ndarray):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def mean_center(x: np.ndarray):
    return x - np.mean(x, axis=0)

def calc_euclid_dist(X: np.ndarray, Y: np.ndarray):

    return euclidean_distances(X, Y)


def calc_cos_sim(v: np.ndarray, M: np.ndarray, zeta: float = 1.0):
    """
    Compute pairwise cosine similarities between an input vector v: (1, l) and a matrix 
    M: (m, l), with parameter zeta for sharpening or softening the similarity comparison, 
    by default 

    K(v, M) = (<v, M> / (||v||*||M||))^zeta

    Parameters:
        v (np.ndarray): vector of size (1, l)
        M (np.ndarray): matrix of size (m, l)
        zeta (float): power of the cos sim calculations, zeta=1.0 is standard cos sim, 
                      but if zeta > 1, you sharpen contrast, and zeta <1.0, softens contrast.

    Returns:
        matrix of (m, l)
    """

    return np.power(cosine_similarity(v, M), zeta)


def calculate_mae(predictions, actuals):
    """Calculate MAE between predictions and actual values."""
    return np.mean(np.abs(predictions - actuals))

def calculate_rmse(predictions, actuals):
    """Calculate RMSE between predictions and actual values."""
    return np.sqrt(np.mean(np.square(predictions - actuals)))

def aligned_rmsd(ref: Atoms,
                pred: Atoms,
                n_atoms: int = None):

    # Remove PBC
    ref.set_pbc(False)
    pred.set_pbc(False)

    if n_atoms is not None:
        ref = ref[:n_atoms]
        pred = pred[:n_atoms]
    else:
        ref = ref.copy()
        pred = pred.copy()

    minimize_rotation_and_translation(ref, pred)

    diff = pred.get_positions() - ref.get_positions()
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    return rmsd

def calculate_rmse_on_errors(errors):
    """Calculate RMSE on precalculated errors."""
    return np.sqrt(np.mean(np.square(errors)))

def calculate_mag_rmse(predictions: np.ndarray, # (n, 3)
                       actuals: np.ndarray): # (n, 3)
    """Calculate RMSE of norms of differences between predicted and actual vectors."""
    diff_norms = np.linalg.norm(predictions - actuals, axis=1)
    return np.sqrt(np.mean(diff_norms** 2))


def calculate_comp_rmse(predictions: np.ndarray, # (n, 3)
                        actuals: np.ndarray): # (n, 3)
    """Calculate component wise RMSE between set of vectors
    See https://pubs.aip.org/aip/jcp/article/158/12/121501/2881528/How-to-validate-machine-learned-interatomic
    section IV B. Is the same as mag_rmse / sqrt(3)
    """
    return calculate_mag_rmse(predictions, actuals) / (3 ** 0.5)


def moving_rmse(predictions: np.ndarray, 
                actuals: np.ndarray, 
                window: int = 10,
                mode: str = 'same'):
    errors = (actuals - predictions)**2
    return np.sqrt(np.convolve(errors, np.ones(window)/window, mode=mode))


def round_sigfigs(x: Union[float, np.ndarray], sigfigs: int = 3):
    
    from math import log10, floor

    if type(x) is np.ndarray:
        print('Rounding numpy array!')
        with np.errstate(divide='ignore', invalid='ignore'):
            mags = np.floor(np.log10(np.abs(x)))
            decimals = sigfigs - mags - 1
            decimals = decimals.astype(int)
            return np.where(x == 0, 0, np.round(x, decimals))
    else:
        if x == 0:
            return 0
        elif math.isnan(x):
            return None
        scale = -floor(log10(abs(x))) + (sigfigs - 1)
        return round(x, scale)


def detect_outliers_iqr(data: np.ndarray, iqr_threshold: float = 2.0) -> np.ndarray:
    """
    Finsds and returns outliers from a numpy array using the Interquartile Range (IQR) method.

    Parameters:
        data (numpy.ndarray): The input array from which outliers are detected.

    Returns:
        - numpy.ndarray: the indexes of the outliers that were removed.
    """

    # calculate q1 (25th percentile) and q3 (75th percentile)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)

    # compute the IQR
    IQR = q3 - q1

    # define lower and upper bounds for outlier detection
    lower_bound = q1 - iqr_threshold * IQR
    upper_bound = q3 + iqr_threshold * IQR

    # identify outlier indexes
    outlier_ids = np.where((data < lower_bound) | (data > upper_bound))[0]
    return outlier_ids

