import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment


def construct_distance_cost(patches_1, patches_2):
    """
    Routine to build a cost matrix, where the entries are L2 distances between patches

    Args:
        patches_1 (np.ndarray): patches for sample 1. Expected dim B1 x latent_dim
        patches_2 (np.ndarray): patches for sample 2. Expected dim B2 x latent_dim

    Returns:
        Cost matrix: B1 x B2
    """
    if (len(patches_1.shape) != 2) and (len(patches_2.shape) != 2):
        raise ValueError("Expected 2-dim arrays as inputs!")

    latent_dim = patches_1.shape[1]

    if patches_2.shape[1] != latent_dim:
        raise ValueError("Patches should have the same latent dimension!")

    cost_matrix = distance_matrix(patches_1, patches_2, p=2)
    return cost_matrix


def match_single_patches(patches_1, patches_2, minimize=True):
    cost_matrix = construct_distance_cost(patches_1, patches_2)
    return linear_sum_assignment(cost_matrix=cost_matrix, maximize=~minimize)
