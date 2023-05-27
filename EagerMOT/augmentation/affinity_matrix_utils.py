"""
Utility functions for creating and manipulating affinity matrices
"""

import numpy as np


def create_affinity_matrix(detections_features, tracklets_features, method_object):
    """
    Create an affinity matrix based on the features of a set of detections and a set of tracklets
    Returns a matrix with detections at rows and tracklets at columns, with the element on row i, column j
    being the similarity score of detection i and tracklet j
    """
    matrix = np.zeros((len(detections_features), len(tracklets_features)), dtype=np.float32)
    for i, element_0 in enumerate(detections_features):
        for j, element_1 in enumerate(tracklets_features):
            matrix[i, j] = method_object.evaluate_score(element_0, element_1)
    return matrix


def unnormalize_matrix(matrix_0, matrix_1):
    """
    Un-normalize matrix_1 to have values in the same range as matrix_0.
    Necessary because EagerMOT uses values in some arbitrary range rather than normalized, and the EagerMOT limit values
    are defined in relation to that arbitrary range.
    """
    return matrix_1*np.max(np.absolute(matrix_0))


def concatenate_matrices(matrix_0, matrix_1, bias_ratio=0.5):
    """
    Take in two matrices and concatenate them by elementwise addition.
    Only concatenates when the element in matrix_1 is nonnegative: negative values signal that the augmentation
    method is invalid for that element.
    (For instance a detection i paired with a tracklet j, where the tracklet does not have a feature vector)
    First, matrix_1 is unnormalized to have values in the same range as matrix_0, then the concatenation is done:
    conc_matrix[i, j] = matrix_0[i, j]*bias_ratio + matrix_1[i, j]*(1-bias_ratio)
    Expects bias_ratio in the interval [0, 1]
    """
    if not matrix_0.shape == matrix_1.shape:
        # Non-matching matrix dimensions
        return matrix_0
    matrix_1 = unnormalize_matrix(matrix_0, matrix_1)
    dims = np.shape(matrix_0)
    conc_matrix = matrix_0.copy()
    for i in range(dims[0]):
        for j in range(dims[1]):
            if matrix_1[i, j] > 0:
                conc_matrix[i, j] = matrix_0[i, j]*bias_ratio + matrix_1[i, j]*(1-bias_ratio)
    return conc_matrix
