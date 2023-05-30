"""
Utility functions for augmenting EagerMOT with results from a visual similarity recognition neural network,
for instance TorchReID.
"""

import numpy as np
from numpy.linalg import norm
from augmentation.augmentation_base_utils import adapt_normalized_score


def cosine_similarity(feature_vector_0, feature_vector_1) -> float:
    """
    Returns the cosine distance, normalized to [0, 1]
    """
    fv_0 = np.array(feature_vector_0)
    fv_1 = np.array(feature_vector_1)
    cos_sim_norm = (1 + np.dot(fv_0, fv_1) / (norm(fv_0) * norm(fv_1))) / 2
    return adapt_normalized_score(cos_sim_norm)


# Once implemented, similarity functions should be added in this dictionary
similarity_functions = {'Type': "similarity function",
                   1: ["Cosine similarity score", False, cosine_similarity]}


def history_equal_weight(element_0, element_1, similarity_function, n):
    """
    History method for assigning equal weight to all feature vectors in the tracklet history
    """
    score = 0
    for k in range(len(element_1)):
        score += similarity_function(element_0, element_1[k])
    return score / len(element_1)


def history_n_last_equal_weight(element_0, element_1, similarity_function, n):
    """
        History method for assigning equal weight to the n most recent feature vectors in the tracklet history,
        or to all if all < n
    """
    score = 0
    all_count = len(element_1)
    if all_count < n:
        n = all_count
    for k in range(n):
        score += similarity_function(element_0, element_1[-k-1])
    return score / n


def history_n_last_diff_weights(element_0, element_1, similarity_function, n):
    """
        History method for assigning different weights the n most recent feature vectors
        in the tracklet history, or to all if all < n
        Assigns 50% weight to the most recent, 25% to the second-most, 12.5% to the third-most, etc.
        until the n-th-most, which receives the remaining weight
    """
    score = 0
    weight = 0.5
    accum_weight = 0
    all_count = len(element_1)
    if all_count < n:
        n = all_count
    for k in range(n-1):
        score += similarity_function(element_0, element_1[-k-1]) / weight
        accum_weight += weight
        weight /= 2
    score += similarity_function(element_0, element_1[-n]) / (1 - accum_weight)
    return score


def history_last_only(element_0, element_1, similarity_function, n):
    """
    History method for only regarding the most recent feature vector in the tracklet history
    """
    return similarity_function(element_0, element_1[-1])


# Once implemented, history methods should be added in this dictionary
history_functions = {'Type': "method for handling the feature vector history of a tracklet",
                     1: ["Equal weight for all feature vectors", False,
                         history_equal_weight],
                     2: ["Equal weight for the n most recent feature vectors", False,
                         history_n_last_equal_weight],
                     3: ["The n most recent feature vectors, with weight 1/2, 1/4, 1/8 etc", False,
                         history_n_last_diff_weights],
                     4: ["Only consider the most recent feature vector", False,
                         history_last_only]}


def setup_n():
    print("Input the desired number n >= 1.")
    while True:
        try:
            n = int(input())
        except ValueError:
            print("Please print an integer.")
        else:
            if n < 1:
                print("Input a value n >= 1")
                continue
            return n