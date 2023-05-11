import numpy as np

from utils.utils_geometry import (iou_3d_from_corners, box_2d_overlap_union,
                                  tracking_center_distance_2d, tracking_distance_2d_dims, tracking_distance_2d_full)


def iou_bbox_3d_matrix(detections, predictions, detections_dims, predictions_dims):
    return generic_similarity_matrix_two_args(detections, predictions,
                                              detections_dims, predictions_dims, iou_3d_from_corners)


def distance_2d_matrix(centers_0, centers_1):
    return generic_similarity_matrix(centers_0, centers_1, tracking_center_distance_2d)


def distance_2d_dims_matrix(coords_0, coords_1):
    return generic_similarity_matrix(coords_0, coords_1, tracking_distance_2d_dims)


def distance_2d_full_matrix(coords_0, coords_1):
    return generic_similarity_matrix(coords_0, coords_1, tracking_distance_2d_full)


def iou_bbox_2d_matrix(det_bboxes, seg_bboxes):
    return generic_similarity_matrix(det_bboxes, seg_bboxes, box_2d_overlap_union)


def generic_similarity_matrix(list_0, list_1, similarity_function):
    """
    Returns a matrix of similarity function values,
    where rows represent detections and columns represent predictions (tracklets)
    """
    matrix = np.zeros((len(list_0), len(list_1)), dtype=np.float32)
    for i, element_0 in enumerate(list_0):
        for j, element_1 in enumerate(list_1):
            matrix[i, j] = similarity_function(element_0, element_1)
            # matrix[i, j] has the similarity score for {list_0[i], list_1[j]}
    return matrix

def generic_similarity_matrix_two_args(list_0, list_1, attrs_0, attrs_1, similarity_function):
    """
    Returns a matrix of similarity function values,
    where rows represent detections and columns represent predictions (tracks)
    """
    matrix = np.zeros((len(list_0), len(list_1)), dtype=np.float32)
    for i, element_0 in enumerate(list_0):
        for j, element_1 in enumerate(list_1):
            matrix[i, j] = similarity_function(element_0, element_1, attrs_0[i], attrs_1[j])
    return matrix

def unnormalize_matrix(matrix_0, matrix_1):
    """
    Un-normalize matrix_1 to have values in the same range as matrix_0
    """
    return matrix_1*np.max(np.absolute(matrix_0))

def concatenate_matrices(matrix_0, matrix_1, bias_ratio=0.5):
    """
    Take in two matrices and concatenate them by elementwise addition.
    First, matrix_1 is unnormalized to have values in the same range as matrix_0, then the concatenation is done:
    conc_matrix[i, j] = matrix_0[i, j]*bias_ratio + matrix_1[i, j]*(1-bias_ratio)
    Expects bias_ratio in the interval [0, 1], reverts to default value 0.5 if an invalid value is given
    """
    print(bias_ratio)
    if not matrix_0.shape == matrix_1.shape:
        # Non-matching matrix dimensions
        print('The matrix dimensions are not the same!')
        return matrix_0
    matrix_1 = unnormalize_matrix(matrix_0, matrix_1)
    return matrix_0*bias_ratio + matrix_1*(1-bias_ratio)

def get_bias_ratio():
    while True:
        try:
            print("Insert bias ratio in the range [0, 1]:")
            bias = float(input())
        except ValueError:
            print("Please print a number.")
            continue
        else:
            if(0 <= bias <= 1):
                return bias
            else:
                print("Bias ratio outside of permitted range")
                continue

