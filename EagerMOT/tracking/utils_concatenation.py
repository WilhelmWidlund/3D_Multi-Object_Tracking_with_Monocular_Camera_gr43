# ----------------- Altered code -----------------------------------------------------
# utility functions for concatenation mode

import numpy as np
from inputs.utils import ask_if_default_folder


def generic_similarity_matrix(detected_feature_vectors, track_feature_vector_histories, similarity_function):
    matrix = np.zeros((len(detected_feature_vectors), len(track_feature_vector_histories)), dtype=np.float32)
    for i, element_0 in enumerate(detected_feature_vectors):
        for j, element_1 in enumerate(track_feature_vector_histories):
            if element_0 and element_1:
                # If both detection and track have feature vector(s), calculate the average score between the
                # detection vector and the vectors associated with the track.
                average_score = 0
                for k in range(len(element_1)):
                    average_score += similarity_function(element_0, element_1[k])
                average_score = average_score/len(element_1)
                matrix[i, j] = average_score
            else:
                # Else, set the score as -1, used as a flag to indicate that this element should not be considered
                # when concatenating matrices.
                matrix[i, j] = -1
    return matrix

def unnormalize_matrix(matrix_0, matrix_1):
    """
    Un-normalize matrix_1 to have values in the same range as matrix_0.
    Necessary because EagerMOT uses values in some arbitrary range rather than normalized, and the EagerMOT limit values
    are defined in relation to that arbitrary range...
    """
    return matrix_1*np.max(np.absolute(matrix_0))


def concatenate_matrices(matrix_0, matrix_1, bias_ratio=0.5):
    """
    Take in two matrices and concatenate them by elementwise addition.
    Only concatenates when the element in matrix_1 is nonnegative: negative values signal that the concatenation
    method is invalid for that element.
    (For instance a detection i paired with a tracklet j, where the tracklet does not have a feature vector)
    First, matrix_1 is unnormalized to have values in the same range as matrix_0, then the concatenation is done:
    conc_matrix[i, j] = matrix_0[i, j]*bias_ratio + matrix_1[i, j]*(1-bias_ratio)
    Expects bias_ratio in the interval [0, 1]
    """
    if not matrix_0.shape == matrix_1.shape:
        # Non-matching matrix dimensions
        # TODO: There is some dimension error: we get different amount of detections in matrix_0 and matrix_1...
        #  but other than that it looks nice, matrix_1 has values that look like they should and stuff...
        return matrix_0
    matrix_1 = unnormalize_matrix(matrix_0, matrix_1)
    dims = np.shape(matrix_0)
    conc_matrix = matrix_0
    for i in dims[0]:
        for j in dims[1]:
            if matrix_1[i, j] > 0:
                conc_matrix[i, j] = matrix_0[i, j]*bias_ratio + matrix_1[i, j]*(1-bias_ratio)
    return conc_matrix


def get_bias_ratio():
    while True:
        try:
            print("Insert bias ratio in the range [0, 1]:\n"
                  "(1 = only use EagerMOT Re-ID, 0 = only use concatenated method Re-ID)")
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


def get_concatenation_source_folder_address(base_addr: str):
    query = "Would you like to use the default concatenation source? \n" \
            "(TorchReID image recognition between current detections and tracklets)"
    prompt = "Write the path to your saved results folder..."
    source_path = ask_if_default_folder(base_addr, query, prompt, False)
    if source_path != base_addr:
        return source_path
    else:
        return base_addr + "/Embeddings/TorchReID/"


def get_feature_vectors(detected_instances, tracks):
    # Get track feature vectors, if available
    tracks_feature_vector_histories = []
    for track in tracks:
        # It should be changed so that each time a Track object is created and/or updated,
        # a check is made for feature vector in its instance.detection_d2.
        # If there is one, then it should be appended to Track_object.feature_vector_history
        # When that's done, we can check here for such a feature_vector_history:
        if track.feature_vector_history:
            tracks_feature_vector_histories.append(track.feature_vector_history)
        else:
            tracks_feature_vector_histories.append(None)
    # Get detection feature vectors, if available
    detection_instances_feature_vectors = []
    for detection_instance in detected_instances:
        if hasattr(detection_instance, 'detection_2d'):
            if hasattr(detection_instance.detection_2d, 'feature_vector'):
                detection_instances_feature_vectors.append(detection_instance.detection_2d.feature_vector)
        else:
            detection_instances_feature_vectors.append(None)
    return detection_instances_feature_vectors, tracks_feature_vector_histories
# ----------------- End altered code -----------------------------------------------------

