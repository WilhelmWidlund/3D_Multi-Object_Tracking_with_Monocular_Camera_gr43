"""
Child class AugMet_2D_Vis_Sim for AugmentationMethod parent class.
Implements a 2D visual similarity method for augmenting to the EagerMOT reidentification model

Requires the features in the extended detection results .json-file to be feature vectors of length n,
where n is the same length for ALL detections.

This method uses a combination of score function and history function to evaluate the similarity between a detection
and a tracklet. Each tracklet may have several feature vectors, from the previous detections it has been matched
with.
* The user may choose a score function for evaluating the similarity score between detection and _one_ previous
feature vector.
* The history function choice governs how the vector of individual scores between detection feature vector,
  and tracklet feature vectors, are summed up into one scalar score
"""

from configs.params import default_settings
from augmentation.augmentation_base import AugmentationMethod
import augmentation.augmentation_base_utils as utils_aug
import augmentation.visual_similarity_2d_utils as utils_vis_2d


class AugMet_Vis_Sim_2D(AugmentationMethod):
    def __init__(self, method_name: str, automatic_init: bool):
        super().__init__(method_name)
        if automatic_init:
            self.n = default_settings["n"]
            self.history_function = utils_vis_2d.history_functions[default_settings["history function"]][1]
            self.similarity_function = \
                utils_vis_2d.similarity_functions[default_settings["similarity function"]][1]
        else:
            self.n = None
            self.history_function = self.get_history_function(utils_vis_2d.history_functions)
            self.similarity_function = self.get_similarity_function(utils_vis_2d.score_functions)

    def get_history_function(self, history_functions):
        history_value = [utils_aug.choose_sub_parameter(history_functions)]
        if history_value == 2 or history_value == 3:
            self.n = utils_vis_2d.get_n()
        return history_functions[history_value][1]

    def get_similarity_function(self, score_functions):
        return score_functions[utils_aug.choose_sub_parameter(score_functions)][1]

    def get_features(self, detected_instances, tracks):
        # Get track feature vectors, if available
        tracks_feature_vector_histories = []
        for track in tracks:
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
            else:
                detection_instances_feature_vectors.append(None)
        return detection_instances_feature_vectors, tracks_feature_vector_histories

    def evaluate_score(self, element_0, element_1):
        if element_0 and element_1:
            return self.history_function(element_0, element_1, self.similarity_function, self.n)
        else:
            return -1


def create_child_object(method_name: str, automatic_init: bool):
    return AugMet_Vis_Sim_2D(method_name, automatic_init)
