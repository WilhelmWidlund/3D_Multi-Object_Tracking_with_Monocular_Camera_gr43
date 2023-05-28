"""
Child class AugMet_2D_Vis_Sim for AugmentationMethod parent class.
Implements a 2D visual similarity method for augmenting to the EagerMOT reidentification model

Requires the features in the extended detection results .json-file to be feature vectors of length n,
where n is the same length for ALL detections.

This method uses a combination of similarity function and history function to evaluate the similarity between a detection
and a tracklet. Each tracklet may have several feature vectors, from the previous detections it has been matched
with. The evaluate_score function required by the AugmentationMethod parent class is here made up of two sub-functions:
* The user may choose a similarity function for evaluating the similarity score between detection and _one_ previous
feature vector.
* The history function choice governs how the vector of individual scores between detection feature vector,
  and tracklet feature vectors, are summed up into one final scalar score
"""

from os import path
from json import dump

from augmentation.augmentation_params import default_settings, VISUAL_SIM_NAME, VIS_SIM_2D_THRESHOLDS
from augmentation.augmentation_base import AugmentationMethod
import augmentation.augmentation_base_utils as utils_aug
import augmentation.visual_similarity_2d_utils as utils_vis_2d


class AugMet_Vis_Sim_2D(AugmentationMethod):
    def __init__(self, method_name: str, automatic_init: bool, eagermot_thresholds: dict):
        super().__init__(method_name, automatic_init, eagermot_thresholds)
        if automatic_init:
            self.n = default_settings["n"]
            self.history_name = utils_vis_2d.history_functions[default_settings["history function"]][0]
            self.similarity_name = utils_vis_2d.similarity_functions[default_settings["similarity function"]][0]
            self.history_function = utils_vis_2d.history_functions[default_settings["history function"]][2]
            self.similarity_function = \
                utils_vis_2d.similarity_functions[default_settings["similarity function"]][2]
        else:
            self.n = None
            self.history_name = None
            self.similarity_name = None
            self.history_function = self.get_history_function(utils_vis_2d.history_functions)
            self.similarity_function = self.get_similarity_function(utils_vis_2d.similarity_functions)

    def setup_map_ratio(self, thresholds_eagermot: dict):
        map_ratio_per_class = {}
        for key in thresholds_eagermot:
            try:
                adapted_threshold_vis_sim_2d = utils_aug.adapt_normalized_score(VIS_SIM_2D_THRESHOLDS[key])
            except KeyError:
                print("Warning: no recognition threshold for class " + str(key) + " found! Using EagerMOT threshold.")
                adapted_threshold_vis_sim_2d = 1
            map_ratio_per_class[key] = thresholds_eagermot[key] / adapted_threshold_vis_sim_2d
        return map_ratio_per_class

    def get_history_function(self, history_functions):
        history_value = utils_aug.choose_hyper_parameter(history_functions)
        if history_value == 2 or history_value == 3:
            self.n = utils_vis_2d.setup_n()
            self.history_name = history_functions[history_value][0]
        else:
            self.history_name = history_functions[history_value][0]
        return history_functions[history_value][2]

    def get_similarity_function(self, similarity_functions):
        similarity_choice = utils_aug.choose_hyper_parameter(similarity_functions)
        self.similarity_name = similarity_functions[similarity_choice][0]
        return similarity_functions[similarity_choice][2]

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

    def save_augmentation_parameters(self, save_path: str):
        params_to_write = {"Hyperparameters": {"Bias ratio": self.bias_ratio,
                                               "Similarity function": self.similarity_name,
                                               "History function": self.history_name,
                                               "n": self.n},
                           "Class thresholds": {"Car": VIS_SIM_2D_THRESHOLDS[1],
                                                "Pedestrian": VIS_SIM_2D_THRESHOLDS[2],
                                                "Bicycle": VIS_SIM_2D_THRESHOLDS[3],
                                                "Bus": VIS_SIM_2D_THRESHOLDS[4],
                                                "Motorcycle": VIS_SIM_2D_THRESHOLDS[5],
                                                "Trailer": VIS_SIM_2D_THRESHOLDS[6],
                                                "Truck": VIS_SIM_2D_THRESHOLDS[7]}}
        params_file = path.join(save_path, (VISUAL_SIM_NAME + "_parameters.json"))
        with open(params_file, 'w') as f:
            dump(params_to_write, f, indent=4)


def create_child_object(method_name: str, automatic_init: bool, eagermot_thresholds: dict):
    return AugMet_Vis_Sim_2D(method_name, automatic_init, eagermot_thresholds)
