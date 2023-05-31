"""
Parameters and variables for implemented augmentation methods
"""

from configs.local_variables import MOUNT_PATH, SPLIT

# Name of the implemented visual_similarity_2d method
VISUAL_SIM_NAME = "TorchReID"
# Choose whether to use default settings for all options, making the entire tracking script automatic
AUTOMATIC_INIT = False
# Add implemented augmentation methods to this dictionary as n: ["Name", create_child_object_function, "Description"]
augmentation_methods = {'Type': "augmentation method",
                        1: [VISUAL_SIM_NAME, " 2D image recognition between detections and tracklets", "vis_sim_2d"]}
# Default settings for augmentation
default_folder = MOUNT_PATH + "/Embeddings/" + VISUAL_SIM_NAME + "/"
default_settings = {"method": "vis_sim_2d", "name": VISUAL_SIM_NAME, "folder": default_folder, "bias ratio": 0.5,
                    "n": None, "history function": 1, "similarity function": 1}
# Threshold parameters
VIS_SIM_2D_THRESHOLDS = {1: 0.65,   # car
                         2: 0.6,   # pedestrian
                         3: 0.6,   # bicycle
                         4: 0.8,   # bus
                         5: 0.5,   # motorcycle
                         6: 0.7,   # trailer
                         7: 0.8}   # truck
