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
# TODO: figure out good values... Could put a breakpoint in the place where both matrices are in memory,
#  and look at what scores there are for augment ones that correspond to EagerMOT ones near the threshold...
#  That ought to get us in the ballpark at least...
#  affinity_matrix_utils row 44 is a good place.
#  It has conc_matrix, matrix_0, and matrix_1 (after normalization but w/e comparisons can be made)
VIS_SIM_2D_THRESHOLDS = {1: 0.7,   # car
                         2: 0.7,   # pedestrian
                         3: 0.7,   # bicycle
                         4: 0.7,   # bus
                         5: 0.7,   # motorcycle
                         6: 0.7,   # trailer
                         7: 0.7}   # truck
