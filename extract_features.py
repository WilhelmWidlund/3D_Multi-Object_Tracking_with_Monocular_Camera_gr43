from torchreid.utils import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

detections_path = 'detections/mmdetection_cascade_x101/val'
# Get which scene we will download
nuscenes_root_path = "C:/Users/Elias/OneDrive/Dokument/LIU/Outgoing/Courses/CIVIL-459/EagerMOT/NuScenes" # Put here path the images
nuscenes_info_path = os.path.join(nuscenes_root_path, "v1.0-mini")

# Load tokens/info for nuscenes dataset
f = open(os.path.join(nuscenes_info_path,'scene.json'))
scene_info = json.load(f)
f.close()

f = open(os.path.join(nuscenes_info_path,'sensor.json'))
sensor_info = json.load(f)
f.close()

f = open(os.path.join(nuscenes_info_path,'sample.json'))
sample_info = json.load(f)
f.close()

f = open(os.path.join(nuscenes_info_path,'sample_data.json'))
sample_data_info = json.load(f)
f.close()

f = open(os.path.join(nuscenes_info_path,'calibrated_sensor.json'))
calibrated_sensor_info = json.load(f)
f.close()

# Create list of valid calibrated sensor tokens
camera_calibrated_sensors = []

for calibrated_sensor in calibrated_sensor_info:
    sensor_token_for_cal = calibrated_sensor['sensor_token']
    sensor_id_in_list = [i for i in range(len(sensor_info)) 
                                  if sensor_info[i]['token'] == sensor_token_for_cal]
    sensor_id_in_list = sensor_id_in_list[0]
    if sensor_info[sensor_id_in_list]['modality'] == 'camera':
        camera_calibrated_sensors.append(calibrated_sensor['token'])

# Load the REID network
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='log/modelzoo/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth',
    device='cpu',
    verbose=False
)

# Loop over scenes in dataset
for scene in scene_info:
    print("New Scene!")
    # Get the scene token
    scene_token = scene['token']
    # Open the correct detection file for this scene
    scene_token_json = scene_token + '_trainval_cascade_mask_rcnn_x101.json'
    scene_json_path = os.path.join(detections_path, scene_token_json)
    if(not os.path.exists(scene_json_path)):
        # Ignore scenes that no detections exists for, AKA train set
        continue

    # Load detections for this scene
    f = open(scene_json_path)
    detections = json.load(f)
    f.close()

    # The dictoinary we save the embeddings in
    detections_with_features = detections.copy()

    sample_token = scene['first_sample_token']
    last_sample_token = scene['last_sample_token']
    sample = {}

    # Loop all frames in this scene
    run = True

    while run:
        print("New Sample!")
        if sample_token == last_sample_token:
            print("Actually the last sample!")
            run = False

        # Find sample info
        sample_id_in_list = [i for i in range(len(sample_info)) 
                             if sample_info[i]['token'] == sample_token]
        sample_id_in_list = sample_id_in_list[0]

        # Get next samples token
        next_sample_token = sample_info[sample_id_in_list]['next']

        sample_data_id_in_list = [i for i in range(len(sample_data_info)) 
                                  if sample_data_info[i]['sample_token'] == sample_token]
        # Loop over all sensors in this frame
        for sample_data_id in sample_data_id_in_list:
            sample_data_frame = sample_data_info[sample_data_id]
            if not(sample_data_frame['calibrated_sensor_token'] in camera_calibrated_sensors):
                # Do nothing if sensor is not a camera
                continue
            print("New frame in the sample! Should be 6 of me per sample")

            # Load detections for this frame
            frame_token = sample_data_frame['token']
            # frame_detections = detections[sample_token][frame_token]
            # HERE IS WEIRD, IN NUSCENES THERE EXISTS A FRAME WITH A TOKEN BUT THE DETECTIONS FOR it DOES NOT EXIST?? MOVE THIS BEFORE WE READ THE IMAGE FOR SPEED
            try:
                frame_detections = detections[sample_token][frame_token]
            except:
                print('Apperently there are no detections for this frame: ')
                print('Sample token: '+ sample_token)
                print('Frame token: ' + frame_token)
                print('So lets skip it for now')
                continue

            # Load the frame
            frame = cv.imread(os.path.join(nuscenes_root_path, sample_data_frame['filename']))

            # Loop over all classes in the frame
            for obj_class, frame_detections in frame_detections.items():
                # Loop over each object in the frame
                for id,detection in enumerate(frame_detections):
                    # IF WRONG CLASS ADD EMPTY LIST
                    x1 = int(detection[0])
                    y1 = int(detection[1])
                    x2 = int(detection[2])
                    y2 = int(detection[3])

                    # Extract the image of the current object
                    object_image = frame[y1:y2, x1:x2, :]

                    # Pass the object image through REID network to create embedding
                    object_embedding = extractor(object_image) 

                    # Add feature vector to detection list
                    detections_with_features[sample_token][frame_token][obj_class][id].append(object_embedding.tolist())
        sample_token = next_sample_token
        

    # Save dictionary as JSON file
    save_root = 'embeddings'
    save_file_name = scene_token + ".json"
    print(os.path.join(save_root, save_file_name ))

    json_obj = json.dumps(detections_with_features)

    with open(os.path.join(save_root, save_file_name ), 'w') as fp:
        fp.write(json_obj)


"""

# Compute similarity matrix
similarity_matrix = (1 + cosine_similarity(features_1, features_1))/2
print(similarity_matrix)

"""