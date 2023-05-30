from torchreid.utils import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import argparse
import torch
import os.path as osp

def main(args):
    """
    Feature extraction specifically made for the NuScenes dataset and structure.
    
    """
    detections_path = args.detections_path
    # Get which scene will be downloaded
    dataset_root_path = args.dataset_path
    dataset_info_path = os.path.join(dataset_root_path, args.conversion_folder_name)
    
    # Load tokens/info for nuscenes dataset
    f = open(os.path.join(dataset_info_path,'scene.json'))
    scene_info = json.load(f)
    f.close()

    f = open(os.path.join(dataset_info_path,'sensor.json'))
    sensor_info = json.load(f)
    f.close()

    f = open(os.path.join(dataset_info_path,'sample.json'))
    sample_info = json.load(f)
    f.close()

    f = open(os.path.join(dataset_info_path,'sample_data.json'))
    sample_data_info = json.load(f)
    f.close()

    f = open(os.path.join(dataset_info_path,'calibrated_sensor.json'))
    calibrated_sensor_info = json.load(f)
    f.close()

    # Create list of valid calibrated sensor tokens which refers to a camera in the dataset
    camera_calibrated_sensors = []

    for calibrated_sensor in calibrated_sensor_info:
        sensor_token_for_cal = calibrated_sensor['sensor_token']
        sensor_id_in_list = [i for i in range(len(sensor_info)) 
                                    if sensor_info[i]['token'] == sensor_token_for_cal]
        sensor_id_in_list = sensor_id_in_list[0]
        if sensor_info[sensor_id_in_list]['modality'] == 'camera':
            camera_calibrated_sensors.append(calibrated_sensor['token'])

    # Load the REID network

    if torch.cuda.is_available():
        current_device = 'cuda'
    else:
        current_device = 'cpu'

    extractor = FeatureExtractor(
        model_name=args.model_name,
        model_path=args.model_path,
        device=current_device,
        verbose=True
    )

    # Loop over scenes in dataset
    for scene in scene_info:
        print("New Scene!")

        # Get the scene token
        scene_token = scene['token']

        # Open the correct detection file for this scene
        scene_token_json = scene_token + '_' + args.detection_name_ending +'.json'
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
        # sample = {}

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

                # Load the frame, this is the real slow part of the script, could be optimized
                frame = cv.imread(os.path.join(dataset_root_path, sample_data_frame['filename']))

                # Loop over all classes in the frame
                for obj_class, frame_detections in frame_detections.items():
                    # Loop over each object in the frame
                    for id,detection in enumerate(frame_detections):
                        
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
            

        detect_folder_name = args.detections_path.split('/')[-1]

        # Save dictionary as JSON file
        save_root = os.path.join(args.save_path,detect_folder_name)
        save_file_name = scene_token + '_' + args.model_name +'.json'
        print(os.path.join(save_root, save_file_name ))

        json_obj = json.dumps(detections_with_features)

        with open(os.path.join(save_root, save_file_name ), 'w') as fp:
            fp.write(json_obj)

if __name__ == "__main__":
    # Arguments defintions
    root = osp.abspath(osp.expanduser(""))
    # Check in which folder the user is currently, this to be able to run script from inside folder or outside
    root_list = root.split("\\")
    if root_list[-1] == "DeepPersonReID":
        # Remove it from root
        root = "\\".join(root_list[:-1])

    parser = argparse.ArgumentParser()
    parser.add_argument('--detections_path', default=os.path.join(root, "Detections/mmdetection_cascade_x101/mini_val"), type=str, help="Path to the 2D detections JSON files for the dataset")
    parser.add_argument('--dataset_path', default=os.path.join(root,"Datasets/NuScenes/mini"), type=str, help="Path to the dataset containing frame images")
    parser.add_argument('--save_path', default=os.path.join(root,"Embeddings/TorchREID"), type=str, help="Where to save the JSON files containing embeddings")
    parser.add_argument('--conversion_folder_name', default="v1.0-mini", type=str, help="Name of folder containing token conversions JSON files for the scenes. This folder should be in the same folder as the dataset.")
    parser.add_argument('--model_path', default=os.path.join(root, 'DeepPersonReID/log/modelzoo/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth'), type=str, help="Path to the model creating embeddings")
    parser.add_argument('--model_name', default="osnet_x1_0", type=str, help="Name of the model used for feature extraction")
    parser.add_argument('--detection_name_ending', default="trainval_cascade_mask_rcnn_x101", type=str, help="The detection files often has an ending of the name of the model used for it. Put this ending here")
    args = parser.parse_args()
    main(args)