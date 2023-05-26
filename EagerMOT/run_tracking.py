import time
from itertools import product
from typing import List, Iterable, Mapping
import os

# for NuScenes eval
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.evaluate import TrackingEval

#import dataset_classes.kitti.mot_kitti as mot_kitti
from dataset_classes.nuscenes.dataset import MOTDatasetNuScenes
from utils import io
from configs.params import TRAIN_SEQ, VAL_SEQ, TRACK_VAL_SEQ, build_params_dict, KITTI_BEST_PARAMS, NUSCENES_BEST_PARAMS, variant_name_from_params
from configs.local_variables import KITTI_WORK_DIR, SPLIT, NUSCENES_WORK_DIR, MOUNT_PATH
import inputs.utils as input_utils
from tracking.utils_concatenation import (get_bias_ratio, get_concatenation_source_folder_address)

# TODO:
    # 1. Setup a framework for taking in another re-identification matrix and fusing with the existing one
    #
#   # Elias:
    #   1.1 Read the detections json file
    #   1.2 ??? image recognition network does stuff ???
    #   1.3 Store a json file with the exact same structure as the detections json file, but each detection now has
    #       the feature vector as value in stead of the bbox stuff in the detections json file
    #
#   # Wilhelm:
    #   1.1 Write a prompt thing for choosing inference json file path, if that option is chosen
    #       DONE: if chosen, params['concatenate'] = [bool=True, str='folder_path', float=bias_ratio]
    #             else, params['concatenate'] = [bool=False]
    #
    #   1.2 Whenever a detection is loaded from the detections json file, also augment it with its feature vector from
    #       the inference json file.
    #   1.3 Add some kinda list of detections_previously_matched_to_tracklet, that is augmented whenever... yeah
    #       1.3.1 Figure out where exactly detection and tracklet are matched
    #       1.3.2 Add the info (possibly requiring to find also the place a tracklet itself is created, to augment with
    #                           the new data member)
    #   1.4 When making the regular affinity matrix, also make one for the recognition network
    #       1.4.1 Loop over active/existing tracklets, extracting the identity of previously-matched-with detections
    #       1.4.2 Triple-loop over detections, tracklets and previous_detections thusly:
    #               for detection, i in enumerate(detections_list):
    #                   for tracklet, j in enumerate(tracklets_list):
    #                       average_score_for_pair = 0
    #                       for previous_detection in tracklet[previous_detections_list]:
    #                           average_score_for_pair += score_function(detection, previous_detection)
    #                       average_score_for_pair = average_score_for_pair / length(previous_detections_list)
    #                       new_affinity_matrix[i, j] = average_score_for_pair
    #
    # 2. Evaluate the HOTA metric (time permitting...)
    #   a) Implement it into the nuscenes-devkit
    #   b) Find another given code that evaluates it on nuscenes datasets

def perform_tracking_full(dataset, params, target_sequences=[], sequences_to_exclude=[], print_debug_info=True):

    if len(target_sequences) == 0:
        target_sequences = dataset.sequence_names(SPLIT)

    total_frame_count = 0
    total_time = 0
    total_time_tracking = 0
    total_time_fusion = 0
    total_time_reporting = 0
    # ----------------- Altered code -----------------------------------------------------
    # Initiate/reset previously neglected accumulative variables
    total_instances_any = 0
    total_instances_both = 0
    total_instances_3d_only = 0
    total_instances_2d_only = 0
    total_matched_tracks_first = 0
    total_unmatched_tracks_first = 0
    total_matched_tracks_second = 0
    total_unmatched_tracks_second = 0
    total_unmatched_dets2d_second = 0
    # Record whether all sequences are skipped or not, for save purposes
    seq_tracked = False
    # Store the extended project base address in params
    params['base_folder_addr'] = dataset.work_dir.rsplit('/', 3)[0]
    # Ask if the user wants to concatenade the EagerMOT affinity matrix with one based on another re-id method
    print("Would you like to concatenade the EagerMOT affinity matrix with one from elsewhere? [y/n]")
    savechoice = str(input())
    if savechoice in ['y', 'Y', 'yes', 'YES', 'Yes', '1']:
        params['concatenate'] = [True]
        # Get address to json file with feature vectors for all detections
        params['concatenate'].append(get_concatenation_source_folder_address(params['base_folder_addr']))
        # Get bias ratio from user
        params['concatenate'].append(get_bias_ratio())
    else:
        params['concatenate'] = [False]

    # ----------------- End altered code -----------------------------------------------------

    for sequence_name in target_sequences:
        if len(sequences_to_exclude) > 0:
            if sequence_name in sequences_to_exclude:
                print(f'Skipped sequence {sequence_name}')
                continue

        print(f'Starting sequence: {sequence_name}')
        start_time = time.time()
        # TODO: Send params here
        sequence = dataset.get_sequence(SPLIT, sequence_name, params['concatenate'])
        sequence.mot.set_track_manager_params(params)
        variant = variant_name_from_params(params)
        # TODO: 1. The path to assignment starts here, calling the perform_tracking_for_eval function in mot_sequence.py
        run_info = sequence.perform_tracking_for_eval(params)
        # ----------------- Altered code -----------------------------------------------------
        # Continue to next sequence if the current one wasn't tracked,
        # else take the input pause time into account, set folder_name, and proceed with the current sequence
        if "total_time_mot" not in run_info:
            continue
        else:
            seq_tracked = True
            folder_name = run_info["mot_3d_file"]
        total_time = time.time() - start_time - run_info['pause_time']
        # ----------------- End altered code -----------------------------------------------------
        if print_debug_info:
            print(f'Sequence {sequence_name} took {total_time:.2f} sec, {total_time / 60.0 :.2f} min')
            print(
                f'Matching took {run_info["total_time_matching"]:.2f} sec, {100 * run_info["total_time_matching"] / total_time:.2f}%')
            print(
                f'Creating took {run_info["total_time_creating"]:.2f} sec, {100 * run_info["total_time_creating"] / total_time:.2f}%')
            print(
                f'Fusion   took {run_info["total_time_fusion"]:.2f} sec, {100 * run_info["total_time_fusion"] / total_time:.2f}%')
            print(
                f'Tracking took {run_info["total_time_mot"]:.2f} sec, {100 * run_info["total_time_mot"] / total_time:.2f}%')

            print(
                f'{run_info["matched_tracks_first_total"]} 1st stage and {run_info["matched_tracks_second_total"]} 2nd stage matches')

        total_time += total_time
        total_time_fusion += run_info["total_time_fusion"]
        total_time_tracking += run_info["total_time_mot"]
        total_time_reporting += run_info["total_time_reporting"]
        total_frame_count += len(sequence.frame_names)
        # ----------------- Altered code -----------------------------------------------------
        # Update previously neglected accumulative variables for the current sequence
        total_instances_any += run_info['instances_both'] + run_info['instances_3d'] + run_info['instances_2d']
        total_instances_both += run_info['instances_both']
        total_instances_3d_only += run_info['instances_3d']
        total_instances_2d_only += run_info['instances_2d']
        total_matched_tracks_first += run_info['matched_tracks_first_total']
        total_unmatched_tracks_first += run_info['unmatched_tracks_first_total']
        total_matched_tracks_second += run_info['matched_tracks_second_total']
        total_unmatched_tracks_second += run_info['unmatched_tracks_second_total']
        total_unmatched_dets2d_second += run_info['unmatched_dets2d_second_total']
        # ----------------- End altered code -----------------------------------------------------

    if not seq_tracked:
        return variant, run_info

    # ------- Altered code -------------------------------------------------------------------------------
    # Get a path from user for where to save the results
    save_folder_query = "Would you like to save the results in the default folder?"
    save_folder_prompt = "Write a path to save the results in..."
    save_path = input_utils.ask_if_default_folder(params['base_folder_addr'], save_folder_query, save_folder_prompt, True)
    # Save results
    dataset.save_all_mot_results(save_path)
    # ------- End altered code -------------------------------------------------------------------------------

    if not print_debug_info:
        return variant, run_info

    # Overall variant stats
    # Timing
    print("\n")
    print(
        f'Fusion    {total_time_fusion: .2f} sec, {(100 * total_time_fusion / total_time):.2f}%')
    print(f'Tracking  {total_time_tracking: .2f} sec, {(100 * total_time_tracking / total_time):.2f}%')
    print(f'Reporting {total_time_reporting: .2f} sec, {(100 * total_time_reporting / total_time):.2f}%')
    print(
        f'Tracking-fusion framerate: {total_frame_count / (total_time_fusion + total_time_tracking):.2f} fps')
    print(f'Tracking-only framerate: {total_frame_count / total_time_tracking:.2f} fps')
    print(f'Total framerate: {total_frame_count / total_time:.2f} fps')
    print()

    # ------- Altered code -------------------------------------------------------------------------------
    # All these total results were previously calculated based only on the last processed sequence, now they take all
    # processed sequences into account.

    # Fused instances stats
    if total_instances_any > 0:
        print(f"Total instances 3D and 2D: {total_instances_both} " +
              f"-> {100.0 * total_instances_both / total_instances_any:.2f}%")
        print(f"Total instances 3D only  : {total_instances_3d_only} " +
              f"-> {100.0 * total_instances_3d_only / total_instances_any:.2f}%")
        print(f"Total instances 2D only  : {total_instances_2d_only} " +
              f"-> {100.0 * total_instances_2d_only / total_instances_any:.2f}%")
        print()

    # Matching stats
    print(f"matched_tracks_first_total {total_matched_tracks_first}")
    print(f"unmatched_tracks_first_total {total_unmatched_tracks_first}")

    print(f"matched_tracks_second_total {total_matched_tracks_second}")
    print(f"unmatched_tracks_second_total {total_unmatched_tracks_second}")
    print(f"unmatched_dets2d_second_total {total_unmatched_dets2d_second}")

    first_matched_percentage = (total_matched_tracks_first /
                                (total_matched_tracks_first + total_unmatched_tracks_first))
    print(f"percentage of all tracks matched in 1st stage {100.0 * first_matched_percentage:.2f}%")

    second_matched_percentage_of_leftovers = (total_matched_tracks_second / total_unmatched_tracks_first)
    print(f"percentage of leftover tracks matched in 2nd stage {100.0 * second_matched_percentage_of_leftovers:.2f}%")

    second_matched_dets2d_second_percentage = (total_matched_tracks_second / 
                                                (total_unmatched_dets2d_second + total_matched_tracks_second))
    print(f"percentage dets 2D matched in 2nd stage {100.0 * second_matched_dets2d_second_percentage:.2f}%")

    final_unmatched_percentage = (total_unmatched_tracks_second / (
        total_matched_tracks_first + total_unmatched_tracks_first))
    print(f"percentage tracks unmatched after both stages {100.0 * final_unmatched_percentage:.2f}%")

    print(f"\n3D MOT saved in {save_path}", end="\n\n")
    # ------- End altered code -------------------------------------------------------------------------------
    return variant, run_info


def perform_tracking_with_params(dataset, params,
                                 target_sequences: Iterable[str] = [],
                                 sequences_to_exclude: Iterable[str] = []):
    start_time = time.time()
    variant, run_info = perform_tracking_full(dataset, params,
                                              target_sequences=target_sequences,
                                              sequences_to_exclude=sequences_to_exclude)
    print(f'Variant {variant} took {(time.time() - start_time) / 60.0:.2f} mins')
    return run_info


def run_on_nuscenes():
    VERSION = "v1.0-mini"
    mot_dataset = MOTDatasetNuScenes(work_dir=NUSCENES_WORK_DIR,
                                     det_source=input_utils.CENTER_POINT,
                                     seg_source=input_utils.MMDETECTION_CASCADE_NUIMAGES,
                                     version=VERSION)

    # if want to run on specific sequences only, add their str names here
    target_sequences: List[str] = []

    # if want to exclude specific sequences, add their str names here
    sequences_to_exclude: List[str] = []

    run_info = perform_tracking_with_params(
        mot_dataset, NUSCENES_BEST_PARAMS, target_sequences, sequences_to_exclude)
    mot_dataset.reset()


def run_on_kitti():
    # To reproduce our test set results run this on the TEST set

    # To reproduce "Ours" results in Table II in the paper run this on the VAL set

    # To reproduce "Ours (dagger)" results in Table II in the paper,
    # change det_source to input_utils.AB3DMOT and run on the VAL set
    mot_dataset = mot_kitti.MOTDatasetKITTI(work_dir=KITTI_WORK_DIR,
                                            det_source=input_utils.POINTGNN_T3,
                                            seg_source=input_utils.TRACKING_BEST)

    # if want to run on specific sequences only, add their str names here
    target_sequences: List[str] = []

    # if want to exclude specific sequences, add their str names here
    sequences_to_exclude: List[str] = []

    perform_tracking_with_params(mot_dataset, KITTI_BEST_PARAMS, target_sequences, sequences_to_exclude)


if __name__ == "__main__":
    run_on_nuscenes()
    # run_on_kitti()
