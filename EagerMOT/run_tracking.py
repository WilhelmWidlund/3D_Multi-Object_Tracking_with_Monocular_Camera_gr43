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
    # ----------------- End altered code -----------------------------------------------------

    for sequence_name in target_sequences:
        if len(sequences_to_exclude) > 0:
            if sequence_name in sequences_to_exclude:
                print(f'Skipped sequence {sequence_name}')
                continue

        print(f'Starting sequence: {sequence_name}')
        start_time = time.time()
        sequence = dataset.get_sequence(SPLIT, sequence_name)
        sequence.mot.set_track_manager_params(params)
        variant = variant_name_from_params(params)
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
    # Choose whether to save in the default folder, or let the user input one
    print("Would you like to save the results in the default folder? [y/n]")
    savechoice = str(input())
    if not savechoice in ['y', 'Y', 'yes', 'YES', 'Yes', '1', 'default', 'DEFAULT', 'Default']:
        # Setup a file path to the project base folder
        dirlist = folder_name.split('\\')
        datasetpath = dirlist[0]
        simplepathlist = datasetpath.split('/')
        simplepath = ""
        for i in range(len(simplepathlist)-2):
            simplepath += (simplepathlist[i] + "/")
        print("Write a path to save the results in...")
        # Take input from user to their folder of choice, staerting from project base folder
        userpath = str(input(simplepath))
        folder_name = simplepath + userpath
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    # Save results
    dataset.save_all_mot_results(folder_name)
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
    # All these total results were previously calculated based only on the last sequence, now they take all
    # processed sequences into account.

    # TODO:
    # 1. Calculate the key metrics we want to compare between methods
    #   1.1 Look up what info is needed
    #   1.2 Gather it if not already available here
    #   1.3 Calculate
    #   1.4 Display
    # 2. Setup a framework for taking in another reidentification matrix and fusing (add? multiply? some kinda normalization?) with the existing one
    #   2.1 Figure out _where_ to fuse them
    #   2.2 Setup the basic framework there
    #   2.3 Setup an extended framework, possibly elsewhere, giving the user a choice whether to do the whole fusion thing at all
    #   2.4 ???
    #   2.5 Profit

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
     # ------- End altered code -------------------------------------------------------------------------------

    print(f"\n3D MOT saved in {folder_name}", end="\n\n")
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
