"""
Run inference: tracking
    -Indata = dataset, set of detections
    -Outdata: * 3D multi object tracking results
              * If this script is run with -eval, then also output NuScenes evaluation of the tracking results

1. Run the detections through TorchReID, generating an augmented .json file with the feature vectors of each detection
2. Run the augmented detections through GuillaumeMOT, performing tracking
    2.1 Use the EagerMOT model of generating an affinity matrix based on geometric similarity
    2.2 Use the feature vectors from TorchReID to generate an affinity matrix based on visual similarity
    2.3 Concatenate the affinity matrices
3. If -eval, run NuScenes Devkit evaluation of tracking results
4. Save results to MOUNT_PATH/Results/<date and time>/
"""

import subprocess
import sys

from GuillaumeMOT.configs.local_variables import MOUNT_PATH


if __name__ == "__main__":
    # Setup paths
    torchreid_feature_extraction = MOUNT_PATH + "/DeepPersonReID/extract_features.py"
    guillaumemot_tracking = MOUNT_PATH + "/GuillaumeMOT/run_tracking.py"
    nuscenes_evaluate_metrics = MOUNT_PATH + "/nuscenes-devkit/python-sdk/nuscenes/eval/tracking/evaluate.py --combined"

    argmnts = []
    if len(sys.argv) > 1:
        argmnts = str(sys.argv)


    # Activate TorchReID conda environment and run TorchReID
    #subprocess.run(f"conda activate torchreid && python {torchreid_feature_extraction} && conda deactivate",
    #               shell=True)

    # Activate GuillaumeMOT conda environment and run GuillaumeMOT
    subprocess.run(f"conda activate DLAV_env && python {guillaumemot_tracking} && conda deactivate",
                   shell=True)

    # Activate NuScenes-DevKit conda environment and run evaluation
    if "-eval" in argmnts:
        subprocess.run(f"conda activate nuscenes && python {nuscenes_evaluate_metrics} && conda deactivate",
                       shell=True)
