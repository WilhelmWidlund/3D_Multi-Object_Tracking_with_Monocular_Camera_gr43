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
    # Setup conda environment names
    CNN_env_name = "torchreid"
    GuillaumeMOT_env_name = "DLAV_env"
    NuScenes_DevKit_env_name = "nuscenes"

    # Setup paths
    model_path = MOUNT_PATH + "/DeepPersonReID/log/osnet_x1_0_nuscenes_tripplet_cosinelr\model\model.pth.tar-250"
    torchreid_feature_extraction = MOUNT_PATH + "/DeepPersonReID/extract_features.py --model_path " + model_path
    guillaumemot_tracking = MOUNT_PATH + "/GuillaumeMOT/run_tracking.py"
    nuscenes_evaluate_metrics = MOUNT_PATH + "/nuscenes-devkit/python-sdk/nuscenes/eval/tracking/evaluate.py --combined"

    argmnts = []
    if len(sys.argv) > 1:
        argmnts = str(sys.argv)

    # Activate TorchReID conda environment and run TorchReID
    subprocess.run(f"conda activate {CNN_env_name} && python {torchreid_feature_extraction} && conda deactivate",
                   shell=True)

    # Activate GuillaumeMOT conda environment and run GuillaumeMOT
    subprocess.run(f"conda activate {GuillaumeMOT_env_name} && python {guillaumemot_tracking} && conda deactivate",
                   shell=True)

    # Activate NuScenes-DevKit conda environment and run evaluation
    if "-eval" in argmnts:
        subprocess.run(f"conda activate {NuScenes_DevKit_env_name} && python {nuscenes_evaluate_metrics} "
                       f"&& conda deactivate", shell=True)
