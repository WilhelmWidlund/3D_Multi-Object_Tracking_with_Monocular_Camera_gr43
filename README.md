# 3D_Multi-Object_Tracking_with_Monocular_Camera_gr43
Course project repository for CIVIL-459 Deep Learning for Autnonmous Vehicles, spring semester 2023, at EPFL. The task is 3D object tracking, by augmenting an existing model with an improvement. The chosen model is EagerMOT and the improvement to add a visual re-identification CNN component. Specifically, the project has implemented 2D visual reidentification using the DeepPersonReid model, and extended the EagerMOT tracker to create a second affinity matrix based on the DeepPersonReid results. The matrices are concatenated, before the EagerMOT tracker continues with the matching process. With the concatenated matrix, it takes both the likelihood of the EagerMOT model and the visual reidentification model into account for the greedy algorithm re-identification step. We call the augmented EagerMOT model GuillaumeMOT.

![Alt text](Documentation/Diagrams/overall_blockscheme.png?raw=true "Title")

This change was chosen because the existing EagerMOT model takes no account of visual information during re-identification, but rather relies only on geometric information of the size and position of detections, and estimated new positions produced by a Kalman filter. Despite this, EagerMOT still gets quite good scores on the NuScenes tracking challenge. Our idea was to further improve these scores by augmenting the model with visual recognition aswell. As visual appearance changes little over time, we aimed to improve recognition of tracklets that might have been occluded for a number of frames. In these cases, the geometric information might be quite different between detection and tracklet, leading to poor performance in a purely geometric re-identifier. To this end, we are focusing on the IDS metric: the number of identity swaps. A lower score is better, indicating fewer times the tracker mistakenly swapped the identities of objects.

As the core of the project task is to improve an existing multi-object tracker, we have chosen to do the work in a modular structure, as far as possible. To this end, we have kept the CNN component and GuillaumeMOT as separate as possible. They rely on the map structure being as in this repo, but otherwise can be seen as interchangeable components. The modular structure continues within the parts of the project as well. For instance,GuillaumeMOT only really relies on each detection being accompanied by a feature of some sort. Any feature that could describe a detection, and be used to create reidentification likelihoods between that detection and a previous one's feature, may easily be implemented. Due to this design choice, more detailed information for the DeepPersonReid network and GuillaumeMOT are available in separate README files in the respective folders. These also contain links to the separate GitHub repositories in which they were created.

The result of this modular development is that our tracker may easily be either extended beyond its current state, or one of the modular parts be extracted and used elsewhere.

Tuning
---------------
After creating the tracker, we moved on to finding appropriate hyperparameters. This was done using a grid search method. However, due to lack of time before the project deadline this process was regrettably very short. We are confident that further searching could have improved the scores significantly.

Results
---------------

Limitations
---------------
While the initial project description called for both detection and tracking, we quickly made the decision to only consider the tracking aspect. This decision was made due to the already large scope of the detection augmentation, and the fact that another project in the same course was concerned with only detection. Quite simply put, doing both would result in an unreasonably large work burden. This decision was made after consulting out Teacher's Assistant.

Datasets
---------------
As our aim has been to improve scores on the NuScenes tracking challenge, we have only used the NuScenes dataset. They may be found at https://www.nuscenes.org/nuscenes#download

Installation
---------------
To install our tracker, the user needs to:
1. Clone the repository.
2. Download the desired dataset.
3. Setup the required conda environments:
   These are made with the requirements.txt files in DeepPersonReID and GuillaumeMOT
   ```ruby
     conda create --name <CNN_environment_name> --file <Git repo base>/DeepPersonReID/requirements.txt
     conda create --name <GuillaumeMOT_environment_name> --file <Git repo base>/GuillaumeMOT/requirements_conda.txt
    ```
    For the evaluation to work, the user also needs a conda environment as described in the NuScenes Devkit repository: https://github.com/nutonomy/nuscenes-devkit
4. Modify the variables in GuillaumeMOT/configs/local_variables.py according to which dataset is used and which path the repository is cloned into
5. Modify the variables in inference.py according to your chosen conda environment names

Run training
---------------


Run inference
---------------
The inference script takes a set of scenes from the NuScenes dataset, and a set of detections for those scenes (.json files), as indata. It first runs the detections through the TorchReID neural network, generating an augmented detections .json file where each detection also has its feature vector. Then, this augmented file is used as indata for the GuillaumeMOT tracker, which performs the tracking. If executed without parameters, the inference stops here, with the results of the tracking saved in the folder <Gitrepo directory>/3D_Multi-Object_Tracking_with_Monocular_Camera_gr43/Results/<date_and_time>/. If the user adds the parameter -eval, the inference is finished with running the tracking results through the NuScenes Devkit tracking evaluation, the results of which are saved alongside the tracking results.
    
```ruby
    python inference.py
    python inference.py -eval   #Run with NuScenes score evaluation
```
