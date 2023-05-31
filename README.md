# 3D_Multi-Object_Tracking_with_Monocular_Camera_gr43
Course project repository for CIVIL-459 Deep Learning for Autnonmous Vehicles, spring semester 2023, at EPFL. The task is 3D object tracking, by augmenting an existing model with an improvement. The chosen model is EagerMOT and the improvement to add a visual re-identification CNN component. Specifically, the project has implemented 2D visual reidentification using the DeepPersonReid, or torchreid, model, and extended the EagerMOT tracker to create a second affinity matrix based on the DeepPersonReid results. The matrices are concatenated, before the EagerMOT tracker continues with the matching process. With the concatenated matrix, it takes both the likelihood of the EagerMOT model and the visual reidentification model into account for the greedy algorithm re-identification step. We call the augmented EagerMOT model GuillaumeMOT.

![Alt text](Documentation/Diagrams/overall_blockscheme.png?raw=true "Title")

This change was chosen because the existing EagerMOT model takes no account of visual information during re-identification, but rather relies only on geometric information of the size and position of detections, and estimated new positions produced by a Kalman filter. Despite this, EagerMOT still gets quite good scores on the NuScenes tracking challenge. Our idea was to further improve these scores by augmenting the model with visual recognition aswell. As visual appearance changes little over time, we aimed to improve recognition of tracklets that might have been occluded for a number of frames. In these cases, the geometric information might be quite different between detection and tracklet, leading to poor performance in a purely geometric re-identifier. To this end, we are focusing on the IDS metric: the number of identity swaps. A lower score is better, indicating fewer times the tracker mistakenly swapped the identities of objects.

As the core of the project task is to improve an existing multi-object tracker, we have chosen to do the work in a modular structure, as far as possible. To this end, we have kept the CNN component and GuillaumeMOT as separate as possible. They rely on the map structure being as in this repo, but otherwise can be seen as interchangeable components. The modular structure continues within the parts of the project as well. For instance,GuillaumeMOT only really relies on each detection being accompanied by a feature of some sort. Any feature that could describe a detection, and be used to create reidentification likelihoods between that detection and a previous one's feature, may easily be implemented. Due to this design choice, more detailed information for the DeepPersonReid network and GuillaumeMOT are available in separate README files in the respective folders. These also contain links to the separate GitHub repositories in which they were created.

The result of this modular development is that our tracker may easily be either extended beyond its current state, or one of the modular parts be extracted and used elsewhere.

Tuning
---------------
After creating the tracker, we moved on to finding appropriate hyperparameters. This was done using a grid search method, evaluating different sets of hyperparameters on the NuScenes-Mini evaluation dataset. However, due to lack of time before the project deadline this process was regrettably very short. We are confident that further searching could have improved the scores significantly. Tuning of the model was done with a focus of improving the IDS score while maintaining a comparable AMOTA score.
![Alt text](Documentation/Diagrams/hyperparams_1.png?raw=true)
As the overall search plot shows, values quickly converged to the same region as EagerMOT. A fine-tuning process led to a set of hyperparameters deemed good enough given the short time spent on the hyperparameter search.
![Alt text](Documentation/Diagrams/hyperparams_2.png?raw=true)
The chosen set of hyperparameters are the second best achieved IDS score, while maintaining an AMOTA score that is very close to the EagerMOT level.
These hyperparameters are:

| Hyperparameter | Value |
| :-----:        | :---: |
| Bias ratio     | 0.55  |
| Similarity score function     | Cosine similarity |
| Feature vector history function | Equal weight for n most recent feature vectors |
| n | 3 |

And for the class thresholds, the chosen hyperparameters are

| Class   | Threshold |
| :-----: | :---: |
| Car | 0.65 |
| Pedestrian | 0.6 |
| Bicycle | 0.6 |
| Bus | 0.8  |
| Motorcycle | 0.5 |
| Trailer | 0.7 |
| Truck | 0.8 |

However, a more thorough hyperparameter search process could most likely result in further improvement.

Training
---------------
In the project two ReID models were trained on the SCITAS gpu clusters. One with triplet loss and softmax loss and one with only softmax loss. More information about this in DeepPersonReID/
The evaluation for these models can be seen at the end of their respective slurm-file. The models can be downloaded and used during inference from https://drive.google.com/drive/folders/1BYgqf6inddm64rKKsxZrkx3DGKotaCQn.
To achive this training, a new dataset class has been created. nuscenes.py unde DeepPersonReID/torchreid/data/datasets/image.

Results
---------------
For the training evalutation, the best model obtained was with combining triplet loss and softmax loss for 250 epochs. We obtained a mAP score of 79.1% and CMC curve rank-1 of 80.2% on the nuscenes_reid evaluation set. The training took approximatly 3 hours on the SCITAS gpu cluster.

Using the hyperparameters chosen in Tuning above, and the model trained on the NuScenes dataset with the triplet softmax loss function, we achieved the following results when using the NuScenes DevKit evaluation script:
![Alt text](Documentation/Diagrams/our_best_result.png?raw=true)
Compared to the EagerMOT score on the same evaluation, we have improved IDS significantly while maintaining the other scores on a comparable level. In the picture below, our scores can be seen on the left, and the EagerMOT scores on the right:
![Alt text](Documentation/Diagrams/compared_results_our_left_EagerMOT_right.png?raw=true)

Limitations
---------------
While the initial project description called for both detection and tracking, we quickly made the decision to only consider the tracking aspect. This decision was made due to the already large scope of the detection augmentation, and the fact that another project in the same course was concerned with only detection. Quite simply put, doing both would result in an unreasonably large work burden. This decision was made after consulting out Teacher's Assistant.

The current solution is an 'off-line' solution where the embeddings for the objects in all frames are created before the tracking algorithm. 

Datasets
---------------
As our aim has been to improve scores on the NuScenes tracking challenge, we have only used the NuScenes dataset. They may be found at https://www.nuscenes.org/nuscenes#download

To train the ReID model a reid datasetis needed. The one used in this project for training was nuscenes_reid2, which is a smaller version of nuscenes_reid. These datasets can be found on SCITAS under /work/scitas-share/datasets/Vita/civil-459/nuscenes_reid

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

Run inference
---------------
The inference script takes a set of scenes from the NuScenes dataset, and a set of detections for those scenes (.json files), as indata. It first runs the detections through the TorchReID neural network, generating an augmented detections .json file where each detection also has its feature vector. Then, this augmented file is used as indata for the GuillaumeMOT tracker, which performs the tracking. If executed without parameters, the inference stops here, with the results of the tracking saved in the folder <Gitrepo directory>/3D_Multi-Object_Tracking_with_Monocular_Camera_gr43/Results/<date_and_time>/. If the user adds the parameter -eval, the inference is finished with running the tracking results through the NuScenes Devkit tracking evaluation, the results of which are saved alongside the tracking results.
    
```ruby
    python inference.py
    python inference.py -eval   #Run with NuScenes score evaluation
```

Run training for the NuScenes ReID dataset
---------------
You can train the ReID network on the NuScenes ReID dataset from your own computer.

Firstly, setup the enviroment for running torchreid. This can be found in DeepPersonReID/readme.rst.

This is a predefined training script for this repo. Run the following command.

.. code-block:: bash

    python train.py


Note that you need to download a folder called nuscenes_reid with the dataset in the folder Datasets/
The model will be saved in log/
A guide on how to train the model on SCITAS can be found in DeepPersonReID/readme.rst.
