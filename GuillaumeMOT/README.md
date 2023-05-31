GuillaumeMOT
================
This is an extension of the EagerMOT multi-object tracker, which can be seen at the repository https://github.com/aleksandrkim61/EagerMOT
The idea of this extension is to allow for using scores from several methods of re-identification, by allowing for the creation of a new affinity matrix. 
This matrix is concatenated with the original EagerMOT affinity matrix, resulting in a new one which is influenced by both.

The guiding principle of the extension is to allow for ease of further extension. To this end, the code is laid out such that a user may easily define their
own re-identification method and use it within the EagerMOT framework. For the purposes of this project, one method was implemented, which may serve as a template 
for further additions.

GuillaumeMOT takes an extended .json file of detections as input, where each detection is also accompanied by a feature. This may be any datatype, representing any value. The key point is that two features should be comparable in some meaningful way, resulting in a numerical score that describes the similarity of the two. The implemented 2d visual similarity method is but one of many possibilities for which the model may be used. Any other source of extracting features may be connected, so long as it provides such a file with the expected structure. 

Code structure
----------------
The bulk of the changes are in the folder augmentation, which is entirely new compared to EagerMOT. The code is class based, with a parent class defining
general functions and data members that must be present for the augmentation framework to function within EagerMOT. This parent class, AugmentationMethod, is abstract, and certain functions must be overridden by the child classes. These functions will do different things depending on the intended method to be implemented. No further description is given here, as the code is easy to follow and the implemented child class AugMet_Vis_Sim_2D provides a template to follow. Additionally, 
the child class DoNotAugment is also implemented with the sole purpose of allowing for running the tracker with only the original EagerMOT framework.

Beyond the new classes, assorted utility functions and parameters found in the augmentation folder, there are also many small changes throughout the EagerMOT code
with the purpose of integrating the new framework into EagerMOT. These changes are very spread out throughout EagerMot, and are therefore marked with
```ruby
    # ------------- Altered code -------------------
    new_variable = new_code(that/does/new/things)
    changed_variable = altered_code(that/does/different/things)
    # ------------- End altered code ---------------
```
in order to be easier to identify.

In order to enable more complex comparisons between detection and tracklet, each tracklet stores the features of all the detections it has been matched with over time. This allows for a larger freedom in similarity score function choices, and larger customisability overall.

Matrix concatenation
----------------
A challenge in the numerical concatenation of the EagerMOT affinity matrix and the augmented one turned out to be the undefined range of the EagerMOT similarity scores. The options available for these all being based on geometric distances, they range from 0 (best) to arbitrarily large. To avoid too much hyperparameter searching, it was desireable to find a decently accurate relationship between the EagerMOT score range and the score range of an augmented model. While the EagerMOT range was unknown, there was however one piece of useful information: the threshold values per class. A similarity score is simply considered too poor if it fails to attain a certain threshold value. These are defined in the EagerMOT code, and their values are in relation to the EagerMOT score range, however high it may go. 

By requiring the user to provide a set of thresholds for the scores of the augmented method, and relating these to the EagerMOT thresholds, a reasonably accurate map was achieved. Having one boundary and one fix value known for each class and each method, it was possible to define this straightforward value map. Of course, this shifts the burden to the user who must define their own thresholds, but at least there is the possibility that this is an easier task for the user's score range. The alternative would have been to leave the map for each class as a hyperparameter to be decided entirely by hyperparameter search. 

General hyperparameters
----------------
Some hyperparameters are common to all methods:
* Bias ratio: how much weight is assigned to the EagerMOT affinity matrix and how much to the augmented one. A value of 1 means that only the EagerMOT matrix is considered, 0 means that only the augmented one is. The final affinity matrix is calculated elementwise as A_final = A_EagerMOT * Bias ratio + A_New * (1 - Bias ratio).
* Map ratio: As described above, the user must provide re-identification threshold values per class.

Tuning
---------------
Tuning of the model was done with a focus of improving the IDS score while maintaining a comparable AMOTA score. A brief grid search was conducted, testing different choices of hyperparameters on the NuScenes-Mini evaluation dataset.
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

Visual Similarity 2D method
----------------
The implemented method uses feature vectors of the detected images, compared to the feature vectors of tracklets, to determine the likelihood of pairing. 
Each possible pair receives a similarity score, which make up the elements of the affinity matrix. At the base is the cosine similarity function, which measures angular distance. This is normalised to the range [0, 1]. However, since a tracklet may contain several feature vectors, each pair of detection-tracklet has several pairs of feature vectors that can be compared: that of the detection with each belonging to the tracklet. Therefore, several options for handling the calculation have been implemented:
* Equal weight: Each pair is considered to be equally important, return the average of their scores
* n last equal weight: the n last, or all if all < n, are considered to be equally important
* n last different weights: the n last, or all if all < n, with falling weight 1/2, 1/4, 1/8 etc. are considered. The n:th has whatever weight is left to make up 100 %
* Last only: none but the most recent are considered

The equal weights and last only options are quite straightforward. The reasoning behind the other two are that the visual appearance of some objects might change 
quite a bit over time, for instance a person drastically changing their post, a cyclist stopping for a red light, etc. Therefore, limiting the time into the past and assigning more weight to recent observations could give a better result.

Augmentation method extension
----------------
The following steps are required when adding a new method:
* Create the new method child class, overriding all functions that must be overridden as defined in augmentation/augmentation_base
* Add the method information, including METHOD_NAME, to augmentation/augmentation_params
* Add default hyperparameters to augmentation/augmentation_params
* For automatic execution: make sure the extended .json file with detections and features can be found in <Git repo base>/Embeddings/<METHOD_NAME>/2*/

Beyond these steps, the user is recommended to follow function calls in general, and the path of the augmentation parameter dictionary param['augment'] in particular, to successfully implement further augmentation methods. Using the implemented child class as a template is also recommended. 

Installation
-----------------
The user must clone the GitHub repository, either the entire one for 3D Multi-Object Tracking, or the GuillaumeMOT repo at https://github.com/WilhelmWidlund/DLAV_GuillaumeMOT. Then, the desired dataset must be downloaded from https://www.nuscenes.org/nuscenes#download. For running the scripts, the necessary conda requirements can be found in GuillaumeMOT/requirements_conda.txt and at the NuScenes DevKit repository: https://github.com/nutonomy/nuscenes-devkit. These can be installed by
```ruby
    conda create --name <NuScenes_environment_name> --file <Git repo base>/nuscenes-devkit/requirements.txt
    conda create --name <GuillaumeMOT_environment_name> --file <Git repo base>/GuillaumeMOT/requirements_conda.txt
```
Then, the variables concerning mount path and dataset paths in configs/local_variables need to be updated to fit the user's map structure.
    
Tracking: automatic and manual execution
-----------------
The user may choose to either run the entire tracking script automatically, which triggers the use of default settings and parameters defined in augmentation/augmentation_params, or manually. In the manual mode, a series of command window prompts require the user to provide paths to indata, choose augmentation method, choose similarity score function etc. To start either mode, take the following steps:
```ruby
    conda activate <GuillaumeMOT_env_name>
    python run_tracking.py              # automatic execution
    python run_tracking.py -manual      # manual execution
```
    
Evaluation
-----------------
After having performed a tracking, the user may evaluate the performance in the NuScenes tracking score metric:
```ruby
    conda activate <NuScenes_env_name>
    python python nuscenes-devkit/python-sdk\nuscenes\eval\tracking\evaluate.py
```

