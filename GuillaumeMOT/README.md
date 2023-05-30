GuillaumeMOT
================
This is an extension of the EagerMOT multi-object tracker, which can be seen at the repository https://github.com/aleksandrkim61/EagerMOT
The idea of this extension is to allow for using scores from several methods of re-identification, by allowing for the creation of a new affinity matrix. 
This matrix is concatenated with the original EagerMOT affinity matrix, resulting in a new one which is influenced by both.

The guiding principle of the extension is to allow for ease of further extension. To this end, the code is laid out such that a user may easily define their
own re-identification method and use it within the EagerMOT framework. For the purposes of this project, one method was implemented, which may serve as a template 
for further additions.

GuillaumeMOT takes an extended .json file of detections as input, where each detection is also accompanied by a feature. This may be any datatype, representing any value. The key point is that two features should be comparable in some meaningful way, resulting in a numerical score that describes the similarity of the two. The implemented 2d visual similarity method is but one of many possibilities for which the model may be used. Any other source of extracting features may be connected, so long as it provides such a file with the expected structure. 

Structure
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


Visual Similarity 2D method
----------------
The implemented method uses feature vectors of the detected images, compared to the feature vectors of tracklets, to determine the likelihood of pairing. 
Each possible pair receives a similarity score, which make up the elements of the affinity matrix. At the base is the cosine similarity function, which measures angular distance. This is normalised to the range [0, 1]. However, since a tracklet may 
