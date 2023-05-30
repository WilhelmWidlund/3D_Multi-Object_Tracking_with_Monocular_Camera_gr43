Torchreid forked project for NuScenes dataset training and inference
===========
This is the REID library used in the course CIVIL-459 at EPFL 2023 for group 43. It is an extension of https://github.com/KaiyangZhou/deep-person-reid. The goal of this project was to add visual feature extraction to a tracking algorithm to make it more stable and robust.
The tracking algorithm is EagerMOT https://github.com/aleksandrkim61/EagerMOT, which is a tracking algorithm that does not use visual features of objects for re-identification.
This is an offline solution for the visual feature implementation. Which means that the embeddings for the object images needs to be created before running the algorithm.
There are two main contributions to this library for better feature extractions of a traffic scenario.
- New dataset class used for training models on a NuScenes REID dataset, can be found at torchreid/data/datasets/image/nuscenes.py
- New feature extraction scripts that creates embeddings for all detected objects in the Nuscenes dataset. extract_features.py

To be able to run this extraction script you need two things:
- A regular Nuscenes dataset that the tracking algorithm should run on.
- 2D detections of this dataset

Installation for feature extraction
---------------
Clone the repository:

.. code-block:: bash

    git clone git@github.com:EliasWilliamGit/deep-person-reid.git

- Please see the original Torchreid github for how to install dependencies in a conda enviroment https://github.com/KaiyangZhou/deep-person-reid.

Run feature extraction
---------------
Run the script extract_features.py from the terminal. Specify path to dataset and detections. For all arguments take a closer look at the script. An example run can be seen below.

.. code-block:: bash

    python extract_features.py --dataset_path C:\Users\Elias\OneDrive\Dokument\LIU\Outgoing\Courses\CIVIL-459\EagerMOT\NuScenes --model_path log\osnet_x1_0_nuscenes_softmax_cosinelr\model\model.pth

The embeddings will be saved in the folder --save_path argument.

Two trained models on the NuScenes REID dataset is available at https://drive.google.com/drive/folders/1BYgqf6inddm64rKKsxZrkx3DGKotaCQn?usp=sharing.
These models where first pretrained in ImageNet and then trained on Nuscenes REID dataset.
There are also alot more models trained on other datsasets at Torchreid modelzoo.

Recreate training on SCITAS GPU cluster
---------------

.. code-block:: bash

    #Connect to the cluster with
    ssh -X GASPAR-username@izar.epfl.ch
    
    #Open your private folder
    cd home/last-name

    #Load python 3.7 and cuda 11.6.
    module load gcc/8.4.0-cuda python/3.7.7 cuda/11.6.2

    #Create a python virtual enviroment.
    python3 -m venv venv/torchreid

    #Enter enviroment.
    source venv/torchreid/bin/activate

    #Clone the repository.
    git clone git@github.com:EliasWilliamGit/deep-person-reid.git
    cd deep-person-reid

    #Install dependencies.
    python3 -m pip install -r requirements.txt

    #Install pytorch version 1.13.1 with cuda.
    python3 -m pip install torch==1.13.1+rocm5.2 torchvision torchaudio

    # install torchreid
    python3 setup.py develop


In the training file you want to run, ex. train_cosine_softmax, change the path to your home directory, then run the script with sbatch.S

The two slurm files for our two models can also be viewed in the repository, it shows how good the training went.

Contact
--------------
For questions, please email: elias.william@epfl.ch

We also want to say thank you to Kaiyang Zhou, the creator of Torchreid, for an easy to work with, open source REID base.