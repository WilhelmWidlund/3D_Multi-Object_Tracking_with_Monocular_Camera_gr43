import os
import os.path as osp
"""
Predefined training script for training the REID network on the NuScenes REID dataset.
Script assumes that the folder nuscenes_reid dataset is available under Datasets/
"""
root = osp.abspath(osp.expanduser(""))
os.system("python " + osp.join(root, "DeepPersonReID/scripts/main.py" ) + " --config-file " +
          osp.join(root, "DeepPersonReID/configs/im_osnet_x1_0_tripplet_cosine_amsgrad_nuscenes.yaml") +
          " --transforms random_flip --root " + osp.join(root, "Datasets"))
