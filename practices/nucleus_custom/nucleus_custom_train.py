"""
Mask R-CNN
Test on the custom nuclei segmentation dataset.

Licensed under the MIT License (see LICENSE for details)
Written by Ocean Star

"""
# Set matplotlib backend
# This has to be done before other importa that might set it, 
# but only if we're running in script mode rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa

from glob import glob

import imageio

# Root directory of the project
# ROOT_DIR = os.path.abspath("../../")
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

from practices.nucleus_custom.nucleus_custom import *

ROOT_DIR = os.path.abspath("./")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
# Main for Training
############################################################
print('Mask R-CNN for nuclei counting and segmentation')
print('Train')

########################################################
# Dataset Info Preparation for training
# Training dataset.
custom_dataset_dir = os.path.join(ROOT_DIR,'datasets/nucleus_custom/') # Root directory of the dataset

dataset_train = NucleusCustomDataset()
dataset_train.load_nucleus_custom(custom_dataset_dir, subset='train')
dataset_train.prepare()

# Validation dataset
dataset_val = NucleusCustomDataset()
dataset_val.load_nucleus_custom(custom_dataset_dir, subset="val")
dataset_val.prepare()

# Image augmentation
# http://imgaug.readthedocs.io/en/latest/source/augmenters.html
augmentation = iaa.SomeOf
(
    (0, 2), 
    [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf
        (
            [
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270)
            ]
        ),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ]
)

###############################################################
# Model Prepariation

# Create model
print("Creating Model ") 
# Configurations
config = NucleusCustomConfig()
config.STEPS_PER_EPOCH  = dataset_train.num_images
config.VALIDATION_STEPS = dataset_val.num_images
config.display()

logs = DEFAULT_LOGS_DIR # Logs and checkpoints directory (default=logs/)
print("Logs: ", logs)

model = modellib.MaskRCNN(mode="training", config=config,
                            model_dir=logs)

# Load weights
print("Loading weights ")
# Select weights file to load
# weights_path ='models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5' # Path to weights .h5 file
weights_path = model.find_last()

# weights_path ='models/mask_rcnn_coco.h5'
model.load_weights(weights_path, by_name=True)

####################################################################
# Model Training

# *** This training schedule is an example. Update to your needs ***

# If starting from imagenet, train heads only for a bit
# since they have random weights
print("Training network heads")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=20,
            augmentation=augmentation,
            layers='heads')

print("Training all layers")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=60,
            augmentation=augmentation,
            layers='all')