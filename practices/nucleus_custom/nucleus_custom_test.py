"""
Mask R-CNN
Test on the custom nuclei segmentation dataset.

Licensed under the MIT License (see LICENSE for details)
Written by Ocean Star

"""
# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import datetime
import numpy as np
import skimage.io

# Root directory of the project
# ROOT_DIR = os.path.abspath("../../")
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

from samples.nucleus_custom.nucleus_custom import *

ROOT_DIR = os.path.abspath("./")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/nucleus_custom/")

############################################################
# Main for Testing
############################################################
print('\nMask R-CNN for nuclei counting and segmentation\n')
print('\nTest\n')
########################################################
# Dataset Preparation for testing
print('\nPreparing dataset for testing...\n')

custom_dataset_dir = 'datasets/nucleus_custom/' # Root directory of the dataset
subset = 'val' # Dataset sub-directory
print("    Dataset: ", custom_dataset_dir)
print("    Subset: ", subset)

# Read dataset
dataset = NucleusCustomDataset()
dataset.load_nucleus_custom(custom_dataset_dir, subset)
dataset.prepare()

print('\n    Dataset for testing prepared.\n')
###############################################################
# Model Prepariation
print('\nPreparing model...')
# Create model
print("\n    Creating Model...\n") 
# Configurations
config = NucleusCustomInferenceConfig()
config.display()

logs = DEFAULT_LOGS_DIR # Logs and checkpoints directory (default=logs/)
print("\n        Logs: ", logs, '\n')

model = modellib.MaskRCNN(mode="inference", config=config,
                            model_dir=logs)
print("\n        Model created.\n") 

# Load weights
print("\n    Loading weights...\n")
# Select weights file to load
# weights_path ='models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5' # Path to weights .h5 file
# weights_path ='models/mask_rcnn_coco.h5'
weights_path ='models/mask_rcnn_nucleus_0040.h5'
model.load_weights(weights_path, by_name=True)
print("\n        Weights loaded.\n")

print('\n    Model prepared.\n')
####################################################################
# Model Inference
print('\nModel Inference...\n')
# Create directory
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
submit_dir = os.path.join(RESULTS_DIR, submit_dir)
os.makedirs(submit_dir)

# Load over images
submission = []
for image_id in dataset.image_ids:
    # Load image and run detection
    image = dataset.load_image(image_id)

    image = normalize_2Dim_uint8(image)

    # Detect objects
    r = model.detect([image], verbose=1)[0]
    # r = model.detect_molded(np.expand_dims(image, 0),verbose=1)

    # Encode image to RLE. Returns a string of multiple lines
    source_id = dataset.image_info[image_id]["id"]
    rle = mask_to_rle(source_id, r["masks"], r["scores"])
    submission.append(rle)
    # Save image with masks
    visualize.display_instances(
        image, r['rois'], r['masks'], r['class_ids'],
        dataset.class_names, r['scores'],
        show_bbox=False, show_mask=False,
        title="Predictions")
    plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

# Save to csv file
submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
file_path = os.path.join(submit_dir, "submit.csv")
with open(file_path, "w") as f:
    f.write(submission)
print("Saved to ", submit_dir)
