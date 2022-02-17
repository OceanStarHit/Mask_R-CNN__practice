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

from practices.nucleus_custom.nucleus_custom import *

ROOT_DIR = os.path.abspath("./")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/nucleus_custom/")


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    fig, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    fig.tight_layout()
    return ax



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
# Model Validation
print('\nModel Validation...\n')
# Create directory
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
submit_dir = "submit_validate_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
submit_dir = os.path.join(RESULTS_DIR, submit_dir)
os.makedirs(submit_dir)

# Load over images
submission = []
for image_id in dataset.image_ids:
    # Load image and run detection
    image = dataset.load_image(image_id)

    image = normalize_2Dim_uint8(image)


    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    image = normalize_2Dim_uint8(image)
    info = dataset.image_info[image_id]
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                        dataset.image_reference(image_id)))
    print("Original image shape: ", modellib.parse_image_meta(image_meta[np.newaxis,...])["original_image_shape"][0])

    # Run object detection
    results = model.detect_molded(np.expand_dims(image, 0), np.expand_dims(image_meta, 0), verbose=1)

    # Display results
    r = results[0]
    # Compute AP over range 0.5 to 0.95 and print it
    utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask,
                        r['rois'], r['class_ids'], r['scores'], r['masks'],
                        verbose=1)
    visualize.display_differences(
        image,
        gt_bbox, gt_class_id, gt_mask,
        r['rois'], r['class_ids'], r['scores'], r['masks'],
        dataset.class_names, ax=get_ax(),
        show_box=False, show_mask=False,
        iou_threshold=0.5, score_threshold=0.5)

    plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

    # Encode image to RLE. Returns a string of multiple lines
    source_id = dataset.image_info[image_id]["id"]
    rle = mask_to_rle(source_id, r["masks"], r["scores"])
    submission.append(rle)

# Save to csv file
submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
file_path = os.path.join(submit_dir, "submit.csv")
with open(file_path, "w") as f:
    f.write(submission)
print("Saved to ", submit_dir)