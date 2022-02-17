"""
Mask R-CNN
Configurations and data loading code for the CustomBB dataset.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)

Edited by Ocean Star ( talkoceanstar@outlook.com )

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

from distutils import command
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from imgaug import augmenters as iaa


from mrcnn.config import Config
from mrcnn import model as modellib

from customBB import *

# Root directory of the project
# ROOT_DIR = os.path.abspath("../../")
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize


# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/customBB/")

# Path to trained weights file

############################################################
#  Command Line Interface
############################################################
if __name__ == '__main__':

    class args_define():
        command = "train"
        dataset = "datasets/customBB"
        # weights = "imagenet"
        weights = "last"
        logs = DEFAULT_LOGS_DIR
        subset = "train"

    args = args_define()    

    if not args.command in ["train", "validate", "detect"]:
        print("'{}' is not recognized. "
              "Use 'train', 'validate' or 'detect'".format(args.command))

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "validate" or "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Train or evaluate
    if args.command == "train":
        train(args)
    elif args.command == "validate":
        validate(args)
    elif args.command == "detect":
        detect(args)
