"""
Mask R-CNN
Configurations and data loading code for the CustomBB dataset.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)

Edited by Ocean Star ( talkoceanstar@outlook.com )

"""

import os
import json
import numpy as np
import skimage.draw

from mrcnn.config import Config
from mrcnn import model as modellib, utils

############################################################
#  Configurations
############################################################


class CustomBBConfig(Config):
    """Configuration for training on the customBB dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "customBB" 

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 11  # Background + number of classes (Here, 2)

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 50% confidence
    DETECTION_MIN_CONFIDENCE = 0.5


    # Number of training and validation steps per epoch
    # STEPS_PER_EPOCH = (657 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
    # VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400


############################################################
#  Dataset
############################################################

class CustomBBDataset(utils.Dataset):

    def load_customBB( self, dataset_dir, subset ):

        # Add classes. smoID has 11 classes - 12/19/21: changed from 0-10, to 1-11, so to match classIds below
        self.add_class( source= "customBB", class_id= 1, class_name=  "bacteria" )
        self.add_class(         "customBB",           2,              "fungi" )
        self.add_class(         "customBB",           3,              "nematode_Bacterivore" )
        self.add_class(         "customBB",           4,              "nematode_Fungivore" )
        self.add_class(         "customBB",           5,              "nematode_Herbivore" )
        self.add_class(         "customBB",           6,              "nematode_Omnivore" )
        self.add_class(         "customBB",           7,              "nematode_Predator" )
        self.add_class(         "customBB",           8,              "organicMatter" )
        self.add_class(         "customBB",           9,              "protozoa_Amoeba" )
        self.add_class(         "customBB",           10,             "protozoa_Ciliate" )
        self.add_class(         "customBB",           11,             "protozoa_Flagellate" )

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves the annotations of images in the form below:
        # {
        # "11.jpg1391949":
        # {   
        #     "filename":"11.jpg",
        #     "size":1391949,
        #     "regions":
        #     [
        #         {   
        #             "shape_attributes":
        #             {
        #                 "name":"circle",
        #                 "cx":1157,
        #                 "cy":137,
        #                 "r":24.099
        #             },
        #             "region_attributes":
        #             {
        #                 "class":"bacteria"
        #             }
        #         }, 
        #         ... more regions
        #     ]
        #    "file_attributes":{}
        # },
        # ... more files
        # }


        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_annotations.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
                classes = [r['region_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 
                classes = [r['region_attributes'] for r in a['regions']] 

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, "images", a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                source="customBB",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                classes=classes
                )


    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a custom dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "customBB":
            return super(self.__class__, self).load_mask(image_id)
        # num_ids = image_info['num_ids']	
        #print("Here is the numID",num_ids)

        # Convert polygons to a bitmap mask of shape
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):

            # Get indexes of pixels inside the polygon and set them to 1
            if p['name'] == 'polygon' or p['name'] == 'polyline':
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])            
            elif p['name'] == 'circle':
                rr, cc = skimage.draw.circle(p['cy'], p['cx'], p['r'])
                # rr, cc = skimage.draw.disk((p['cy'], p['cx']), p['r'])
            else: 
                rr, cc = skimage.draw.ellipse(p['cy'], p['cx'], p['ry'], p['rx'], rotation=np.deg2rad(p['theta']))  

            rr[ (rr > info['height']-1) ] = info['height']-1
            cc[ (cc > info['width']-1)] = info['width']-1

            mask[rr, cc, i] = 1

        class_ids = np.array([self.class_names.index(s['class']) for s in info['classes']])

        return mask, class_ids.astype(np.int32)

