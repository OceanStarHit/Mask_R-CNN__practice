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
# if __name__ == '__main__':
#     import matplotlib
#     # Agg backend runs without a display
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt

# if __name__ == '__main__':
import matplotlib
# Agg backend runs without a display
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from imgaug import augmenters as iaa


from mrcnn.config import Config
from mrcnn import model as modellib


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize

from glob import glob


# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/customBB/")

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

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


# Override the training configurations with a few changes for inferencing.
class CustomBBInferenceConfig(CustomBBConfig):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


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

        if subset in ["train", "val"]:
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
        elif subset=="test":
            # Get image ids from directory names
            print(dataset_dir)

            image_path = os.path.join(dataset_dir, "images")
            # image_ids = next(os.walk(image_path))[0]
            
            image_ids = glob(os.path.join(image_path, '*.jpg'))
            
            # Add images
            for image_id in image_ids:

                image_id = image_id.split('/')[-1]
                image_id = image_id.split('\\')[-1].split('.')[0]
        
                self.add_image(
                    source="customBB",
                    image_id=image_id,
                    path=os.path.join(dataset_dir, "images/{}.jpg".format(image_id)))


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


    def image_reference(self, image_id):
        """Return the customBB data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "customBB":
            return info["path"]
        else:
            super(self.__class__).image_reference(self, image_id)


############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)


############################################################

def load_model_weight(model, weights):
    # Select weights file to load
    if weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    elif weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    else:
        weights_path = weights

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)


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
#  Training
############################################################
def train(args):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomBBDataset()
    dataset_train.load_customBB(args.dataset, args.subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomBBDataset()
    dataset_val.load_customBB(args.dataset, "val")
    dataset_val.prepare()

    config = CustomBBConfig()
    config.STEPS_PER_EPOCH  = dataset_train.num_images
    config.VALIDATION_STEPS = dataset_val.num_images
    config.display()

    model = modellib.MaskRCNN(
        mode="training", 
        config=config,
        model_dir=args.logs)

    load_model_weight(model, args.weights)

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                augmentation=augmentation,
                layers='all')


############################################################
#  Detection
############################################################
def validate(args):
    """Run detection on images in the given directory."""
    dataset_dir = args.dataset
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(args.results):
        os.makedirs(args.results)
    submit_dir = "submit_validation_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(args.results, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = CustomBBDataset()
    dataset.load_customBB(dataset_dir, args.subset)
    dataset.prepare()

    # Configuring and Creating a model
    config = CustomBBInferenceConfig()
    config.display()

    model = modellib.MaskRCNN(
        mode="inference", 
        config=config,
        model_dir=args.logs)

    # Loading model weights
    load_model_weight(model, args.weights)

    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        # image_id = random.choice(dataset.image_ids)
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        info = dataset.image_info[image_id]
        print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                            dataset.image_reference(image_id)))
        print("Original image shape: ", modellib.parse_image_meta(image_meta[np.newaxis,...])["original_image_shape"][0])

        # Run object detection
        results = model.detect_molded(np.expand_dims(image, 0), np.expand_dims(image_meta, 0), verbose=1)

        # Display results
        r = results[0]
        # log("gt_class_id", gt_class_id)
        # log("gt_bbox", gt_bbox)
        # log("gt_mask", gt_mask)

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


############################################################
#  Detection
############################################################
def detect(args):
    """Run detection on images in the given directory."""
    dataset_dir = args.dataset
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(args.results):
        os.makedirs(args.results)
    submit_dir = "submit_detection_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(args.results, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = CustomBBDataset()
    dataset.load_customBB(dataset_dir, args.subset)
    dataset.prepare()

    # Configuring and Creating a model
    config = CustomBBInferenceConfig()
    config.display()

    model = modellib.MaskRCNN(
        mode="inference", 
        config=config,
        model_dir=args.logs)

    # Loading model weights
    load_model_weight(model, args.weights)

    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
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


############################################################
#  Command Line Interface
############################################################
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for customBB object counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train', 'validate' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'imagenet' or 'last' to be used to initialize the model")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--results', required=False,
                        default=RESULTS_DIR,
                        metavar="/path/to/results/",
                        help='Results directory (default=results/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run training or prediction on")
    args = parser.parse_args()

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
