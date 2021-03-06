import os

# from practices.customBB.customBB import *
from customBB import *

############################################################
# Root directory of the project
# ROOT_DIR = os.path.abspath("../../")
ROOT_DIR = os.path.abspath("./")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/customBB/")


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
