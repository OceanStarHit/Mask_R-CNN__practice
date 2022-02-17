# CustomBB

This sample segments individual objects in microscopy images.
The `customBB.py` file contains the main parts of the code.


## Command line Usage
Train a new model with starting from ImageNet weights using `train` dataset.
```
python customBB.py train --dataset=/path/to/dataset --subset=train --weights=imagenet
```

Train a new model with starting from specific weights file using `train` dataset.
```
python customBB.py train --dataset=/path/to/dataset --subset=stage1_train --weights=/path/to/weights.h5
```

Resume training a model that you have ever trained earlier
```
python customBB.py train --dataset=/path/to/dataset --subset=train --weights=last
```

Generate submission file from `val` images
```
python customBB.py validate --dataset=/path/to/dataset --subset=val --weights=<last or /path/to/weights.h5>
```

Generate submission file from `test` images
```
python customBB.py detect --dataset=/path/to/dataset --subset=test --weights=<last or /path/to/weights.h5>
```

### Examples (Tested):
```
(mrcnn_env) E:\Working\Mask_R-CNN__practice\practices\customBB>python customBB.py train --datasets=E:/Working/Mask_R-CNN__practice/datasets/customBB --subset=train --weights=imagenet
```
```
(mrcnn_env) E:\Working\Mask_R-CNN__practice\practices\customBB>python customBB.py train --datasets=E:/Working/Mask_R-CNN__practice/datasets/customBB --subset=train --weights=last
```
```
(mrcnn_env) E:\Working\Mask_R-CNN__practice\practices\customBB>python customBB.py validate --datasets=E:/Working/Mask_R-CNN__practice/datasets/customBB --subset=val --weights=last
```
```
(mrcnn_env) E:\Working\Mask_R-CNN__practice\practices\customBB>python customBB.py detect --datasets=E:/Working/Mask_R-CNN__practice/datasets/customBB --subset=test --weights=last
```

### The Easiest Ways without arguments (Tested) 
```
(mrcnn_env) E:\Working\Mask_R-CNN__practice>python -m practices.customBB.customBB_train
```
```
(mrcnn_env) E:\Working\Mask_R-CNN__practice>python -m practices.customBB.customBB_validate
```
```
(mrcnn_env) E:\Working\Mask_R-CNN__practice>python -m practices.customBB.customBB_test
```


## Jupyter notebooks
Two Jupyter notebooks are provided as well: `inspect_customBB_data.ipynb` and `inspect_customBB_model.ipynb`.
They explore the dataset, run stats on it, and go through the detection process step by step.

