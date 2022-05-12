# BraiNN
Convolutional Neural Network for classifying sex from MRI

Find the preprint of the related paper here: https://medrxiv.org/cgi/content/short/2022.04.27.22274355v1

### Project Overview

* `brainn.py` Use this script to train BraiNN on your data and to make predictions
* `helper/createDataset.py` Helper module to create TensorFlow data sets
* `sample/brainn_config.json` Configuration file for BraiNN model
* `sample/logreg_config.json` Configuration file for LogReg model
* `sample/sample_GMV.[nii|pickle4.gz]` Preprocessed example brain (only GMV) in Nifty format and as pickled Numpy array

## Usage

Here is how to train BraiNN with your MRI data

### Dependencies

You need to install these third-party Python libraries

* `numpy`
* `pandas`
* `tensorflow` (>= 2.2.0 should work)

### Data Preparation

Your MRI images must be stored as gzipped and pickled Numpy arrays. The shape must be the same for all images. If you want to apply binary masks (e.g. to train only on regions of interest (ROI)), the masks must be of the same shape as the MR images. 

By default, each MR image is getting z-normalized, i.e. the mean and standard deviation of all voxels in the image are computed and then, for each voxel, the mean is subtracted and the difference is divided by the standard deviation. You can turn this behaviour off with the `--normalized-input` option to use your images as they are.

The script `brainn.py` accepts one or more tables in CSV format that specify the _sex_ and the _image path_ for each sample:

| sex | path | ... |
| --- | ---- | --- |
| 1   | /path/to/file1.pickle.gz | ... |
| 2   | /path/to/file2.pickle.gz | ... |
| 1   | /path/to/file3.pickle.gz | ... |
| ...   | ... | ... |

The first line must contain the column names, and the columns `sex` and `path` must be present. The values in `sex` must be either `1` or `2` and encode the proband's sex. Note that during training, the `sex` is subtracted by one, so the model output is in the range `[0, 1]`

The values in `path` must be valid paths to the respective pickled and gzipped numpy arrays of the MR images.

#### Using ROI Masks

If you specify mask(s) via the `--roi` command line argument, each image is multiplied with the mask before training and prediction. If you specify more than one mask, a full training round will be performed for each mask.

#### Example Scan

The directory `sample/` contains a MRI scan of a brain that has been preprocessed as described in the publication for reference. Note that this image was not yet Z-score normalized, but all other preprocessing steps have been performed. The true class of this brain is "male".

### Run Training

Run `python brainn.py -h` to get an overview of the command line arguments. We describe some of them in more detail now.

#### Grid Configuration

`--parameters` requires a JSON formatted text file that contains a JSON object with model and training configuration details. See the files `sample/brainn_config.json` and `sample/logreg_config.json` for an example. Required fields missing in your configuration will be set to default values (that are the values in `sample/brainn_config.json`), unknown fields will be ignored but may lead to a larger grid than neccessary, so please avoid this.

For most fields, you can specify more than one option. This then creates a grid where _every combination_ of parameters is trained.

#### Single Training

If you leave the argument `--k` to its default (`0`), a single training run is performed for each configuration (and ROI). In this case, you _must_ split up your data into training, validation and test set and supply the respective CSV files via the appropriate arguments.

#### k\*l-fold Cross Validation

Here, for each configuration (and ROI), multiple models are trained, depending on your choice for _k_ and _l_

If `--k` is greater than `0`, a k-fold cross validation is performed, resulting in _k_ trained models. Note that no validation set is used during training for this setting.

If both `--k` and `--l` are greater than `0`, a k\*l-fold cross validation is performed, resulting in _k\*l_ trained models.

In both cases, `--testdata` and `--valdata` are ignored! You only need to specify a CSV to `--traindata`, the splits will be done automatically.

#### Cross-Cohort Prediction

You can optionally specify a CSV table to MR images from a differen cohort via `--cpdata`. The options `--normalized-input` and `--roi` apply here as well! If given, after each training, the model is used to predict all MR images from the cross cohort data set.

#### Other Options

If `--perform-class-predictions` is specified, the predictions of every MR image in the test data (and the cross cohort data) will be stored for each trained model.

### Output

Depending on the arguments, a number of output files is created in the directory specified by `--out`. They are all prefixed by `brainn-mm-dd-HH-MM`, where the last four fields are the time stamp of the overall training start. Most files also get a `gridIndex` that specifies the configuration and optionally the _k_ and _l_ split, all starting at `0`.

* `prefix_gridIndex.h5` the TensorFlow model(s)
* `prefix_history_gridIndex.csv` a table containing the training history for the respective model
* `prefix_result.csv` a table containing an overview of all trained models and their performances
* `prefix_predictions.json` a JSON file containing the predictions and true classes for each test set image for each model, the format is
  ```JSON
  {"test": {"gridIndex": {"true": [...],
                          "pred": [...]},
            ...},
   "cross_population": {"true": [...],
                        "gridIndex": [...],
                        ...}}
  ```
* `prefix_trainset.csv`, `prefix_valset.csv`, `prefix_testset.csv` the same tables as were the input for single training runs
* `prefix_testset_gridIndex.csv` table with test set split for the respective model in a k(\*l)-cross-validation
* `prefix_trainset_gridIndex.csv` table with training set split for the respective model in a k(\*l)-cross-validation
* `prefix_valset_gridIndex.csv` table with validation set split for the respective model in a k\*l-cross-validation
* `prefix_parameters_gridIndex.json` JSON file with the respective model configuration
* `prefix_modelconf_gridIndex.json` TensorFlow-readable model configuration, useful to manually re-create the model later
