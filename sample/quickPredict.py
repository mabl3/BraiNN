import argparse
import datetime
import gzip
import json
import math
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf

# print if GPU is available
print("", flush=True) # flush: force immediate print, for live monitoring
print("Tensorflow version", tf.__version__, flush=True)
print("tf.config.list_physical_devices('GPU'): ", tf.config.list_physical_devices('GPU'), flush=True)
print("tf.test.is_built_with_gpu_support(): ", tf.test.is_built_with_gpu_support(), flush=True)
print("tf.test.is_built_with_cuda()", tf.test.is_built_with_cuda(), flush=True)
print("", flush=True)



# ## Parse Command Line Parameters

parser = argparse.ArgumentParser(description = "Run test CNN",
                                 formatter_class = argparse.RawTextHelpFormatter)
scriptArgs = parser.add_argument_group("Script Arguments")
scriptArgs.add_argument("--mri",
                        dest = "img", 
                        metavar = "PATH", 
                        type=argparse.FileType("rt"),
                        required=True,
                        help="Path to mri as pickled, gzipped numpy-array")
scriptArgs.add_argument("--model",
                        dest="model",
                        metavar="FILE",
                        type=argparse.FileType("rb"),
                        required=True,
                        help="h5 model file with trained parameters")
scriptArgs.add_argument("--modelconfig",
                        dest="modelconfig",
                        metavar="JSON FILE",
                        type=argparse.FileType("rt"),
                        required=True,
                        help="TensorFlow file in JSON format with the model configuration")
scriptArgs.add_argument("--parameters",
                        dest="parameters",
                        metavar="JSON dictionary",
                        type=argparse.FileType("rt"),
                        required=True,
                        help="JSON formatted file containing a dictionary with the parameters for the model")
args = parser.parse_args()

modelfile = args.model.name
modelconfigfile = args.modelconfig.name
jparameters = args.parameters.name
imgfile = args.img.name

assert os.path.exists(modelfile), "Cannot find file "+modelfile
assert os.path.isfile(modelfile), modelfile+" is not a file"

assert os.path.exists(modelconfigfile), "Cannot find file "+modelconfigfile
assert os.path.isfile(modelconfigfile), modelconfigfile+" is not a file"

assert os.path.exists(jparameters), "Cannot find file "+jparameters
assert os.path.isfile(jparameters), jparameters+" is not a file"

assert os.path.exists(imgfile), "Cannot find file "+imgfile
assert os.path.isfile(imgfile), imgfile+" is not a file"



## Load Model

with open(jparameters, "rt") as fh:
    parameters = json.load(fh)

with open(modelconfigfile, "rt") as fh:
    modelconfig = fh.read()
    
# create model without loading trained parameters
model = tf.keras.models.model_from_json(modelconfig)
model.compile(optimizer = tf.keras.optimizers.get(parameters['optimizer']),
              loss = tf.keras.losses.get(parameters['loss']), 
              metrics = [tf.keras.metrics.get(m) for m in parameters['metrics']])

summaryStr = model.summary()
print(summaryStr, flush=True)
parameters['model_summary'] = summaryStr

# load trained parameters
model.load_weights(modelfile)



# ## Preprocess Images

# this is fixed and needs to be known
mri_shape = (121, 145, 121)

with gzip.open(imgfile) as fh:
    img = pickle.load(fh)

assert img.shape == mri_shape, "[ERROR] >>> Image shape "+str(img.shape)+" invalid, must be "+str(mri_shape)

vmean = np.mean(img)
if not math.isclose(vmean, 0):
    print("[WARNING] >>> Image voxel mean is", vmean, "-- you may need to perform z-score normalization!")
    inp = input("Do you want to z-score normalize the image now? [Y|n] ")
    if inp in ["", "y", "Y"]:
        print("[INFO] >>> Performing z-score normalization")
        vmean = np.mean(img)
        vstdev = np.std(img)
        img = (img-vmean)/vstdev



## Run Prediction

ts = datetime.datetime.now()
print("[INFO] >>> Starting prediction", flush=True)

p = model(np.expand_dims(img, axis=0)).numpy()[0]

tsEnd = datetime.datetime.now()
tsDuration = tsEnd - ts
print("[INFO] >>> Finished", tsEnd, flush=True)
print("[INFO] >>> Took", tsDuration, flush=True)

# print result
print("Femaleness probability of the image is", p)
