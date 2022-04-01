#!/usr/bin/env python

# MIT License
#
# Copyright (c) 2022 Matthis Ebel, Mario Stanke
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



import argparse
import datetime
import json
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
import time
from helper import createDataset as ds
from tensorflow.keras.layers import Dense



# Parse Command Line Parameters
# -----------------------------

parser = argparse.ArgumentParser(description = "Train and test BraiNN",
                                 formatter_class = argparse.RawTextHelpFormatter)
scriptArgs = parser.add_argument_group("Script Arguments")
scriptArgs.add_argument("--cpdata",
                        dest = "cpds", 
                        metavar = "PATH", 
                        type = argparse.FileType("rt"),
                        help = "Path to cross population data csv (optional). If given, whole dataset is classified \
                                after each model training.")
scriptArgs.add_argument("--k",
                        dest = "k", 
                        metavar = "INT", 
                        type = int,
                        default = 0,
                        help = "Parameter for k-fold cross-validation. Set to 0 for no CV (default)")
scriptArgs.add_argument("--l",
                        dest = "l", 
                        metavar = "INT", 
                        type = int,
                        default = 0,
                        help = "Parameter for k*l-fold cross-validation. Set to 0 for only k-fold CV (default)")
scriptArgs.add_argument("--normalized-input",
                        dest = "normInput",
                        action = 'store_true',
                        help = "Do not apply Z-normalization and assume that input images are already normalized")
scriptArgs.add_argument("--out",
                        dest = "out", 
                        metavar = "PATH", 
                        type = str,
                        default = ".",
                        help = "Path to directory to store the model and history in (default: current dir)")
scriptArgs.add_argument("--parameters",
                        dest = "parameters",
                        metavar = "JSON dictionary",
                        type = argparse.FileType("rt"),
                        required = True,
                        help = "JSON formatted file containing a dictionary with the parameter grid")
scriptArgs.add_argument("--perform-class-predictions",
                        dest = "classPred",
                        action = 'store_true',
                        help = "If set, generate and store class predictions for each training run on test items and \
                                possibly cross-population data (if given)")
scriptArgs.add_argument("--roi",
                        dest = "roi", 
                        metavar = "PATH", 
                        type = str,
                        help = "Optional path to dir with ROI files or complete path to a single ROI file")
scriptArgs.add_argument("--testdata",
                        dest = "testds", 
                        metavar = "PATH", 
                        type = argparse.FileType("rt"),
                        help = "Optional path to test data csv. Not used in cross validation!")
scriptArgs.add_argument("--traindata",
                        dest = "trainds", 
                        metavar = "PATH", 
                        type = argparse.FileType("rt"),
                        required = True,
                        help = "Path to training data csv")
scriptArgs.add_argument("--valdata",
                        dest = "valds", 
                        metavar = "PATH", 
                        type = argparse.FileType("rt"),
                        help = "Path to validation data csv. Not used in cross validation!")
args = parser.parse_args()

outpath = args.out
jparameters = args.parameters.name
trainfile = args.trainds.name
kCV = args.k
lCV = args.l
rescaling = False if args.normInput else True
print("[INFO] >>> Image rescaling is set to", rescaling, flush = True)

assert os.path.exists(outpath), "Cannot find directory "+outpath
assert os.path.isdir(outpath), outpath+" is not a directory"

assert os.path.exists(jparameters), "Cannot find file "+jparameters
assert os.path.isfile(jparameters), jparameters+" is not a file"

assert os.path.exists(trainfile), "Cannot find file "+trainfile
assert os.path.isfile(trainfile), trainfile+" is not a file"

assert kCV >= 0, "Negative '--k' not allowed"
assert lCV >= 0, "Negative '--l' not allowed"

valfile = args.valds.name if (kCV == 0) else ""
if kCV == 0:
    assert os.path.exists(valfile), "Cannot find file "+valfile
    assert os.path.isfile(valfile), valfile+" is not a file"

testfile = args.testds.name if args.testds and kCV == 0 else ""
if kCV == 0:
    assert os.path.exists(testfile), "Cannot find file "+testfile
    assert os.path.isfile(testfile), testfile+" is not a file"

cpfile = args.cpds.name if args.cpds else ""
if cpfile:
    assert os.path.exists(cpfile), "Cannot find file "+cpfile
    assert os.path.isfile(cpfile), cpfile+" is not a file"

if args.roi:
    roipath = args.roi
    assert os.path.isdir(roipath) or os.path.isfile(roipath), str(roipath)+" is neither a directory nor a file"
    rois = {}
    if os.path.isdir(roipath):
        for f in os.listdir(roipath):
            roifile = os.path.join(roipath, f)
            if os.path.isfile(roifile) and f[-len("pickle4"):] == "pickle4":
                with open(roifile, 'rb') as fh:
                    roi = pickle.load(fh)
                    
                key = f.replace(".pickle4", "")
                rois[key] = roi

    elif os.path.isfile(roipath):
        _, roifile = os.path.split(roipath)
        assert roifile[-len("pickle4"):] == "pickle4", str(roipath)+" is invalid"
        with open(roipath, 'rb') as fh:
            roi = pickle.load(fh)
            
        key = roifile.replace(".pickle4", "")
        rois[key] = roi

    assert len(rois) > 0, str(os.listdir(roipath))

else:
    rois = {'noroi': np.empty(())}



# Load Configuration and Data
# ---------------------------

with open(jparameters, "rt") as fh:
    grid = json.load(fh) # contains user defined parameter grid

# used to prefix output files
outfileBase = "brainn-" + time.strftime("%m-%d-%H-%M")
print("[INFO] >>> Using", outfileBase, "as output file prefix", flush = True)

# load datasets
dfCrossPop = pd.read_csv(cpfile) if cpfile else None
if kCV == 0:
    dfTrain = pd.read_csv(trainfile)
    dfVal = pd.read_csv(valfile)
    dfTest = pd.read_csv(testfile) if testfile else None
    trainSize = len(dfTrain.index)
    valSize = len(dfVal.index)
else:
    df = pd.read_csv(trainfile)
    df = df.sample(frac=1).reset_index(drop=True) # shuffle df
    # do k(*l) splits
    datasets = []
    for ind in np.array_split(df.index, kCV):
        outerTestset = df.loc[ind]
        outerTrainset = df.loc[df.index.difference(ind)]
        if lCV > 0:
            dataset = {'testset': outerTestset, 'inner': []}
            for ind2 in np.array_split(outerTrainset.index, lCV):
                innerValset = outerTrainset.loc[ind2]
                innerTrainset = outerTrainset.loc[outerTrainset.index.difference(ind2)]
                dataset['inner'].append({'valset': innerValset, 'trainset': innerTrainset})

        else:
            dataset = {'testset': outerTestset, 'trainset': outerTrainset} # no valset for k-CV

        datasets.append(dataset)

    trainSize = len(datasets[0]['inner'][0]['trainset'].index) if lCV > 0 else len(datasets[0]['trainset'].index)

# Create Parameter Dicts From Grid
#
# Valid fields and their default values
#   Each key gets a list(!) of values (which might be again lists), each possible combination of values
#     is executed in grid search
#
#   Copy this dict to a .json text file and adjust value lists to create a grid
#
#   The models do not need to specify input and output layers. A layer is specified with the same
#     dict structure like optimizer etc., i.e. a class_name entry and a config dict listing the
#     layer arguments (see tf docs). Important: for string values, use escaped quotes \'...\'!
#     Layers can use the variable `initializer` (no quotes here) to obtain the respective 
#     initialization function for their grid batch

defaultConfig = {
    "epochs": [100],
    "batchsize": [32], # global, only one value allowed
    "stepsPerEpoch": [trainSize/32], # global, only one value allowed
    "initializers": ["he_normal"],
    "metrics": [["BinaryAccuracy"]],
    "model": [
        [{"class_name": "MaxPool3D", "config": {"pool_size": "(6,6,6)", "strides": "(6,6,6)"}},
         {"class_name": "Conv3D", 
          "config": {"filters": 32, "kernel_size": "(7,7,7)", "strides": "(1,1,1)", "activation": "\"linear\"", 
                     "kernel_initializer": "initializer"}},
         {"class_name": "PReLU", "config": {}},
         {"class_name": "MaxPool3D", "config": {"pool_size": "(2,2,2)", "strides": "(2,2,2)"}},
         {"class_name": "Flatten", "config": {}},
         {"class_name": "Dense", 
          "config": {"units": 128, "activation": "\"linear\"", "kernel_initializer": "initializer"}},
         {"class_name": "PReLU", "config": {}},
         {"class_name": "Dropout", "config": {"rate": 0.5}}]
    ],
    "loss": ["binary_crossentropy"],
    "optimizer": [{"class_name": "adam", "config": {"learning_rate": 0.0001}}]     
}

# set defaults if not set by user
for field in defaultConfig:
    if field not in grid:
        grid[field] = defaultConfig[field]

assert len(grid['batchsize']) == 1, "The 'batchsize' value in config must be a single-element list"
assert len(grid['stepsPerEpoch']) == 1, "The 'stepsPerEpoch' value in config must be a single-element list"

# create all possible parameter combinations from grid dict recursively
#   this results in a list of dicts such that each parameter combination is covered
def createParameterDicts(currentDictRef, parameterDictRef, parameterDictListRef):
    currentDict = dict(currentDictRef) # true copys
    parameterDict = dict(parameterDictRef)
    
    if parameterDict:
        key = list(parameterDict.keys())[0]
        values = parameterDict[key]
        parameterDict.pop(key)  # remove key for recursion
        for value in values:
            currentDict[key] = value                    
            createParameterDicts(currentDict, parameterDict, parameterDictListRef) # recursion
                
    else:
        # dict is empty, so store complete parameter string
        parameterDictListRef.append(currentDict)

parameterList = []
createParameterDicts({}, grid, parameterList)

print("[INFO] >>> All configurations:", flush = True)
for p in parameterList: 
    print(p, end = "\n\n", flush = True)



# Functions to Create a Model
# ---------------------------

# this is fixed and needs to be known
mri_shape = (121, 145, 121)

def getLayer(layerdict, initializer):
    # creates e.g. a code string like
    #   `tf.keras.layers.Conv3D(filters = 64, kernel_size = 7, activation = 'linear', kernel_initializer = initializer)`
    codestr = "layer = tf.keras.layers." + layerdict['class_name'] + "("
    paramstr = []
    for key in layerdict['config']:
        paramstr.append(key + " = " + str(layerdict['config'][key]))

    codestr += ", ".join(paramstr) + ")"

    loc = {'initializer': initializer}
    exec(codestr, globals(), loc)
    return loc['layer']



# always adds input and output layer
def create_model(parameters, input_shape = mri_shape):
    initializer = tf.keras.initializers.get(parameters['initializers'])
    model = tf.keras.models.Sequential() # sequential stack of layers
    
    # make input 4-dimensional as expected by Conv3d
    # in the input the 4-th dimension has size 1 (grayscale)
    model.add( tf.keras.layers.Reshape(input_shape + (1,), input_shape = input_shape))
    
    for layerstr in parameters['model']:
        model.add( getLayer(layerstr, initializer) )

    model.add( Dense (1, activation = "sigmoid", kernel_initializer = initializer))

    return model



# Functions to Actually Train a Model
# -----------------------------------

def createDatasets(batchSize, mask, dfTrain, 
                   dfVal = None, dfTest = None, dfCrossPop = None,
                   mri_shape = mri_shape, rescaling = rescaling):
    train_ds = ds.createDataset(dfTrain, repeat = True, batchSize = batchSize, prefetchSize = 1, 
                                mri_shape = mri_shape, mask = mask, rescaling = rescaling)
    val_ds = ds.createDataset(dfVal, repeat = False, batchSize = 1, prefetchSize = min(batchSize, len(dfVal.index)), 
                              mri_shape = mri_shape, mask = mask, rescaling = rescaling) if dfVal is not None else None
    test_ds = ds.createDataset(dfTest, repeat = False, batchSize = 1, prefetchSize = min(batchSize, len(dfTest.index)), 
                               mri_shape = mri_shape, mask = mask, rescaling = rescaling) \
                                   if dfTest is not None else None
    crossPop_ds = ds.createDataset(dfCrossPop, repeat = False, batchSize = 1, 
                                   prefetchSize = min(batchSize, len(dfCrossPop.index)), 
                                   mri_shape = mri_shape, mask = mask, rescaling = rescaling) \
                                       if dfCrossPop is not None else None
    return train_ds, val_ds, test_ds, crossPop_ds



# this is where training and test prediction happen
def runGrid(parameters, gridIdx, train_ds, 
            val_ds = None, valSize = None, test_ds = None, crossPop_ds = None,
            outpath = outpath, outfileBase = outfileBase, classPred = args.classPred):
    # possible that OOM or other errors occur, avoid failing the entire grid search
    ts = datetime.datetime.now()
    try:
        model = create_model(parameters)

        # define the loss, optimization algorithm and prepare the model for gradient computation 
        model.compile(optimizer = tf.keras.optimizers.get(parameters['optimizer']),
                      loss = tf.keras.losses.get(parameters['loss']), 
                      metrics = [tf.keras.metrics.get(m) for m in parameters['metrics']])

        modelconf = model.to_json()
        modelSummaryStr = model.summary()
        print(modelSummaryStr, flush = True)
        
        # Set up Training

        modelfname = os.path.join(outpath, outfileBase + "_" + str(gridIdx) + ".h5")

        # Callbacks
        # Function to store model to file, if validation loss has a new record
        # Check always after having seen at least another save_freq examples.
        if val_ds:
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                modelfname, monitor = 'val_loss', mode = 'min', 
                save_best_only = True, save_weights_only = False, verbose = 1)
        else:
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                modelfname, monitor = 'loss', mode = 'min', 
                save_best_only = True, save_weights_only = False, verbose = 1)

        # Function to decrease learning rate by 'factor'
        # when there has been no significant improvement in the last 'patience' epochs.
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor = 'val_loss', mode = 'min', factor = 0.75, patience = 4, verbose = 1)

        # Run Training and Testing
    
        print("[INFO] >>> Calling fit()", flush = True)
        if val_ds:
            history = model.fit(
                train_ds,
                epochs = parameters['epochs'], 
                steps_per_epoch = parameters['stepsPerEpoch'],
                validation_data = val_ds,
                validation_steps = valSize, # validation data has single-element batches, use all data
                verbose = 2,
                callbacks = [checkpoint, reduce_lr]
            )
        else:
            history = model.fit(
                train_ds,
                epochs = parameters['epochs'], 
                steps_per_epoch = parameters['stepsPerEpoch'],
                verbose = 2,
                callbacks = [checkpoint]
            )

        if test_ds:
            # Load the parameters with the best validation accuracy during training
            model.load_weights(modelfname)
            test_loss, test_acc = model.evaluate(test_ds, verbose = 0)
            test_pred = np.reshape(model.predict(test_ds, verbose = 0), -1).tolist() if classPred else list()
        else:
            test_loss, test_acc = 0, 0
            test_pred = list()

        if crossPop_ds:
            # Load the parameters with the best validation accuracy during training
            model.load_weights(modelfname)
            cp_loss, cp_acc = model.evaluate(crossPop_ds, verbose = 0)
            cp_pred = np.reshape(model.predict(crossPop_ds, verbose = 0), -1).tolist() if classPred else list()
        else:
            cp_loss, cp_acc = 0, 0
            cp_pred = list()

        # Store Training History

        historyfname = os.path.join(outpath, outfileBase + "_history_" + str(gridIdx) + ".csv")
        historyDF = pd.DataFrame.from_dict(history.history)
        historyDF.to_csv(historyfname, index = False)

        tsEnd = datetime.datetime.now()
        tsDuration = tsEnd - ts
        print("[INFO] >>> Finished", tsEnd, flush = True)
        print("[INFO] >>> Took", tsDuration, flush = True)

    except Exception as e:
        print("[WARNING] >>> Training failed for model", gridIdx, flush = True)
        print("[WARNING] >>> Exception: " + repr(e), flush = True)
        tsEnd = datetime.datetime.now()
        tsDuration = tsEnd - ts
        print("[INFO] >>> Finished", tsEnd, flush = True)
        print("[INFO] >>> Took", tsDuration, flush = True)

    return modelconf, modelSummaryStr, historyDF, tsDuration, test_loss, test_acc, test_pred, cp_loss, cp_acc, cp_pred



# Start Grid Search
# -----------------

resultfname = os.path.join(outpath, outfileBase + "_result.csv")
predictfname = os.path.join(outpath, outfileBase + "_predictions.json")
print ("\n[INFO] >>> Using", outfileBase, "to store outfiles.", end = "\n\n", flush = True)

# store values for "best" model weights for each configuration
resultDict = {'accuracy': [],
              'gridIdx': [],
              'loss': [],
              'time': [],
              'val_accuracy': [],
              'val_loss': [],
              'test_accuracy': [],
              'test_loss': [],
              'cross_population_accuracy': [],
              'cross_population_loss': []}

predictionDict = {'test': {},
                  'cross_population': {'true': list()}}
if dfCrossPop is not None:
    predictionDict["cross_population"]["true"] = (np.array(dfCrossPop['sex'])-1).tolist() 



for roiKey in rois: # if no ROI were specified, this is "noroi" and no ROI mask is applied
    roi = rois[roiKey]
    gridIdx = 0
    print("[INFO] >>> Starting runs for ROI", roiKey, flush = True)
    for parameters in parameterList:
        ts = datetime.datetime.now()
        print("[INFO] >>> Starting run on parameter configuration", (gridIdx+1), "/", len(parameterList), "at", ts, 
              flush = True)

        # 'accuracy' key in history varies given the accuracy measure 
        #   (e.g. it's 'binary_accuracy' for BinaryAccuracy metric)
        #   For now, hardcode this
        acckey = 'binary_accuracy' if "BinaryAccuracy" in parameters['metrics'] else "accuracy"
        print("[INFO] >>> Using", acckey, " to get accuracy measures from training history", flush = True)

        idx = roiKey+"_"+str(gridIdx)

        if kCV == 0: # single training run if no k(*l)-cross-validation is desired
            # for later eval
            dfTrain.to_csv(os.path.join(outpath, outfileBase + "_trainset.csv"), index=False)
            dfVal.to_csv(os.path.join(outpath, outfileBase + "_valset.csv"), index=False)
            if dfTest is not None:
                dfTest.to_csv(os.path.join(outpath, outfileBase + "_testset.csv"), index=False)

            train_ds, val_ds, test_ds, crossPop_ds = createDatasets(batchSize = grid['batchsize'][0], mask = roi, 
                                                                    dfTrain = dfTrain, dfVal = dfVal, dfTest = dfTest, 
                                                                    dfCrossPop = dfCrossPop)
            
            # quick check that not all elements are the same
            checkElems = [x[0][0] for x in train_ds.take(1)]
            checkElems.extend([x[0][0] for x in val_ds.take(1)])
            assert (checkElems[0] != checkElems[1]).numpy().any()

            # perform training and predictions
            modelconf, modelSummaryStr, historyDF, duration, \
                testLoss, testAcc, testPred, cpLoss, cpAcc, cpPred = runGrid(parameters = parameters, 
                                                                             gridIdx = idx, 
                                                                             train_ds = train_ds, 
                                                                             val_ds = val_ds, 
                                                                             valSize = len(dfVal.index), 
                                                                             test_ds = test_ds, 
                                                                             crossPop_ds = crossPop_ds)

            # store history and training performance
            bestHistoryDF = historyDF[historyDF['val_loss'] == min(historyDF['val_loss'])]
            resultDict['accuracy'].append(bestHistoryDF.iloc[0][acckey])
            resultDict['gridIdx'].append(idx)
            resultDict['loss'].append(bestHistoryDF.iloc[0]['loss'])
            resultDict['time'].append(duration)
            resultDict['val_accuracy'].append(bestHistoryDF.iloc[0]['val_'+acckey])
            resultDict['val_loss'].append(bestHistoryDF.iloc[0]['val_loss'])
            if test_ds is not None:
                resultDict['test_accuracy'].append(testAcc)
                resultDict['test_loss'].append(testLoss)
                predictionDict["test"][idx] = {'true': (np.array(dfTest['sex'])-1).tolist(), 
                                               'pred': testPred}
            else:
                resultDict['test_accuracy'].append("")
                resultDict['test_loss'].append("")

            if crossPop_ds is not None:
                resultDict['cross_population_accuracy'].append(cpAcc)
                resultDict['cross_population_loss'].append(cpLoss)
                predictionDict["cross_population"][idx] = cpPred
            else:
                resultDict['cross_population_accuracy'].append("")
                resultDict['cross_population_loss'].append("")

            # store intermediate result
            pd.DataFrame.from_dict(resultDict).to_csv(resultfname, index = False)
            if args.classPred:
                with open(predictfname, 'wt') as fh:
                    json.dump(predictionDict, fh)



        else: # do a k(*l)-cross-validation
            kcount = 0
            for dataset in datasets:
                # store testset for later evaluation
                kInd = idx+"k"+str(kcount)
                testsetfname = os.path.join(outpath, outfileBase + "_testset_" + kInd + ".csv")
                dataset['testset'].to_csv(testsetfname, index=False)

                if lCV > 0:
                    lcount = 0
                    for ids in dataset['inner']:
                        lkInd = kInd+"l"+str(lcount)
                        train_ds, val_ds, test_ds, crossPop_ds = createDatasets(batchSize = grid['batchsize'][0], 
                                                                                mask = roi, 
                                                                                dfTrain = ids['trainset'], 
                                                                                dfVal = ids['valset'], 
                                                                                dfTest = dataset['testset'], 
                                                                                dfCrossPop = dfCrossPop)
                                                                                
                        # quick check that not all elements are the same
                        checkElems = [x[0][0] for x in train_ds.take(1)]
                        checkElems.extend([x[0][0] for x in val_ds.take(1)])
                        assert (checkElems[0] != checkElems[1]).numpy().any()

                        # perform training and predictions
                        modelconf, modelSummaryStr, historyDF, duration, \
                            testLoss, testAcc, testPred, \
                                cpLoss, cpAcc, cpPred = runGrid(parameters = parameters,
                                                                gridIdx = lkInd,
                                                                train_ds = train_ds,
                                                                val_ds = val_ds, 
                                                                valSize = len(ids['valset'].index), 
                                                                test_ds = test_ds, 
                                                                crossPop_ds = crossPop_ds)

                        bestHistoryDF = historyDF[historyDF['val_loss'] == min(historyDF['val_loss'])]
                        resultDict['accuracy'].append(bestHistoryDF.iloc[0][acckey])
                        resultDict['gridIdx'].append(lkInd)
                        resultDict['loss'].append(bestHistoryDF.iloc[0]['loss'])
                        resultDict['time'].append(duration)
                        resultDict['val_accuracy'].append(bestHistoryDF.iloc[0]['val_'+acckey])
                        resultDict['val_loss'].append(bestHistoryDF.iloc[0]['val_loss'])
                        resultDict['test_accuracy'].append(testAcc)
                        resultDict['test_loss'].append(testLoss)
                        predictionDict["test"][lkInd] = {'true': (np.array(dataset['testset']['sex'])-1).tolist(), 
                                                         'pred': testPred}
                        if crossPop_ds is not None:
                            resultDict['cross_population_accuracy'].append(cpAcc)
                            resultDict['cross_population_loss'].append(cpLoss)
                            predictionDict["cross_population"][lkInd] = cpPred
                        else:
                            resultDict['cross_population_accuracy'].append("")
                            resultDict['cross_population_loss'].append("")


                        # store intermediate result
                        pd.DataFrame.from_dict(resultDict).to_csv(resultfname, index = False)
                        if args.classPred:
                            with open(predictfname, 'wt') as fh:
                                json.dump(predictionDict, fh)

                        trainsetfname = os.path.join(outpath, outfileBase + "_trainset_" + lkInd + ".csv")
                        valsetfname = os.path.join(outpath, outfileBase + "_valset_" + lkInd + ".csv")
                        ids['trainset'].to_csv(trainsetfname, index = False)
                        ids['valset'].to_csv(valsetfname, index = False)

                        lcount += 1

                else: # only k-cross-validation without validation set
                    train_ds, _, test_ds, crossPop_ds = createDatasets(batchSize = grid['batchsize'][0], mask = roi, 
                                                                       dfTrain = dataset['trainset'], dfVal = None, 
                                                                       dfTest = dataset['testset'], 
                                                                       dfCrossPop = dfCrossPop)

                    # perform training and predictions
                    modelconf, modelSummaryStr, historyDF, duration, \
                        testLoss, testAcc, testPred, cpLoss, cpAcc, cpPred = runGrid(parameters = parameters, 
                                                                                     gridIdx = kInd,
                                                                                     train_ds = train_ds, 
                                                                                     test_ds = test_ds, 
                                                                                     crossPop_ds = crossPop_ds)

                    bestHistoryDF = historyDF[historyDF['loss'] == min(historyDF['loss'])]
                    resultDict['accuracy'].append(bestHistoryDF.iloc[0][acckey])
                    resultDict['gridIdx'].append(kInd)
                    resultDict['loss'].append(bestHistoryDF.iloc[0]['loss'])
                    resultDict['time'].append(duration)
                    resultDict['val_accuracy'].append("")
                    resultDict['val_loss'].append("")
                    resultDict['test_accuracy'].append(testAcc)
                    resultDict['test_loss'].append(testLoss)
                    predictionDict["test"][kInd] = {'true': (np.array(dataset['testset']['sex'])-1).tolist(), 
                                                    'pred': testPred}
                    if crossPop_ds is not None:
                        resultDict['cross_population_accuracy'].append(cpAcc)
                        resultDict['cross_population_loss'].append(cpLoss)
                        predictionDict["cross_population"][kInd] = cpPred
                    else:
                        resultDict['cross_population_accuracy'].append("")
                        resultDict['cross_population_loss'].append("")


                    # store intermediate result
                    pd.DataFrame.from_dict(resultDict).to_csv(resultfname, index = False)
                    if args.classPred:
                        with open(predictfname, 'wt') as fh:
                            json.dump(predictionDict, fh)

                    trainsetfname = os.path.join(outpath, outfileBase + "_trainset_" + kInd + ".csv")
                    dataset['trainset'].to_csv(trainsetfname, index = False)

                kcount += 1

        # save model configuration -- needed for workaround to load model for predictions later
        #   (loading via large h5 file is buggy)
        parameters['model_summary'] = modelSummaryStr
        with open(os.path.join(outpath, outfileBase + "_parameters_" + idx + ".json"), "wt") as fh:
            json.dump(parameters, fh, indent=4)

        modelconffname = os.path.join(outpath, outfileBase + "_modelconf_" + idx + ".json")
        with open(modelconffname, "wt") as fh:
            fh.write(modelconf)

        tsEnd = datetime.datetime.now()
        tsDuration = tsEnd - ts
        print("[INFO] >>> run on parameter configuration", (gridIdx+1), "/", len(parameterList), "finished", tsEnd, 
              flush = True)
        print("[INFO] >>> run on parameter configuration", (gridIdx+1), "/", len(parameterList), "took", tsDuration, 
              flush = True)

        gridIdx += 1



# store final result
pd.DataFrame.from_dict(resultDict).to_csv(resultfname, index = False)
if args.classPred:
    with open(predictfname, 'wt') as fh:
        json.dump(predictionDict, fh)