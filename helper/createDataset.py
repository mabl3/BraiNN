import gzip
import numpy as np
import os
import pickle
import tensorflow as tf



# expect images as pickled and gzipped numpy arrays of shape (121, 145, 121)
def loadImg(path):
    assert os.path.isfile(path), str(path) + " does not exist"
    with gzip.open(path, 'rb') as fh:
        img = pickle.load(fh)

    return img.astype(np.float64)



def applyMask(img, mask):
    assert type(img) == np.ndarray, "Image has wrong type " + str(type(img))
    assert type(mask) == np.ndarray, "Mask has wrong type " + str(type(mask))
    assert img.shape == mask.shape, "Image and mask dimensions differ: "+str(img.shape)+" != "+str(mask.shape)
    dtype = img.dtype
    if mask.dtype != dtype:
        mask = mask.astype(dtype)

    assert mask.dtype == dtype, "Image (" + str(dtype) + ") and mask (" + str(mask.dtype) + ") have different dtypes"
    maskedImg = img * mask
    return maskedImg



def createLabel(label):
    lab = label - 1 # df['sex'] is in [1,2] -> [0,1]
    return [lab]



def generateElement(path, label, mask = np.empty(()), rescaling: bool = True):
    lab = createLabel(label)
    img = loadImg(path)
    if mask.shape != ():
        img = applyMask(img, mask)

    if rescaling:
        mean = np.mean(img)
        stdev = np.std(img)
        img = (img-mean)/stdev

    yield img, lab



def createDataset(df, repeat = False, batchSize = 32, prefetchSize = 1, mri_shape = (121, 145, 121), 
                  mask = np.empty(()), rescaling: bool = True):
    generateDataLambda = lambda pathTensor, labelTensor : tf.data.Dataset.from_generator(
        generateElement,
        output_types = (tf.float64, tf.float64),
        output_shapes = (tf.TensorShape(mri_shape), tf.TensorShape((1))),
        args=[pathTensor, labelTensor, mask, rescaling]
    )
    
    # dataset with paths and labels represented as tensors
    ds_path = tf.data.Dataset.from_tensor_slices((df['path'], df['sex']))

    # load images by calling generateElement on each path tensor, creating the actual dataset
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    if repeat:
        # use for training set to enable arbitrary number of training epochs
        ds = ds_path.interleave(generateDataLambda, num_parallel_calls=AUTOTUNE).repeat()
    else:
        ds = ds_path.interleave(generateDataLambda, num_parallel_calls=AUTOTUNE)

    # quick check that not all elements are the same
    if len(df.index) > 1:
        checkElems = [x[0] for x in ds.take(2)]
        assert (checkElems[0] != checkElems[1]).numpy().any(), str(checkElems)

    if batchSize:
        ds = ds.batch(batchSize)

    if prefetchSize:
        ds = ds.prefetch(prefetchSize)

    return ds