"""
Created on Mon Jul  6 27:38:05 2015
 Kaggle facial landmark detection Neural Network class
 Tutorial: http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
@author: dwu
"""
# file kfkd.py
import cPickle as pickle
from datetime import datetime
import os
import sys
#from matplotlib import pyplot
import numpy as np
from lasagne import layers
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet
from pandas import DataFrame
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import theano

#Our network is sufficiently large now to crash Python's pickle with a maximum
# recursion error. Therefore we have to increase Python's recursion limit
# before we save it:
import sys
sys.setrecursionlimit(10000)


try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
except ImportError:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer

def float32(k):
    return np.cast['float32'](k)


def load(fname, test=False, cols=None):
    import h5py
    # load pandas dataframe
    f = h5py.File(fname, "r") 
    x_train_image_croped = f["x_train_image_croped"][:]
    x_valid_image_croped = f["x_valid_image_croped"][:]
    y_train_age = f["y_train_age"][:]
    y_train_variance = f["y_train_variance"][:]


    # only FTRAIN has any target columns
    if not test:  
        X= np.expand_dims(x_train_image_croped,axis=1) 
        X = X / 255.  
        X = X.astype(np.float32)
        y = y_train_age
        # scale target coordinates to around [-1, 1]
        ymean = y_train_age.mean()
        y = (y - ymean) / ymean
        # shuffle train data
        X, y = shuffle(X, y, random_state=42)  
        y = y.astype(np.float32)
    else:
        X= np.expand_dims(x_valid_image_croped,axis=1) 
        X = X / 255.  
        X = X.astype(np.float32)
        y = None
    return X, y


class FlipBatchIterator(BatchIterator):

    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        return Xb, yb
        
        
class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)



class ConvNet_Naive_Age_Regression(object):
    """
    This is a fixed NN
    """
    def __init__(self):        
        self.net = NeuralNet(
            layers=[
                ('input', layers.InputLayer),
                ('conv1', layers.Conv2DLayer),
                ('pool1', layers.MaxPool2DLayer),
                ('dropout1', layers.DropoutLayer),  # !
                ('conv2', layers.Conv2DLayer),
                ('pool2', layers.MaxPool2DLayer),
                ('dropout2', layers.DropoutLayer),  # !
                ('conv3', layers.Conv2DLayer),
                ('pool3', layers.MaxPool2DLayer),
                ('dropout3', layers.DropoutLayer),  # !
                ('hidden4', layers.DenseLayer),
                ('dropout4', layers.DropoutLayer),  # !
                ('hidden5', layers.DenseLayer),
                ('output', layers.DenseLayer),
                ],
            input_shape=(None, 1, 96, 96),
            conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
            dropout1_p=0.1,  # !
            conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
            dropout2_p=0.2,  # !
            conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
            dropout3_p=0.3,  # !
            hidden4_num_units=1000,
            dropout4_p=0.5,  # !
            hidden5_num_units=1000,
            output_num_units=1, output_nonlinearity=None,
              
            eval_size=0.1,
            
            update_learning_rate=theano.shared(float32(0.03)),
            update_momentum=theano.shared(float32(0.9)),
        
            regression=True,
            batch_iterator_train=FlipBatchIterator(batch_size=128),
            on_epoch_finished=[
                AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
                AdjustVariable('update_momentum', start=0.9, stop=0.999),
                ],      
            max_epochs=10000,
            verbose=1,
            )

