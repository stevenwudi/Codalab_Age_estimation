"""
Use trained face detection NN to detect face,
"""
# we need to import all the classes
# pre-defined NN class 

import cPickle as pickle
import os
import cv2
import sys
from matplotlib import pyplot


sys.path.append('/idiap/user/dwu/spyder/Codalab_Age_estimation/Codalab_Age_estimation/')
from Functions.utils import *
# load training data
import h5py
file = h5py.File("/idiap/user/dwu/spyder/Codalab_Age_estimation/Codalab_Age_estimation/data%d.hdf5", "r", driver="family", memb_size=2**32-1)      
x_temp= file["x_train"]
print x_temp.shape
X = x_temp[:]
X = X.reshape(x_temp.shape[0],1,96,96)
X = X /255.

# if we using trained network to predict face landmark
if False:
    from NN.NeuralNetworks_kfkd import *
    net_temp = NeuralNetworks_kfkd()
    load_path= '/idiap/user/dwu/spyder/KaggleFacialKeyPointDetection/net.pickle'
    net_temp.load_params(load_path)
    y_pred = net_temp.net.predict(X)    
    # we save the prediction landmard  
    pickle.dump( y_pred, open(y_pred_save_path+ "y_pred.pkl", "wb" ) )
else:# we load  
    y_pred_save_path = '/idiap/user/dwu/spyder/Codalab_Age_estimation/Codalab_Age_estimation/'
    y_pred = pickle.load( open(y_pred_save_path+  "y_pred.pkl", "rb" ) )


if False:
    
    from Functions.utils import plot_sample
    fig = pyplot.figure(figsize=(6, 6))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    
    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(X[i], y_pred[i], ax)

    pyplot.show()

### collect statistics for face Affine transformation
if False:
    from pandas.io.parsers import read_csv
    data_dir ="/idiap/user/dwu/spyder/KaggleFacialKeyPointDetection/data/"
    FTRAIN = os.path.join(data_dir,'training.csv')
    df = read_csv(os.path.expanduser(FTRAIN))
    df = df.dropna()
    y = df[df.columns[:-1]].values
    y = (y - 48) / 48 
    y_mean = y.mean(axis=0)
    pickle.dump( y_mean, open(y_pred_save_path+ "y_mean.pkl", "wb" ) )
else:
    y_mean = pickle.load( open(y_pred_save_path+  "y_mean.pkl", "rb" ) )
    
if False: # plot the mean face landmark
    fig = pyplot.figure(figsize=(6, 6))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
          
    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample_average(X[i], y_mean, ax)
        
    pyplot.show()


## windows load
##################################################################
import h5py
file = h5py.File("D:ChalearnAge\data%d.hdf5", "r", driver="family", memb_size=2**32-1)      
x_temp= file["x_train"]
print x_temp.shape

X = x_temp[:]
X = X.reshape(x_temp.shape[0],1,96,96)
X = X /255.

import cPickle as pickle
y_pred = pickle.load( open('D:\ChalearnAge\y_pred.pkl', "rb" ) )

def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)


fig = pyplot.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)       
for i in range(8):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_sample(X[i], y_pred[i], ax)
    
for i in range(8,16):
    warp_matrix = compute_affine_transformation(y_pred[i-8], y_mean)
    dst = cv2.warpAffine(X[i-8][0,:], warp_matrix, X[i-8][0,:].shape, cv2.cv.CV_INTER_LINEAR, cv2.cv.CV_WARP_FILL_OUTLIERS, 0)
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(dst, cmap='gray')

pyplot.show()
warp_matrix = compute_affine_transformation(y_pred[0], y_mean)
dst = cv2.warpAffine(X[0][0,:], warp_matrix, X[0][0,:].shape, cv2.cv.CV_INTER_LINEAR, cv2.cv.CV_WARP_FILL_OUTLIERS, 0)



