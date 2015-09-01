"""
Use trained face detection NN to detect face,
"""
# we need to import all the classes
# pre-defined NN class 

import cPickle as pickle
import os
import cv2
import sys
import numpy
from matplotlib import pyplot
import h5py
import numpy


pc = "linux"
pc = "windows"
pc = "virtualbox"
if pc=="linux": 
    sys.path.append('/idiap/user/dwu/spyder/Codalab_Age_estimation/Codalab_Age_estimation/')
    load_path= '/idiap/user/dwu/spyder/KaggleFacialKeyPointDetection/net.pickle'
    y_pred_save_path = '/idiap/user/dwu/spyder/Codalab_Age_estimation/Codalab_Age_estimation/'
elif pc=="virtualbox": 
    sys.path.append('/home/user/Codalab_Age_estimation/Codalab_Age_estimation')
    load_path = '/home/user/Codalab_Age_Data/net.pickle'
elif pc=="windows":
    file = h5py.File('data%d.hdf5', "r", driver="family", memb_size=2**32-1)


from Functions.utils import *
# load training data
import h5py
if pc=="linux": data_load_path = "/idiap/user/dwu/spyder/Codalab_Age_estimation/Codalab_Age_estimation/"
if pc=="virtualbox": data_load_path = '/home/user/Codalab_Age_Data/'
#file = h5py.File(os.path,join(data_load_path,"data%d.hdf5", "r", driver="family", memb_size=2**32-1))
#file = h5py.File("/home/user/Codalab_Age_Data/Train_crop/data0.hdf5", "r", driver="family", memb_size=2**32-1)
#file = h5py.File('/home/user/Codalab_Age_Data/data%d.hdf5', "r", driver="family", memb_size=2**32-1)
#file = h5py.File('/home/user/Codalab_Age_estimation/data0.hdf5', "r", driver="family", memb_size=2**32-1)

if pc=="virtualbox":
    X = pickle.load( open("/home/user/Codalab_Age_estimation/Codalab_Age_estimation/Codalab_Age_X_norm.pkl", "rb" ) )
if pc=="windows":
    X = pickle.load(open('/idiap/user/dwu/spyder/KaggleFacialKeyPointDetection/Codalab_Age_X_norm.pkl',"rb"))

X = numpy.array(X, dtype=numpy.float32)


x_temp= file["x_train"]
print x_temp.shape
X = x_temp[:]
X = X.reshape(x_temp.shape[0],1,96,96)
X = X /255.

# if we using trained network to predict face landmark
if pc=="linux":
    from NN.NeuralNetworks_kfkd import *
    net_temp = NeuralNetworks_kfkd()    
    net_temp.net.load_params_from(r'D:\ChalearnAge\net_params.pickle')
    net_temp.load_params(load_path)
    y_pred = net_temp.net.predict(X)    
    # we save the prediction landmard  
    pickle.dump( y_pred, open(y_pred_save_path+ "y_pred.pkl", "wb" ) )
else:# we load  
    y_pred_save_path = r'D:\ChalearnAge/'
    y_pred = pickle.load( open(y_pred_save_path+  "y_pred.pkl", "rb" ) )


if pc=="windows":  
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

file = h5py.File("D:ChalearnAge\data%d.hdf5", "r", driver="family", memb_size=2**32-1)      
x_temp= file["x_train"]
print x_temp.shape

# we only plot left eye center, right eye center and mouth center bottom lip
plot_idx_x = [0,2,28]
plot_idx_y = [1,3,29]
import cPickle as pickle
y_pred = pickle.load( open('D:\ChalearnAge\y_pred.pkl', "rb" ) )


pst1 = numpy.zeros(shape=(3,2), dtype=numpy.float32)
pst2 = numpy.zeros(shape=(3,2),dtype=numpy.float32)
for i in range(3):
    pst1[i][0] = y_mean[plot_idx_x[i]]
    pst1[i][1] = y_mean[plot_idx_y[i]]


def plot_sample(x, y, axis, plot_idx_x, plot_idx_y):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    #axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)
    # we only plot left eye center, right eye center and mouth center bottom lip
    axis.scatter(y[plot_idx_x]*48 + 48, y[plot_idx_y]*48+48, marker='x', s=10,color='r')


X = x_temp[:]
X = X.reshape(x_temp.shape[0],1,96,96)
X = X /255.


##plotting constant
SCALE = 45
START_FRAME = 10

fig = pyplot.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)       
for i in range(8):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_sample(X[i+START_FRAME], y_pred[i+START_FRAME], ax,plot_idx_x, plot_idx_y)



for i in range(8,16):
    for p in range(3):
        pst2[p][0] = y_pred[i+START_FRAME-8][plot_idx_x[p]]
        pst2[p][1] = y_pred[i+START_FRAME-8][plot_idx_y[p]]
    M = cv2.getAffineTransform(pst2*48+48 ,pst1*SCALE+SCALE)
    dst = cv2.warpAffine(X[i+START_FRAME-8][0,:], M, X[i-8][0,:].shape)
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(dst, cmap='gray')
    ax.scatter(y_mean[0::2] * SCALE + SCALE, y_mean[1::2] * SCALE + SCALE, marker='x', s=10, color='r')
 
pyplot.show()
      
#for i in range(8,16):
#    warp_matrix, scale, angle = compute_affine_transformation(y_pred[i-8],y_mean)
#    dst = cv2.warpAffine(X[i-8][0,:], warp_matrix, X[i-8][0,:].shape, cv2.cv.CV_INTER_LINEAR, cv2.cv.CV_WARP_FILL_OUTLIERS, 0)
#    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
#    ax.imshow(dst, cmap='gray')
#    ax.scatter(y_mean[0::2] * 48 + 48, y_mean[1::2] * 48 + 48, marker='x', s=10, color='r')




warp_matrix = compute_affine_transformation(y_pred[0], y_mean)
dst = cv2.warpAffine(X[0][0,:], warp_matrix, X[0][0,:].shape, cv2.cv.CV_INTER_LINEAR, cv2.cv.CV_WARP_FILL_OUTLIERS, 0)



