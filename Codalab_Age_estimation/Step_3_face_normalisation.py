"""
Use trained face detection NN to detect face,
"""
# we need to import all the classes
# pre-defined NN class 

import cPickle as pickle
import cv2
import sys
from matplotlib import pyplot
import h5py
import numpy
from Functions.utils import *

pc = "linux"
sys.path.append('/idiap/user/dwu/spyder/Codalab_Age_estimation/Codalab_Age_estimation/')
y_pred_save_path = '/idiap/user/dwu/spyder/Codalab_Age_estimation/Codalab_Age_estimation/'
load_path = '/idiap/user/dwu/spyder/Codalab_Age_estimation/'

f = h5py.File(load_path+"data_with_label_DPM_includeTraining.hdf5", "r")
x_train_image_croped = f["x_train_image_croped"][:]
X_mean = x_train_image_croped.mean(axis=0)
X_std = x_train_image_croped.std(axis=0)

x_valid_image_croped = f["x_valid_image_croped"][:]
y_train_age = f["y_train_age"][:]
y_train_variance = f["y_train_variance"][:]
y_mean = y_train_age.mean()
y_std = y_train_age.std()


X = x_train_image_croped[:]
X = x_train_image_croped[:, 14:-15, 14:-15]
X = X /255.
X = X.astype(numpy.float32)
X = numpy.expand_dims(X,1)

X_valid = x_valid_image_croped[:, 14:-15, 14:-15]
X_valid = X_valid/255.
X_valid = X_valid.astype(numpy.float32)
X_valid = numpy.expand_dims(X_valid,1)
# if we using trained network to predict face landmark
if False:
    from NN.NeuralNetworks_kfkd import *
    net_temp = NeuralNetworks_kfkd()
    net_temp.net.load_params_from('/idiap/user/dwu/spyder/KaggleFacialKeyPointDetection/net_params.pickle')
    y_pred = net_temp.net.predict(X)
    # we save the prediction landmard  
    pickle.dump( y_pred, open(y_pred_save_path+ "y_pred.pkl", "wb"))
    y_pred_valid = net_temp.net.predict(X_valid)
    # we save the prediction landmard
    pickle.dump( y_pred, open(y_pred_save_path+ "y_valid_pred.pkl", "wb"))
else:# we load
    y_pred = pickle.load( open(y_pred_save_path+  "y_pred.pkl", "rb"))
    y_pred_valid = pickle.load( open(y_pred_save_path+  "y_valid_pred.pkl", "rb"))


plot_shift = 20
if False:
    from Functions.utils import plot_sample
    fig = pyplot.figure(figsize=(6, 6))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    
    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(X[i+plot_shift], y_pred[i+plot_shift], ax)

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
        plot_sample_average(X[i+plot_shift], y_mean, ax)
        
    pyplot.show()


## windows load
##################################################################

X = x_train_image_croped[:]
X  = X [:, 14:-15, 14:-15]
X = numpy.expand_dims(X,1)

# we only plot left eye center, right eye center and mouth center bottom lip
plot_idx_x = [0,2,28]
plot_idx_y = [1,3,29]
pst1 = numpy.zeros(shape=(3,2), dtype=numpy.float32)
pst2 = numpy.zeros(shape=(3,2), dtype=numpy.float32)

for i in range(3):
    pst1[i][0] = y_mean[plot_idx_x[i]]
    pst1[i][1] = y_mean[plot_idx_y[i]]

def plot_sample(x, y, axis, plot_idx_x, plot_idx_y):
    WIDTH =x.shape[-1]
    img = x.reshape(WIDTH, WIDTH)
    axis.imshow(img, cmap='gray')
    #axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)
    # we only plot left eye center, right eye center and mouth center bottom lip
    axis.scatter(y[plot_idx_x]*WIDTH/2 + WIDTH/2, y[plot_idx_y]*WIDTH/2+WIDTH/2, marker='x', s=10,color='r')


START_FRAME = 10

fig = pyplot.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)       
for i in range(8):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_sample(X[i+START_FRAME], y_pred[i+START_FRAME], ax,plot_idx_x, plot_idx_y)

#pyplot.show()
##plotting constant
SCALE = 48
WIDTH = X.shape[-1]
for i in range(8,16):
    for p in range(3):
        pst2[p][0] = y_pred[i+START_FRAME-8][plot_idx_x[p]]
        pst2[p][1] = y_pred[i+START_FRAME-8][plot_idx_y[p]]
    M = cv2.getAffineTransform(pst2*SCALE+SCALE, pst1*SCALE+SCALE)
    img = X[i+START_FRAME-8][0,:]
    print type(img)
    print img.shape
    #dst = cv2.warpAffine(img, M, img.shape[:-1])
    dst = img
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(dst, cmap='gray')
    ax.scatter(y_mean[0::2] * WIDTH/2 + WIDTH/2, y_mean[1::2] * WIDTH/2 + WIDTH/2, marker='x', s=10, color='r')
 
pyplot.show()
      
#for i in range(8,16):
#    warp_matrix, scale, angle = compute_affine_transformation(y_pred[i-8],y_mean)
#    dst = cv2.warpAffine(X[i-8][0,:], warp_matrix, X[i-8][0,:].shape, cv2.cv.CV_INTER_LINEAR, cv2.cv.CV_WARP_FILL_OUTLIERS, 0)
#    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
#    ax.imshow(dst, cmap='gray')
#    ax.scatter(y_mean[0::2] * 48 + 48, y_mean[1::2] * 48 + 48, marker='x', s=10, color='r')




warp_matrix = compute_affine_transformation(y_pred[0], y_mean)
dst = cv2.warpAffine(X[0][0,:], warp_matrix, X[0][0,:].shape, cv2.cv.CV_INTER_LINEAR, cv2.cv.CV_WARP_FILL_OUTLIERS, 0)



