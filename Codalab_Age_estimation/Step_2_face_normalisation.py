"""
Use trained face detection NN to detect face
"""
# we need to import all the classes
from NN.NeuralNetworks_kfkd import *


# pre-defined NN class 
net_temp = NeuralNetworks_kfkd()
load_path= '/idiap/user/dwu/spyder/KaggleFacialKeyPointDetection/net.pickle'
net_temp.load_params(load_path)

import h5py
file = h5py.File("/idiap/user/dwu/spyder/Codalab_Age_estimation/Codalab_Age_estimation/data%d.hdf5", "r", driver="family", memb_size=2**32-1)      
x_temp= file["x_train"]
print x_temp.shape


X = x_temp[:]
X = X.reshape(x_temp.shape[0],1,96,96)

y_pred = net_temp.net.predict(X)

# we save the prediction landmard
import cPickle as pickle
pickle.dump( y_pred, open(y_pred_save_path+ "y_pred.pkl", "wb" ) )


y_pred_save_path = '/idiap/user/dwu/spyder/Codalab_Age_estimation/Codalab_Age_estimation/'
y_pred = pickle.load( open(y_pred_save_path+  "y_pred.pkl", "rb" ) )

def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

from matplotlib import pyplot

fig = pyplot.figure(figsize=(6, 6))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_sample(X[i], y_pred[i], ax)

pyplot.show()