__author__ = 'dwu'
# -*- coding: utf-8 -*-

import sys
import cPickle as pickle

sys.path.append('/idiap/user/dwu/spyder/Codalab_Age_estimation/Codalab_Age_estimation/')
net_params ='/idiap/user/dwu/spyder/KaggleFacialKeyPointDetection/net_params.pickle'
load_path = '/idiap/user/dwu/spyder/Codalab_Age_estimation/'

fname = load_path+"data_with_label_DPM_includeTraining.hdf5"


from NN.Convnet_age_ony_regression import *
net_temp = ConvNet_Naive_Age_Regression()

#This is Di Wu's adaptation that because we have different last layer
#Though there is no difference between the "load_params_from" method
#because the constraint p1.get_value().shape == p2v.shape:
#net_temp.net.load_params_from(load_path  + 'net_params_DPM.pickle')

net_temp.net.load_params_from(load_path  + 'net_params_DPM_NoShuffle.pickle')



X, y, y_mean, y_std = load(fname)
y_pred = net_temp.net.predict(X)
y_dict = {}
y_dict['y_pred'] = y_pred
y_dict['y_mean'] = y_mean
y_dict['y_std'] = y_std
with open(load_path+'train_y_pred.pkl','wb') as handle:
    pickle.dump(y_dict, handle)


X, _, y_mean, y_std = load(fname, test=True)
y_pred = net_temp.net.predict(X)
y_dict = {}
y_dict['y_pred'] = y_pred
y_dict['y_mean'] = y_mean
y_dict['y_std'] = y_std
with open(load_path+'valid_y_pred.pkl','wb') as handle:
    pickle.dump(y_dict, handle)

