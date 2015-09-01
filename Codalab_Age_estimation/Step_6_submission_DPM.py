
import time
import os
import cPickle as pickle
import cv2
import h5py
import pandas
import numpy


train_image_dir = '/idiap/user/dwu/spyder/Codalab_Age_estimation/Train'
valid_image_dir = '/idiap/user/dwu/spyder/Codalab_Age_estimation/Validation'
load_path = '/idiap/user/dwu/spyder/Codalab_Age_estimation/'

######################################################
# training, write to a file

with open(load_path+'train_y_pred.pkl','rb') as handle:
    y_dict = pickle.load(handle)
y_age = y_dict['y_pred'] * y_dict['y_std'] + y_dict['y_mean']

# load bounding box using bob, the saved .pkl files are supposed to be in local dir
train_prediction_file = open(load_path+"train_y_pred.csv", "w")
count = 0
train_image_list = '/idiap/user/dwu/spyder/Codalab_Age_estimation/train_matlab_squence.csv'
with open(train_image_list, 'r') as train_list_file:
    for image_name in train_list_file:
        print image_name[:-1]
        train_prediction_file.write(image_name[:-1]+';'+ numpy.array_str(y_age[count][0])+'\n')
        count += 1

train_prediction_file.close()

######################################################
# validation
with open(load_path+'valid_y_pred.pkl','rb') as handle:
    y_dict = pickle.load(handle)
y_age = y_dict['y_pred'] * y_dict['y_std'] + y_dict['y_mean']
# load bounding box using bob, the saved .pkl files are supposed to be in local dir

valid_prediction_file = open(load_path+"valid_y_pred.csv", "w")
count = 0

valid_image_list = '/idiap/user/dwu/spyder/Codalab_Age_estimation/valid_matlab_squence.csv'
with open(valid_image_list, 'r') as valid_list_file:
    for image_name in valid_list_file:
        print image_name[:-1]
        valid_prediction_file.write(image_name[:-1]+';'+ numpy.array_str(y_age[count][0])+'\n')
        count += 1
            

valid_prediction_file.close()

print "finish DPM writing submission!"