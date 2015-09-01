
import time
import os
import cPickle as pickle
import cv2
import h5py
import pandas
import numpy

pc="linux"
#pc ="virtualbox"
#pc="windows"
if pc=="windows":
    train_image_dir = r'D:\ChalearnAge/Train'
    valid_image_dir = r'D:\ChalearnAge/Validation'
    train_save_dir = r'D:\ChalearnAge\Train_crop'
    valid_save_dir = r'D:\ChalearnAge\Validation_crop'
    load_path = r'D:\ChalearnAge/'
elif pc=="linux":
    train_image_dir = '/idiap/user/dwu/spyder/Codalab_Age_estimation/Train'
    valid_image_dir = '/idiap/user/dwu/spyder/Codalab_Age_estimation/Validation'
    load_path = '/idiap/user/dwu/spyder/Codalab_Age_estimation/'
elif pc=="virtualbox":
    train_image_dir = '/home/user/Codalab_Age_Data/Train/'
    valid_image_dir = '/home/user/Codalab_Age_Data/Train/'
    load_path = '/home/user/Codalab_Age_Data/'

######################################################
# training, write to a file

with open(load_path+'train_y_pred.pkl','rb') as handle:
    y_dict = pickle.load(handle)
y_age = y_dict['y_pred'] * y_dict['y_mean'] + y_dict['y_mean']

# load bounding box using bob, the saved .pkl files are supposed to be in local dir
train_prediction_file = open(load_path+"train_y_pred.csv", "w")
count = 0
for file in os.listdir(train_image_dir):
     if file.endswith(".jpg"):
         train_prediction_file.write(file+';'+ numpy.array_str(y_age[count][0])+'\n')
         count += 1


train_prediction_file.close()

######################################################
# validation
with open(load_path+'valid_y_pred.pkl','rb') as handle:
    y_dict = pickle.load(handle)
y_age = y_dict['y_pred'] * y_dict['y_mean'] + y_dict['y_mean']
# load bounding box using bob, the saved .pkl files are supposed to be in local dir

valid_prediction_file = open(load_path+"valid_y_pred.csv", "w")
count = 0
for file in os.listdir(valid_image_dir):
     if file.endswith(".jpg"):
         valid_prediction_file.write(file+';'+ numpy.array_str(y_age[count][0])+'\n')
         count += 1

valid_prediction_file.close()

print "finish writing submission!"