'''
visualising the bob face detection code

'''

train_image_dir = r'D:\ChalearnAge/Train'
valid_image_dir = r'D:\ChalearnAge/Validation'
train_save_dir = r'D:\ChalearnAge\Train_crop'
valid_save_dir = r'D:\ChalearnAge\Validation_crop'
# iterate train folder
image_count = 0


import time
import os
import cPickle as pickle

bounding_boxes= {}

# load bounding box using bob, the saved .pkl files are supposed to be in local dir
train_bounding_box = pickle.load( open( "train_bounding_box.pkl", "rb" ) )

valid_bounding_box = pickle.load( open( "valid_bounding_box.pkl", "rb" ) )

import numpy as np
import cv2

for file in os.listdir(train_image_dir):
    if file.endswith(".jpg"):
        start = time.time()
        print file
        image_count = image_count + 1
        color_image = cv2.imread(os.path.join(train_image_dir, file))
        pt1_bob, pt2_bob = train_bounding_box[file]

        # bob and cv2 has different coordinate system
        pt1 = tuple(reversed(pt1_bob))
        pt2 = tuple(reversed(pt2_bob))
        height, width, channels = color_image.shape
        cv2.rectangle(color_image, pt1, pt2 ,color=(0,255,0))
        cv2.imshow(file,color_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(os.path.join(train_save_dir, file), color_image)
        # detect single face
        end = time.time()
        print end - start
        

        