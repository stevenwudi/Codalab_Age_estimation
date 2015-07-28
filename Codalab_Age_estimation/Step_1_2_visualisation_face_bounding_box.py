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

import h5py

bounding_boxes= {}

# load bounding box using bob, the saved .pkl files are supposed to be in local dir
train_bounding_box = pickle.load( open( "train_bounding_box.pkl", "rb" ) )

valid_bounding_box = pickle.load( open( "valid_bounding_box.pkl", "rb" ) )

import numpy as np
import cv2

start_flag = True

f = h5py.File("D:\ChalearnAge\data%d.hdf5", "w", driver="family", memb_size=2**32-1)
x_train = f.create_dataset("x_train", (2477,96,96), dtype='uint8', chunks=True)
count = 0

for file in os.listdir(train_image_dir):
    if file.endswith(".jpg") and (file=='image_1175.jpg' or start_flag):
        start_flag = True
        start = time.time()
        print file
        image_count = image_count + 1
        color_image = cv2.imread(os.path.join(train_image_dir, file))
        image_origin = color_image.copy()
        pt1_bob, pt2_bob = train_bounding_box[file]

        # bob and cv2 has different coordinate system
        pt1 = tuple(reversed(pt1_bob))
        pt2 = tuple(reversed(pt2_bob))

        # draw original face
        cv2.rectangle(color_image, pt1, pt2 ,(0,255,0), 5)
        height, width, channels = color_image.shape

        # bob has very constraint face detected images, we need to expand a bit
        # for face alignment, but this expansion doesn't need to be strict
        face_height = abs(pt1[0] - pt2[0])
        face_width = abs(pt1[1] - pt2[1])
        expand_up = face_height * 0.4
        expand_down = face_height * 0.2
        expand_sides = face_width * 0.2

        pt1 = tuple([int(max(0, pt1[0] - expand_sides )), int(max(0, pt1[1] - expand_up))])
        pt2 = tuple([int(min(width, pt2[0] +expand_sides )), int(min(height, pt2[1] + expand_down))])

        # detect single face


        # let's crop the image area only contains face
        # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
        crop_img = image_origin[pt1[1]:pt2[1], pt1[0]:pt2[0],:]

        # resize the image to 96*96 according to kaggle facial keypoint detection
        # http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
        resize_img = cv2.resize(crop_img, (96,96))
        # And for face landmark detction, we use only gray image
        gray_image = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
        x_train[count, :, :] = gray_image
        count += 1
        

        if True:
            
            cv2.rectangle(color_image, pt1, pt2 , (255,0,0), 5)
            cv2.imshow(file,color_image)
            cv2.imshow(file+'_cropped', gray_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # draw enlarged face    
            cv2.imwrite(os.path.join(train_save_dir, file), color_image)

end = time.time()
print end - start
f.close()       
print "done"
  
file = h5py.File("D:\ChalearnAge\data%d.hdf5", "r")      
x_temp = file["x_train"][:]
file.close()

