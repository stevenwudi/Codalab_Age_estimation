__author__ = 'dwu'

__author__ = 'dwu'

'''
visualising the bob face detection code and save the result into a single file

'''

import time
import os
import cv2
import h5py
import pandas


train_image_dir = '/idiap/user/dwu/spyder/Codalab_Age_estimation/Train'
valid_image_dir = '/idiap/user/dwu/spyder/Codalab_Age_estimation/Validation'
train_save_dir = '/idiap/user/dwu/spyder/Codalab_Age_estimation/Train_crop'
valid_save_dir = '/idiap/user/dwu/spyder/Codalab_Age_estimation/Valid_crop'
load_path = '/idiap/user/dwu/spyder/Codalab_Age_estimation/'


import scipy.io

train_bounding_box_mat = scipy.io.loadmat('/idiap/user/dwu/spyder/Codalab_Age_estimation/boundingbox_train.mat')
train_image_list = '/idiap/user/dwu/spyder/Codalab_Age_estimation/train_matlab_squence.csv'
train_bounding_box = train_bounding_box_mat['boundingbox']
valid_bounding_box_mat = scipy.io.loadmat('/idiap/user/dwu/spyder/Codalab_Age_estimation/boundingbox_valid.mat')
valid_bounding_box = valid_bounding_box_mat['boundingbox']
valid_image_list = '/idiap/user/dwu/spyder/Codalab_Age_estimation/valid_matlab_squence.csv'

def load_age_deviation(file):
    names = ['imageID', 'age', 'variance']
    return pandas.read_table(file, sep=';', names=names)

grounth_truth = load_age_deviation(train_image_dir+'/Train.csv')

f = h5py.File(load_path+"data_with_label_DPM_Color_NoCROP.hdf5", "w")
x_train_image_croped = f.create_dataset("x_train_image_croped", (2476,96,96, 3), dtype='i', chunks=True)
y_train_age = f.create_dataset("y_train_age", (2476,1), dtype="f",chunks=True)
y_train_variance = f.create_dataset("y_train_variance", (2476,1), dtype="f", chunks=True)
x_valid_image_croped = f.create_dataset("x_valid_image_croped", (1136,96,96, 3), dtype='i',chunks=True)
# iterate train folder
count = 0
count_bb = -1
if True:
    with open(train_image_list, 'r') as train_list_file:
        for image_name in train_list_file:
            print image_name
            count_bb += 1
            start = time.time()
            color_image = cv2.imread(os.path.join(train_image_dir, image_name[:-1]))
            image_origin = color_image.copy()
            pt1_dpm_x, pt1_dpm_y, pt2_dpm_x, pt2_dpm_y = train_bounding_box[count_bb]
            height, width, channels = color_image.shape

            # for training images, if we didn't detect any faces, the matlab DPM calling actually set detected coordinates
            # to 0, we exclude them from the training set
            if sum([pt1_dpm_x, pt1_dpm_y, pt2_dpm_x, pt2_dpm_y])==0:
                print " undetected faces, continue without face:"+image_name[:-1]
            else:
                pt1 = tuple(((int(pt1_dpm_x), int(pt1_dpm_y))))
                pt2 = tuple(((int(pt2_dpm_x), int(pt2_dpm_y))))
                # draw original face
                cv2.rectangle(color_image, pt1, pt2, (0,255,0), 5)
                # we need to expand a bit for face alignment, but this expansion doesn't need to be strict
                face_height = abs(pt1[0] - pt2[0])
                face_width = abs(pt1[1] - pt2[1])
                expand_up = face_height * 0
                expand_down = face_height * 0
                expand_sides = face_width * 0

                pt1 = tuple([int(max(0, pt1[0] - expand_sides)), int(max(0, pt1[1] - expand_up))])
                pt2 = tuple([int(min(width, pt2[0] + expand_sides)), int(min(height, pt2[1] + expand_down))])

                # let's crop the image area only contains face
                # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
                crop_img = image_origin[pt1[1]:pt2[1], pt1[0]:pt2[0], :]

                # resize the image to 96*96 according to kaggle facial keypoint detection
                # http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
                resize_img = cv2.resize(crop_img, (96,96))
                # And for face landmark detction, we use only gray image
                x_train_image_croped[count, :] = resize_img
                y_train_age[count] = float(grounth_truth['age'][grounth_truth['imageID']==image_name[:-1]].values)
                y_train_variance[count] = float(grounth_truth['variance'][grounth_truth['imageID']==image_name[:-1]].values)
                count += 1

            if False:
                #cv2.rectangle(color_image, pt1, pt2, (255,0,0), 5)
                #cv2.imshow(image_name[:-1],color_image)
                #cv2.imshow(image_name[:-1]+'_cropped', gray_image)
                #cv2.waitKey(1000)
                #cv2.destroyAllWindows()
                # draw enlarged face
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(resize_img, str(y_train_age[count-1]), (0,50), font, 1, (255,0,255))
                cv2.imwrite(os.path.join(train_save_dir, image_name[:-1]), resize_img)
                cv2.imshow(image_name[:-1]+'_cropped', resize_img)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()

x_train_image_croped.resize((count, 96, 96, 3))
y_train_age.resize((count,1))
y_train_variance.resize((count,1))

count = 0
if True:
    with open(valid_image_list, 'r') as valid_list_file:
        for image_name in valid_list_file:
            print image_name
            start = time.time()
            color_image = cv2.imread(os.path.join(valid_image_dir, image_name[:-1]))
            image_origin = color_image.copy()
            pt1_dpm_x, pt1_dpm_y, pt2_dpm_x, pt2_dpm_y = valid_bounding_box[count]
            height, width, channels = color_image.shape

            # for validation images, if we didn't detect any faces, the matlab DPM calling actually set detected coordinates
            # to 0, we set the whole image as input
            if sum([pt1_dpm_x, pt1_dpm_y, pt2_dpm_x, pt2_dpm_y])==0:
                print " undetected faces, continue without face:"+image_name[:-1]
                (pt1_dpm_x, pt1_dpm_y, pt2_dpm_x, pt2_dpm_y) = (0, 0, width, height)

            pt1 = tuple(((int(pt1_dpm_x), int(pt1_dpm_y))))
            pt2 = tuple(((int(pt2_dpm_x), int(pt2_dpm_y))))
            # draw original face
            cv2.rectangle(color_image, pt1, pt2, (0,255,0), 5)
            # we need to expand a bit for face alignment, but this expansion doesn't need to be strict
            face_height = abs(pt1[0] - pt2[0])
            face_width = abs(pt1[1] - pt2[1])
            expand_up = face_height * 0
            expand_down = face_height * 0
            expand_sides = face_width * 0

            pt1 = tuple([int(max(0, pt1[0] - expand_sides)), int(max(0, pt1[1] - expand_up))])
            pt2 = tuple([int(min(width, pt2[0] + expand_sides)), int(min(height, pt2[1] + expand_down))])

            # let's crop the image area only contains face
            # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
            crop_img = image_origin[pt1[1]:pt2[1], pt1[0]:pt2[0], :]

            # resize the image to 96*96 according to kaggle facial keypoint detection
            # http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
            resize_img = cv2.resize(crop_img, (96,96))
            x_valid_image_croped[count, :] = resize_img
            count += 1

            if False:
                #cv2.rectangle(color_image, pt1, pt2 , (255,0,0), 5)
                #cv2.imshow(image_name[:-1],color_image)
                #cv2.imshow(image_name[:-1]+'_cropped', gray_image)
                #cv2.waitKey(1000)
                #cv2.destroyAllWindows()
                # draw enlarged face
                cv2.imwrite(os.path.join(valid_save_dir, image_name[:-1]), resize_img)
                cv2.imshow(image_name[:-1]+'_cropped', resize_img)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()

x_valid_image_croped.resize((count, 96, 96, 3))