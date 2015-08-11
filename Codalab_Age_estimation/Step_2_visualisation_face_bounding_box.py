'''
visualising the bob face detection code and save the result into a single file

'''

import time
import os
import cPickle as pickle
import cv2
import h5py
import pandas

pc="linux"
pc ="virtualbox"
pc="windows"
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

# iterate train folder
image_count = 0
bounding_boxes= {}

# load bounding box using bob, the saved .pkl files are supposed to be in local dir
train_bounding_box = pickle.load( open(load_path+"train_bounding_box.pkl", "rb" ) )
valid_bounding_box = pickle.load( open(load_path+ "valid_bounding_box.pkl", "rb" ) )

def load_age_deviation(file):
    names = ['imageID', 'age', 'variance']
    return pandas.read_table(file, sep=';', names=names)

grounth_truth = load_age_deviation(train_image_dir+'/Train.csv')

start_flag = True
f = h5py.File("data_with_label.hdf5", "w")
x_train_image_croped = f.create_dataset("x_train_image_croped", (2476,96,96), dtype='i')
y_train_age = f.create_dataset("y_train_age", (2476,1), dtype="f")
y_train_variance = f.create_dataset("y_train_variance", (2476,1), dtype="f")

x_valid_image_croped = f.create_dataset("x_valid_image_croped", (1136,96,96), dtype='i')

count = 0

for file in os.listdir(train_image_dir):
    if file.endswith(".jpg"):# and (file=='image_1175.jpg' or start_flag):
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
        x_train_image_croped[count, :, :] = gray_image
        y_train_age[count] =  float(grounth_truth['age'][grounth_truth['imageID']==file].values)
        y_train_variance[count] = float(grounth_truth['variance'][grounth_truth['imageID']==file].values)
        count += 1
        

        if False:
            
            cv2.rectangle(color_image, pt1, pt2 , (255,0,0), 5)
            cv2.imshow(file,color_image)
            cv2.imshow(file+'_cropped', gray_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # draw enlarged face    
            cv2.imwrite(os.path.join(train_save_dir, file), color_image)

count = 0
for file in os.listdir(valid_image_dir):
    if file.endswith(".jpg"):# and (file=='image_1175.jpg' or start_flag):
        start_flag = True
        start = time.time()
        print file
        image_count = image_count + 1
        color_image = cv2.imread(os.path.join(valid_image_dir, file))
        image_origin = color_image.copy()
        pt1_bob, pt2_bob = valid_bounding_box[file]
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
        x_valid_image_croped[count, :, :] = gray_image
        count += 1
        
        if False:            
            cv2.rectangle(color_image, pt1, pt2 , (255,0,0), 5)
            cv2.imshow(file,color_image)
            cv2.imshow(file+'_cropped', gray_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # draw enlarged face    
            cv2.imwrite(os.path.join(train_save_dir, file), color_image)


end = time.time()
f.close()       
print "done"


