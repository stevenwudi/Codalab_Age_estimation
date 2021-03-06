'''
use bob for face detection
http://pythonhosted.org/bob.ip.facedetect/guide.html
'''

import bob.io.base
import bob.io.base.test_utils
import bob.io.image
import bob.ip.facedetect
import bob.ip.draw

pc = "linux"
pc = "virtualbox"

if pc=="linux":  
    train_image_dir = '/idiap/user/dwu/spyder/Codalab_Age_estimation/Train'
    valid_image_dir = '/idiap/user/dwu/spyder/Codalab_Age_estimation/Validation'
    save_dir = '/idiap/user/dwu/spyder/Codalab_Age_estimation'
elif pc=="virtualbox":
    train_image_dir = '/home/user/Codalab_Age_Data/Train'
    valid_image_dir = '/home/user/Codalab_Age_Data/Validation'
    save_dir = '/home/user/Codalab_Age_Data/Train'
    
# iterate train folder
image_count = 0


import time
import os
import cPickle as pickle

bounding_boxes= {}

for file in os.listdir(train_image_dir):
    if file.endswith(".jpg"):
        start = time.time()
        print file
        image_count = image_count + 1
        color_image = bob.io.base.load(os.path.join(train_image_dir, file))
        # create figure
        from matplotlib import pyplot
        import numpy

        # detect single face
        bounding_box, score = bob.ip.facedetect.detect_single_face(color_image)
        bounding_boxes[file] =[bounding_box.topleft, bounding_box.bottomright]
        end = time.time()
        if True:
            bob.ip.draw.box(color_image, bounding_box.topleft, bounding_box.size, color=(0,255,0))
            pyplot.imshow(numpy.rollaxis(numpy.rollaxis(color_image, 2),2))
            pyplot.show()
        print end - start
        print "score is:",score
        
with open(os.path.join(save_dir,'train_bounding_box.pkl'), 'wb') as f:
    pickle.dump(bounding_boxes, f, pickle.HIGHEST_PROTOCOL)
        
bounding_boxes= {}      
for file in os.listdir(valid_image_dir):
    if file.endswith(".jpg"):
        start = time.time()
        print file
        image_count = image_count + 1
        color_image = bob.io.base.load(os.path.join(valid_image_dir, file))
        
        # detect single face
        bounding_box, _ = bob.ip.facedetect.detect_single_face(color_image)
        bounding_boxes[file] =[bounding_box.topleft, bounding_box.bottomright]
        end = time.time()
        print end - start
        
with open(os.path.join(save_dir,'valid_bounding_box.pkl'), 'wb') as f:
    pickle.dump(bounding_boxes, f, pickle.HIGHEST_PROTOCOL)
        
    