
image_dir = '/idiap/user/dwu/spyder/Codalab_Age_estimation/Train'

import cv2
import os
import bob.ip.facedetect

face_image = bob.io.base.load(os.path.join(image_dir,'image_2584.jpg'))

img = cv2.imread(os.path.join(image_dir,'image_5.jpg'))
cv2.imshow('img',img)

bounding_box, quality = bob.ip.facedetect.detect_single_face(face_image)


bounding_box, quality = bob.ip.facedetect.detect_single_face(img.swapaxes(2,0))



(61, 47), (186, 151)
(47, 61), (151,186)

cv2.rectangle(img, (47, 61), (151,186), (255,0,0),2)
cv2.imshow('img',img)

cv2.rectangle(img, (90, 152), (133, 188), (255,0,0),2)



cv2.imshow('img',img.swapaxes(0,1))
