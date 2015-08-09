"""
This file is used to read REPERE test 2 dataset and normalise the data using face detection
"""
import cv2
import os, csv
import pandas as pd

video_path = "D:\ChalearnAge"
video_name = 'BFMTV_CultureEtVous_2012-12-17_064700.MPG'

face_detecton_file = 'faces.norm'

def loadFaceDetection(file):
    names = ['frameNum', 'top', 'left', 'bottom', 'right', 'conf', 'd', 'cluster']
    return pd.read_table(file, sep=' ', names=names)

# read the detected faces from Nam's face detection and tracking result
detected_faces = loadFaceDetection(os.path.join(video_path, face_detecton_file))

# frames that have faces detected:
FaceFrame = detected_faces['frameNum'].unique()

# color set
ColorSet = [(255,0,0), (0,255,0), (0,0, 255), (128,128,0), (0,128,128), (128,0,128)]

frame_number = 0



# read video fils
cap = cv2.VideoCapture(os.path.join(video_path, video_name))
while(cap.isOpened()):
    ret, frame = cap.read()
    frame_number += 1
    # if this frame has detectd faces, we show the image
    if frame_number in FaceFrame:
        # draw rectangles:
        for f in range(len(detected_faces[detected_faces['frameNum']==frame_number])):
            top =    detected_faces[detected_faces['frameNum']==frame_number].iloc[f].top
            left =   detected_faces[detected_faces['frameNum']==frame_number].iloc[f].left
            bottom = detected_faces[detected_faces['frameNum']==frame_number].iloc[f].bottom
            right =  detected_faces[detected_faces['frameNum']==frame_number].iloc[f].right

            height = abs(bottom-top)
            width = abs(left-right)
            clusterNo = int(detected_faces[detected_faces['frameNum']==frame_number].iloc[f].cluster[-3:])
            color_chosen = ColorSet[clusterNo % len(ColorSet)]
            #cv2.rectangle(frame, (left, top), (width, height), color_chosen, 5)
            #cv2.circle(frame, (right, bottom), 60, color_chosen, 5)
            #cv2.rectangle(frame,(bottom, right) , (left, top), color_chosen, 5)
            cv2.rectangle(frame, (top, left), (bottom, right), color_chosen, 5)
        
        cv2.imshow(str(frame_number), frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




import h5py
file = h5py.File("D:\ChalearnAge\data%d.hdf5", "r", driver="family", memb_size=2**32-1)      
x_temp= file["x_train"]
print x_temp.shape

X = x_temp[:]
X = X.reshape(x_temp.shape[0],1,96,96)
X = X /255.

import cPickle as pickle
y_pred = pickle.load( open('D:\ChalearnAge\y_pred.pkl', "rb" ) )

from matplotlib import pyplot
def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

fig = pyplot.figure(figsize=(6, 6))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_sample(X[i], y_pred[i], ax)

pyplot.show()