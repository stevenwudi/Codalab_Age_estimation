# -------------------------------------------------------------------------------
# Name:        Chalearn LAP evaluation scripts
# Purpose:     Provide an evaluation script for Age Estimation
#
# Authors:     Pablo Pardo
# 	           Xavier Baro
#              HupBA
#
# Created:     19/01/2015
# Copyright:   (c) Chalearn LAP 2015
# Licence:     GPL
#
#  Modified by Di Wu: 
#
# To run this script: python program/evaluate.py input output
#
#-------------------------------------------------------------------------------

import os,sys,glob
import math
import numpy as np

def normal(x, mu, sig):
    return 1 - math.exp(-(x - mu)**2 / (2*sig**2))

def load_predictions(path):
    f = open(path, 'r')
    lines = f.readlines()
    f.close()
    img_name=[l.split(';')[0] for l in lines]
    img_pred=[float(l.split(';')[1]) for l in lines]
    return dict(zip(img_name,img_pred))

def load_gt(path):
    f = open(path, 'r')
    lines = f.readlines()
    f.close()
    img_name=[l.split(';')[0] for l in lines]
    img_mu=[float(l.split(';')[1]) for l in lines]
    img_sig=[float(l.split(';')[2]) for l in lines]
    return dict(zip(img_name,zip(img_mu,img_sig)))

#This folder contains the files with the ground truth
dirRef = sys.argv[1] + "/ref/"

#This folder contains the files with the confidences of your classifier
dirRes = sys.argv[1] + "/res/"

#This script creates a file (scores.txt) containing the final mean average precision (mAP)
dirOut = sys.argv[2] + "/"

# Load data
gt_data=load_gt(dirRef + "Reference.csv")
pred_data=load_predictions(dirRes + "Predictions.csv")

# For each image in the GT, find its prediction
score_arr = []
for key in gt_data:
    # Assign the not predicted error
    img_score=1.0

    # If prediction exists, compute the real prediction error
    if key in pred_data:
        img_score=normal(pred_data[key],gt_data[key][0],gt_data[key][1])
    score_arr.append(img_score)

score = np.mean(score_arr)

fileO = open(dirOut + "scores.txt", "wb")
fileO.write("Error: " + str(score))
fileO.close()