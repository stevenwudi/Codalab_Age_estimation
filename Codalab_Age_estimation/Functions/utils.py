# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 22:52:25 2015

@author: dwu
"""
import math
class Point2D32f:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
def compute_center(pts, N=15):     # ok
    center = Point2D32f(0.0,0.0)
    for i in range(N):
        center.x += pts[i*2]
        center.y += pts[i*2+1]
    center.x /= N
    center.y /= N
    return center            
            
def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)
    
def plot_sample_average(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10, color='r')
    
def compute_affine_transformation(src, dst, N=15):
    import numpy as np
    import cv2, math

    # this function compute a affine transformation matrix      
    src_center = compute_center(src)
    dst_center = compute_center(dst)
    X = np.empty((2, N))
    Y = np.empty((2, N))   
        
    for i in range(N):
        Y[0, i] = src[i*2] - src_center.x
        Y[1, i] = src[i*2+1] - src_center.y
        X[0, i] = dst[i*2] - dst_center.x
        X[1, i] = dst[i*2+1] - dst_center.y


    #Get transpose matrix
    YXt = np.dot( Y, X.T)

    #The flags cause U to be returned transposed (does not work well without the transpose flags).
    W, Ut, V = cv2.SVDecomp(YXt, cv2.cv.CV_SVD_U_T)

    #Compute s = sum(sum( X.*(R*Y) )) / sum(sum( Y.^2 ));
    R = np.dot(V, Ut)
    RY = np.dot(R, Y)
    XRY = X * RY
    YY = Y * Y

    #Compute scale, sum, angle
    scale = 0.0
    sum = 0.0
    for i in range(N):
        scale += XRY[0, i] + XRY[1, i] 
        sum += YY[0, i] + YY[1, i]

    if sum != 0:
        scale = scale/sum
    else:
        scale = 1   
    angle = math.atan(R[0, 1]/R[0, 0]) * 180.0 / cv2.cv.CV_PI
    #angle = math.atan(R[0, 0]/R[0, 1]) * 180.0 / cv2.cv.CV_PI
    #Compute matrix
    warp_matrix = cv2.getRotationMatrix2D((src_center.x, src_center.y), -angle, scale)
    return warp_matrix, scale, angle