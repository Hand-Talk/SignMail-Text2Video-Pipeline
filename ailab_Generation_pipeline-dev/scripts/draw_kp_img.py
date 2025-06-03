import cv2
import os
#import time
import shutil
import pickle
import json
#import gc
import numpy as np
import matplotlib
#import matplotlib.pyplot as plt
#import tensorflow as tf
import random
#import pathlib
#import json
import mediapipe as mp
import math

from PIL import Image
#from tensorflow import keras
#from tensorflow.keras import layers
#from sklearn.utils import shuffle
from copy import deepcopy
from scipy.interpolate import CubicSpline
from mediapipe.framework.formats import landmark_pb2
#from keras import backend as K
#from IPython import display
#from datetime import datetime
#from functools import partial
from glob import glob
from collections import defaultdict
from pathlib import Path



#############################
####draw image functions#####

def zero_rate(arr):
    """
    Compute the rate of zeros in a given 1D numpy array.

    Parameters:
    - arr (np.array): 1D numpy array

    Returns:
    - float: rate of zeros in the array
    """
    num_zeros = np.sum(arr == 0)
    rate = num_zeros / len(arr)
    return rate


#function to change array to mediapipe NormalizedLandmark formula
def array_to_landmarks(array):
    
    List = list()
    for i in range( int( array.shape[0]/3 ) ):
        List.append({'x':array[3*i], 'y':array[3*i+1], 'z':array[3*i+2], 'visibility':1.})

    landmarks_loaded = landmark_pb2.NormalizedLandmarkList(landmark = List)
    
    return(landmarks_loaded)

########################
def draw_styled_landmarks_for_face_only(image, face_lm, mp_drawing):
    
    # Draw face connections
    mp_drawing.draw_landmarks(image, face_lm, None, 
                             mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=0), 
                             mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=0)
                             )


def draw_lines(image, keypoints, dim_list, tag, p_, color, thick):
    if tag not in ["segment", "circle"]:
        raise ValueError("Invalid tag. Choose either 'segment' or 'circle'.")

    # Helper function to draw a line between two keypoints
    def draw_between_dims(dim_1, dim_2):
        
        x1 = keypoints[3*dim_1]
        y1 = keypoints[3*dim_1+1]
        x2 = keypoints[3*dim_2]
        y2 = keypoints[3*dim_2+1]
        
        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
        # Check if any coordinate is zero
        if (keypoints[3*dim_1] != 0 and 
            keypoints[3*dim_1+1] != 0 and 
            keypoints[3*dim_2] != 0 and 
            keypoints[3*dim_2+1] != 0) and (
            keypoints[3*dim_1] != -1 and 
            keypoints[3*dim_1+1] != -1 and 
            keypoints[3*dim_2] != -1 and 
            keypoints[3*dim_2+1] != -1) and (
            distance <= 0.2):

            cv2.line(image, 
                     (int(keypoints[3*dim_1]*p_), int(keypoints[3*dim_1+1]*p_)), 
                     (int(keypoints[3*dim_2]*p_), int(keypoints[3*dim_2+1]*p_)), 
                     color, thick)
    
    # Drawing segment lines
    for i in range(len(dim_list) - 1):
        draw_between_dims(dim_list[i], dim_list[i+1])
    
    # If tag is circle, draw a line from the last to the first keypoint
    if tag == "circle":
        draw_between_dims(dim_list[-1], dim_list[0])
    
    return image


def draw_lines_pose(image, keypoints, dim_list, tag, p_, color, thick):
    if tag not in ["segment", "circle"]:
        raise ValueError("Invalid tag. Choose either 'segment' or 'circle'.")

    # Helper function to draw a line between two keypoints
    def draw_between_dims(dim_1, dim_2):
        
        x1 = keypoints[3*dim_1]
        y1 = keypoints[3*dim_1+1]
        x2 = keypoints[3*dim_2]
        y2 = keypoints[3*dim_2+1]
        
        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
        # Check if any coordinate is zero
        if (keypoints[3*dim_1] != 0 and 
            keypoints[3*dim_1+1] != 0 and 
            keypoints[3*dim_2] != 0 and 
            keypoints[3*dim_2+1] != 0) and (
            keypoints[3*dim_1] != -1 and 
            keypoints[3*dim_1+1] != -1 and 
            keypoints[3*dim_2] != -1 and 
            keypoints[3*dim_2+1] != -1) and (
            distance <= 0.7):

            cv2.line(image, 
                     (int(keypoints[3*dim_1]*p_), int(keypoints[3*dim_1+1]*p_)), 
                     (int(keypoints[3*dim_2]*p_), int(keypoints[3*dim_2+1]*p_)), 
                     color, thick)
    
    # Drawing segment lines
    for i in range(len(dim_list) - 1):
        draw_between_dims(dim_list[i], dim_list[i+1])
    
    # If tag is circle, draw a line from the last to the first keypoint
    if tag == "circle":
        draw_between_dims(dim_list[-1], dim_list[0])
    
    return image


def draw_lines_upper_arm(image, keypoints, dim_list, tag, p_, color, thick):
    if tag not in ["segment", "circle"]:
        raise ValueError("Invalid tag. Choose either 'segment' or 'circle'.")

    # Helper function to draw a line between two keypoints
    def draw_between_dims(dim_1, dim_2):
        
        x1 = keypoints[3*dim_1]
        y1 = keypoints[3*dim_1+1]
        x2 = keypoints[3*dim_2]
        y2 = keypoints[3*dim_2+1]
        
        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
        # Check if any coordinate is zero
        #!!!!! the threshold 0.44 is obtained from downloaded sent 324, frame 172, 173
        if (keypoints[3*dim_1] != 0 and 
            keypoints[3*dim_1+1] != 0 and 
            keypoints[3*dim_2] != 0 and 
            keypoints[3*dim_2+1] != 0) and (
            keypoints[3*dim_1] != -1 and 
            keypoints[3*dim_1+1] != -1 and 
            keypoints[3*dim_2] != -1 and 
            keypoints[3*dim_2+1] != -1) and (
            distance <= 0.44):

            cv2.line(image, 
                     (int(keypoints[3*dim_1]*p_), int(keypoints[3*dim_1+1]*p_)), 
                     (int(keypoints[3*dim_2]*p_), int(keypoints[3*dim_2+1]*p_)), 
                     color, thick)
    
    # Drawing segment lines
    for i in range(len(dim_list) - 1):
        draw_between_dims(dim_list[i], dim_list[i+1])
    
    # If tag is circle, draw a line from the last to the first keypoint
    if tag == "circle":
        draw_between_dims(dim_list[-1], dim_list[0])
    
    return image


#########################
def actual_draw_line(image, x1, y1, x2, y2, color, thickness, p_):
    x1 = int(x1 * p_)
    y1 = int(y1 * p_)
    x2 = int(x2 * p_)
    y2 = int(y2 * p_)
    cv2.line(image, (x1, y1), (x2, y2), color, thickness)


########################
#draw line segments between specific two points
def draw_specific_line(ept_image, x1, y1, x2, y2, color, thick, p_):
    
    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    #we hope the elbow connect to hand wrist, not pose wrist!!!
    if (x1 != 0 and y1 != 0 and x2 != 0 and y2 != 0) and (
        x1 != -1 and y1 != -1 and x2 != -1 and y2 != -1) and (
        distance <= 0.7): #used on pose
        
        actual_draw_line(ept_image, x1, y1, x2, y2, color, thick, p_)
        

########################
#draw line segments between specific two points (elbow and wrist)
def draw_specific_line_upper_arm(ept_image, x1, y1, x2, y2, color, thick, p_):
    
    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    #we hope the elbow connect to hand wrist, not pose wrist!!!
    if (x1 != 0 and y1 != 0 and x2 != 0 and y2 != 0) and (
        x1 != -1 and y1 != -1 and x2 != -1 and y2 != -1) and (
        distance <= 0.44): #used on upper arm
        
        actual_draw_line(ept_image, x1, y1, x2, y2, color, thick, p_)


def append_keypoint_image(mp_drawing, inp_tensor, pose_lm, face_lm, lh_lm, rh_lm, parsing_resolution, draw_hands=True):
    
    # Draw landmarks on iput image
    ept_image = inp_tensor.astype(np.float32)
    ept_image = ept_image/255. #normalize to 1
    
    draw_styled_landmarks_for_face_only(ept_image, face_lm, mp_drawing)

    ####manually draw pose points#####
    pose = np.array([[res.x, res.y, res.z] for res in pose_lm.landmark]).flatten() if pose_lm else np.zeros(33 * 3)
    face = np.array([[res.x, res.y, res.z] for res in face_lm.landmark]).flatten() if face_lm else np.zeros(478 * 3)
    left = np.array([[res.x, res.y, res.z] for res in lh_lm.landmark]).flatten() if lh_lm else np.zeros(21 * 3)
    right = np.array([[res.x, res.y, res.z] for res in rh_lm.landmark]).flatten() if rh_lm else np.zeros(21 * 3)
    p_ = parsing_resolution
            
    #draw pose
    ept_image = draw_lines_pose(ept_image, pose, [11, 13], 'segment', p_, (0,0,0), 2)
    ept_image = draw_lines_pose(ept_image, pose, [12, 14], 'segment', p_, (0,0,0), 2)
    ept_image = draw_lines_pose(ept_image, pose, [11, 12], 'segment', p_, (0,0,0), 2)
    ept_image = draw_lines_pose(ept_image, pose, [11, 23], 'segment', p_, (0,0,0), 2)
    ept_image = draw_lines_pose(ept_image, pose, [12, 24], 'segment', p_, (0,0,0), 2)
    
    if draw_hands:

        #draw left hand
        ept_image = draw_lines(ept_image, left, [0, 1], 'segment', p_, (0,0,0), 2)
        ept_image = draw_lines(ept_image, left, [1, 2, 3, 4], 'segment', p_, (0,0,0), 2)
        ept_image = draw_lines(ept_image, left, [0, 5, 9, 13, 17], 'circle', p_, (0,0,0), 2)
        ept_image = draw_lines(ept_image, left, [5, 6, 7, 8], 'segment', p_, (0,0,0), 2)
        ept_image = draw_lines(ept_image, left, [9, 10, 11, 12], 'segment', p_, (0,0,0), 2)
        ept_image = draw_lines(ept_image, left, [13, 14, 15, 16], 'segment', p_, (0,0,0), 2)
        ept_image = draw_lines(ept_image, left, [17, 18, 19, 20], 'segment', p_, (0,0,0), 2)

        #draw right hand
        ept_image = draw_lines(ept_image, right, [0, 1], 'segment', p_, (0,0,0), 2)
        ept_image = draw_lines(ept_image, right, [1, 2, 3, 4], 'segment', p_, (0,0,0), 2)
        ept_image = draw_lines(ept_image, right, [0, 5, 9, 13, 17], 'circle', p_, (0,0,0), 2)
        ept_image = draw_lines(ept_image, right, [5, 6, 7, 8], 'segment', p_, (0,0,0), 2)
        ept_image = draw_lines(ept_image, right, [9, 10, 11, 12], 'segment', p_, (0,0,0), 2)
        ept_image = draw_lines(ept_image, right, [13, 14, 15, 16], 'segment', p_, (0,0,0), 2)
        ept_image = draw_lines(ept_image, right, [17, 18, 19, 20], 'segment', p_, (0,0,0), 2)

    #connect the elbow to hand wrist, not pose wrist!
    '''
    However, if the hands are all zeros, we still connect pose wrist.
    Only keep in mind: there are still chances the hand is so far away from pose wrist.
    In this case, we still skip drawing (threshold is specific draw_lines_upper_arm).
    '''
    if zero_rate(left) > 0.5:
        ept_image = draw_lines_upper_arm(ept_image, pose, [13, 15], 'segment', p_, (0,0,0), 2)
    else:
        #print('left', zero_rate(left), pose[15*3 + 1])
        draw_specific_line_upper_arm(ept_image, pose[3*13], pose[3*13+1], left[3*0], left[3*0+1],
                                     (0,0,0), 2, p_)

    if zero_rate(right) > 0.5:
        ept_image = draw_lines_upper_arm(ept_image, pose, [14, 16], 'segment', p_, (0,0,0), 2)
    else:
        #print('right', zero_rate(right), pose[16*3 + 1])
        draw_specific_line_upper_arm(ept_image, pose[3*14], pose[3*14+1], right[3*0], right[3*0+1], 
                                     (0,0,0), 2, p_)

    #draw the middle line segment from mid-shouder to mid-back
    x1_mid = (pose[3*11] + pose[3*12])/2.
    y1_mid = (pose[3*11+1] + pose[3*12+1])/2.
    x2_mid = (pose[3*23] + pose[3*24])/2.
    y2_mid = (pose[3*23+1] + pose[3*24+1])/2.

    draw_specific_line(ept_image, x1_mid, y1_mid, x2_mid, y2_mid, 
                       (0,0,0), 2, p_)
    
    #draw face countor
    ept_image = draw_lines(ept_image, face, [10, 338, 297, 332, 284, 251,
                                             389, 356, 454, 323, 361, 288,
                                             397, 365, 379, 378, 400, 377,
                                             152, 148, 176, 149, 150, 136,
                                             172, 58, 132, 93, 234, 127,
                                             162, 21, 54, 103, 67, 109], 'circle', p_, (0,0,0), 2)

    #draw right eyebrow
    ept_image = draw_lines(ept_image, face, [70, 63, 105, 66, 107], 'segment', p_, (0,0,0), 2)
    ept_image = draw_lines(ept_image, face, [46, 53, 52, 65, 55], 'segment', p_, (0,0,0), 2)

    #draw left eyebrow
    ept_image = draw_lines(ept_image, face, [300, 293, 334, 296, 336], 'segment', p_, (0,0,0), 2)
    ept_image = draw_lines(ept_image, face, [276, 283, 282, 295, 285], 'segment', p_, (0,0,0), 2)

    #draw right eye
    ept_image = draw_lines(ept_image, face, [246, 161, 160, 159, 158, 157, 173,
                                             133, 155, 154, 153, 145, 144, 163, 7, 33], 'circle', p_, (0,0,0), 2)
    #ept_image = draw_lines(ept_image, face, [247, 30, 29, 27, 28, 56, 190,
    #                                         243, 112, 26, 22, 23, 24, 110, 25, 130], 'circle', p_, (0,0,0), 2)
    
    #draw left eye
    ept_image = draw_lines(ept_image, face, [466, 388, 387, 386, 385, 384, 398,
                                             362, 382, 381, 380, 374, 373, 390, 249, 263], 'circle', p_, (0,0,0), 2)
    #ept_image = draw_lines(ept_image, face, [467, 260, 259, 257, 258, 286, 414,
    #                                         463, 341, 256, 252, 253, 254, 339, 255, 359], 'circle', p_, (0,0,0), 2)

    #draw bridge of the nose
    ept_image = draw_lines(ept_image, face, [6, 197, 195, 5, 4], 'segment', p_, (0,0,0), 2)
    
    #draw lower bound of nose
    ept_image = draw_lines(ept_image, face, [240, 99, 2, 328, 460], 'segment', p_, (0,0,0), 2)
    
    #draw lips outer
    ept_image = draw_lines(ept_image, face, [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                                             375, 321, 405, 314, 17, 84, 181, 91, 146], 'circle', p_, (0,0,0), 2)

    #draw lips inner
    ept_image = draw_lines(ept_image, face, [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                                             324, 318, 402, 317, 14, 87, 178, 88, 95], 'circle', p_, (0,0,0), 2)

    
    return(ept_image*255.)
    


############################
#the function just used to draw DW other keypoints (face & hand)
def actual_draw_line_DW(image, x1, y1, x2, y2, color, thickness, p_):
    x1 = int(x1 * p_)
    y1 = int(y1 * p_)
    x2 = int(x2 * p_)
    y2 = int(y2 * p_)
    cv2.line(image, (x1, y1), (x2, y2), color, thickness)


def draw_lines_DW(image, keypoints, dim_list, tag, p_, color, thick):

    def draw_between_dims_DW(keypoints, dim_1, dim_2, image, color, thick, p_):
        x1 = keypoints[2*dim_1]
        y1 = keypoints[2*dim_1+1]
        x2 = keypoints[2*dim_2]
        y2 = keypoints[2*dim_2+1]
        
        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
        #-1 is the NaN in DWPose, and also we don't want the distance
        #also, some points are not quite -1, but still out-range
        #so, we take care those keypoints by set a maximum bound
        #we use lower threshold for other keypoint 
        if (x1 != -1 and y1 != -1 and x2 != -1 and y2 != -1) and (
            distance <= 0.2): 
            actual_draw_line_DW(image, x1, y1, x2, y2, color, thick, p_)

    for i in range(len(dim_list) - 1):
        draw_between_dims_DW(keypoints, dim_list[i], dim_list[i+1], image, color, thick, p_)

    if tag == "circle":
        draw_between_dims_DW(keypoints, dim_list[-1], dim_list[0], image, color, thick, p_)

    return image


#############################
#the function just used to draw DW pose (body) keypoints
def draw_lines_DW_pose(image, keypoints, dim_list, tag, p_, color, thick):

    def draw_between_dims_DW_pose(keypoints, dim_1, dim_2, image, color, thick, p_):
        x1 = keypoints[2*dim_1]
        y1 = keypoints[2*dim_1+1]
        x2 = keypoints[2*dim_2]
        y2 = keypoints[2*dim_2+1]
        
        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
        #-1 is the NaN in DWPose, and also we don't want the distance
        #also, some points are not quite -1, but still out-range
        #so, we take care those keypoints by set a maximum bound
        #since this is to draw pose, the threshold is 0.7
        if (x1 != -1 and y1 != -1 and x2 != -1 and y2 != -1) and (
            distance <= 0.7): 
            actual_draw_line_DW(image, x1, y1, x2, y2, color, thick, p_)

    for i in range(len(dim_list) - 1):
        draw_between_dims_DW_pose(keypoints, dim_list[i], dim_list[i+1], image, color, thick, p_)

    if tag == "circle":
        draw_between_dims_DW_pose(keypoints, dim_list[-1], dim_list[0], image, color, thick, p_)

    return image


def append_keypoint_image_DW(pose, face, left, right, parsing_resolution, thick):
    
    # Draw landmarks on empty image
    ept_image = np.ones((parsing_resolution, parsing_resolution, 3), dtype='float32')
    
    p_ = parsing_resolution
            
    #draw pose
    ept_image = draw_lines_DW_pose(ept_image, pose, [1, 2, 3, 4], 'segment', p_, (0,0,0), thick)
    ept_image = draw_lines_DW_pose(ept_image, pose, [1, 5, 6, 7], 'segment', p_, (0,0,0), thick)
    ept_image = draw_lines_DW_pose(ept_image, pose, [1, 8], 'segment', p_, (0,0,0), thick)
    ept_image = draw_lines_DW_pose(ept_image, pose, [1, 9], 'segment', p_, (0,0,0), thick)

    #draw left hand
    ept_image = draw_lines_DW(ept_image, left, [0, 1, 2, 3, 4], 'segment', p_, (0,0,0), thick)
    ept_image = draw_lines_DW(ept_image, left, [0, 5, 6, 7, 8], 'segment', p_, (0,0,0), thick)
    ept_image = draw_lines_DW(ept_image, left, [0, 9, 10, 11, 12], 'segment', p_, (0,0,0), thick)
    ept_image = draw_lines_DW(ept_image, left, [0, 13, 14, 15, 16], 'segment', p_, (0,0,0), thick)
    ept_image = draw_lines_DW(ept_image, left, [0, 17, 18, 19, 20], 'segment', p_, (0,0,0), thick)
    
    #draw right hand
    ept_image = draw_lines_DW(ept_image, right, [0, 1, 2, 3, 4], 'segment', p_, (0,0,0), thick)
    ept_image = draw_lines_DW(ept_image, right, [0, 5, 6, 7, 8], 'segment', p_, (0,0,0), thick)
    ept_image = draw_lines_DW(ept_image, right, [0, 9, 10, 11, 12], 'segment', p_, (0,0,0), thick)
    ept_image = draw_lines_DW(ept_image, right, [0, 13, 14, 15, 16], 'segment', p_, (0,0,0), thick)
    ept_image = draw_lines_DW(ept_image, right, [0, 17, 18, 19, 20], 'segment', p_, (0,0,0), thick)
    
    #draw face contour
    ept_image = draw_lines_DW(ept_image, face, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], 'segment', p_, (0,0,0), thick)

    #draw right eyebrow
    ept_image = draw_lines_DW(ept_image, face, [17,18,19,20,21], 'segment', p_, (0,0,0), thick)

    #draw left eyebrow
    ept_image = draw_lines_DW(ept_image, face, [22,23,24,25,26], 'segment', p_, (0,0,0), thick)

    #draw nose bridge
    ept_image = draw_lines_DW(ept_image, face, [27,28,29,30], 'segment', p_, (0,0,0), thick)

    #draw nose lower edge
    ept_image = draw_lines_DW(ept_image, face, [31,32,33,34,35], 'segment', p_, (0,0,0), thick)

    #draw right eye
    ept_image = draw_lines_DW(ept_image, face, [36,37,38,39,40,41], 'circle', p_, (0,0,0), thick)

    #draw left eye
    ept_image = draw_lines_DW(ept_image, face, [42,43,44,45,46,47], 'circle', p_, (0,0,0), thick)

    #draw outer lip
    ept_image = draw_lines_DW(ept_image, face, [48,49,50,51,52,53,54,55,56,57,58,59], 'circle', p_, (0,0,0), thick)

    #draw inner lip
    ept_image = draw_lines_DW(ept_image, face, [60,61,62,63,64,65,66,67], 'circle', p_, (0,0,0), thick)

    return(ept_image*255.)
############################



####################################
####DWPose offcial drawing code#####
def draw_bodypose(canvas, candidate, subset):
    
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    return canvas


def draw_handpose(canvas, all_hand_peaks):
    eps = 0.01

    H, W, C = canvas.shape

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for peaks in all_hand_peaks:
        peaks = np.array(peaks)

        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255, thickness=2)

        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    return canvas


def draw_facepose(canvas, all_lmks):
    eps = 0.01

    H, W, C = canvas.shape
    for lmks in all_lmks:
        lmks = np.array(lmks)
        for lmk in lmks:
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    return canvas


def draw_pose(pose, H, W):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = draw_bodypose(canvas, candidate, subset)

    canvas = draw_handpose(canvas, hands)

    canvas = draw_facepose(canvas, faces)

    return canvas
#################################


def load_pickles(folder_path):
    # Convert to Path object
    folder = Path(folder_path)
    
    # Get all pickle files
    pickle_files = list(folder.glob('*.pickle'))
    
    # List to store (id, path) pairs
    valid_pickles = []
    id_check = {}
    
    for pickle_path in pickle_files:
        # Get filename without extension
        name = pickle_path.stem
        
        # Check if filename is integer
        try:
            file_id = int(name)
        except ValueError:
            raise ValueError(f"Non-integer filename found: {name}")
            
        # Check for duplicates
        if file_id in id_check:
            raise ValueError(f"Duplicate ID found: {file_id}")
            
        id_check[file_id] = True
        valid_pickles.append((file_id, pickle_path))
    
    # Sort by ID
    valid_pickles.sort(key=lambda x: x[0])
    
    # Return just the filenames (ID.pickle)
    return [f"{id}.pickle" for id, _ in valid_pickles]



#################################
####class function###############
class CombineKeyPoint_and_DrawKeyPointVideo:
        
    def __init__(self, path, H, W):
        
        self.path = path + 'raw/individual_pickles/'
        self.path_root = path
        self.path_vid = path + 'raw/individual_kp_videos/'
        self.sorted_filenames = load_pickles(self.path)
        
        if H is not None:
            self.H = H
        else:
            self.H = 1024

        if W is not None:
            self.W = W
        else:
            self.W = 1024

        os.makedirs(self.path_vid, exist_ok=True)

    def run(self):

        #we draw videos and also keep all normalized array in one file
        image_list = list()
        total_array_list = list()
        ave_fps = 0.
        count_fps = 0
        for ele in self.sorted_filenames:

            in_path = self.path + ele
            vid_id = int(ele.split('.')[0])

            image_list_temp = list()

            with open(in_path, 'rb') as handle:
                Dict = pickle.load(handle)

            array_list = Dict['keypoint']
            
            if 'fps' in Dict['info'] and Dict['info']['fps'] <= 62: #if the frame rate is too high we still refuse to use
                fps = float(Dict['info']['fps'])
                count_fps += 1
                ave_fps += fps
            else:
                fps = 30.

            #draw keypoints
            for i_ in range(len(array_list)):
                pose_image = draw_pose(array_list[i_], self.H, self.W)
                image_list.append(pose_image)
                image_list_temp.append(pose_image)
                total_array_list.append(array_list[i_])
            
            #os.remove(self.path + ele) #we don't need each file anymore

            #save the individual video
            out_temp = cv2.VideoWriter(self.path_vid + str(vid_id) + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (self.W, self.H))
            for pose_image in image_list_temp:
                out_temp.write(pose_image)
            out_temp.release()

        if count_fps > 0:
            ave_fps = ave_fps/count_fps
        else:
            ave_fps = 30.
        print('average fps', ave_fps, 'number of fps recorded', count_fps)

        # Define the codec and create a video writer object
        out = cv2.VideoWriter(self.path_root + 'out_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), ave_fps, (self.W, self.H))

        for pose_image in image_list:
            out.write(pose_image)
    
        # Release the video objects and close the windows
        out.release()

        #save array into one file
        with open(self.path_root + 'normalized_keypoints.pickle', 'wb') as handle:
            pickle.dump(total_array_list, handle, protocol=pickle.HIGHEST_PROTOCOL)






