import sys
import os
#import time
import pickle
import cv2
#import matplotlib.pyplot as plt
import mediapipe as mp
import shutil
#import multiprocessing
#import logging
import numpy as np
#import gc
#import traceback

from collections import Counter, defaultdict
from scipy.interpolate import CubicSpline
from mediapipe.framework.formats import landmark_pb2
from copy import deepcopy
from tqdm import tqdm
#from multiprocessing import Pool, Manager
from pathlib import Path



'''
!!!!!
Note that code here is not up to date. 
Please see kp_normalization.py which is the latest normalization code.
!!!!!
'''



'''
Instead of given input video, sometimes people directly work with keypoints.
In this case, we use this script to load the kp and normalize.

This is mainly for Acore/OmniBridge dataset.
In fact, this script, kp_normalization_for_npy is designed to run by itself, rather than
be stitched together into the main function. 
So, this is just kind of a proto-type script for OmniBridge dataset kp normalization.
Their data format is like:

Given a numpy array of shape [num_frames, 544, 4], it contains the human face/hands/pose keypoints extracted from a video segment. 
Each element in the first dimension contains the [544, 4] array for the keypoints. 544 is divided like this:

First index is just the frame_id compared to the original video.
next 33 body
next 468 face
next 21 left hand
next 21 right hand
4 contains [x, y, z, visibility]. 
 
Then, we just have this data reformatted:
For the [544, 4] array of each frame, I want to keep them in a dictionary: 
The key 'pose_mp' will store the keypoint parameters for body pose, which is the dim[1:34, :] 
(ID is taken according to Python hobby, so 1, 2, ..., 33) in the [544,4] array. 
But moreover, I want to flatten the xyz and remove the visibility. 
So finally, 'pose_mp' will store (x1, y1, z1, ..., x33, y33, z33). 

In the same format, we have 'face_mp' store (x1, y1, z1, ..., x468, y468, z468), 
note that x1, y1, z1 starts from dim-34 in the original [544, 4] array. 
And we have 'left_hand_mp' and 'right_hand_mp' store (x1, y1, z1, ..., x21, y21, z21) accordingly. 
We call this Dict_1

In a separate dictionary, we have the visibility stored under the same key: 
'pose_mp' store (v1, v2, ..., v33). 'face_mp' store (v1, v2, ..., v468), etc. We call this Dict_2

After that, we will have a root dictionary {'keypoint': list(), 'visibility': list()}. 
Then, append Dict_1 to keypoint, append Dict_2 to visibility. 

Finally, we return the root dict as the output, which is the correct format for normalization.
'''


################################
def reformat(data):
    """
    Reformat the keypoints and visibility into the desired dictionary structure.

    Returns:
        dict: A dictionary with keys 'keypoint' and 'visibility', each containing a list of dictionaries.
    """
    # Initialize the root dictionary
    root_dict = {'keypoint': [], 'visibility': [], 'frame_id': []}

    # Iterate over each frame
    for frame in data:
        # Initialize Dict_1 and Dict_2 for the current frame
        Dict_1 = {}
        Dict_2 = {}

        # Extract pose keypoints (1:34 in Python indexing)
        pose_keypoints = frame[1:34, :3].flatten()  # Flatten x, y, z
        pose_visibility = frame[1:34, 3]           # Extract visibility
        Dict_1['pose_mp'] = pose_keypoints
        Dict_2['pose_mp'] = pose_visibility

        # Extract face keypoints (34:502 in Python indexing)
        face_keypoints = frame[34:502, :3].flatten()
        face_visibility = frame[34:502, 3]
        Dict_1['face_mp'] = face_keypoints
        Dict_2['face_mp'] = face_visibility

        # Extract left hand keypoints (502:523 in Python indexing)
        left_hand_keypoints = frame[502:523, :3].flatten()
        left_hand_visibility = frame[502:523, 3]
        Dict_1['left_hand_mp'] = left_hand_keypoints
        Dict_2['left_hand_mp'] = left_hand_visibility

        # Extract right hand keypoints (523:544 in Python indexing)
        right_hand_keypoints = frame[523:544, :3].flatten()
        right_hand_visibility = frame[523:544, 3]
        Dict_1['right_hand_mp'] = right_hand_keypoints
        Dict_2['right_hand_mp'] = right_hand_visibility

        # Append Dict_1 and Dict_2 to the root dictionary
        root_dict['keypoint'].append(Dict_1)
        root_dict['visibility'].append(Dict_2)
        root_dict['frame_id'].append(frame[0])

    return root_dict


def revert_reformat(root_dict):
    """
    Revert the reformatted dictionary back to the original numpy array format (k, 544, 4).

    Args:
        root_dict (dict): A dictionary with keys 'keypoint', 'visibility', and 'frame_id'.

    Returns:
        numpy.ndarray: A numpy array of shape (k, 544, 4), where k is the number of frames.
    """
    # Extract the number of frames
    num_frames = len(root_dict['keypoint'])

    # Initialize the output array
    reverted_data = np.zeros((num_frames, 544, 4))

    # Iterate over each frame
    for i in range(num_frames):
        # Set the entire first row (frame ID vector)
        frame_id_vector = root_dict['frame_id'][i]
        reverted_data[i, 0, :] = frame_id_vector  # Set the entire 4-dimensional vector

        # Reconstruct pose keypoints (1:34 in Python indexing)
        pose_keypoints = root_dict['keypoint'][i]['pose_mp'].reshape(-1, 3)  # Reshape to (33, 3)
        pose_visibility = root_dict['visibility'][i]['pose_mp']             # Visibility (33,)
        reverted_data[i, 1:34, :3] = pose_keypoints
        reverted_data[i, 1:34, 3] = pose_visibility

        # Reconstruct face keypoints (34:502 in Python indexing)
        face_keypoints = root_dict['keypoint'][i]['face_mp'].reshape(-1, 3)  # Reshape to (468, 3)
        face_visibility = root_dict['visibility'][i]['face_mp']             # Visibility (468,)
        reverted_data[i, 34:502, :3] = face_keypoints
        reverted_data[i, 34:502, 3] = face_visibility

        # Reconstruct left hand keypoints (502:523 in Python indexing)
        left_hand_keypoints = root_dict['keypoint'][i]['left_hand_mp'].reshape(-1, 3)  # Reshape to (21, 3)
        left_hand_visibility = root_dict['visibility'][i]['left_hand_mp']             # Visibility (21,)
        reverted_data[i, 502:523, :3] = left_hand_keypoints
        reverted_data[i, 502:523, 3] = left_hand_visibility

        # Reconstruct right hand keypoints (523:544 in Python indexing)
        right_hand_keypoints = root_dict['keypoint'][i]['right_hand_mp'].reshape(-1, 3)  # Reshape to (21, 3)
        right_hand_visibility = root_dict['visibility'][i]['right_hand_mp']             # Visibility (21,)
        reverted_data[i, 523:544, :3] = right_hand_keypoints
        reverted_data[i, 523:544, 3] = right_hand_visibility

    return reverted_data


'''
We need to build ('estimate' or 'mimic') DWPose body keypoint from mediapipe body keypoint.
This is because the normalization is based on DWpose keypoint, which is not included in Acorn kp.

That is, I have the 33 human body keypoints extracted by mediapipe. They are
0 - nose
1 - left eye (inner)
2 - left eye
3 - left eye (outer)
4 - right eye (inner)
5 - right eye
6 - right eye (outer)
7 - left ear
8 - right ear
9 - mouth (left)
10 - mouth (right)
11 - left shoulder
12 - right shoulder
13 - left elbow
14 - right elbow
15 - left wrist
16 - right wrist
17 - left pinky
18 - right pinky
19 - left index
20 - right index
21 - left thumb
22 - right thumb
23 - left hip
24 - right hip
25 - left knee
26 - right knee
27 - left ankle
28 - right ankle
29 - left heel
30 - right heel
31 - left foot index
32 - right foot index
Each keypoint has x, y, z value. The x value ranges from 0 to 1 as from left to right, 
y value ranges from 0 to 1 as from top to bottom. 
Z is the distance from the camera, which can be either positive or negative. 
Then, the mediapipe keypoints are stored in a 1D array (x0, y0, z0, x1, y1, z1, ..., x32, y32, z32). 

Given this 1D array, I need to build the DWpose keypoint from it. 
DWpose body keypoint contains 18 keypoints. 
Here is a one-to-one map of DWpose kp by mediapipe kp (left is DWpose ID, right is mediapipe ID):

0 - 0
1 - average of 11 and 12
2 - 12
3 - 14
4 - 16
5 - 11
6 - 13
7 - 15
8 - 24
9 - 26
10 - 28
11 - 23
12 - 25
13 - 27
14 - 5
15 - 2
16 - 8
17 - 7

You may notice 1 - average of 11 and 12. 
That is, 1 is the neck-end of DWpose, which is not a recognized keypoint in mediapipe. 
So instead, we use the middle of two should keypoint in mediapipe to calculate it.

For each 0 to 17 of the target DWpose keypoint, 
I want to extract the corresponding x and y value and store in [x, y]. 
For ID 1, you will get it by [(x11+x12)/2, (y11+y12)/2]. 
After that. please keep the target keypoints in 2D array: [[x0, y0], [x1, y1], ... [x17,y17]]. 

Finally, the output is a dict, which contains only one key 'bodies'.
Under this key, the value is still a dictionary:

{'candidate': target 2D array of shape (18,2),
'subset': array([[ 0., 1., 2., 3., 4., 5., 6., 7., 8., -1., -1., 11., -1., -1., 14., 15., 16., 17.]])},

The subset indicate which keypoint in DWpose is valid. 
We just assume that the leg is not predicted (even it is in mediapipe). 
So, we always assign the exactly same value as in above to subset.
'''
def mediapipe_to_dwpose(mediapipe_keypoints):
    """
    Convert Mediapipe keypoints to DWpose keypoints.

    Args:
        mediapipe_keypoints (list or np.ndarray): A 1D array of shape (33*3,) containing Mediapipe keypoints
                                                  in the format [x0, y0, z0, x1, y1, z1, ..., x32, y32, z32].

    Returns:
        dict: A dictionary with the DWpose keypoints in the required format.
    """
    # Ensure the input is a numpy array
    mediapipe_keypoints = np.array(mediapipe_keypoints).reshape(-1, 3)

    # DWpose keypoint mapping from Mediapipe
    dwpose_map = {
        0: 0,
        1: (11, 12),  # Average of left shoulder (11) and right shoulder (12)
        2: 12,
        3: 14,
        4: 16,
        5: 11,
        6: 13,
        7: 15,
        8: 24,
        9: 26,
        10: 28,
        11: 23,
        12: 25,
        13: 27,
        14: 5,
        15: 2,
        16: 8,
        17: 7
    }

    # Initialize the DWpose keypoints array
    dwpose_keypoints = []

    for dw_id, mp_id in dwpose_map.items():
        if isinstance(mp_id, tuple):  # If the DWpose keypoint is an average of two Mediapipe keypoints
            x = (mediapipe_keypoints[mp_id[0], 0] + mediapipe_keypoints[mp_id[1], 0]) / 2
            y = (mediapipe_keypoints[mp_id[0], 1] + mediapipe_keypoints[mp_id[1], 1]) / 2
        else:  # Direct mapping
            x = mediapipe_keypoints[mp_id, 0]
            y = mediapipe_keypoints[mp_id, 1]
        dwpose_keypoints.append([x, y])

    # Convert to numpy array
    dwpose_keypoints = np.array(dwpose_keypoints)

    # Subset array (legs are not predicted, so fixed as given)
    subset = np.array([[0., 1., 2., 3., 4., 5., 6., 7., 8., -1., -1., 11., -1., -1., 14., 15., 16., 17.]])

    # Build the final dictionary
    result = {
        'bodies': {
            'candidate': dwpose_keypoints,
            'subset': subset
        }
    }

    return result

#######################################
def compute_l2_distance(array1, array2):
    """
    Compute L2 (Euclidean) distance between two 1D arrays
    
    Parameters:
    array1, array2: arrays of same length
    
    Returns:
    float: L2 distance between the arrays
    """
    # Convert inputs to numpy arrays if they aren't already
    a1 = np.array(array1)
    a2 = np.array(array2)
    
    # Compute L2 distance
    distance = np.sqrt(np.sum((a1 - a2) ** 2))
    
    return distance


def calculate_3d_distance(point1, point2):
    # Convert inputs to numpy arrays for easier calculation
    p1 = np.array(point1)
    p2 = np.array(point2)
    
    # Calculate distance using numpy's built-in norm function
    distance = np.linalg.norm(p2 - p1)
    
    return distance


#old function without mask option
def process_keypoints(keypoints_dict, resize_height_rate=1.0, resize_width_rate=1.0, move_x=0.0, move_y=0.0):
    """
    Resize and move keypoints in the dictionary while maintaining specific formats and rules.
    
    Args:
        keypoints_dict: Dictionary containing different types of keypoints
        resize_height_rate: float, rate to resize in height (y-direction)
        resize_width_rate: float, rate to resize in width (x-direction)
        move_x: float, displacement in x direction
        move_y: float, displacement in y direction
    
    Returns:
        Dictionary with processed keypoints
    """
    result_dict = {}
    
    # Helper function to process 2D array of shape (..., 2)
    def process_2d_points(points):
        # Check for invalid points (x or y < -0.9)
        invalid_mask = np.any(points < -0.9, axis=-1)
        
        # Process valid points
        processed = points.copy()
        processed[..., 0] = points[..., 0] * resize_width_rate + move_x
        processed[..., 1] = points[..., 1] * resize_height_rate + move_y
        
        # Reset invalid points to [-1, -1]
        processed[invalid_mask] = [-1, -1]
        return processed
    
    # Helper function to process 1D array with z values
    def process_1d_points_with_z(points):
        if np.all(points == 0):  # If all zeros, return as is
            return points
            
        n_points = len(points) // 3
        reshaped = points.reshape(n_points, 3)
        
        # Process x and y, keep z unchanged
        reshaped[:, 0] = reshaped[:, 0] * resize_width_rate + move_x
        reshaped[:, 1] = reshaped[:, 1] * resize_height_rate + move_y
        
        return reshaped.reshape(-1)
    
    # Process DWPose body keypoints
    if 'bodies' in keypoints_dict:
        result_dict['bodies'] = {
            'candidate': process_2d_points(keypoints_dict['bodies']['candidate']),
            'subset': keypoints_dict['bodies']['subset'].copy()  # Keep subset unchanged
        }
    
    # Process DWPose hands
    if 'hands' in keypoints_dict:
        result_dict['hands'] = process_2d_points(keypoints_dict['hands'])
    
    # Process DWPose faces (unstack first dim)
    if 'faces' in keypoints_dict:
        faces = keypoints_dict['faces']
        processed_faces = process_2d_points(faces[0])  # Process the unstacked array
        result_dict['faces'] = processed_faces[np.newaxis, ...]  # Stack back
    
    # Process MediaPipe keypoints (all ending with _mp)
    for key in keypoints_dict:
        if key.endswith('_mp'):
            if key == 'confidence_score_mp':
                result_dict[key] = keypoints_dict[key]  # Keep confidence score unchanged
            else:
                result_dict[key] = process_1d_points_with_z(keypoints_dict[key])
    
    return result_dict


#new process funciton with mask option
def process_keypoints_with_mask(keypoints_dict, mask_dict, resize_height_rate=1.0, resize_width_rate=1.0, move_x=0.0, move_y=0.0):
    """
    Resize and move keypoints in the dictionary while maintaining specific formats and rules.
    
    Args:
        keypoints_dict: Dictionary containing different types of keypoints
        resize_height_rate: float, rate to resize in height (y-direction)
        resize_width_rate: float, rate to resize in width (x-direction)
        move_x: float, displacement in x direction
        move_y: float, displacement in y direction
        mask_dict contains the mask for specific key. 
        That is, we put the mask to the keypoints under the same key in the mask_dict as the key in keypoints_dict
    
    Returns:
        Dictionary with processed keypoints
    """
    result_dict = {}
    
    # Helper function to process 2D array of shape (..., 2)
    def process_2d_points(points, mask_indices=None):
        # Check for invalid points (x or y < -0.9)
        invalid_mask = np.any(points < -0.9, axis=-1)
        
        # Process valid points
        processed = points.copy()
        
        if mask_indices is not None:
            # Create a boolean mask for points to change
            change_mask = np.ones(points.shape[0], dtype=bool)
            change_mask[mask_indices] = False
            
            # Only process non-masked points
            processed[change_mask, 0] = points[change_mask, 0] * resize_width_rate + move_x
            processed[change_mask, 1] = points[change_mask, 1] * resize_height_rate + move_y
        else:
            # Process all points
            processed[..., 0] = points[..., 0] * resize_width_rate + move_x
            processed[..., 1] = points[..., 1] * resize_height_rate + move_y
        
        # Reset invalid points to [-1, -1]
        processed[invalid_mask] = [-1, -1]
        return processed
    
    # Helper function to process 1D array with z values
    def process_1d_points_with_z(points, mask_indices=None):
        if np.all(points == 0):  # If all zeros, return as is
            return points
            
        n_points = len(points) // 3
        reshaped = points.reshape(n_points, 3)
        processed = reshaped.copy()
        
        if mask_indices is not None:
            # Create a boolean mask for points to change
            change_mask = np.ones(n_points, dtype=bool)
            change_mask[mask_indices] = False
            
            # Only process non-masked points
            processed[change_mask, 0] = reshaped[change_mask, 0] * resize_width_rate + move_x
            processed[change_mask, 1] = reshaped[change_mask, 1] * resize_height_rate + move_y
        else:
            # Process all points
            processed[:, 0] = reshaped[:, 0] * resize_width_rate + move_x
            processed[:, 1] = reshaped[:, 1] * resize_height_rate + move_y
        
        return processed.reshape(-1)
    
    def get_mask(mask_dict, key):
        if key in mask_dict:
            return(mask_dict[key])
        else:
            return(None)
    
    # Process DWPose body keypoints
    if 'bodies' in keypoints_dict:
        result_dict['bodies'] = {
            'candidate': process_2d_points(keypoints_dict['bodies']['candidate'], 
                                           mask_indices=get_mask(mask_dict, 'bodies')),
            'subset': keypoints_dict['bodies']['subset'].copy()  # Keep subset unchanged
        }
    
    # Process DWPose hands (process all points)
    if 'hands' in keypoints_dict:
        left_hand_DW = process_2d_points(keypoints_dict['hands'][0],
                                                 mask_indices=get_mask(mask_dict, 'left_hand_DW'))
        right_hand_DW = process_2d_points(keypoints_dict['hands'][1],
                                                 mask_indices=get_mask(mask_dict, 'right_hand_DW'))
        
        result_dict['hands'] = np.stack([left_hand_DW, right_hand_DW], axis=0)
    
    # Process DWPose faces (unstack first dim)
    if 'faces' in keypoints_dict:
        faces = keypoints_dict['faces']
        processed_faces = process_2d_points(faces[0], 
                                            mask_indices=get_mask(mask_dict, 'faces'))  # Process the unstacked array
        result_dict['faces'] = processed_faces[np.newaxis, ...]  # Stack back
    
    # Process MediaPipe keypoints (all ending with _mp)
    for key in keypoints_dict:
        if key.endswith('_mp'):

            if key == 'confidence_score_mp':
                result_dict[key] = keypoints_dict[key]  # Keep confidence score unchanged
            
            else:
                # Convert mask to array indices if provided
                array_mask = None
                if key in mask_dict:
                    array_mask = [i for point in mask_dict[key] for i in range(point, point+1)]
                result_dict[key] = process_1d_points_with_z(keypoints_dict[key], 
                                                            mask_indices=array_mask)
    
    return result_dict



# function to actually implement normalization of keypoints
'''
std_sholder is the sholder L-2 distance (consider both x and y) of the DWPose.
That is, the dist between Dict['body']['candidate'][2] and Dict['body']['candidate'][5]

std_d_y is the neck-end to average ear y-value difference of the DWPose.
That is, the difference Dict['body']['candidate'][0][1] and 
   average of (Dict['body']['candidate'][16][1], Dict['body']['candidate'][17][1])

std_sholder=0.323, std_d_y=0.208 is obtained from
../../data/SigningSavvy_Dict/new_2_key_vectors_indpt_h_adjust_DWPose/a/1-#-the-letter-a/2.pickle
That is, the first frame of lady Brenda signing letter 'a' in fingerspelling.

Similarly, we use this frame of lady Branda to decide the center (neck end, note 1 in DWPose body)
x=0.50217276, y=0.55913485
We accordingly decide the move direction.

Finally, we decide to expand/shrink the x of the face and hands according to the ear distance
since in many cases the face is still too narrow. We use 
the x-value distance between Lady Brenda as std_ear_x=0.159
this is also taken from
../../data/SigningSavvy_Dict/new_2_key_vectors_indpt_h_adjust_DWPose/a/1-#-the-letter-a/2.pickle 
'''
def do_normalization(numpy_array_in, num_insert_interpolation, draw_video,
                     std_sholder=0.323, std_d_y=0.208, std_x=0.502, std_y=0.559, std_ear_x=0.159):
    
    #extract keypoint
    format_dict = reformat(numpy_array_in)

    for i in range(len(format_dict['keypoint'])):
        dwpose_result = mediapipe_to_dwpose(format_dict['keypoint'][i]['pose_mp'])
        format_dict['keypoint'][i]['bodies'] = dwpose_result['bodies']

    vec_array = format_dict['keypoint']
    
    #calculate ave sholder distance
    ave_sholder = 0.
    for i in range(len(vec_array)):
        x1 = vec_array[i]['bodies']['candidate'][2][0]
        y1 = vec_array[i]['bodies']['candidate'][2][1]
        x2 = vec_array[i]['bodies']['candidate'][5][0]
        y2 = vec_array[i]['bodies']['candidate'][5][1]
        distance = compute_l2_distance([x1, y1], [x2, y2])
        ave_sholder += distance
    ave_sholder = ave_sholder/len(vec_array)
    
    #obtain the resize rate
    r_w = std_sholder/ave_sholder

    #resize mp and DW
    vec_array_2 = list()
    for i in range(len(vec_array)):
        processed = process_keypoints(vec_array[i], 
                                      resize_height_rate=r_w, resize_width_rate=r_w,
                                      move_x=0, move_y=0)
        vec_array_2.append(processed)
    
    #calculate average neck end to ear y-difference (only consider y)
    max_d_y = 0.
    for i in range(len(vec_array_2)):
        y1 = vec_array_2[i]['bodies']['candidate'][1][1]
        y2 = vec_array_2[i]['bodies']['candidate'][16][1]
        y3 = vec_array_2[i]['bodies']['candidate'][17][1]
        y4 = (y2 + y3)/2
        distance = abs(y1 - y4)
        max_d_y = max(distance, max_d_y)
    
    #print('max_d_y', max_d_y)

    #obtain the resize rate
    r_y = std_d_y/max_d_y

    #resize mp and DW
    vec_array_3 = list()
    for i in range(len(vec_array_2)):
        processed = process_keypoints(vec_array_2[i], 
                                      resize_height_rate=r_y, resize_width_rate=1,
                                      move_x=0, move_y=0)
        vec_array_3.append(processed)

    #unlike resize rate, we only use the first frame to decide movement direction
    mv_x = std_x - vec_array_3[0]['bodies']['candidate'][1][0]
    mv_y = std_y - vec_array_3[0]['bodies']['candidate'][1][1]
    
    #move mp and DW
    vec_array_4 = list()
    for i in range(len(vec_array_3)):
        processed = process_keypoints(vec_array_3[i], 
                                      resize_height_rate=1, resize_width_rate=1,
                                      move_x=mv_x, move_y=mv_y)
        vec_array_4.append(processed)

    #only change face and hands now: if the face is too narrow or two wide, change it, too long, change it
    #hands follow face re-size factor.
    max_d_ear = 0.
    for i in range(len(vec_array_4)):
        x1 = vec_array_4[i]['bodies']['candidate'][16][0]
        x2 = vec_array_4[i]['bodies']['candidate'][17][0]
        distance = abs(x1 - x2)
        max_d_ear = max(distance, max_d_ear)

    r_x_ear = std_ear_x/max_d_ear

    #we also need to record the nose point before resize, so that we can move back after resizing
    #also need to record hand wrist point for moving back
    nose_x_list, nose_y_list = list(), list()
    for i in range(len(vec_array_4)):
        nose_x = vec_array_4[i]['bodies']['candidate'][0][0]
        nose_y = vec_array_4[i]['bodies']['candidate'][0][1]
        
        nose_x_list.append(nose_x)
        nose_y_list.append(nose_y)

    #resize mp and DW for ear (face) width
    vec_array_5 = list()
    for i in range(len(vec_array_4)):
        processed = process_keypoints_with_mask(vec_array_4[i], 
                                                {'pose_mp':[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
                                                 'bodies':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                                                 'left_hand_DW':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                                 'right_hand_DW':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                                 'left_hand_mp':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                                 'right_hand_mp':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                                 'left_hand_holistic_mp':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                                 'right_hand_holistic_mp':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]},
                                                resize_height_rate=1, resize_width_rate=r_x_ear,
                                                move_x=0, move_y=0)
        vec_array_5.append(processed)

    #move back to original nose location
    nose_x_new_list, nose_y_new_list = list(), list()
    for i in range(len(vec_array_5)):
        nose_x_new = vec_array_5[i]['bodies']['candidate'][0][0]
        nose_y_new = vec_array_5[i]['bodies']['candidate'][0][1]

        nose_x_new_list.append(nose_x_new)
        nose_y_new_list.append(nose_y_new)

    vec_array_6 = list()
    for i in range(len(vec_array_5)):

        mv_x_new = nose_x_list[i] - nose_x_new_list[i]
        mv_y_new = nose_y_list[i] - nose_y_new_list[i]

        processed = process_keypoints_with_mask(vec_array_5[i], 
                                                {'pose_mp':[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
                                                 'bodies':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                                                 'left_hand_DW':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                                 'right_hand_DW':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                                 'left_hand_mp':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                                 'right_hand_mp':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                                 'left_hand_holistic_mp':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                                 'right_hand_holistic_mp':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]},
                                                resize_height_rate=1, resize_width_rate=1,
                                                move_x=mv_x_new, move_y=mv_y_new)
        vec_array_6.append(processed)

    #also, we believe that r_x_ear (ear distance) somehow reflects how large hands and arms should be sized
    #So, we will use it to resize left arm (lower and upper), right arm (lower and upper), left hand and right hand
    #record sholder x and y first
    l_position_list = list()
    for i in range(len(vec_array_6)):
        l_x = vec_array_6[i]['bodies']['candidate'][5][0]
        l_y = vec_array_6[i]['bodies']['candidate'][5][1]

        l_position_list.append([l_x, l_y])

    #resize left arm and left hand
    vec_array_7 = list()
    for i in range(len(vec_array_6)):

        face_mp_mask = [j for j in range( int(len(vec_array_6[0]['face_mp'])/3) )]
        face_DW_mask = [j for j in range(68)]

        processed = process_keypoints_with_mask(vec_array_6[i], 
                                                {'pose_mp':[0,1,2,3,4,5,6,7,8,9,10, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
                                                 'bodies':[0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                                                 'faces':face_DW_mask,
                                                 'face_mp':face_mp_mask,
                                                 'right_hand_DW':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                                 'right_hand_mp':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                                 'right_hand_holistic_mp':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]},
                                                resize_height_rate=r_x_ear, resize_width_rate=r_x_ear,
                                                move_x=0, move_y=0)
        vec_array_7.append(processed)

    l_position_new_list = list()
    for i in range(len(vec_array_7)):
        l_new_x = vec_array_7[i]['bodies']['candidate'][5][0]
        l_new_y = vec_array_7[i]['bodies']['candidate'][5][1]

        l_position_new_list.append([l_new_x, l_new_y])

    #move back to original
    vec_array_8 = list()
    for i in range(len(vec_array_7)):

        face_mp_mask = [j for j in range( int(len(vec_array_7[0]['face_mp'])/3) )]
        face_DW_mask = [j for j in range(68)]
        
        mv_x = l_position_list[i][0] - l_position_new_list[i][0]
        mv_y = l_position_list[i][1] - l_position_new_list[i][1]

        processed = process_keypoints_with_mask(vec_array_7[i], 
                                                {'pose_mp':[0,1,2,3,4,5,6,7,8,9,10, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
                                                 'bodies':[0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                                                 'faces':face_DW_mask,
                                                 'face_mp':face_mp_mask,
                                                 'right_hand_DW':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                                 'right_hand_mp':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                                 'right_hand_holistic_mp':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]},
                                                resize_height_rate=1, resize_width_rate=1,
                                                move_x=mv_x, move_y=mv_y)
        vec_array_8.append(processed)

    ################################
    ##work on right hand & arm
    r_position_list = list()
    for i in range(len(vec_array_8)):
        r_x = vec_array_8[i]['bodies']['candidate'][2][0]
        r_y = vec_array_8[i]['bodies']['candidate'][2][1]

        r_position_list.append([r_x, r_y])

    #resize left arm and left hand
    vec_array_9 = list()
    for i in range(len(vec_array_8)):

        face_mp_mask = [j for j in range( int(len(vec_array_8[0]['face_mp'])/3) )]
        face_DW_mask = [j for j in range(68)]

        processed = process_keypoints_with_mask(vec_array_8[i], 
                                                {'pose_mp':[0,1,2,3,4,5,6,7,8,9,10,11, 13, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
                                                 'bodies':[0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                                                 'faces':face_DW_mask,
                                                 'face_mp':face_mp_mask,
                                                 'left_hand_DW':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                                 'left_hand_mp':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                                 'left_hand_holistic_mp':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]},
                                                resize_height_rate=r_x_ear, resize_width_rate=r_x_ear,
                                                move_x=0, move_y=0)
        vec_array_9.append(processed)

    r_position_new_list = list()
    for i in range(len(vec_array_9)):
        r_new_x = vec_array_9[i]['bodies']['candidate'][2][0]
        r_new_y = vec_array_9[i]['bodies']['candidate'][2][1]

        r_position_new_list.append([r_new_x, r_new_y])

    #move back to original
    vec_array_10 = list()
    for i in range(len(vec_array_9)):

        face_mp_mask = [j for j in range( int(len(vec_array_9[0]['face_mp'])/3) )]
        face_DW_mask = [j for j in range(68)]
        
        mv_x = r_position_list[i][0] - r_position_new_list[i][0]
        mv_y = r_position_list[i][1] - r_position_new_list[i][1]

        processed = process_keypoints_with_mask(vec_array_9[i], 
                                                {'pose_mp':[0,1,2,3,4,5,6,7,8,9,10,11, 13, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
                                                 'bodies':[0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                                                 'faces':face_DW_mask,
                                                 'face_mp':face_mp_mask,
                                                 'left_hand_DW':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                                 'left_hand_mp':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                                 'left_hand_holistic_mp':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]},
                                                resize_height_rate=1, resize_width_rate=1,
                                                move_x=mv_x, move_y=mv_y)
        vec_array_10.append(processed)

    if num_insert_interpolation > 0:
        final = interpolate_dicts(vec_array_10, num_insert_interpolation)
    else:
        final = vec_array_10

    #save to file
    Dict_ = defaultdict()
    
    Dict_['keypoint'] = final

    if draw_video:

        mp_drawing = mp.solutions.drawing_utils
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 30  # Frames per second

        list_before = []
        list_after = []

        for i_0 in range(len(format_dict['keypoint'])):

            ept_image = np.ones((1024, 1024, 3), dtype='float32')
            ept_image = ept_image*255.

            pose_lm = array_to_landmarks(format_dict['keypoint'][i_0]['pose_mp'])
            face_lm = array_to_landmarks(format_dict['keypoint'][i_0]['face_mp'])
            lh_lm = array_to_landmarks(format_dict['keypoint'][i_0]['left_hand_mp'])
            rh_lm = array_to_landmarks(format_dict['keypoint'][i_0]['right_hand_mp'])
            
            out_image = append_keypoint_image(mp_drawing, ept_image, 
                                                pose_lm, 
                                                face_lm, 
                                                lh_lm, 
                                                rh_lm, 
                                                1024, draw_hands=True)
            
            list_before.append(out_image)

        for i_0 in range(len(final)):

            ept_image = np.ones((1024, 1024, 3), dtype='float32')
            ept_image = ept_image*255.

            pose_lm = array_to_landmarks(final[i_0]['pose_mp'])
            face_lm = array_to_landmarks(final[i_0]['face_mp'])
            lh_lm = array_to_landmarks(final[i_0]['left_hand_mp'])
            rh_lm = array_to_landmarks(final[i_0]['right_hand_mp'])
            
            out_image = append_keypoint_image(mp_drawing, ept_image, 
                                                pose_lm, 
                                                face_lm, 
                                                lh_lm, 
                                                rh_lm, 
                                                1024, draw_hands=True)
            
            list_after.append(out_image)

        video_writer_before = cv2.VideoWriter('./before.mp4', fourcc, fps, (1024, 1024))

        # Write each frame to the video
        for frame in list_before:
            video_writer_before.write(np.uint8(frame))  # Ensure the frame is in uint8 format

        # Release the VideoWriter
        video_writer_before.release()

        video_writer_after = cv2.VideoWriter('./after.mp4', fourcc, fps, (1024, 1024))

        # Write each frame to the video
        for frame in list_after:
            video_writer_after.write(np.uint8(frame))  # Ensure the frame is in uint8 format

        # Release the VideoWriter
        video_writer_after.release()


    Dict_['visibility'] = format_dict['visibility']
    Dict_['frame_id'] = format_dict['frame_id']

    reverted_data = revert_reformat(Dict_)
    
    #by doing 2*ID, the name of the extracted kp will be 0.pickle, 2.pickle, ....
    #then, interpolation pickle will be like 1.pickle, 3.pickle, ....
    return(reverted_data)


def clean_and_create_folder(folder_path):
    if os.path.exists(folder_path):
        # Remove all contents inside the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    else:
        # Create the folder if it doesn't exist
        os.makedirs(folder_path)


##############################
'''
To better suit for the HT ComfyUI pipeline, I will further interpolate the movement
between each real frames.
That is, between frame i and i+1, I will interpolate the keypoint movement by n frames
Then, the n interpolated frames will be inserted between i and i+1
In this case, the movement will be smoother.
We simply use cubic spline linear interpolation.
'''

#function to interpolate w.r.t 1D and 2D and 3D kp format
def cubic_spline_interpolation(keypoint1, keypoint2, num_interpolation_frames):
    keypoint1 = np.array(keypoint1)
    keypoint2 = np.array(keypoint2)

    # Ensure the input shapes are the same
    if keypoint1.shape != keypoint2.shape:
        raise ValueError("keypoint1 and keypoint2 must have the same shape.")

    # Create time points for interpolation
    t = np.linspace(0, 1, num=num_interpolation_frames + 2)

    # Initialize the list to store interpolated keypoints
    interpolated_keypoints = []

    # Handle 1D keypoints (e.g., (x1, y1, z1, ..., xn, yn, zn))
    if keypoint1.ndim == 1:
        for dim in range(keypoint1.shape[0]):
            if keypoint1[dim] == 0 or keypoint2[dim] == 0:
                # If either value in the dimension is zero, set all interpolated values for that dimension to zero
                interpolated_values = np.zeros_like(t)[1:-1]
            else:
                # Perform cubic spline interpolation
                cubic_spline = CubicSpline([0, 1], [keypoint1[dim], keypoint2[dim]])
                interpolated_values = cubic_spline(t)[1:-1]
            interpolated_keypoints.append(interpolated_values)

        # Convert to list of 1D arrays
        interpolated_keypoints = np.array(interpolated_keypoints).T
        return [frame.tolist() for frame in interpolated_keypoints]

    # Handle 2D keypoints (e.g., (n, 2))
    elif keypoint1.ndim == 2:
        n, _ = keypoint1.shape
        for i in range(n):
            interpolated_node = []
            for dim in range(2):  # x and y
                if (keypoint1[i, 0] == -1 and keypoint1[i, 1] == -1) or (keypoint2[i, 0] == -1 and keypoint2[i, 1] == -1):
                    # If both x and y are -1, set all interpolated values for this node to [-1, -1]
                    interpolated_values = np.full((num_interpolation_frames,), -1)
                elif keypoint1[i, dim] == 0 or keypoint2[i, dim] == 0:
                    # If either value in the dimension is zero, set all interpolated values for that dimension to zero
                    interpolated_values = np.zeros_like(t)[1:-1]
                else:
                    # Perform cubic spline interpolation
                    cubic_spline = CubicSpline([0, 1], [keypoint1[i, dim], keypoint2[i, dim]])
                    interpolated_values = cubic_spline(t)[1:-1]
                interpolated_node.append(interpolated_values)
            # Combine x and y for each node
            interpolated_node = np.array(interpolated_node).T
            interpolated_keypoints.append(interpolated_node)

        # Combine all nodes and return as a list of 2D arrays
        interpolated_keypoints = np.array(interpolated_keypoints).transpose(1, 0, 2)
        return [frame.tolist() for frame in interpolated_keypoints]

    # Handle 3D keypoints (e.g., (k, n, 2))
    elif keypoint1.ndim == 3:
        k, n, _ = keypoint1.shape
        for i in range(k):
            interpolated_part = []
            for j in range(n):
                interpolated_node = []
                for dim in range(2):  # x and y
                    if (keypoint1[i, j, 0] == -1 and keypoint1[i, j, 1] == -1) or (keypoint2[i, j, 0] == -1 and keypoint2[i, j, 1] == -1):
                        # If both x and y are -1, set all interpolated values for this node to [-1, -1]
                        interpolated_values = np.full((num_interpolation_frames,), -1)
                    elif keypoint1[i, j, dim] == 0 or keypoint2[i, j, dim] == 0:
                        # If either value in the dimension is zero, set all interpolated values for that dimension to zero
                        interpolated_values = np.zeros_like(t)[1:-1]
                    else:
                        # Perform cubic spline interpolation
                        cubic_spline = CubicSpline([0, 1], [keypoint1[i, j, dim], keypoint2[i, j, dim]])
                        interpolated_values = cubic_spline(t)[1:-1]
                    interpolated_node.append(interpolated_values)
                # Combine x and y for each node
                interpolated_node = np.array(interpolated_node).T
                interpolated_part.append(interpolated_node)
            # Combine all nodes for each keypoint part
            interpolated_part = np.array(interpolated_part).transpose(1, 0, 2)
            interpolated_keypoints.append(interpolated_part)

        # Combine all keypoint parts and return as a list of 3D arrays
        interpolated_keypoints = np.array(interpolated_keypoints).transpose(1, 0, 2, 3)
        return [frame.tolist() for frame in interpolated_keypoints]

    else:
        raise ValueError("Unsupported keypoint format. Must be 1D, 2D, or 3D.")


#interpolate bween consecutive dicts
def interpolate_dicts(dict_list, num_interpolation_frames):
    def interpolate_keypoints(keypoint1, keypoint2, num_frames):
        """Helper function to interpolate keypoints using cubic spline."""
        if keypoint1 is None or keypoint2 is None:
            return [keypoint1] * num_frames  # If keypoints are missing, replicate the first keypoint
        return cubic_spline_interpolation(keypoint1, keypoint2, num_frames)

    # Initialize the final output list
    interpolated_list = []

    # Iterate through the list of dictionaries
    for i in tqdm(range(len(dict_list) - 1), desc='insert interpolating'):
        dict_i = dict_list[i]
        dict_i_plus_1 = dict_list[i + 1]

        # Add the current dictionary to the output
        interpolated_list.append(dict_i)

        # Create interpolated dictionaries
        for frame_idx in range(1, num_interpolation_frames + 1):
            interpolated_dict = {}

            for key in dict_i.keys():
                if key == 'bodies':
                    # Interpolate 'candidate' keypoints
                    candidate1 = dict_i[key]['candidate']
                    candidate2 = dict_i_plus_1[key]['candidate']
                    interpolated_candidate = interpolate_keypoints(candidate1, candidate2, num_interpolation_frames)

                    # Copy 'subset' directly from the first dictionary
                    subset = dict_i[key]['subset']

                    # Add to the interpolated dictionary
                    interpolated_dict[key] = {
                        'candidate': np.array(interpolated_candidate[frame_idx - 1]),  # Convert to NumPy array
                        'subset': subset  # Copy as is
                    }

                elif key == 'hands' or key == 'faces':
                    # Interpolate 2D keypoints
                    keypoints1 = dict_i[key]
                    keypoints2 = dict_i_plus_1[key]
                    interpolated_keypoints = interpolate_keypoints(keypoints1, keypoints2, num_interpolation_frames)
                    interpolated_dict[key] = np.array(interpolated_keypoints[frame_idx - 1])  # Convert to NumPy array

                else:
                    # Interpolate 1D keypoints
                    keypoints1 = dict_i[key]
                    keypoints2 = dict_i_plus_1[key]
                    interpolated_keypoints = interpolate_keypoints(keypoints1, keypoints2, num_interpolation_frames)
                    interpolated_dict[key] = np.array(interpolated_keypoints[frame_idx - 1])  # Convert to NumPy array

            # Add the interpolated dictionary to the output
            interpolated_list.append(interpolated_dict)

    # Add the last dictionary to the output
    interpolated_list.append(dict_list[-1])

    return interpolated_list



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



##############################
#########the class############
class KeyPointNormalization:
    
    def __init__(self):
        self.hello = 'hello'

    def run(self, numpy_array_in, num_insert_interpolation=0, draw_video=False):
        normalized = do_normalization(numpy_array_in, num_insert_interpolation, draw_video)
        return(normalized)






