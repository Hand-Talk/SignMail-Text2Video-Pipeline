#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:01:24 2024

@author: czhang
"""


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


'''
One issue we should pay extra attention:
face_mp (from FaceMesh of mediapipe) is not stable. The zero face rate is nearly 40%. 
But the face_holistice_mp is okay, with zero rate less than 0.1% 
'''


##################################
####original functions############

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


def cubic_spline_interpolation(keypoint1, keypoint2, num_interpolation_frames):
    keypoint1 = np.array(keypoint1)
    keypoint2 = np.array(keypoint2)

    # Create time points for interpolation. Note: linspace should include the endpoints 
    # but we'll discard them in the result
    t = np.linspace(0, 1, num=num_interpolation_frames + 2)

    interpolated_keypoints = []
    for dim in range(keypoint1.shape[0]):
        # If either value in the dimension is zero, set all interpolated values for that dimension to zero
        if (keypoint1[dim] in {0,-1}) or (keypoint2[dim] in {0,-1}):
            interpolated_keypoints.append(np.zeros_like(t)[1:-1])
        else:
            # The interpolator should only be provided with the start and end times (0 and 1) 
            # and the start and end values (keypoint1[dim] and keypoint2[dim]).
            cubic_spline = CubicSpline([0, 1], [keypoint1[dim], keypoint2[dim]])
            interpolated_keypoints.append(cubic_spline(t)[1:-1])

    interpolated_keypoints = np.array(interpolated_keypoints).T

    return interpolated_keypoints


'''
Given two segments [(x1,y1,z1), (x2,y2,z2)] and [(x3,y3,z3), (x4,y4,z4)], 
first, please move (x3,y3,z3) to (x1,y1,z1) without changing the direction of 
[(x3,y3,z3), (x4,y4,z4)]. 
Then, the two segments are within the same plane, with the first point coincide. 

Then, you may want to rotate [(x1,y1,z1), (x2,y2,z2)] 
towards the moved second segment in the plane by a ratio of the angle between them. 
For instance, if I say the ratio is 0.2, then [(x1,y1,z1), (x2,y2,z2)] 
will be rotate 20% towards the moved second segment. 
Finally, please return such rotated first segment in the function.

One more update: Suppose the length of segment 1 and segment 2 is different. 
Then, given the ratio, I want not only rotate segment 1 accordingly, 
but also longer or shorter segment 1 accordingly. 
That is, if the ratio is 0.2, len_seg_1=1 and len_seg_2=2. 
Then after rotate 20%, the length of segment 1 should also en-longered by (2-1)*0.2. 
That is, make sure 20% of the length difference is mitigated.
'''

def safe_acos(x):
    # Clamp the input to the range [-1, 1] to ensure the domain is valid for acos
    return np.arccos(np.clip(x, -1, 1))

def rotate_and_scale_segment(segment1, segment2, ratio):
    (x1, y1, z1), (x2, y2, z2) = segment1
    (x3, y3, z3), (x4, y4, z4) = segment2
    
    # Translate the endpoint of segment2 to start from the start point of segment1
    x4_new = x4 - x3 + x1
    y4_new = y4 - y3 + y1
    z4_new = z4 - z3 + z1
    
    # Vectors for the segments
    v1 = np.array([x2 - x1, y2 - y1, z2 - z1])
    v2 = np.array([x4_new - x1, y4_new - y1, z4_new - z1])
    
    # Lengths of the vectors
    len_v1 = np.linalg.norm(v1)
    len_v2 = np.linalg.norm(v2)
    
    if len_v1 == 0 or len_v2 == 0:
        # If either vector length is zero, return the original segment scaled by ratio
        new_length = len_v1 + ratio * (len_v2 - len_v1)
        scaled_vector = v1 * (new_length / len_v1) if len_v1 != 0 else v1
        scaled_endpoint = scaled_vector + np.array([x1, y1, z1])
        return [(x1, y1, z1), tuple(scaled_endpoint.tolist())]

    # Compute the cross product (rotation axis) and angle
    axis = np.cross(v1, v2)
    angle = safe_acos(np.dot(v1, v2) / (len_v1 * len_v2))
    
    # Scale the angle by the ratio
    angle *= ratio
    
    # Rodrigues' rotation formula for rotation around a unit vector
    axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) != 0 else axis
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    cross_matrix = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    # Compute the rotated vector
    rotated_vector = (cos_angle * v1) + (sin_angle * np.cross(axis, v1)) + ((1 - cos_angle) * np.dot(v1, axis) * axis)
    
    # Scale the rotated vector by the length ratio
    length_difference = len_v2 - len_v1
    new_length = len_v1 + ratio * length_difference
    scaled_vector = rotated_vector * (new_length / np.linalg.norm(rotated_vector)) if np.linalg.norm(rotated_vector) != 0 else rotated_vector
    rotated_endpoint = scaled_vector + np.array([x1, y1, z1])
    
    return [(x1, y1, z1), tuple(rotated_endpoint.tolist())]


#function to move segment without changing its direction.
def translate_segment(seg, tip):
    # Unpack the segment and the new start tip
    (x1, y1, z1), (x2, y2, z2) = seg
    x3, y3, z3 = tip
    
    # Calculate the translation vector
    dx = x3 - x1
    dy = y3 - y1
    dz = z3 - z1
    
    # Calculate the new coordinates for the endpoint
    x4 = x2 + dx
    y4 = y2 + dy
    z4 = z2 + dz
    
    # Create the new segment
    out_seg = [(x3, y3, z3), (x4, y4, z4)]
    return out_seg


def interpolate_triangles(triangle1, triangle2, t):
    # Triangle vertices
    p1_1, p1_2, p1_3 = np.array(triangle1)
    p2_1, p2_2, p2_3 = np.array(triangle2)

    # 1. Calculate translation vector for first vertices
    translation = p2_1 - p1_1

    # Calculate normals of both triangles
    normal1 = np.cross(p1_2 - p1_1, p1_3 - p1_1)
    normal2 = np.cross(p2_2 - p2_1, p2_3 - p2_1)

    # 2. Calculate rotation to align normals
    axis = np.cross(normal1, normal2)
    if np.linalg.norm(axis) != 0:
        axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.dot(normal1, normal2) / (np.linalg.norm(normal1) * np.linalg.norm(normal2)))

    # Create rotation matrix
    from scipy.spatial.transform import Rotation as R
    rotation = R.from_rotvec(axis * angle)

    # 3. Interpolate translation and rotation
    interpolated_translation = p1_1 + translation * t
    interpolated_rotation = R.from_rotvec(axis * angle * t).as_matrix()

    # Apply translation and rotation to triangle1 vertices
    interpolated_triangle = interpolated_rotation.dot((np.array(triangle1).T - p1_1[:, None])) + interpolated_translation[:, None]

    return interpolated_triangle.T


'''
Given two 3D points (x1,y1,z1) and (x2,y2,z2), please interpolate them linearly.
'''
def interpolate_points(p1, p2, t):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    
    # Calculate the interpolated point
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    z = z1 + t * (z2 - z1)
    
    return (x, y, z)


def interpolate_points_series(p1, p2, n):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    
    # List to hold all interpolated points
    interpolated_points = []

    
    '''updated to consider 0 or -1'''
    # If p1 is zero, fill the series with zeros except for the last element
    if p1 == (0, 0, 0) or (p1[0] == -1 and p1[1] == -1):
        if p1 == (0, 0, 0):
            interpolated_points = [(0, 0, 0)] * (n-1) + [p2]
        else:
            interpolated_points = [(-1, -1, -1)] * (n-1) + [p2]
    # If p2 is zero, fill the series with zeros except for the first element
    elif p2 == (0, 0, 0) or (p2[0] == -1 and p2[1] == -1):
        if p2 == (0, 0, 0):
            interpolated_points = [p1] + [(0, 0, 0)] * (n-1)
        else:
            interpolated_points = [p1] + [(-1, -1, -1)] * (n-1)
    else:
        # Add the starting point
        interpolated_points.append((x1, y1, z1))

        # Generate n-2 interpolated points
        for i in range(1, n-1):
            t = i / (n-1)  # Calculate the interpolation parameter
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            z = z1 + t * (z2 - z1)
            interpolated_points.append((x, y, z))
        
        # Add the endpoint
        interpolated_points.append((x2, y2, z2))
    
    return interpolated_points


#given the upper point list containing (x0,y0,z0), (x1,y1,z1), (x2,y2,z2), ... (xn,yn,zn)
#which is the interpolated points for the upper point from previous step,
#the function will interpolate from down 1 to down 2 accordingly.

#the interpolation will take rotation and length into consideration:
#the angle between two segments: [(x0,y0,z0), down_0], [(xn,yn,zn), down_1]
#will be linearly interpolated.
#the length different between these two segments will be linearly interpolated
#the interpolation will be moved along (x0,y0,z0), (x1,y1,z1), (x2,y2,z2), ...
#functions will return the down list (end point list for the next level)
'''our interpolation always includes begining and end original points'''
def forward_kinematics(up_list, down_0, down_1):
    
    '''updated to consider 0 or -1'''
    # Special case: when up list is invalid, the down points are interpolated linearly
    if up_list[0] == (0, 0, 0) or (up_list[0][0] == -1 and up_list[0][1] == -1):
        return( interpolate_points_series(down_0, down_1, len(up_list)) )
            
    if up_list[-1] == (0, 0, 0) or (up_list[-1][0] == -1 and up_list[-1][1] == -1):
        return( interpolate_points_series(down_0, down_1, len(up_list)) )
    
    
    '''updated to consider 0 or -1'''
    p1 = down_0
    p2 = down_1
    n = len(up_list)
    # If p1 is zero, fill the series with zeros except for the last element
    if p1 == (0, 0, 0) or (p1[0] == -1 and p1[1] == -1):
        
        if p1 == (0, 0, 0):
            interpolated_points = [(0, 0, 0)] * (n-1) + [p2]
            return(interpolated_points)
        else:
            interpolated_points = [(-1, -1, -1)] * (n-1) + [p2]
            return(interpolated_points)
            
    # If p2 is zero, fill the series with zeros except for the first element
    elif p2 == (0, 0, 0) or (p2[0] == -1 and p2[1] == -1):
        
        if p2 == (0, 0, 0):
            interpolated_points = [p1] + [(0, 0, 0)] * (n-1)
            return(interpolated_points)
        else:
            interpolated_points = [p1] + [(-1, -1, -1)] * (n-1)
            return(interpolated_points)
    
    
    #compute the ratio: len(up_list) points, then should be len(up_list)-1 in-between, 
    #hence, should be len(up_list)-2 interpolations. Anyway, the ratio should be 1/(len(up_list)-1)
    ratio = float( 1. / float( len(up_list)-1 ) )
    
    down_list = list()
    
    #first append the begining down point, no need to interpolate
    down_list.append(down_0)
    
    seg_1 = [up_list[0],down_0]
    seg_2 = [up_list[-1],down_1]
    
    #start interpolation:
    for i in range(1, len(up_list)-1):
        
        #do the rotation and length adjust
        int_seg = rotate_and_scale_segment(seg_1, seg_2, float(i)*ratio)
        
        #move the the starting tip
        moved_seg = translate_segment(int_seg, up_list[i])
        
        #append the interpolated down point
        down_list.append(moved_seg[1])
    
    #append the ending down point, no need to interpolation
    down_list.append(down_1)
    
    return(down_list)


'''
Me: Suppose I have a plane in the 3D space, which is located by a triangle, 
or the three tips of the triangle: [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)]. 
Please obtain the normal direction using the right hand rule following point 1 to 2 to 3. 
That is, using the right hand with four fingers along the direction 
(x1, y1, z1) --> (x2, y2, z2) --> (x3, y3, z3), 
the thumb should point to the normal direction.
--------------------------
ChatGPT: To find the normal vector of a plane defined by three points in 3D space using 
the right-hand rule, you need to compute the cross product of two vectors that lie on the plane. 
Specifically, these vectors can be formed using the points in the order provided, 
from point 1 to point 2 and then from point 2 to point 3. 
The cross product of these vectors will give you the normal vector pointing in the direction 
determined by the right-hand rule.

Here’s a Python function:
'''
def normal_vector_triangle(triangle):
    # Unpack the triangle points
    (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = triangle

    # Vector from point 1 to point 2
    vector1 = (x2 - x1, y2 - y1, z2 - z1)
    # Vector from point 2 to point 3
    vector2 = (x3 - x2, y3 - y2, z3 - z2)

    # Cross product of vector1 and vector2
    i = vector1[1] * vector2[2] - vector1[2] * vector2[1]
    j = vector1[2] * vector2[0] - vector1[0] * vector2[2]
    k = vector1[0] * vector2[1] - vector1[1] * vector2[0]

    # Normal vector
    normal = (i, j, k)
    return normal


'''
Function to calculate the angle between the two normal direction. 
That is, I will give triangle 1 as [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)]. 
I will give triangle 2 as [(x4, y4, z4), (x5, y5, z5), (x6, y6, z6)]. 
Then, you may calculate the normal direction following right hand rule 
with direction 1 --> 2 --> 3; 4 --> 5 --> 6. 
Then, please return the angle between the two normal directions.
'''

def angle_between_normals(triangle1, triangle2):
    # Calculate normals
    normal1 = normal_vector_triangle(triangle1)
    normal2 = normal_vector_triangle(triangle2)

    # Convert to vectors for dot product calculation
    norm1 = math.sqrt(sum(n**2 for n in normal1))
    norm2 = math.sqrt(sum(n**2 for n in normal2))
    
    # Add a small value to the norms to avoid division by zero
    eps = 1e-10
    norm1 = max(norm1, eps)
    norm2 = max(norm2, eps)
    
    # Normalize vectors
    normal1 = tuple(n / norm1 for n in normal1)
    normal2 = tuple(n / norm2 for n in normal2)

    # Dot product
    dot_product = sum(n1 * n2 for n1, n2 in zip(normal1, normal2))

    # Clamp dot_product to avoid numerical issues
    dot_product = max(min(dot_product, 1.0), -1.0)

    # Calculate angle (in radians)
    angle_rad = math.acos(dot_product)

    # Convert to degrees
    angle_deg = math.degrees(angle_rad)
    return angle_deg


'''
This looks great. Now, one more update: 
That is, given a list of 3D points [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3), ... (xn,yn,zn)], 
we need to rotate them along the normal direction of the plane formed by 
seg_1 = [(x1,y1,z1),(x2,y2,z2)]
seg_2 = [(x1,y1,z1),(x3,y3,z3)]
The rotation should make seg_1 rotate towards seg_2 by angle theta. 
Please update the function accordingly. The input is points, seg_1, seg_2 and theta.
-----------------------
I'll create a combined function that first calculates the normal vector to 
the plane defined by two segments, 
and then uses this normal vector as the axis of rotation to rotate a list of 3D points 
by a specified angle around that axis. 
Here's the updated function:
'''

def normal_vector(seg_1, seg_2):
    """
    Helper function to calculate the normal vector to the plane defined by two segments.
    """
    p1 = np.array(seg_1[0])  # Common start point
    p2 = np.array(seg_1[1])  # End point of first segment
    p3 = np.array(seg_2[1])  # End point of second segment
    
    vector1 = p2 - p1
    vector2 = p3 - p1
    
    return np.cross(vector1, vector2)

def rotate_points_around_normal(points, seg_1, seg_2, theta):
    """
    Rotate a list of 3D points around the normal direction of the plane defined by seg_1 and seg_2.
    
    Parameters:
    - points: List of tuples representing the points [(x1, y1, z1), ..., (xn, yn, zn)]
    - seg_1: A list containing two tuples, representing the first segment.
    - seg_2: A list containing two tuples, representing the second segment.
    - theta: Rotation angle in degrees.
    
    Returns:
    - List of tuples representing the rotated points.
    """
    # Calculate the normal vector from the segments
    direction = normal_vector(seg_1, seg_2)
    
    # Anchor point from the first segment's start point
    anchor = np.array(seg_1[0])
    
    # Convert degrees to radians
    theta_rad = np.radians(theta)
    
    # Add a small value to the norm to avoid division by zero
    norm = np.linalg.norm(direction)
    eps = 1e-10
    norm = max(norm, eps)
    
    # Normalize the direction vector
    axis = direction / norm
    
    # Cosine and sine calculations
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    
    # Rodrigues' rotation formula components
    cross_prod_matrix = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    rotation_matrix = cos_theta * np.eye(3) + sin_theta * cross_prod_matrix + (1 - cos_theta) * np.outer(axis, axis)
    
    rotated_points = []
    for point in points:
        point_array = np.array(point)
        point_relative = point_array - anchor
        rotated_point = np.dot(rotation_matrix, point_relative) + anchor
        rotated_points.append(tuple(rotated_point))
    
    return rotated_points


'''
Rotate the points according to the given line 
(that is, the normal direction and the anchor point is directly given). 
The rotation is by angle theta. The direction is formed by two points: 
Rotate so that point 1 will move toward point 2.

That is, we have the input list of 3D points: 
[(x1,y1,z1), (x2,y2,z2), (x3,y3,z3), ... (xn,yn,zn)], 
the anchor point (x,y,z) and the normal direction (x_l,y_l,z_l). 
The 3D points will be rotated by theta degree according to the axis, 
which is the line pointing toward (x_l,y_l,z_l) and passing through (x,y,z). 
The rotation direction is so that point 1 (xa, ya, za) will move toward (xb,yb,zb).

To be specific, it has nothing to do with hand rule this time: 
The axis is given, and the rotation direction is given. 
You do NOT need to make sure the rotation direction forms a right hand rule 
with the normal direction (x_l,y_l,z_l) ! 
Just make sure the rotation is making point 1 move toward point 2.
'''

def rotate_points(points, anchor, axis_direction, theta, point_a, point_b):
    """
    Rotate a list of 3D points around a specified axis by a given angle,
    ensuring the rotation moves point_a towards point_b.

    Parameters:
    - points: List of tuples representing the points [(x1, y1, z1), ..., (xn, yn, zn)]
    - anchor: Tuple (x, y, z) representing the anchor point of the rotation axis.
    - axis_direction: Tuple (x_l, y_l, z_l) representing the direction vector of the rotation axis.
    - theta: Rotation angle in degrees.
    - point_a: Tuple (xa, ya, za), starting point of the direction check.
    - point_b: Tuple (xb, yb, zb), target point of the direction check.

    Returns:
    - List of tuples representing the rotated points.
    """
    # Convert degrees to radians
    theta_rad = np.radians(theta)
    
    # Normalize the axis direction vector
    axis = np.array(axis_direction)
    norm = np.linalg.norm(axis)
    eps = 1e-10  # small value to prevent division by zero
    norm = max(norm, eps)
    axis = axis / norm
    
    # Direction vector from point_a to point_b
    direction = np.array(point_b) - np.array(point_a)
    
    # Cross product of axis and direction
    rotation_direction = np.cross(axis, direction)
    
    # Dot product to determine if the rotation should be reversed
    if np.dot(rotation_direction, axis) < 0:
        theta_rad = -theta_rad
    
    # Rodrigues' rotation formula components
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    cross_prod_matrix = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    rotation_matrix = cos_theta * np.eye(3) + sin_theta * cross_prod_matrix + (1 - cos_theta) * np.outer(axis, axis)
    
    # Rotate points
    anchor = np.array(anchor)
    rotated_points = []
    for point in points:
        point_array = np.array(point)
        point_relative = point_array - anchor
        rotated_point = np.dot(rotation_matrix, point_relative) + anchor
        rotated_points.append(tuple(rotated_point))
    
    return rotated_points


def translate_points(points, target):
    """
    Translate a list of 3D points such that the first point moves to the target location,
    and all other points move in parallel by the same vector.

    Parameters:
    - points: List of tuples representing the points [(x1, y1, z1), ..., (xn, yn, zn)]
    - target: Tuple (x0, y0, z0) representing the target location for the first point.

    Returns:
    - List of tuples representing the translated points.
    """
    if not points:
        return []

    # Calculate the translation vector
    translation_vector = [t - p for t, p in zip(target, points[0])]

    # Apply the translation vector to all points
    translated_points = [(x + translation_vector[0], y + translation_vector[1], z + translation_vector[2]) for (x, y, z) in points]

    return translated_points


def angle_between_segments(seg_1, seg_2):
    """
    Calculate the angle between two 3D segments that share a common starting point.
    
    Parameters:
    - seg_1: List containing two tuples, representing the first segment [(x1, y1, z1), (x2, y2, z2)]
    - seg_2: List containing two tuples, representing the second segment [(x1, y1, z1), (x3, y3, z3)]
    
    Returns:
    - The angle between the two segments in degrees.
    """
    # Convert segment points to numpy arrays
    p1 = np.array(seg_1[0])
    p2 = np.array(seg_1[1])
    p3 = np.array(seg_2[1])
    
    # Create vectors from the common start point to the other points
    vector1 = p2 - p1
    vector2 = p3 - p1
    
    # Calculate the dot product of vectors
    dot_product = np.dot(vector1, vector2)
    
    # Calculate the norms of each vector
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    
    # Add a small value to the norms to prevent division by zero
    eps = 1e-10  # small value to prevent division by zero
    norm1 = max(norm1, eps)
    norm2 = max(norm2, eps)
    
    # Calculate the cosine of the angle using the dot product and norms
    cos_theta = dot_product / (norm1 * norm2)
    
    # Handling potential numerical issues that might push the value slightly out of range
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Calculate the angle in radians and then convert to degrees
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)
    
    return theta_deg


def array_to_tuples(arr):
    """
    Convert a 1D numpy array with length 3*n into a list of tuples.
    Each tuple contains three elements, representing the 3D coordinates of a point.
    
    Parameters:
    - arr: 1D numpy array with length 3*n
    
    Returns:
    - List of tuples representing the points.
    """
    # Ensure the array has a length that is a multiple of 3
    if len(arr) % 3 != 0:
        raise ValueError("The length of the array must be a multiple of 3.")

    # Reshape the array to (-1, 3), where -1 infers the necessary number of rows based on the length of the array
    reshaped_array = arr.reshape(-1, 3)
    
    # Convert the reshaped array into a list of tuples
    tuple_list = [tuple(point) for point in reshaped_array]

    return tuple_list


def tuples_to_array(tuple_list):
    """
    Convert a list of tuples into a 1D numpy array.
    Each tuple is assumed to contain three elements, representing the 3D coordinates of a point.
    
    Parameters:
    - tuple_list: List of tuples, where each tuple contains three elements.
    
    Returns:
    - 1D numpy array constructed from the list of tuples.
    """
    # Flatten the list of tuples
    flat_list = [element for tup in tuple_list for element in tup]
    
    # Convert the flattened list to a numpy array
    arr = np.array(flat_list)

    return arr


def add_tuples(tup1, tup2):
    """Add two tuples element-wise."""
    return tuple(a + b for a, b in zip(tup1, tup2))

def subtract_tuples(tup1, tup2):
    """Subtract two tuples element-wise."""
    return tuple(a - b for a, b in zip(tup1, tup2))


'''
Suppose I form a line from point c to point d. 
That is, the line pass through point c and d, and point into the c --> d direction.

Then, suppose I have another two points, a and b. I can form two triangles: a,c,d and b,c,d.
Then, I want to know the angle between the two triangle planes. 
'''

def normal_vector_triangle_new(p1, p2, p3):
    """
    Calculate the normal vector of the triangle defined by points p1, p2, and p3.
    """
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    return np.cross(v1, v2)

def angle_and_rotation_direction(c, d, a, b):
    """
    Calculate the left-hand and right-hand rotation angles from the plane of triangle a,c,d to the plane of triangle b,c,d.

    Parameters:
    - c, d: Points defining the direction (thumb direction using right-hand rule).
    - a, b: Points forming triangles with c and d.
    
    Returns:
    - angle_left: The angle for the left-hand rotation in degrees.
    - angle_right: The angle for the right-hand rotation in degrees.
    """
    # Calculate normal vectors of the planes
    n_acd = normal_vector_triangle_new(c, d, a)
    n_bcd = normal_vector_triangle_new(c, d, b)

    # Normalize the normal vectors
    norm_acd = np.linalg.norm(n_acd)
    norm_bcd = np.linalg.norm(n_bcd)
    n_acd = n_acd / max(norm_acd, 1e-10)
    n_bcd = n_bcd / max(norm_bcd, 1e-10)

    # Calculate the angle between the normal vectors
    dot_product = np.dot(n_acd, n_bcd)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)

    # Determine the direction of rotation
    rotation_direction = np.dot(np.cross(n_acd, n_bcd), np.array(d) - np.array(c))

    # Define angles based on rotation direction
    if rotation_direction > 0:  # n_acd to n_bcd is counter-clockwise
        angle_left = 360 - angle_deg
        angle_right = angle_deg
    else:  # n_acd to n_bcd is clockwise
        angle_left = angle_deg
        angle_right = 360 - angle_deg

    return angle_left, angle_right


'''
Previously, I want to rotate the points according to the given line 
(that is, the normal direction and the anchor point is directly given). 
The rotation is by angle theta. The direction is formed by two points: 
Rotate so that point 1 will move toward point 2.

But now, I don't want to decide the direction by a --> b. 
Instead, I will just give 'left' or 'right' to the function. 
If 'right' is given, then the rotation is around the line by the right hand rule. 
Similarly, 'left' will trigger the rotation in the left hand rule.
'''

def rotate_points_by_direction(points, anchor, axis_direction, theta, direction):
    """
    Rotate a list of 3D points around a specified axis by a given angle,
    ensuring the rotation direction is either 'left' or 'right' according to the right-hand rule.

    Parameters:
    - points: List of tuples representing the points [(x1, y1, z1), ..., (xn, yn, zn)]
    - anchor: Tuple (x, y, z) representing the anchor point of the rotation axis.
    - axis_direction: Tuple (x_l, y_l, z_l) representing the direction vector of the rotation axis.
    - theta: Rotation angle in degrees.
    - direction: 'left' or 'right' indicating the direction of rotation according to the right-hand rule.

    Returns:
    - List of tuples representing the rotated points.
    """
    # Convert degrees to radians
    theta_rad = np.radians(theta)
    
    # Normalize the axis direction vector
    axis = np.array(axis_direction)
    norm = np.linalg.norm(axis)
    eps = 1e-10  # small value to prevent division by zero
    norm = max(norm, eps)
    axis = axis / norm

    # Check the rotation direction and adjust the sign of theta_rad accordingly
    if direction == 'left':
        theta_rad = -theta_rad  # For left-hand rule, we rotate in the opposite direction
    
    # Rodrigues' rotation formula components
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    cross_prod_matrix = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    rotation_matrix = cos_theta * np.eye(3) + sin_theta * cross_prod_matrix + (1 - cos_theta) * np.outer(axis, axis)
    
    # Rotate points
    anchor = np.array(anchor)
    rotated_points = []
    for point in points:
        point_array = np.array(point)
        point_relative = point_array - anchor
        rotated_point = np.dot(rotation_matrix, point_relative) + anchor
        rotated_points.append(tuple(rotated_point))
    
    return rotated_points


'''
Given a sequence of normal directions [(x1,y1,z1),....,(xn,yn,zn)] 
and another one normal direction (x,y,z). 
I want to know the average angle between each normal direction 
in the sequence and the one normal direction.
'''

def average_angle_with_normal(normals, single_normal):
    """
    Calculate the average angle between each normal in a list and a single normal vector.

    Parameters:
    - normals: List of tuples or lists representing normal vectors [(x1, y1, z1), ..., (xn, yn, zn)].
    - single_normal: A tuple or list representing a single normal vector (x, y, z).

    Returns:
    - Average angle in degrees.
    """
    # Convert lists to numpy arrays
    normals = np.array(normals)
    single_normal = np.array(single_normal)

    # Normalize the single normal vector
    single_normal = single_normal / np.linalg.norm(single_normal)

    # Normalize each vector in the list
    norms = np.linalg.norm(normals, axis=1)
    # Adding small epsilon to avoid division by zero
    normals_normalized = normals / (norms[:, np.newaxis] + 1e-10)

    # Calculate the dot products
    dot_products = np.dot(normals_normalized, single_normal)

    # Clip values to avoid numerical errors outside the domain of arccos
    dot_products = np.clip(dot_products, -1.0, 1.0)

    # Calculate angles in radians and then convert to degrees
    angles = np.degrees(np.arccos(dot_products))

    # Compute the average angle
    average_angle = np.mean(angles)

    return average_angle


#updated from the average_angle_with_normal
def max_angle_between_normals(normals1, normals2):
    """
    Calculate the maximum angle between corresponding pairs of normal vectors
    from two sequences using a for loop.

    Parameters:
    - normals1: List of tuples or lists representing normal vectors [(x1, y1, z1), ..., (xn, yn, zn)].
    - normals2: List of tuples or lists representing another sequence of normal vectors [(x1', y1', z1'), ..., (xn', yn', zn')].

    Returns:
    - Maximum angle between corresponding normal vectors in degrees.
    """
    # Convert lists to numpy arrays
    normals1 = np.array(normals1)
    normals2 = np.array(normals2)

    # Normalize the vectors in both lists
    norms1 = np.linalg.norm(normals1, axis=1)
    normals1_normalized = normals1 / (norms1[:, np.newaxis] + 1e-10)
    
    norms2 = np.linalg.norm(normals2, axis=1)
    normals2_normalized = normals2 / (norms2[:, np.newaxis] + 1e-10)

    # Initialize list to store angles
    angles = []

    # Calculate the dot products using a for loop
    for norm1, norm2 in zip(normals1_normalized, normals2_normalized):
        dot_product = np.dot(norm1, norm2)
        dot_product = np.clip(dot_product, -1.0, 1.0)  # Clip to avoid numerical issues
        angle = np.degrees(np.arccos(dot_product))
        angles.append(angle)

    # Find the maximum angle
    max_angle = max(angles)

    return max_angle


#interpolation of hands: rotat palm.
#both first_hand and second_hand are 1D array in 63 dim.

def rotate_hand_palm(first_hand, second_hand, out_chain, l_or_r_hand, triangles_input):
    
    n_interp = len(out_chain)
    
    ##change 1d array into list of tuples
    first_hand = array_to_tuples(first_hand)
    second_hand = array_to_tuples(second_hand)
    
    #list to hold normal direction of interpolation rotated in two different directions
    #note that first and last frame (original start/end) is not included
    normal_1, normal_2 = list(), list()
    
    ##output list##
    output_1, output_2 = list(), list()
    output_1.append(first_hand)
    output_2.append(first_hand)
    
    ##move second hand to first wrist, share the same wrist
    #but we still keep the original second hand
    second_hand_0 = translate_points(second_hand, first_hand[0])
    
    ##calculate the angle between two index fingers palm bones (0 to 5)
    seg_1 = [first_hand[0], first_hand[5]]
    seg_2 = [first_hand[0], second_hand_0[5]]
    angle = angle_between_segments(seg_1, seg_2)
    
    ##rotate first hand along the normal direction of seg_1, seg_2 plane (0 to 5 plane)
    #seg 1 rotate to seg 2
    first_hand_0 = rotate_points_around_normal(first_hand, seg_1, seg_2, angle)
    
    #calculate the angle between segond hand 0 and first hand 0 palm normal direction.
    '''
    angle_new is the angle difference after seg_1 rotated to seg_2,   
    So, the rotation along seg_1 should according to angle_new
    '''
    #triangle1 = [first_hand_0[0], first_hand_0[5], first_hand_0[17]]
    #triangle2 = [second_hand_0[0], second_hand_0[5], second_hand_0[17]]
    #angle_new = angle_between_normals(triangle1, triangle2)
    c_ = second_hand_0[0]
    d_ = second_hand_0[5]
    a_ = first_hand_0[17]
    b_ = second_hand_0[17]
    angle_left, angle_right = angle_and_rotation_direction(c_, d_, a_, b_)
    
    direction_1, direction_2 = 'left', 'right'
    
    #finally, we rotate the first hand 0 in both directions, the output will be used
    #for finger movement in the next function
    anchor_ = first_hand_0[0]
    axis_ = subtract_tuples(first_hand_0[5], first_hand_0[0])
    first_hand_rotate_1 = rotate_points_by_direction(first_hand_0, anchor_, axis_, 
                                                     angle_left, direction_1)
    first_hand_rotate_2 = rotate_points_by_direction(first_hand_0, anchor_, axis_, 
                                                     angle_right, direction_2)

    '''
    The direction may or may not be correct. So, we will rotate in both directions!
    Then, we select the one that consistent with elbow normal direction, 
    so that the hand will not rotate from outside the wrist.
    '''

    ############################################
    ##rotate the first_hand toward second_hand_0
    #in each step, first hand will rotate along seg_1 seg_2 normal to seg_1 by angle/n_interp
    #then, first hand will rotate along seg_1 by angle_new/n_interp
    
    angle_each = angle/float(n_interp-1)
    angle_left_each = angle_left/float(n_interp-1)
    angle_right_each = angle_right/float(n_interp-1)
    
    for i in range(1, n_interp-1):
        
        #rotate first hand from seg1 toward seg2 by i*angle_each
        first_hand_each = rotate_points_around_normal(first_hand, seg_1, seg_2, 
                                                      i*angle_each)
        
        ####################################
        #rotate first_hand along seg1 by the i*angle_new_each in both direction
        anchor = first_hand_each[0]
        axis = subtract_tuples(first_hand_each[5], first_hand_each[0])
        #first_hand_rotate_each = rotate_points(first_hand_each, anchor, axis, 
        #                                       i*angle_new_each, point_a_, point_b_)
        first_hand_rotate_each_1 = rotate_points_by_direction(first_hand_each, anchor, axis, 
                                                            i*angle_left_each, direction_1)
        
        #normal direction of first rotation
        #left hand, 0 --> 17 --> 5 with right hand rule, so normal direction point out from palm
        if l_or_r_hand == 'left':
            triangle_ = [first_hand_rotate_each_1[0], first_hand_rotate_each_1[17], first_hand_rotate_each_1[5]]
            normal_ = normal_vector_triangle(triangle_)
            normal_1.append(normal_)
        else:#right hand, 0-->5-->17
            triangle_ = [first_hand_rotate_each_1[0], first_hand_rotate_each_1[5], first_hand_rotate_each_1[17]]
            normal_ = normal_vector_triangle(triangle_)
            normal_1.append(normal_)            
        
        first_hand_rotate_each_2 = rotate_points_by_direction(first_hand_each, anchor, axis, 
                                                            i*angle_right_each, direction_2)
        
        #similarly, left hand
        if l_or_r_hand == 'left':
            triangle_ = [first_hand_rotate_each_2[0], first_hand_rotate_each_2[17], first_hand_rotate_each_2[5]]
            normal_ = normal_vector_triangle(triangle_)
            normal_2.append(normal_)
        else:#right hand
            triangle_ = [first_hand_rotate_each_2[0], first_hand_rotate_each_2[5], first_hand_rotate_each_2[17]]
            normal_ = normal_vector_triangle(triangle_)
            normal_2.append(normal_)   
        
        temp_1 = translate_points(first_hand_rotate_each_1, out_chain[i])
        output_1.append(temp_1)
        temp_2 = translate_points(first_hand_rotate_each_2, out_chain[i])
        output_2.append(temp_2)
        ##################################

    output_1.append(second_hand)
    output_2.append(second_hand)
    
    '''
    Since the hand may be replace by DW pose due to wrong normal direction,
    we also need this in the mediapipe hand rotate.
    '''
    #before return output, we will change the interpolated dim to [-1,-1]
    #if either the start or end corresponding dim is [-1, -1]
    for dim in range(len(first_hand)):
        if (first_hand[dim][0] == -1 and first_hand[dim][1] == -1) or (
            second_hand[dim][0] == -1 and second_hand[dim][1] == -1):
            
            first_hand_rotate_1[dim] = deepcopy((-1,-1,-1))
            first_hand_rotate_2[dim] = deepcopy((-1,-1,-1))
            second_hand_0[dim] = deepcopy((-1,-1,-1))
            
            for i_0 in range(len(output_1)):
                output_1[i_0][dim] = deepcopy((-1,-1,-1))
                output_2[i_0][dim] = deepcopy((-1,-1,-1))
                
    #!!!!! modified: We first use angle_left/right
    #directly to measure how much rotation we need in total.
    #then choose the rotation within 120 degrees
    #but if both angles are too large (>120), then we refer to input sholder-elbow-wrist triangle
    #to get the normal direction and use the maximum angle distence to decide
    
    #get normal direction for each triangle in the list
    triangle_normals = list()
    for triangle_input in triangles_input:
        triangle_normal = normal_vector_triangle(triangle_input)
        triangle_normals.append(triangle_normal)
        
    ##now we decide which direction is correct
    #that is, which rotation direction is more align with the inner normal of elbow.
    angle_1 = max_angle_between_normals(normal_1, triangle_normals)
    angle_2 = max_angle_between_normals(normal_2, triangle_normals)
    
    
    if angle_left < 120:
        return(output_1, first_hand_rotate_1, second_hand_0)
    if angle_right < 120:
        return(output_2, first_hand_rotate_2, second_hand_0)
    
    if angle_1 < angle_2:
        return(output_1, first_hand_rotate_1, second_hand_0)
    return(output_2, first_hand_rotate_2, second_hand_0)


#interpolation of hands: rotate palm.
#both first_hand and second_hand are 1D array in 63 dim.

#this is for DW pose keypoints. We will consider individual [-1,-1] dimensions, 
#which makes the function more complicated.

#first p_wrist is the tuple containing the 3D value of the starting wrist from pose
#second p_wrist is that of the ending wrist from pose

def rotate_hand_palm_DW(first_hand, second_hand, first_p_wrist, second_p_wrist,
                        out_chain, l_or_r_hand, triangles_input):
    
    n_interp = len(out_chain)
    
    #do NOT need to change array to list of tuples 
    #since it is already by the combine_keypoints function
    
    #replace wrist if the hand wrist is -1,-1
    if (first_hand[0][0] == -1 and first_hand[0][1] == -1) and (
        first_p_wrist[0] != -1 or first_p_wrist[1] != -1):
        first_hand[0] = deepcopy(first_p_wrist)
        
    #similarly, replace second wrist
    if (second_hand[0][0] == -1 and second_hand[0][1] == -1) and (
        second_p_wrist[0] != -1 or second_p_wrist[1] != -1):
        second_hand[0] = deepcopy(second_p_wrist)
    
    #list to hold normal direction of interpolation rotated in two different directions
    #note that first and last frame (original start/end) is not included
    normal_1, normal_2 = list(), list()
    
    ##output list##
    output_1, output_2 = list(), list()
    
    output_1.append(first_hand)
    output_2.append(first_hand)
    
    ##move second hand to first wrist, share the same wrist
    #but we still keep the original second hand
    second_hand_0 = translate_points(second_hand, first_hand[0])
    
    ##calculate the angle between two index fingers palm bones (0 to 5)
    seg_1 = [first_hand[0], first_hand[5]]
    seg_2 = [first_hand[0], second_hand_0[5]]
    angle = angle_between_segments(seg_1, seg_2)
    
    ##rotate first hand along the normal direction of seg_1, seg_2 plane (0 to 5 plane)
    #seg 1 rotate to seg 2
    first_hand_0 = rotate_points_around_normal(first_hand, seg_1, seg_2, angle)
    
    #calculate the angle between segond hand 0 and first hand 0 palm normal direction.
    '''
    angle_new is the angle difference after seg_1 rotated to seg_2,   
    So, the rotation along seg_1 should according to angle_new
    '''
    #triangle1 = [first_hand_0[0], first_hand_0[5], first_hand_0[17]]
    #triangle2 = [second_hand_0[0], second_hand_0[5], second_hand_0[17]]
    #angle_new = angle_between_normals(triangle1, triangle2)
    c_ = second_hand_0[0]
    d_ = second_hand_0[5]
    a_ = first_hand_0[17]
    b_ = second_hand_0[17]
    angle_left, angle_right = angle_and_rotation_direction(c_, d_, a_, b_)
    
    direction_1, direction_2 = 'left', 'right'
    
    #finally, we rotate the first hand 0 in both directions, the output will be used
    #for finger movement in the next function
    anchor_ = first_hand_0[0]
    axis_ = subtract_tuples(first_hand_0[5], first_hand_0[0])
    first_hand_rotate_1 = rotate_points_by_direction(first_hand_0, anchor_, axis_, 
                                                     angle_left, direction_1)
    first_hand_rotate_2 = rotate_points_by_direction(first_hand_0, anchor_, axis_, 
                                                     angle_right, direction_2)

    '''
    The direction may or may not be correct. So, we will rotate in both directions!
    Then, we select the one that consistent with elbow normal direction, 
    so that the hand will not rotate from outside the wrist.
    '''

    ############################################
    ##rotate the first_hand toward second_hand_0
    #in each step, first hand will rotate along seg_1 seg_2 normal to seg_1 by angle/n_interp
    #then, first hand will rotate along seg_1 by angle_new/n_interp
    
    angle_each = angle/float(n_interp-1)
    angle_left_each = angle_left/float(n_interp-1)
    angle_right_each = angle_right/float(n_interp-1)
    
    for i in range(1, n_interp-1):
        
        #rotate first hand from seg1 toward seg2 by i*angle_each
        first_hand_each = rotate_points_around_normal(first_hand, seg_1, seg_2, 
                                                      i*angle_each)
        
        ####################################
        #rotate first_hand along seg1 by the i*angle_new_each in both direction
        anchor = first_hand_each[0]
        axis = subtract_tuples(first_hand_each[5], first_hand_each[0])
        #first_hand_rotate_each = rotate_points(first_hand_each, anchor, axis, 
        #                                       i*angle_new_each, point_a_, point_b_)
        first_hand_rotate_each_1 = rotate_points_by_direction(first_hand_each, anchor, axis, 
                                                            i*angle_left_each, direction_1)
        
        #normal direction of first rotation
        #left hand, 0 --> 17 --> 5 with right hand rule, so normal direction point out from palm
        if l_or_r_hand == 'left':
            triangle_ = [first_hand_rotate_each_1[0], first_hand_rotate_each_1[17], first_hand_rotate_each_1[5]]
            normal_ = normal_vector_triangle(triangle_)
            normal_1.append(normal_)
        else:#right hand, 0-->5-->17
            triangle_ = [first_hand_rotate_each_1[0], first_hand_rotate_each_1[5], first_hand_rotate_each_1[17]]
            normal_ = normal_vector_triangle(triangle_)
            normal_1.append(normal_)            
        
        first_hand_rotate_each_2 = rotate_points_by_direction(first_hand_each, anchor, axis, 
                                                            i*angle_right_each, direction_2)
        
        #similarly, left hand
        if l_or_r_hand == 'left':
            triangle_ = [first_hand_rotate_each_2[0], first_hand_rotate_each_2[17], first_hand_rotate_each_2[5]]
            normal_ = normal_vector_triangle(triangle_)
            normal_2.append(normal_)
        else:#right hand
            triangle_ = [first_hand_rotate_each_2[0], first_hand_rotate_each_2[5], first_hand_rotate_each_2[17]]
            normal_ = normal_vector_triangle(triangle_)
            normal_2.append(normal_)   
        
        temp_1 = translate_points(first_hand_rotate_each_1, out_chain[i])
        output_1.append(temp_1)
        temp_2 = translate_points(first_hand_rotate_each_2, out_chain[i])
        output_2.append(temp_2)
        ##################################

    output_1.append(second_hand)
    output_2.append(second_hand)
    
    #before return output, we will change the interpolated dim to [-1,-1]
    #if either the start or end corresponding dim is [-1, -1]
    for dim in range(len(first_hand)):
        if (first_hand[dim][0] == -1 and first_hand[dim][1] == -1) or (
            second_hand[dim][0] == -1 and second_hand[dim][1] == -1):
            
            first_hand_rotate_1[dim] = deepcopy((-1,-1,-1))
            first_hand_rotate_2[dim] = deepcopy((-1,-1,-1))
            second_hand_0[dim] = deepcopy((-1,-1,-1))
            
            for i_0 in range(len(output_1)):
                output_1[i_0][dim] = deepcopy((-1,-1,-1))
                output_2[i_0][dim] = deepcopy((-1,-1,-1))
    
    #!!!!! modified: We first use angle_left/right
    #directly to measure how much rotation we need in total.
    #then choose the rotation within 120 degrees
    #but if both angles are too large (>120), then we refer to input sholder-elbow-wrist triangle
    #to get the normal direction and use the maximum angle distence to decide
    
    #get normal direction for each triangle in the list
    triangle_normals = list()
    for triangle_input in triangles_input:
        triangle_normal = normal_vector_triangle(triangle_input)
        triangle_normals.append(triangle_normal)
        
    ##now we decide which direction is correct
    #that is, which rotation direction is more align with the inner normal of elbow.
    angle_1 = max_angle_between_normals(normal_1, triangle_normals)
    angle_2 = max_angle_between_normals(normal_2, triangle_normals)
    
    
    if angle_left < 120:
        return(output_1, first_hand_rotate_1, second_hand_0)
    if angle_right < 120:
        return(output_2, first_hand_rotate_2, second_hand_0)
    
    if angle_1 < angle_2:
        return(output_1, first_hand_rotate_1, second_hand_0)
    return(output_2, first_hand_rotate_2, second_hand_0)


def find_rotation_matrix(A, B):
    """
    Find the rotation matrix that aligns matrix A to matrix B.
    A and B are n x 3 matrices representing n points in 3D space.
    """
    assert A.shape == B.shape, "Both point sets must have the same shape."

    # Subtract centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # Compute the covariance matrix
    H = np.dot(AA.T, BB)

    # SVD to find the rotation
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Ensure a right-handed coordinate system
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    return R


def rotate_point(point, rotation_matrix):
    """Rotate a point using a given rotation matrix."""
    return np.dot(rotation_matrix, point)


def rotate_point_to_match_list(point, list_1, list_2):
    """
    Rotate a point so that its relative position to list_1 matches its new position relative to list_2.

    Parameters:
    - point: Tuple (x, y, z), the point to rotate.
    - list_1: List of tuples [(x1, y1, z1), ..., (xn, yn, zn)], the original points.
    - list_2: List of tuples [(x1, y1, z1), ..., (xn, yn, zn)], the rotated points.

    Returns:
    - The new position of the point as a tuple (x', y', z').
    """
    # Convert lists to numpy arrays
    list_1 = np.array(list_1)
    list_2 = np.array(list_2)

    # Calculate the rotation matrix
    rotation_matrix = find_rotation_matrix(list_1, list_2)

    # Rotate the point
    new_position = rotate_point(np.array(point), rotation_matrix)
    return tuple(new_position)


def move_point_along_vector(point, point_1, point_2):
    """
    Move a point along the vector defined by (point_2 - point_1).

    Parameters:
    - point: Tuple (x, y, z), the point to move.
    - point_1: Tuple (x1, y1, z1), start of the vector.
    - point_2: Tuple (x2, y2, z2), end of the vector.

    Returns:
    - Tuple (x', y', z'), the new position of the point.
    """
    # Convert points to numpy arrays
    point = np.array(point)
    point_1 = np.array(point_1)
    point_2 = np.array(point_2)

    # Calculate the vector from point_1 to point_2
    vector = point_2 - point_1

    # Move the point along the vector
    new_point = point + vector

    return tuple(new_point)


def translate_points_by_vector(points, origin, target):
    # Calculate the translation vector
    x_o, y_o, z_o = origin
    x_t, y_t, z_t = target
    translation_vector = (x_t - x_o, y_t - y_o, z_t - z_o)
    
    # Create a new list to store the translated points
    translated_points = []
    
    # Apply the translation to each point in the list
    for (x, y, z) in points:
        new_x = x + translation_vector[0]
        new_y = y + translation_vector[1]
        new_z = z + translation_vector[2]
        translated_points.append((new_x, new_y, new_z))
    
    return translated_points


#function to move the entire finger (or partial finger) in parallel, 
#given the movement interpolation chain of the fist node
#will return the finger movement chain as a dict

#finger contains three nodes: [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3)]
#which are the nodes from fist node to finger tip 
#(not include fist node, but include tip node)

def move_finger_parallel(finger, in_node_chain):
    
    #list containing list of tuples
    out_chain = list()
    out_chain.append(finger)
    
    #start from the first interpolated movement
    for i in range(1, len(in_node_chain)):
        finger_out = translate_points_by_vector(finger, in_node_chain[0], in_node_chain[i])
        out_chain.append(finger_out)
        
    return(out_chain)


def apply_vector_transformations(list1, list2, list3):
    # Calculate vector differences between corresponding points in list1 and list2
    vector_differences = [(x3 - x1, y3 - y1, z3 - z1) for (x1, y1, z1), (x3, y3, z3) in zip(list1, list2)]
    
    # Add these vector differences to the corresponding points in list3
    transformed_points = [(x5 + vx, y5 + vy, z5 + vz) for (x5, y5, z5), (vx, vy, vz) in zip(list3, vector_differences)]
    
    return transformed_points


def segment_length_difference(seg_1, seg_2):
    # Calculate the length of the first segment
    x1, y1, z1 = seg_1[0]
    x2, y2, z2 = seg_1[1]
    length_1 = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

    # Calculate the length of the second segment
    x3, y3, z3 = seg_2[0]
    x4, y4, z4 = seg_2[1]
    length_2 = math.sqrt((x4 - x3)**2 + (y4 - y3)**2 + (z4 - z3)**2)

    # Calculate the difference in lengths
    length_difference = length_2 - length_1

    return length_difference


def move_points_by_distance(points, direction, distance):
    # Calculate the magnitude of the direction vector
    x, y, z = direction
    magnitude = math.sqrt(x**2 + y**2 + z**2) + 0.0000000001

    # Normalize the direction vector to get a unit vector
    unit_vector = (x / magnitude, y / magnitude, z / magnitude)

    # Calculate the movement vector
    movement_vector = (unit_vector[0] * distance, unit_vector[1] * distance, unit_vector[2] * distance)

    # Move each point by the movement vector
    moved_points = [(px + movement_vector[0], py + movement_vector[1], pz + movement_vector[2]) for px, py, pz in points]

    return moved_points


#function to rotate finger (entire or partial)
#the input is a list of list of tuples
#the last list of tuple in the entire list will be rotated toward the target position
#then, the rotation will be added to each list of tuples in the list

#the first node (x1,y1,z1) in target segment [(x1,y1,z1), (x2,y2,z2)]
#should be coincide with the first node (first tuple) in the last list of tuples
#this is guaranteed by the above function move_finger_parallel

# the rotation is calculated using the last list of tuples in the finger_list:
# the first and second tuple in the last list of tuples form another segment
# the angle between two segments are calculated
# the last list of tuple is rotated toward the target segment, with the coincide point
# to be the anchor
# in each interpolation, the angle is equally divided.
# rotation is made

#finally, the rotated last list of tuple (rotation difference) is added to the corresponding
#list of tuple in the entire list
def rotate_finger(finger_list, target_segment, in_node_chain):
    
    seg_1 = [target_segment[0], finger_list[-1][0]]
    seg_2 = target_segment
    theta = angle_between_segments(seg_1, seg_2)
    itp_theta = theta/(len(finger_list) - 1)
    
    #temporary list to hold the list of tuples, 
    #which will be added to the finger_list
    temp = list()
    temp.append(finger_list[-1])
    
    for i in range(1, len(finger_list)):
        rotated = rotate_points_around_normal(finger_list[-1], seg_1, seg_2, itp_theta*i)
        temp.append(rotated)
    
    #again, the output is a list of list of tuples
    #add the rotation to each list of tuples in finger_list
    temp_2 = list()
    for i in range(len(temp)):
        item = temp[i]
        base_list = finger_list[i]
        added_list = apply_vector_transformations(finger_list[-1], item, base_list)
        temp_2.append(added_list)
        
    #finally adjust the distance
    output_list = list()
    distance = segment_length_difference(seg_2, seg_1)
    for i in range(len(temp_2)):
        
        direction = (in_node_chain[i][0] - temp_2[i][0][0], 
                     in_node_chain[i][1] - temp_2[i][0][1],
                     in_node_chain[i][2] - temp_2[i][0][2])
        
        moved_points = move_points_by_distance(temp_2[i], direction, i*distance/(len(temp_2)-1))
        output_list.append(moved_points)
        
    return(output_list)


#the function to handle finger tip, second tip and third tip movements
#id_1 is the third-last tip, id_2 is the second-last tip, id_3 is the tip
#id_0 is the palm edge/fist node id

#first_hand_ and second_hand_ is similar to the next function

#(first_hand_ is the output from the previous function, 
#which is the first hand rotated so that the palm coincide with the palm plane of the second hand
#second_hand_ is the one moved in parallel to first_hand_ wrist point.)

#out_chain is the palm edge/fist node movement interpolation

#temp_dict is the dict holder for final interpolation

def finger_tip_three_movements(id_0, id_1, id_2, id_3, first_hand_, second_hand_, 
                               out_chain, key, temp_dict):
    
    '''to be commented out in practice'''
    #print(first_hand_, second_hand_)
    
    #for DWpose, make sure all four id nodes are not [-1,-1], are valid
    #we don't need to further consider mediapipe, since all will be zeros at one time
    #which will re-assign interpolation to all zeros at the very end
    #so we do not need to care about interplation on all zero hands for mediapipe
    if (first_hand_[id_0][0] != -1 and first_hand_[id_0][1] != -1 and second_hand_[id_0][0] != -1 and second_hand_[id_0][1] != -1) and (
        first_hand_[id_1][0] != -1 and first_hand_[id_1][1] != -1 and second_hand_[id_1][0] != -1 and second_hand_[id_1][1] != -1) and (
        first_hand_[id_2][0] != -1 and first_hand_[id_2][1] != -1 and second_hand_[id_2][0] != -1 and second_hand_[id_2][1] != -1) and (
        first_hand_[id_3][0] != -1 and first_hand_[id_3][1] != -1 and second_hand_[id_3][0] != -1 and second_hand_[id_3][1] != -1):
        
        ####this is node 1##############
        finger = [first_hand_[id_1], first_hand_[id_2], first_hand_[id_3]]
        out_finger_parallel = move_finger_parallel(finger, out_chain)
        out_finger = rotate_finger(out_finger_parallel, 
                                   [second_hand_[id_0], second_hand_[id_1]], 
                                   out_chain)

        temp_dict[key + str(id_1)] = deepcopy([out_finger[i][0] for i in range(len(out_finger))])

        ####this is for node 2##########
        '''although the base out_finger is no more parallel, but the function still work'''
        out_chain_new = deepcopy(temp_dict[key + str(id_1)])
        
        out_finger_mid = [[out_finger[i][1], out_finger[i][2]] for i in range(len(out_finger))]
        out_finger_new = rotate_finger(out_finger_mid, 
                                       [second_hand_[id_1], second_hand_[id_2]], 
                                       out_chain_new)

        temp_dict[key + str(id_2)] = deepcopy([out_finger_new[i][0] for i in range(len(out_finger_new))])

        ####this is for node 3############
        out_chain_new_2 = deepcopy(temp_dict[key + str(id_2)])
        
        out_finger_mid_2 = [[out_finger_new[i][1]] for i in range(len(out_finger_new))]
        out_finger_new_tip = rotate_finger(out_finger_mid_2, 
                                           [second_hand_[id_2], second_hand_[id_3]], 
                                           out_chain_new_2)

        temp_dict[key + str(id_3)] = deepcopy([out_finger_new_tip[i][0] for i in range(len(out_finger_new_tip))])

        return(temp_dict)
    
    #when the fist/hand palm edge node is [-1,-1]
    elif (first_hand_[id_1][0] != -1 and first_hand_[id_1][1] != -1 and second_hand_[id_1][0] != -1 and second_hand_[id_1][1] != -1) and (
          first_hand_[id_2][0] != -1 and first_hand_[id_2][1] != -1 and second_hand_[id_2][0] != -1 and second_hand_[id_2][1] != -1) and (
          first_hand_[id_3][0] != -1 and first_hand_[id_3][1] != -1 and second_hand_[id_3][0] != -1 and second_hand_[id_3][1] != -1):
                
        #then, we linearly interpolate the third-last tip, and start from there
        out_chain_new = interpolate_points_series(first_hand_[id_1], second_hand_[id_1], len(out_chain))

        ####node 1 is linearly interpolated########
        temp_dict[key + str(id_1)] = deepcopy(out_chain_new)
        
        ####node 2 is now interpolated as previous node 1##########
        finger = [first_hand_[id_2], first_hand_[id_3]]
        out_finger_parallel = move_finger_parallel(finger, out_chain_new)
        out_finger = rotate_finger(out_finger_parallel, 
                                   [second_hand_[id_1], second_hand_[id_2]], 
                                   out_chain_new)

        temp_dict[key + str(id_2)] = deepcopy([out_finger[i][0] for i in range(len(out_finger))])
        
        ####node 3 finally##############
        out_chain_new = deepcopy(temp_dict[key + str(id_2)])
        
        out_finger_mid = [[out_finger[i][1]] for i in range(len(out_finger))]
        out_finger_new = rotate_finger(out_finger_mid, 
                                       [second_hand_[id_2], second_hand_[id_3]], 
                                       out_chain_new)

        temp_dict[key + str(id_3)] = deepcopy([out_finger_new[i][0] for i in range(len(out_finger_new))])
        
        return(temp_dict)
        
    #when both fist/hand palm edge node and third-last tip are [-1,-1]
    elif (first_hand_[id_2][0] != -1 and first_hand_[id_2][1] != -1 and second_hand_[id_2][0] != -1 and second_hand_[id_2][1] != -1) and (
          first_hand_[id_3][0] != -1 and first_hand_[id_3][1] != -1 and second_hand_[id_3][0] != -1 and second_hand_[id_3][1] != -1):
        
        ####node 0 and 1 are just invalid in all interpolation#########
        temp_dict[key + str(id_1)] = deepcopy([(-1,-1,-1) for i in range(len(out_chain))])

        ####node 2 linear interpolation############
        out_chain_new = interpolate_points_series(first_hand_[id_2], second_hand_[id_2], len(out_chain))
        temp_dict[key + str(id_2)] = deepcopy(out_chain_new)
        
        ####node 3 interpolation#############
        finger = [first_hand_[id_3]]
        out_finger_parallel = move_finger_parallel(finger, out_chain_new)
        out_finger = rotate_finger(out_finger_parallel, 
                                   [second_hand_[id_2], second_hand_[id_3]], 
                                   out_chain_new)

        temp_dict[key + str(id_3)] = deepcopy([out_finger[i][0] for i in range(len(out_finger))])

        return(temp_dict)
    
    #when both 0, 1 and 2 nodes are [-1,-1]
    elif first_hand_[id_3][0] != -1 and first_hand_[id_3][1] != -1 and second_hand_[id_3][0] != -1 and second_hand_[id_3][1] != -1:
        
        temp_dict[key + str(id_1)] = deepcopy([(-1,-1,-1) for i in range(len(out_chain))])
        temp_dict[key + str(id_2)] = deepcopy([(-1,-1,-1) for i in range(len(out_chain))])
        
        out_chain_new = interpolate_points_series(first_hand_[id_3], second_hand_[id_3], len(out_chain))
        temp_dict[key + str(id_3)] = deepcopy(out_chain_new)
        
        return(temp_dict)
    
    #all [-1,-1] ^_^
    else:
        
        temp_dict[key + str(id_1)] = deepcopy([(-1,-1,-1) for i in range(len(out_chain))])
        temp_dict[key + str(id_2)] = deepcopy([(-1,-1,-1) for i in range(len(out_chain))])
        temp_dict[key + str(id_3)] = deepcopy([(-1,-1,-1) for i in range(len(out_chain))])

        return(temp_dict)


'''
This is the function to add finger movements (by forward kinematics) 
to each rotated hand palm.

first_hand_ is the output from the previous function, 
which is the first hand rotated so that the palm coincide with the palm plane of the second hand
second_hand_ is the one moved in parallel to first_hand_ wrist point.

We first get the finger movements. Then, these movements will be transformed into co-ordinate
based on hand-formed axis. Then, the movements will be added to palm rotations.
'''

def add_finger_movements(palm_rotation_list, first_hand_, second_hand_, l_or_r, key_type):
    
    n_interp = len(palm_rotation_list)
    temp_dict = dict()
    
    if l_or_r not in {'left','right'}:
        raise ValueError('variable l_or_r must be either left or right')
        
    if key_type not in {'mp', 'DW'}:
        raise ValueError('variable key_type must be either mp or DW')
        
    if l_or_r == 'left':
        if key_type == 'mp':
            key = 'left_hand_mp_'
        else:
            key = 'left_hand_DW_'
    else:
        if key_type == 'mp':
            key = 'right_hand_mp_'
        else:
            key = 'right_hand_DW_'
    
    ##########################
    # interpolate the hand wrist to creat the out_chain
    out_chain = interpolate_points_series(first_hand_[0], second_hand_[0], n_interp)
    temp_dict[key + str(0)] = deepcopy(out_chain)
    
    #forward alongside 0,9,10,11,12
    #0,9,5,6,7,8
    #0,9,13,14,15,16
    #0,9,17,18,19,20
    #0,1,2,3,4
    
    ####this is node 9#################
    out_chain = forward_kinematics(out_chain, first_hand_[9], second_hand_[9])
    temp_dict[key + str(9)] = deepcopy(out_chain)
    
    ####implement 10,11,12#############
    temp_dict = finger_tip_three_movements(9, 10, 11, 12, first_hand_, second_hand_, 
                                           out_chain, key, temp_dict)
    
    ####this is node 5#################
    out_chain = deepcopy(temp_dict[key + str(9)]) #reload node 9
    out_chain = forward_kinematics(out_chain, first_hand_[5], second_hand_[5])
    temp_dict[key + str(5)] = deepcopy(out_chain)
    
    ####implement 6,7,8################
    temp_dict = finger_tip_three_movements(5, 6, 7, 8, first_hand_, second_hand_, 
                                           out_chain, key, temp_dict)
    
    ####this is node 13################
    out_chain = deepcopy(temp_dict[key + str(9)]) #reload node 9
    out_chain = forward_kinematics(out_chain, first_hand_[13], second_hand_[13])
    temp_dict[key + str(13)] = deepcopy(out_chain)

    ####implement 14,15,16#############
    temp_dict = finger_tip_three_movements(13, 14, 15, 16, first_hand_, second_hand_, 
                                           out_chain, key, temp_dict)
    
    ####this is node 17################
    out_chain = deepcopy(temp_dict[key + str(9)]) #reload node 9
    out_chain = forward_kinematics(out_chain, first_hand_[17], second_hand_[17])
    temp_dict[key + str(17)] = deepcopy(out_chain)

    ####implement 18,19,20#############
    temp_dict = finger_tip_three_movements(17, 18, 19, 20, first_hand_, second_hand_, 
                                           out_chain, key, temp_dict)
    
    ####this is node 1#################
    out_chain = deepcopy(temp_dict[key + str(0)]) #reload node 0!
    out_chain = forward_kinematics(out_chain, first_hand_[1], second_hand_[1])
    temp_dict[key + str(1)] = deepcopy(out_chain)

    ####implement 2,3,4################
    temp_dict = finger_tip_three_movements(1, 2, 3, 4, first_hand_, second_hand_, 
                                           out_chain, key, temp_dict)

    
    ##################################
    ###do co-ordinate transfer########
    temp_dict_new = dict()
    
    for real_key in temp_dict:
        temp_dict_new[real_key] = list()
        
        #the current keypoint node we are working on
        cur_node = int(real_key.split('_')[-1])
        
        # i_point stands for each frame
        for i_point in range(len(temp_dict[real_key])):
            
            if i_point == 0 or i_point == len(temp_dict[real_key])-1:
                
                point_ini = palm_rotation_list[i_point][cur_node]
                temp_dict_new[real_key].append(point_ini)
            
            else:
                point = temp_dict[real_key][i_point]
                
                target_hand = palm_rotation_list[i_point]
                target_hand_at_0 = translate_points(target_hand, (0,0,0))
                first_hand_at_0 = translate_points(first_hand_, (0,0,0))
                point_at_0 = move_point_along_vector(point, first_hand_[0], (0,0,0))
                
                new_point_at_0 = rotate_point_to_match_list(point_at_0, first_hand_at_0, target_hand_at_0)
                new_point = move_point_along_vector(new_point_at_0, (0,0,0), target_hand[0])

                temp_dict_new[real_key].append(new_point)    
            
    return(temp_dict_new)


def move_keypoints(keypoints, x_target, y_target):
    # Convert the list to a numpy array for easier manipulation (okay if already array)
    keypoints_array = np.array(keypoints)
    
    # Extract x, y coordinates
    x_coords = keypoints_array[0::3]  # x1, x2, ..., xn
    y_coords = keypoints_array[1::3]  # y1, y2, ..., yn
    
    # Calculate the direction vector
    direction_vector = np.array([x_target - x_coords[0], y_target - y_coords[0]])
    
    # Move all x and y coordinates
    x_coords += direction_vector[0]
    y_coords += direction_vector[1]
    
    # Update the original keypoints array
    keypoints_array[0::3] = x_coords
    keypoints_array[1::3] = y_coords
    
    return np.array(keypoints_array)


'''
sometimes especially the left hand mediapipe kp is not captured correct:
Its palm is toward outside, which is not posible for human arm.
In this case, we use DW pose kp to replace the xy-value of mediapipe, 
That says, we do exactly the interpolate_DW on one hand in this case!
'''
def whether_replace_mp_hand_by_DW(Dict, hand_tag, hand_kp, pose_kp):
    
    i_1, i_2, i_3 = hand_kp[0], hand_kp[1], hand_kp[2]
    j_1, j_2, j_3 = pose_kp[0], pose_kp[1], pose_kp[2]
    
    #test normal direction
    temp = Dict[hand_tag]
    triangle_hand = [(temp[3*i_1],temp[3*i_1 + 1],temp[3*i_1 + 2]), 
                     (temp[3*i_2],temp[3*i_2 + 1],temp[3*i_2 + 2]), 
                     (temp[3*i_3],temp[3*i_3 + 1],temp[3*i_3 + 2])]

    temp = Dict['pose_mp']
    triangle_pose = [(temp[3*j_1],temp[3*j_1 + 1],temp[3*j_1 + 2]), 
                     (temp[3*j_2],temp[3*j_2 + 1],temp[3*j_2 + 2]), 
                     (temp[3*j_3],temp[3*j_3 + 1],temp[3*j_3 + 2])]
    
    angle_0 = angle_between_normals(triangle_hand, triangle_pose)

    #if the hand normal direction (point out from palm) is 120 degree more than inner 
    #inner normal direction of arm, then change left hand.
    if angle_0 >= 120:
        return(True)
    
    return(False)
        

#the function to decide whether a hand is below the screen
'''
For mediapipe, a hand is below the screen if
(pose wrist y > 0.94) and ((hand is zero) or (hand minimum y > 0.98))

For DWPose, a hand is below screen if
(pose wrist y > 0.94) and (among non [-1,-1] hand dims, minimum y > 0.98)

If either mediapipe or DWpose says hand is blow screen, then hand is below screen
'''

def find_min_y(array):
    # Ensure that the array length is a multiple of 3
    if len(array) % 3 != 0:
        raise ValueError("Array length must be a multiple of 3")
    
    # Extract y values which are every 3rd element starting from the second element
    y_values = array[1::3]
    
    # Return the minimum y value
    return min(y_values)


def find_min_y_2d(array):
    # Check for empty numpy array
    if array.size == 0:
        raise ValueError("The array cannot be empty")
    
    # Initialize min_y with a very large value
    min_y = float('inf')
    
    # Loop through each pair and update min_y if a smaller y is found
    for x, y in array:
        if (x, y) != (-1, -1) and y < min_y:
            min_y = y
    
    # Handle the case where all coordinates were invalid
    if min_y == float('inf'):
        return 1.2  # or raise an exception if no valid coordinates are found
    
    return min_y


def check_hand_below_screen(Dict, l_or_r):
    
    below_screen = False
    
    if l_or_r == 'left':
        
        #check mediapipe
        if (Dict['pose_mp'][46] > 0.94) and (
            zero_rate(Dict['left_hand_mp']) > 0.5 or find_min_y(Dict['left_hand_mp']) > 0.98):
            below_screen = True
        
        #check DWPose
        elif (Dict['bodies']['candidate'][7][1] > 0.94) and (
              find_min_y_2d(Dict['hands'][0]) > 0.98):
            below_screen = True
            
    elif l_or_r == 'right':
        
        #check mediapipe
        if (Dict['pose_mp'][49] > 0.94) and (
            zero_rate(Dict['right_hand_mp']) > 0.5 or find_min_y(Dict['right_hand_mp']) > 0.98):
            below_screen = True
        
        #check DWPose
        elif (Dict['bodies']['candidate'][4][1] > 0.94) and (
              find_min_y_2d(Dict['hands'][1]) > 0.98):
            below_screen = True
            
    else:
        raise ValueError('l_or_r has to be left or right when check hand below screen')
        
    return(below_screen)


#function to extract sholder-elbow-wrist triangle from interpolated series
def build_triangles(list1, list2, list3):
    triangles = []
    for point1, point2, point3 in zip(list1, list2, list3):
        triangles.append([point1, point2, point3])
    return triangles


'''
We actively calculate the sholder-elbow-wrist triangle and its normal direction
elbow normal is always inner to body
'''
#function to interpolate face, body and hand keypoints of mediapipe
def interpolate_mp(Dict_1_, Dict_2_,
                   l_1_below, l_2_below, r_1_below, r_2_below,
                   s_e_file, n_interp=12):
    
    '''
    Before everything starts, we want to extend the length of the arm,
    if the wrist is below the screen.
    This is because the added standard hand is usually partially in the screen,
    which is unwanted.
    '''
    
    Dict_1 = deepcopy(Dict_1_)
    Dict_2 = deepcopy(Dict_2_)
    
    #l_1: either mp or DW pose wrist out
    if l_1_below:
        Dict_1['pose_mp'][46] = 1.2
        Dict_1['bodies']['candidate'][7][1] = 1.2
        
    #l_2: either mp or DW pose wrist out
    if l_2_below:
        Dict_2['pose_mp'][46] = 1.2
        Dict_2['bodies']['candidate'][7][1] = 1.2
        
    #r_1: either mp or DW
    if r_1_below:
        Dict_1['pose_mp'][49] = 1.2
        Dict_1['bodies']['candidate'][4][1] = 1.2
        
    #r_2: either mp or DW
    if r_2_below:
        Dict_2['pose_mp'][49] = 1.2
        Dict_2['bodies']['candidate'][4][1] = 1.2
        
    
    #the temp dict with keypoint name as key, and interpolations as value
    temp_dict = dict()
    
    #decide whether change each hand in each of start/end frame
    l_1_change = whether_replace_mp_hand_by_DW(Dict_1, 'left_hand_mp', [0,17,5], [11,13,15])
    
    l_2_change = whether_replace_mp_hand_by_DW(Dict_2, 'left_hand_mp', [0,17,5], [11,13,15])

    r_1_change = whether_replace_mp_hand_by_DW(Dict_1, 'right_hand_mp', [0,5,17], [12,16,14])
    
    r_2_change = whether_replace_mp_hand_by_DW(Dict_2, 'right_hand_mp', [0,5,17], [12,16,14])

    
    ###whether change hand############
    if not l_1_change:
        #deal with hand outside issue
        if l_1_below:
            left_hand_in_1 = s_e_file['standard']['standard_left_hand']
            left_hand_in_1 = move_keypoints(left_hand_in_1, Dict_1['pose_mp'][45], 
                                                            Dict_1['pose_mp'][46])
        else:
            left_hand_in_1 = Dict_1['left_hand_mp']
    else:
        #deal with hand outside issue
        if l_1_below:
            left_hand_in_1 = s_e_file['standard']['standard_left_hand']
            left_hand_in_1 = move_keypoints(left_hand_in_1, Dict_1['bodies']['candidate'][7][0], 
                                                            Dict_1['bodies']['candidate'][7][1])
            #left_hand_in_1 = array_to_tuples(left_hand_in_1)
        else:
            left_hand_in_1 = combine_keypoints(Dict_1['hands'][0], Dict_1['left_hand_mp'])
            left_hand_in_1 = tuples_to_array(left_hand_in_1)
        
    if not l_2_change:
        if l_2_below:
            left_hand_in_2 = s_e_file['standard']['standard_left_hand']
            left_hand_in_2 = move_keypoints(left_hand_in_2, Dict_2['pose_mp'][45], 
                                                            Dict_2['pose_mp'][46])
        else:
            left_hand_in_2 = Dict_2['left_hand_mp']
    else:
        if l_2_below:
            left_hand_in_2 = s_e_file['standard']['standard_left_hand']
            left_hand_in_2 = move_keypoints(left_hand_in_2, Dict_2['bodies']['candidate'][7][0], 
                                                            Dict_2['bodies']['candidate'][7][1])
            #left_hand_in_2 = array_to_tuples(left_hand_in_2)
        else:
            left_hand_in_2 = combine_keypoints(Dict_2['hands'][0], Dict_2['left_hand_mp'])
            left_hand_in_2 = tuples_to_array(left_hand_in_2)
        
    if not r_1_change:
        #deal with hand outside issue
        if r_1_below:
            right_hand_in_1 = s_e_file['standard']['standard_right_hand']
            right_hand_in_1 = move_keypoints(right_hand_in_1, Dict_1['pose_mp'][48], 
                                                              Dict_1['pose_mp'][49])
        else:
            right_hand_in_1 = Dict_1['right_hand_mp']
    else:
        #deal with hand outside issue
        if r_1_below:
            right_hand_in_1 = s_e_file['standard']['standard_right_hand']
            right_hand_in_1 = move_keypoints(right_hand_in_1, Dict_1['bodies']['candidate'][4][0], 
                                                        Dict_1['bodies']['candidate'][4][1])
            #right_hand_in_1 = array_to_tuples(right_hand_in_1)
        else:
            right_hand_in_1 = combine_keypoints(Dict_1['hands'][1], Dict_1['right_hand_mp'])
            right_hand_in_1 = tuples_to_array(right_hand_in_1)
        
    if not r_2_change:
        if r_2_below:
            right_hand_in_2 = s_e_file['standard']['standard_right_hand']
            right_hand_in_2 = move_keypoints(right_hand_in_2, Dict_2['pose_mp'][48], 
                                                              Dict_2['pose_mp'][49])
        else:
            right_hand_in_2 = Dict_2['right_hand_mp']
    else:
        if r_2_below:
            right_hand_in_2 = s_e_file['standard']['standard_right_hand']
            right_hand_in_2 = move_keypoints(right_hand_in_2, Dict_2['bodies']['candidate'][4][0], 
                                                        Dict_2['bodies']['candidate'][4][1])
            #right_hand_in_2 = array_to_tuples(right_hand_in_2)
        else:
            right_hand_in_2 = combine_keypoints(Dict_2['hands'][1], Dict_2['right_hand_mp'])
            right_hand_in_2 = tuples_to_array(right_hand_in_2)
    

    ####start interpolation############
    #left hip
    hip_1 = (Dict_1['pose_mp'][69], Dict_1['pose_mp'][70], Dict_1['pose_mp'][71])
    hip_2 = (Dict_2['pose_mp'][69], Dict_2['pose_mp'][70], Dict_2['pose_mp'][71])
    out_chain_ = interpolate_points_series(hip_1, hip_2, n_interp)
    temp_dict['pose_mp_23'] = deepcopy(out_chain_)
    
    #right hip
    hip_1 = (Dict_1['pose_mp'][72], Dict_1['pose_mp'][73], Dict_1['pose_mp'][74])
    hip_2 = (Dict_2['pose_mp'][72], Dict_2['pose_mp'][73], Dict_2['pose_mp'][74])
    out_chain_ = interpolate_points_series(hip_1, hip_2, n_interp)
    temp_dict['pose_mp_24'] = deepcopy(out_chain_)

    #interpolate other pose_mp keypoints by linear (left and right)
    for pose_mp_id in range(33):
        if pose_mp_id not in {11, 12, 13, 14, 15, 16, 23, 24}:
            pose_mp_1 = (Dict_1['pose_mp'][3*pose_mp_id], Dict_1['pose_mp'][3*pose_mp_id+1], Dict_1['pose_mp'][3*pose_mp_id+2])
            pose_mp_2 = (Dict_2['pose_mp'][3*pose_mp_id], Dict_2['pose_mp'][3*pose_mp_id+1], Dict_2['pose_mp'][3*pose_mp_id+2])           
            out_chain_ = interpolate_points_series(pose_mp_1, pose_mp_2, n_interp)
            temp_dict['pose_mp_' + str(pose_mp_id)] = deepcopy(out_chain_)

    #interpolate other pose_holistic_mp keypoints by linear (left and right)
    for pose_mp_id in range(33):
        pose_mp_1 = (Dict_1['pose_holistic_mp'][3*pose_mp_id], Dict_1['pose_holistic_mp'][3*pose_mp_id+1], Dict_1['pose_holistic_mp'][3*pose_mp_id+2])
        pose_mp_2 = (Dict_2['pose_holistic_mp'][3*pose_mp_id], Dict_2['pose_holistic_mp'][3*pose_mp_id+1], Dict_2['pose_holistic_mp'][3*pose_mp_id+2])           
        out_chain_ = interpolate_points_series(pose_mp_1, pose_mp_2, n_interp)
        temp_dict['pose_holistic_mp_' + str(pose_mp_id)] = deepcopy(out_chain_)
    
    ##########################
    # start from interpolating left sholder
    sholder_1 = (Dict_1['pose_mp'][33], Dict_1['pose_mp'][34], Dict_1['pose_mp'][35])
    sholder_2 = (Dict_2['pose_mp'][33], Dict_2['pose_mp'][34], Dict_2['pose_mp'][35])
    out_chain = interpolate_points_series(sholder_1, sholder_2, n_interp)
    temp_dict['pose_mp_11'] = deepcopy(out_chain)
    
    '''in order to mitigate invalid z, we still use linear interpolation for elbow'''
    down_1 = (Dict_1['pose_mp'][39], Dict_1['pose_mp'][40], Dict_1['pose_mp'][41])
    down_2 = (Dict_2['pose_mp'][39], Dict_2['pose_mp'][40], Dict_2['pose_mp'][41])
    out_chain = interpolate_points_series(down_1, down_2, n_interp)
    temp_dict['pose_mp_13'] = deepcopy(out_chain)
    
    '''in order to mitigate invalid z, we still use linear interpolation for elbow'''
    #followed by the left wrist (wrist of pose)
    down_1 = (Dict_1['pose_mp'][45], Dict_1['pose_mp'][46], Dict_1['pose_mp'][47])
    down_2 = (Dict_2['pose_mp'][45], Dict_2['pose_mp'][46], Dict_2['pose_mp'][47])
    out_chain_l_temp = interpolate_points_series(down_1, down_2, n_interp)
    temp_dict['pose_mp_15'] = deepcopy(out_chain_l_temp)
    
    #build the sholder-elbow-wrist triangle list for left side
    triangles_left = build_triangles(temp_dict['pose_mp_11'], temp_dict['pose_mp_13'], temp_dict['pose_mp_15'])
    
    #followed by the left wrist (wrist of hand)
    down_1 = (left_hand_in_1[0], left_hand_in_1[1], left_hand_in_1[2])
    down_2 = (left_hand_in_2[0], left_hand_in_2[1], left_hand_in_2[2])
    out_chain = interpolate_points_series(down_1, down_2, n_interp)
    temp_dict['left_hand_mp_0'] = deepcopy(out_chain)
    
    #obtain hand interpolation
    hand_interp, first_hand_l, second_hand_l = rotate_hand_palm(left_hand_in_1, 
                                   left_hand_in_2, out_chain, 'left', triangles_left)
    
    #get finger movement and add to each rotated hand
    out_temp_dict = add_finger_movements(hand_interp, first_hand_l, second_hand_l, 'left', 'mp')
    
    #combine dict
    for key_out in out_temp_dict:
        temp_dict[key_out] = deepcopy(out_temp_dict[key_out])
            
    del(hand_interp, out_temp_dict)
    
    ##########################
    # start from interpolating right sholder
    sholder_1 = (Dict_1['pose_mp'][36], Dict_1['pose_mp'][37], Dict_1['pose_mp'][38])
    sholder_2 = (Dict_2['pose_mp'][36], Dict_2['pose_mp'][37], Dict_2['pose_mp'][38])
    out_chain = interpolate_points_series(sholder_1, sholder_2, n_interp)
    temp_dict['pose_mp_12'] = deepcopy(out_chain)
    
    '''similarly, we do point-wise linear interpolation to avoid inaccurate z'''
    #followed by the right elbow
    down_1 = (Dict_1['pose_mp'][42], Dict_1['pose_mp'][43], Dict_1['pose_mp'][44])
    down_2 = (Dict_2['pose_mp'][42], Dict_2['pose_mp'][43], Dict_2['pose_mp'][44])
    out_chain = interpolate_points_series(down_1, down_2, n_interp)
    temp_dict['pose_mp_14'] = deepcopy(out_chain)
    
    #followed by the right wrist (wrist of pose)
    down_1 = (Dict_1['pose_mp'][48], Dict_1['pose_mp'][49], Dict_1['pose_mp'][50])
    down_2 = (Dict_2['pose_mp'][48], Dict_2['pose_mp'][49], Dict_2['pose_mp'][50])
    out_chain_r_temp = interpolate_points_series(down_1, down_2, n_interp)
    temp_dict['pose_mp_16'] = deepcopy(out_chain_r_temp)
    
    #build the sholder-wrist-elbow triangle list for right side
    triangles_right = build_triangles(temp_dict['pose_mp_12'], temp_dict['pose_mp_16'], temp_dict['pose_mp_14'])
    
    #followed by the right wrist (wrist of hand)
    down_1 = (right_hand_in_1[0], right_hand_in_1[1], right_hand_in_1[2])
    down_2 = (right_hand_in_2[0], right_hand_in_2[1], right_hand_in_2[2])
    out_chain = interpolate_points_series(down_1, down_2, n_interp)
    temp_dict['right_hand_mp_0'] = deepcopy(out_chain)
    
    #obtain hand interpolation
    hand_interp, first_hand_r, second_hand_r = rotate_hand_palm(right_hand_in_1, 
                                   right_hand_in_2, out_chain, 'right', triangles_right)
    
    #get finger movement and add to each rotated hand
    out_temp_dict = add_finger_movements(hand_interp, first_hand_r, second_hand_r, 'right', 'mp')
    
    #combine dict
    for key_out in out_temp_dict:
        temp_dict[key_out] = deepcopy(out_temp_dict[key_out])
    
    
    #############################
    ####interpolate face#########
    out_ = cubic_spline_interpolation(Dict_1['face_mp'], Dict_2['face_mp'], n_interp)
    temp_dict['face_mp'] = deepcopy(out_)
        
    out_ = cubic_spline_interpolation(Dict_1['face_holistic_mp'], Dict_2['face_holistic_mp'], n_interp)
    temp_dict['face_holistic_mp'] = deepcopy(out_)
    
    for key in temp_dict:
        if len(temp_dict[key]) != n_interp:
            print(key, len(temp_dict[key]))
            raise ValueError('temp_dict key length not equal to n_interp')
    
    #############################
    ####write into arrays########
    output = list()
    for i in range(n_interp):
        
        ####entire holder###
        temp_holder = dict()
        
        ####face###########
        if zero_rate(Dict_1['face_mp']) > 0.5 or zero_rate(Dict_2['face_mp']) > 0.5:
            temp_holder['face_mp'] = np.zeros_like(Dict_1['face_mp'])
        else:
            temp_holder['face_mp'] = deepcopy(temp_dict['face_mp'][i])

        if zero_rate(Dict_1['face_holistic_mp']) > 0.5 or zero_rate(Dict_2['face_holistic_mp']) > 0.5:
            temp_holder['face_holistic_mp'] = np.zeros_like(Dict_1['face_holistic_mp'])
        else:
            temp_holder['face_holistic_mp'] = deepcopy(temp_dict['face_holistic_mp'][i])
        
        ####pose###########
        if zero_rate(Dict_1['pose_mp']) > 0.5 or zero_rate(Dict_2['pose_mp']) > 0.5:
            temp_holder['pose_mp'] = np.zeros_like(Dict_1['pose_mp'])
        else:
            temp_ = list()
            for j in range(33):
                search_key = 'pose_mp_' + str(j)
                if search_key in temp_dict:
                    temp_.append(temp_dict[search_key][i][0])
                    temp_.append(temp_dict[search_key][i][1])
                    temp_.append(temp_dict[search_key][i][2])
                else:
                    temp_.append(0.)
                    temp_.append(0.)
                    temp_.append(0.)
            temp_holder['pose_mp'] = np.array(temp_)

        if zero_rate(Dict_1['pose_holistic_mp']) > 0.5 or zero_rate(Dict_2['pose_holistic_mp']) > 0.5:
            temp_holder['pose_holistic_mp'] = np.zeros_like(Dict_1['pose_holistic_mp'])
        else:
            temp_ = list()
            for j in range(33):
                search_key = 'pose_holistic_mp_' + str(j)
                if search_key in temp_dict:
                    temp_.append(temp_dict[search_key][i][0])
                    temp_.append(temp_dict[search_key][i][1])
                    temp_.append(temp_dict[search_key][i][2])
                else:
                    temp_.append(0.)
                    temp_.append(0.)
                    temp_.append(0.)
            temp_holder['pose_holistic_mp'] = np.array(temp_)


        ###################
        ####left hand######
        ###################
        # !!!!! holistic hands are always duplicated from mp_hands
        # !!!!! This is because anyway we select non-zero hands from both mp_hand and holistic hand
        # !!!!! at each frame.
        if l_1_change or l_2_change: #one left hand replaced by DWpose
            
            if zero_rate(left_hand_in_1) > 0.5 or zero_rate(left_hand_in_2) > 0.5:
                temp_holder['left_hand_mp'] = np.zeros_like(left_hand_in_1)
                temp_holder['left_hand_holistic_mp'] = np.zeros_like(left_hand_in_1)
            else:
                temp_ = list()

                '''if wrist, hand 5 or hand 17 (any of them) is [-1,-1], 
                   we know the palm rotation doesn't work. 
                   Then, the entire hand interpolation is skipped.

                   To be specific, both pose wrist and hand wrist are [-1,-1], then we think
                   the wrist is [-1,-1].
                '''
                if (left_hand_in_1[0] == -1 and left_hand_in_1[1] == -1) or (
                    left_hand_in_1[15] == -1 and left_hand_in_1[16] == -1) or (
                    left_hand_in_1[51] == -1 and left_hand_in_1[52] == -1) or (
                    left_hand_in_2[0] == -1 and left_hand_in_2[1] == -1) or (
                    left_hand_in_2[15] == -1 and left_hand_in_2[16] == -1) or (
                    left_hand_in_2[51] == -1 and left_hand_in_2[52] == -1):
                    for j in range(21):
                        temp_.append(-1)
                        temp_.append(-1)
                        temp_.append(-1)

                else:
                    for j in range(21):
                        search_key = 'left_hand_mp_' + str(j)
                        if search_key in temp_dict:

                            #change interpolation to -1 if either start or end dim is -1,-1
                            if (left_hand_in_1[3*j] == -1 and left_hand_in_1[3*j+1] == -1) or (
                                left_hand_in_2[3*j] == -1 and left_hand_in_2[3*j+1] == -1):
                                temp_.append(-1)
                                temp_.append(-1)
                                temp_.append(-1)
                            else:
                                temp_.append(temp_dict[search_key][i][0])
                                temp_.append(temp_dict[search_key][i][1])
                                temp_.append(temp_dict[search_key][i][2])
                        else:
                            temp_.append(-1)
                            temp_.append(-1)
                            temp_.append(-1)

                temp_holder['left_hand_mp'] = np.array(temp_)
                temp_holder['left_hand_holistic_mp'] = np.array(temp_)
        
        else: #no hand replaced by DWPose
            
            if zero_rate(left_hand_in_1) > 0.5 or zero_rate(left_hand_in_2) > 0.5:
                temp_holder['left_hand_mp'] = np.zeros_like(left_hand_in_1)
                temp_holder['left_hand_holistic_mp'] = np.zeros_like(left_hand_in_1)
            else:
                temp_ = list()
                for j in range(21):
                    search_key = 'left_hand_mp_' + str(j)
                    if search_key in temp_dict:
                        #print('found')
                        temp_.append(temp_dict[search_key][i][0])
                        temp_.append(temp_dict[search_key][i][1])
                        temp_.append(temp_dict[search_key][i][2])
                    else:
                        temp_.append(0.)
                        temp_.append(0.)
                        temp_.append(0.)
                        print('left hand key not found', search_key)
                temp_holder['left_hand_mp'] = np.array(temp_)
                temp_holder['left_hand_holistic_mp'] = np.array(temp_)
        
        
        ####################
        ####right hand######
        ####################
        if r_1_change or r_2_change: #one right hand replaced by DWpose
            if zero_rate(right_hand_in_1) > 0.5 or zero_rate(right_hand_in_2) > 0.5:
                temp_holder['right_hand_mp'] = np.zeros_like(right_hand_in_1)
                temp_holder['right_hand_holistic_mp'] = np.zeros_like(right_hand_in_1)
            else:
                temp_ = list()

                '''similarly, check the palm rotation validity for right hand'''
                if (right_hand_in_1[0] == -1 and right_hand_in_1[1] == -1) or (
                    right_hand_in_1[15] == -1 and right_hand_in_1[16] == -1) or (
                    right_hand_in_1[51] == -1 and right_hand_in_1[52] == -1) or (
                    right_hand_in_2[0] == -1 and right_hand_in_2[1] == -1) or (
                    right_hand_in_2[15] == -1 and right_hand_in_2[16] == -1) or (
                    right_hand_in_2[51] == -1 and right_hand_in_2[52] == -1):
                    for j in range(21):
                        temp_.append(-1)
                        temp_.append(-1)
                        temp_.append(-1)

                else:
                    for j in range(21):
                        search_key = 'right_hand_mp_' + str(j)
                        if search_key in temp_dict:

                            #change interpolation to -1 if either start or end dim is -1,-1
                            if (right_hand_in_1[3*j] == -1 and right_hand_in_1[3*j+1] == -1) or (
                                right_hand_in_2[3*j] == -1 and right_hand_in_2[3*j+1] == -1):
                                temp_.append(-1)
                                temp_.append(-1)
                                temp_.append(-1)
                            else:
                                temp_.append(temp_dict[search_key][i][0])
                                temp_.append(temp_dict[search_key][i][1])
                                temp_.append(temp_dict[search_key][i][2])
                        else:
                            temp_.append(-1)
                            temp_.append(-1)
                            temp_.append(-1)

                temp_holder['right_hand_mp'] = np.array(temp_)
                temp_holder['right_hand_holistic_mp'] = np.array(temp_)

        else: #no right hand is changed
            if zero_rate(right_hand_in_1) > 0.5 or zero_rate(right_hand_in_2) > 0.5:
                temp_holder['right_hand_mp'] = np.zeros_like(right_hand_in_1)
                temp_holder['right_hand_holistic_mp'] = np.zeros_like(right_hand_in_1)
            else:
                temp_ = list()
                for j in range(21):
                    search_key = 'right_hand_mp_' + str(j)
                    if search_key in temp_dict:
                        temp_.append(temp_dict[search_key][i][0])
                        temp_.append(temp_dict[search_key][i][1])
                        temp_.append(temp_dict[search_key][i][2])
                    else:
                        temp_.append(0.)
                        temp_.append(0.)
                        temp_.append(0.)
                        print('right hand key not found', search_key)
                temp_holder['right_hand_mp'] = np.array(temp_)
                temp_holder['right_hand_holistic_mp'] = np.array(temp_)
        
        output.append(temp_holder)
        
    #print('l_1_changed:', l_1_change)
    #print('l_2_changed:', l_2_change)
    #print('r_1_changed:', r_1_change)
    #print('r_2_changed:', r_2_change)
    
    del(Dict_1, Dict_2)

    return(output[1:-1]) #!!!!! we removed the original start/end frames, this is different from main pipeline 


def combine_keypoints(A, B):
    # Check if dimensions are correct
    if A.shape != (21, 2) or B.shape != (63,):
        raise ValueError("Input arrays have incorrect shape.")
    
    # Reshape B to match the structure needed for combining
    B_reshaped = B.reshape(21, 3)[:, 2]  # Only take the z values
    
    # Combine x, y from A and z from B_reshaped into tuples
    combined = [(x, y, z) for (x, y), z in zip(A, B_reshaped)]
    
    return combined



def transform_keypoints(input_dict):
    # Reshape the 1D arrays into the desired (n, 2) format
    face = input_dict['face_DW'].reshape(1, -1, 2)  # Expand dimensions to (1, 68, 2)
    pose = input_dict['pose_DW'].reshape(-1, 2)
    left_hand = input_dict['left_hand_DW'].reshape(-1, 2)
    right_hand = input_dict['right_hand_DW'].reshape(-1, 2)

    # Build the 'bodies' dictionary
    bodies = {
        'candidate': pose,
        'subset': np.array([[0., 1., 2., 3., 4., 5., 6., 7., 8., -1., -1., 11., -1., -1., 14., 15., 16., 17.]])
    }

    # Build the 'hands' array
    hands = np.array([left_hand, right_hand])

    # Combine everything into the final dictionary
    result = {
        'bodies': bodies,
        'hands': hands,
        'faces': face
    }

    return result


'''
We actively calculate the sholder-elbow-wrist triangle and its normal direction.
Elbow normal is always inner to body
'''
#function to interpolate face, body and hand keypoints of mediapipe
def interpolate_DW(Dict_1_, Dict_2_, 
                   l_1_below, l_2_below, r_1_below, r_2_below,
                   s_e_file, n_interp=12):
    
    Dict_1 = deepcopy(Dict_1_)
    Dict_2 = deepcopy(Dict_2_)
    
    #l_1: either mp or DW pose wrist out
    if l_1_below:
        Dict_1['pose_mp'][46] = 1.2
        Dict_1['bodies']['candidate'][7][1] = 1.2
        
    #l_2: either mp or DW pose wrist out
    if l_2_below:
        Dict_2['pose_mp'][46] = 1.2
        Dict_2['bodies']['candidate'][7][1] = 1.2
        
    #r_1: either mp or DW
    if r_1_below:
        Dict_1['pose_mp'][49] = 1.2
        Dict_1['bodies']['candidate'][4][1] = 1.2
        
    #r_2: either mp or DW
    if r_2_below:
        Dict_2['pose_mp'][49] = 1.2
        Dict_2['bodies']['candidate'][4][1] = 1.2
    
    #the temp dict with keypoint name as key, and interpolations as value
    temp_dict = dict()
    
    #deal with hand outside issue
    if l_1_below:
        left_hand_1 = s_e_file['standard']['standard_left_hand']
        left_hand_1 = move_keypoints(left_hand_1, Dict_1['bodies']['candidate'][7][0], 
                                                  Dict_1['bodies']['candidate'][7][1])
        left_hand_1 = array_to_tuples(left_hand_1)
    else:
        left_hand_1 = combine_keypoints(Dict_1['hands'][0], Dict_1['left_hand_mp'])

    if l_2_below:
        left_hand_2 = s_e_file['standard']['standard_left_hand']
        left_hand_2 = move_keypoints(left_hand_2, Dict_2['bodies']['candidate'][7][0], 
                                                  Dict_2['bodies']['candidate'][7][1])
        left_hand_2 = array_to_tuples(left_hand_2)
    else:
        left_hand_2 = combine_keypoints(Dict_2['hands'][0], Dict_2['left_hand_mp'])
        
    #deal with hand outside issue
    if r_1_below:
        right_hand_1 = s_e_file['standard']['standard_right_hand']
        right_hand_1 = move_keypoints(right_hand_1, Dict_1['bodies']['candidate'][4][0], 
                                                    Dict_1['bodies']['candidate'][4][1])
        right_hand_1 = array_to_tuples(right_hand_1)
    else:
        right_hand_1 = combine_keypoints(Dict_1['hands'][1], Dict_1['right_hand_mp'])

    if r_2_below:
        right_hand_2 = s_e_file['standard']['standard_right_hand']
        right_hand_2 = move_keypoints(right_hand_2, Dict_2['bodies']['candidate'][4][0], 
                                                    Dict_2['bodies']['candidate'][4][1])
        right_hand_2 = array_to_tuples(right_hand_2)
    else:
        right_hand_2 = combine_keypoints(Dict_2['hands'][1], Dict_2['right_hand_mp'])
    
    ####start interpolation##########
    #left hip
    hip_1 = (Dict_1['bodies']['candidate'][11][0], Dict_1['bodies']['candidate'][11][1], Dict_1['pose_mp'][71])
    hip_2 = (Dict_2['bodies']['candidate'][11][0], Dict_2['bodies']['candidate'][11][1], Dict_2['pose_mp'][71])
    out_chain_ = interpolate_points_series(hip_1, hip_2, n_interp)
    temp_dict['pose_DW_11'] = deepcopy(out_chain_)

    #left knee
    knee_1 = (Dict_1['bodies']['candidate'][12][0], Dict_1['bodies']['candidate'][12][1], Dict_1['pose_mp'][77])
    knee_2 = (Dict_2['bodies']['candidate'][12][0], Dict_2['bodies']['candidate'][12][1], Dict_2['pose_mp'][77])
    out_chain_ = interpolate_points_series(knee_1, knee_2, n_interp)
    temp_dict['pose_DW_12'] = deepcopy(out_chain_)

    #left ankle
    ankle_1 = (Dict_1['bodies']['candidate'][13][0], Dict_1['bodies']['candidate'][13][1], Dict_1['pose_mp'][83])
    ankle_2 = (Dict_2['bodies']['candidate'][13][0], Dict_2['bodies']['candidate'][13][1], Dict_2['pose_mp'][83])
    out_chain_ = interpolate_points_series(ankle_1, ankle_2, n_interp)
    temp_dict['pose_DW_13'] = deepcopy(out_chain_)

    
    #right hip
    hip_1 = (Dict_1['bodies']['candidate'][8][0], Dict_1['bodies']['candidate'][8][1], Dict_1['pose_mp'][74])
    hip_2 = (Dict_2['bodies']['candidate'][8][0], Dict_2['bodies']['candidate'][8][1], Dict_2['pose_mp'][74])
    out_chain_ = interpolate_points_series(hip_1, hip_2, n_interp)
    temp_dict['pose_DW_8'] = deepcopy(out_chain_)

    #right knee
    knee_1 = (Dict_1['bodies']['candidate'][9][0], Dict_1['bodies']['candidate'][9][1], Dict_1['pose_mp'][80])
    knee_2 = (Dict_2['bodies']['candidate'][9][0], Dict_2['bodies']['candidate'][9][1], Dict_2['pose_mp'][80])
    out_chain_ = interpolate_points_series(knee_1, knee_2, n_interp)
    temp_dict['pose_DW_9'] = deepcopy(out_chain_)

    #right ankle
    ankle_1 = (Dict_1['bodies']['candidate'][10][0], Dict_1['bodies']['candidate'][10][1], Dict_1['pose_mp'][86])
    ankle_2 = (Dict_2['bodies']['candidate'][10][0], Dict_2['bodies']['candidate'][10][1], Dict_2['pose_mp'][86])
    out_chain_ = interpolate_points_series(ankle_1, ankle_2, n_interp)
    temp_dict['pose_DW_10'] = deepcopy(out_chain_)

    
    #!!!!! middle face/nose , we don't have this in the Sorenson SL Generation code.
    f_mid_1 = (Dict_1['bodies']['candidate'][0][0], Dict_1['bodies']['candidate'][0][1], Dict_1['pose_mp'][2])
    f_mid_2 = (Dict_2['bodies']['candidate'][0][0], Dict_2['bodies']['candidate'][0][1], Dict_2['pose_mp'][2])
    out_chain_ = interpolate_points_series(f_mid_1, f_mid_2, n_interp)
    temp_dict['pose_DW_0'] = deepcopy(out_chain_)
    
    #!!!!! left eye , we don't have this in the Sorenson SL Generation code.
    l_eye_1 = (Dict_1['bodies']['candidate'][15][0], Dict_1['bodies']['candidate'][15][1], Dict_1['pose_mp'][8])
    l_eye_2 = (Dict_2['bodies']['candidate'][15][0], Dict_2['bodies']['candidate'][15][1], Dict_2['pose_mp'][8])
    out_chain_ = interpolate_points_series(l_eye_1, l_eye_2, n_interp)
    temp_dict['pose_DW_15'] = deepcopy(out_chain_)
    
    #!!!!! right eye , we don't have this in the Sorenson SL Generation code.
    r_eye_1 = (Dict_1['bodies']['candidate'][14][0], Dict_1['bodies']['candidate'][14][1], Dict_1['pose_mp'][17])
    r_eye_2 = (Dict_2['bodies']['candidate'][14][0], Dict_2['bodies']['candidate'][14][1], Dict_2['pose_mp'][17])
    out_chain_ = interpolate_points_series(r_eye_1, r_eye_2, n_interp)
    temp_dict['pose_DW_14'] = deepcopy(out_chain_)
    
    #!!!!! left ear , we don't have this in the Sorenson SL Generation code.
    l_ear_1 = (Dict_1['bodies']['candidate'][17][0], Dict_1['bodies']['candidate'][17][1], Dict_1['pose_mp'][23])
    l_ear_2 = (Dict_2['bodies']['candidate'][17][0], Dict_2['bodies']['candidate'][17][1], Dict_2['pose_mp'][23])
    out_chain_ = interpolate_points_series(l_ear_1, l_ear_2, n_interp)
    temp_dict['pose_DW_17'] = deepcopy(out_chain_)
    
    #!!!!! right ear , we don't have this in the Sorenson SL Generation code.
    r_ear_1 = (Dict_1['bodies']['candidate'][16][0], Dict_1['bodies']['candidate'][16][1], Dict_1['pose_mp'][26])
    r_ear_2 = (Dict_2['bodies']['candidate'][16][0], Dict_2['bodies']['candidate'][16][1], Dict_2['pose_mp'][26])
    out_chain_ = interpolate_points_series(r_ear_1, r_ear_2, n_interp)
    temp_dict['pose_DW_16'] = deepcopy(out_chain_)
    
    #neck
    '''we use the average of mediapipe z value of sholders as the z-value here'''
    neck_1 = (Dict_1['bodies']['candidate'][1][0], Dict_1['bodies']['candidate'][1][1], 0.5*(Dict_1['pose_mp'][35] + Dict_1['pose_mp'][38]))
    neck_2 = (Dict_2['bodies']['candidate'][1][0], Dict_2['bodies']['candidate'][1][1], 0.5*(Dict_2['pose_mp'][35] + Dict_2['pose_mp'][38]))
    out_chain_ = interpolate_points_series(neck_1, neck_2, n_interp)
    temp_dict['pose_DW_1'] = deepcopy(out_chain_)

    
    ##########################
    # start from interpolating left sholder
    sholder_1 = (Dict_1['bodies']['candidate'][5][0], Dict_1['bodies']['candidate'][5][1], Dict_1['pose_mp'][35])
    sholder_2 = (Dict_2['bodies']['candidate'][5][0], Dict_2['bodies']['candidate'][5][1], Dict_2['pose_mp'][35])
    out_chain = interpolate_points_series(sholder_1, sholder_2, n_interp)
    temp_dict['pose_DW_5'] = deepcopy(out_chain)
    #print(temp_dict['pose_DW_5'])
    
    '''similarly, we do linear interpolation to avoid inaccurate z'''
    #followed by the left elbow
    down_1 = (Dict_1['bodies']['candidate'][6][0], Dict_1['bodies']['candidate'][6][1], Dict_1['pose_mp'][41])
    down_2 = (Dict_2['bodies']['candidate'][6][0], Dict_2['bodies']['candidate'][6][1], Dict_2['pose_mp'][41])
    out_chain = interpolate_points_series(down_1, down_2, n_interp)
    temp_dict['pose_DW_6'] = deepcopy(out_chain)
    
    #followed by the left wrist (wrist of pose)
    down_1 = (Dict_1['bodies']['candidate'][7][0], Dict_1['bodies']['candidate'][7][1], Dict_1['pose_mp'][47])
    down_2 = (Dict_2['bodies']['candidate'][7][0], Dict_2['bodies']['candidate'][7][1], Dict_2['pose_mp'][47])
    out_chain_l_temp = interpolate_points_series(down_1, down_2, n_interp)
    temp_dict['pose_DW_7'] = deepcopy(out_chain_l_temp)
    
    #obtain sholder-elbow-wrist triangles for left side
    triangles_left = build_triangles(temp_dict['pose_DW_5'], temp_dict['pose_DW_6'], temp_dict['pose_DW_7'])
    
    #followed by the left wrist (wrist of hand)
    down_1 = (left_hand_1[0][0], left_hand_1[0][1], left_hand_1[0][2])
    down_2 = (left_hand_2[0][0], left_hand_2[0][1], left_hand_2[0][2])
    out_chain = interpolate_points_series(down_1, down_2, n_interp)
    temp_dict['left_hand_DW_0'] = deepcopy(out_chain)
    
    #left pose wrist, first and second hand
    left_p_w_1 = deepcopy((Dict_1['bodies']['candidate'][7][0], Dict_1['bodies']['candidate'][7][1], Dict_1['pose_mp'][47]))
    left_p_w_2 = deepcopy((Dict_2['bodies']['candidate'][7][0], Dict_2['bodies']['candidate'][7][1], Dict_2['pose_mp'][47]))
    
    #obtain hand interpolation
    hand_interp, first_hand_l, second_hand_l = rotate_hand_palm_DW(
                                    left_hand_1, left_hand_2, left_p_w_1, left_p_w_2,
                                    out_chain, 'left', triangles_left)
    
    #get finger movement and add to each rotated hand
    out_temp_dict = add_finger_movements(hand_interp, first_hand_l, second_hand_l, 'left', 'DW')
    
    #combine dict
    for key_out in out_temp_dict:
        temp_dict[key_out] = deepcopy(out_temp_dict[key_out])
            
    del(hand_interp, out_temp_dict)
    
    ##########################
    # start from interpolating right sholder
    sholder_1 = (Dict_1['bodies']['candidate'][2][0], Dict_1['bodies']['candidate'][2][1], Dict_1['pose_mp'][38])
    sholder_2 = (Dict_2['bodies']['candidate'][2][0], Dict_2['bodies']['candidate'][2][1], Dict_2['pose_mp'][38])
    out_chain = interpolate_points_series(sholder_1, sholder_2, n_interp)
    temp_dict['pose_DW_2'] = deepcopy(out_chain)
    
    #followed by the right elbow
    down_1 = (Dict_1['bodies']['candidate'][3][0], Dict_1['bodies']['candidate'][3][1], Dict_1['pose_mp'][44])
    down_2 = (Dict_2['bodies']['candidate'][3][0], Dict_2['bodies']['candidate'][3][1], Dict_2['pose_mp'][44])
    out_chain = interpolate_points_series(down_1, down_2, n_interp)
    temp_dict['pose_DW_3'] = deepcopy(out_chain)
    
    #followed by the left wrist (wrist of pose)
    down_1 = (Dict_1['bodies']['candidate'][4][0], Dict_1['bodies']['candidate'][4][1], Dict_1['pose_mp'][50])
    down_2 = (Dict_2['bodies']['candidate'][4][0], Dict_2['bodies']['candidate'][4][1], Dict_2['pose_mp'][50])
    out_chain_r_temp = interpolate_points_series(down_1, down_2, n_interp)
    temp_dict['pose_DW_4'] = deepcopy(out_chain_r_temp)
    
    #obtain sholder-wrist-elbow triangles for right side
    triangles_right = build_triangles(temp_dict['pose_DW_2'], temp_dict['pose_DW_4'], temp_dict['pose_DW_3'])
    
    #followed by the right wrist (wrist of hand)
    down_1 = (right_hand_1[0][0], right_hand_1[0][1], right_hand_1[0][2])
    down_2 = (right_hand_2[0][0], right_hand_2[0][1], right_hand_2[0][2])
    out_chain = interpolate_points_series(down_1, down_2, n_interp)
    temp_dict['right_hand_DW_0'] = deepcopy(out_chain)
    
    #get the first and second right hand pose wrist
    right_p_w_1 = deepcopy((Dict_1['bodies']['candidate'][4][0], Dict_1['bodies']['candidate'][4][1], Dict_1['pose_mp'][50]))
    right_p_w_2 = deepcopy((Dict_2['bodies']['candidate'][4][0], Dict_2['bodies']['candidate'][4][1], Dict_2['pose_mp'][50]))

    #obtain hand interpolation
    hand_interp, first_hand_r, second_hand_r = rotate_hand_palm_DW(
                                    right_hand_1, right_hand_2, right_p_w_1, right_p_w_2,
                                    out_chain, 'right', triangles_right)
    
    #get finger movement and add to each rotated hand
    out_temp_dict = add_finger_movements(hand_interp, first_hand_r, second_hand_r, 'right', 'DW')
    
    #combine dict
    for key_out in out_temp_dict:
        temp_dict[key_out] = deepcopy(out_temp_dict[key_out])
    
    
    #############################
    ####interpolate face#########
    out_ = cubic_spline_interpolation(Dict_1['faces'].flatten(), 
                                      Dict_2['faces'].flatten(), n_interp)

    temp_dict['face_DW'] = deepcopy(out_)
    
    for key in temp_dict:
        if len(temp_dict[key]) != n_interp:
            print(key, len(temp_dict[key]))
            raise ValueError('temp_dict key length not equal to n_interp')
    
    #############################
    ####write into arrays########
    output = list()
    for i in range(n_interp):
        
        ####entire holder###
        temp_holder = dict()
        
        ####face###########
        temp_ = list()
        for j in range(Dict_1['faces'].shape[1]):
            
            #change the interpolation to -1 if either start or end dim is (-1,-1)
            if (Dict_1['faces'][0][j][0] == -1 and Dict_1['faces'][0][j][1] == -1) or (
                Dict_2['faces'][0][j][0] == -1 and Dict_2['faces'][0][j][1] == -1):
                temp_.append(-1)
                temp_.append(-1)
            else:
                temp_.append(temp_dict['face_DW'][i][2*j])
                temp_.append(temp_dict['face_DW'][i][2*j+1])
        
        temp_holder['face_DW'] = np.array(temp_)

        
        ####pose###########
        temp_ = list()
        #we know some DW body pose has 36 keypoints with the latter 18 wrongly extracted
        #so, we fixt the range to 18. So always the first 18 correct ones are built
        for j in range(18):
            
            search_key = 'pose_DW_' + str(j)
            if search_key in temp_dict:
                
                #change the interpolation dim to -1 if either start or end dim is (-1,-1)
                if ((Dict_1['bodies']['candidate'][j][0] == -1) and (
                     Dict_1['bodies']['candidate'][j][1] == -1)) or (
                    (Dict_2['bodies']['candidate'][j][0] == -1) and (
                     Dict_2['bodies']['candidate'][j][1] == -1)):
                    temp_.append(-1)
                    temp_.append(-1)
                else:
                    temp_.append(temp_dict[search_key][i][0])
                    temp_.append(temp_dict[search_key][i][1])
            else:
                temp_.append(-1)
                temp_.append(-1)
        
        temp_holder['pose_DW'] = np.array(temp_)


        ####left hand######
        temp_ = list()
        
        '''if wrist, hand 5 or hand 17 (any of them) is [-1,-1], 
           we know the palm rotation doesn't work. 
           Then, the entire hand interpolation is skipped.
           
           To be specific, both pose wrist and hand wrist are [-1,-1], then we think
           the wrist is [-1,-1].
        '''
        if ((left_p_w_1[0] == -1 and left_p_w_1[1] == -1) and (
            left_hand_1[0][0] == -1 and left_hand_1[0][1] == -1)) or (
            left_hand_1[5][0] == -1 and left_hand_1[5][1] == -1) or (
            left_hand_1[17][0] == -1 and left_hand_1[17][1] == -1) or (
            (left_p_w_2[0] == -1 and left_p_w_2[1] == -1) and (
            left_hand_2[0][0] == -1 and left_hand_2[0][1] == -1)) or (
            left_hand_2[5][0] == -1 and left_hand_2[5][1] == -1) or (
            left_hand_2[17][0] == -1 and left_hand_2[17][1] == -1):
            for j in range(21):
                temp_.append(-1)
                temp_.append(-1)
        
        else:
            for j in range(21):
                search_key = 'left_hand_DW_' + str(j)
                if search_key in temp_dict:

                    #change interpolation to -1 if either start or end dim is -1,-1
                    if (left_hand_1[j][0] == -1 and left_hand_1[j][1] == -1) or (
                        left_hand_2[j][0] == -1 and left_hand_2[j][1] == -1):
                        temp_.append(-1)
                        temp_.append(-1)                    
                    else:
                        temp_.append(temp_dict[search_key][i][0])
                        temp_.append(temp_dict[search_key][i][1])
                else:
                    temp_.append(-1)
                    temp_.append(-1)
                
        temp_holder['left_hand_DW'] = np.array(temp_)
        
        ####right hand######
        temp_ = list()
        
        '''similarly, check the palm rotation validity for right hand'''
        if ((right_p_w_1[0] == -1 and right_p_w_1[1] == -1) and (
            right_hand_1[0][0] == -1 and right_hand_1[0][1] == -1)) or (
            right_hand_1[5][0] == -1 and right_hand_1[5][1] == -1) or (
            right_hand_1[17][0] == -1 and right_hand_1[17][1] == -1) or (
            (right_p_w_2[0] == -1 and right_p_w_2[1] == -1) and (
            right_hand_2[0][0] == -1 and right_hand_2[0][1] == -1)) or (
            right_hand_2[5][0] == -1 and right_hand_2[5][1] == -1) or (
            right_hand_2[17][0] == -1 and right_hand_2[17][1] == -1):
            for j in range(21):
                temp_.append(-1)
                temp_.append(-1)
        
        else:
            for j in range(21):
                search_key = 'right_hand_DW_' + str(j)
                if search_key in temp_dict:

                    #change interpolation to -1 if either start or end dim is -1,-1
                    if (right_hand_1[j][0] == -1 and right_hand_1[j][1] == -1) or (
                        right_hand_2[j][0] == -1 and right_hand_2[j][1] == -1):
                        temp_.append(-1)
                        temp_.append(-1)
                    else:
                        temp_.append(temp_dict[search_key][i][0])
                        temp_.append(temp_dict[search_key][i][1])
                else:
                    temp_.append(-1)
                    temp_.append(-1)

        temp_holder['right_hand_DW'] = np.array(temp_)
        
        #!!!!! transfer to DWPose format for easier official drawing code.
        #This is different from the main pipeline in Sorenson SL Generation
        temp_holder = transform_keypoints(temp_holder)
        
        output.append(temp_holder)
        
    return(output[1:-1]) #!!!!! we removed the original frames, this is different from main pipeline 



#############################
####draw image functions#####

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


def actual_draw_line_DW(image, x1, y1, x2, y2, color, thickness, p_):
    x1 = int(x1 * p_)
    y1 = int(y1 * p_)
    x2 = int(x2 * p_)
    y2 = int(y2 * p_)
    cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    

############################
#the function just used to draw DW other keypoints (face & hand)
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


# !!!!! This combine_image function is a little bit different from the one in 1.2 and in PART_1:
#in this one, we do NOT need real image, so both style img and real img are given style image
def combine_images_intpl(file, style_file, fst_image_path, sec_image_path, comb_image_path, target_size):

    # Combine images as before
    images = [Image.open(x) for x in [style_file,
                                      fst_image_path + file,
                                      sec_image_path + file,
                                      style_file]]
    
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    new_im = new_im.resize((4*target_size, target_size))
    new_im.save(comb_image_path + file)


def combine_opencv_images(variable_1, variable_2, variable_3, comb_image_path, file, target_size):
    # List of images to combine
    images = [variable_1, variable_2, variable_3, variable_1]

    # Calculate the total width and maximum height of the new image
    total_width = sum(image.shape[1] for image in images)
    max_height = max(image.shape[0] for image in images)

    # Create a new empty image with the total width and maximum height
    new_im = np.zeros((max_height, total_width, 3), dtype=np.uint8)

    # Position for the next image to be pasted in new_im
    x_offset = 0
    for image in images:
        new_im[:image.shape[0], x_offset:x_offset + image.shape[1]] = image
        x_offset += image.shape[1]

    # Resize the combined image
    new_im = cv2.resize(new_im, (4*target_size, target_size))

    # Save the combined image
    cv2.imwrite(comb_image_path + file, new_im)


'''
Some times mediapipe hand is all zero. 
In this case, we will replace the all zero hand as much as possible by
no_holistic hand, or hand from nearby frames.
   
Same story, if there are too many [-1,-1] dim in a mediapipe hand,
it will also be replaced.
   
In this way, we will build a final Dict for actual interpolation.
'''
def replace_hand_and_build_actual_dict(Dict_1, Dict_2):
    
    actual_dict_1 = deepcopy(Dict_1['keypoint'][-1])
    actual_dict_2 = deepcopy(Dict_2['keypoint'][0])
    
    if zero_rate(actual_dict_1['left_hand_mp']) > 0.5:
        
        if zero_rate(actual_dict_1['left_hand_holistic_mp']) < 0.25:
            actual_dict_1['left_hand_mp'] = deepcopy(actual_dict_1['left_hand_holistic_mp'])
            
        elif zero_rate(Dict_1['keypoint'][-2]['left_hand_mp']) < 0.25:
            actual_dict_1['left_hand_mp'] = deepcopy(Dict_1['keypoint'][-2]['left_hand_mp'])
            actual_dict_1['hands'][0] = Dict_1['keypoint'][-2]['hands'][0]
            
        elif zero_rate(Dict_1['keypoint'][-2]['left_hand_holistic_mp']) < 0.25:
            actual_dict_1['left_hand_mp'] = deepcopy(Dict_1['keypoint'][-2]['left_hand_holistic_mp'])
            actual_dict_1['hands'][0] = Dict_1['keypoint'][-2]['hands'][0]
            
        elif zero_rate(Dict_1['keypoint'][-3]['left_hand_mp']) < 0.25:
            actual_dict_1['left_hand_mp'] = deepcopy(Dict_1['keypoint'][-3]['left_hand_mp'])
            actual_dict_1['hands'][0] = Dict_1['keypoint'][-3]['hands'][0]
            
        elif zero_rate(Dict_1['keypoint'][-3]['left_hand_holistic_mp']) < 0.25:
            actual_dict_1['left_hand_mp'] = deepcopy(Dict_1['keypoint'][-3]['left_hand_holistic_mp'])
            actual_dict_1['hands'][0] = Dict_1['keypoint'][-3]['hands'][0]

    if zero_rate(actual_dict_1['right_hand_mp']) > 0.5:
        
        if zero_rate(actual_dict_1['right_hand_holistic_mp']) < 0.25:
            actual_dict_1['right_hand_mp'] = deepcopy(actual_dict_1['right_hand_holistic_mp'])
            
        elif zero_rate(Dict_1['keypoint'][-2]['right_hand_mp']) < 0.25:
            actual_dict_1['right_hand_mp'] = deepcopy(Dict_1['keypoint'][-2]['right_hand_mp'])
            actual_dict_1['hands'][1] = Dict_1['keypoint'][-2]['hands'][1]
            
        elif zero_rate(Dict_1['keypoint'][-2]['right_hand_holistic_mp']) < 0.25:
            actual_dict_1['right_hand_mp'] = deepcopy(Dict_1['keypoint'][-2]['right_hand_holistic_mp'])
            actual_dict_1['hands'][1] = Dict_1['keypoint'][-2]['hands'][1]
            
        elif zero_rate(Dict_1['keypoint'][-3]['right_hand_mp']) < 0.25:
            actual_dict_1['right_hand_mp'] = deepcopy(Dict_1['keypoint'][-3]['right_hand_mp'])
            actual_dict_1['hands'][1] = Dict_1['keypoint'][-3]['hands'][1]
            
        elif zero_rate(Dict_1['keypoint'][-3]['right_hand_holistic_mp']) < 0.25:
            actual_dict_1['right_hand_mp'] = deepcopy(Dict_1['keypoint'][-3]['right_hand_holistic_mp'])
            actual_dict_1['hands'][1] = Dict_1['keypoint'][-3]['hands'][1]

    if zero_rate(actual_dict_2['left_hand_mp']) > 0.5:
        
        if zero_rate(actual_dict_2['left_hand_holistic_mp']) < 0.25:
            actual_dict_2['left_hand_mp'] = deepcopy(actual_dict_2['left_hand_holistic_mp'])
            
        elif zero_rate(Dict_2['keypoint'][1]['left_hand_mp']) < 0.25:
            actual_dict_2['left_hand_mp'] = deepcopy(Dict_2['keypoint'][1]['left_hand_mp'])
            actual_dict_2['hands'][0] = Dict_2['keypoint'][1]['hands'][0]
            
        elif zero_rate(Dict_2['keypoint'][1]['left_hand_holistic_mp']) < 0.25:
            actual_dict_2['left_hand_mp'] = deepcopy(Dict_2['keypoint'][1]['left_hand_holistic_mp'])
            actual_dict_2['hands'][0] = Dict_2['keypoint'][1]['hands'][0]
            
        elif zero_rate(Dict_2['keypoint'][2]['left_hand_mp']) < 0.25:
            actual_dict_2['left_hand_mp'] = deepcopy(Dict_2['keypoint'][2]['left_hand_mp'])
            actual_dict_2['hands'][0] = Dict_2['keypoint'][2]['hands'][0]
            
        elif zero_rate(Dict_2['keypoint'][2]['left_hand_holistic_mp']) < 0.25:
            actual_dict_2['left_hand_mp'] = deepcopy(Dict_2['keypoint'][2]['left_hand_holistic_mp'])
            actual_dict_2['hands'][0] = Dict_2['keypoint'][2]['hands'][0]

    if zero_rate(actual_dict_2['right_hand_mp']) > 0.5:
        
        if zero_rate(actual_dict_2['right_hand_holistic_mp']) < 0.25:
            actual_dict_2['right_hand_mp'] = deepcopy(actual_dict_2['right_hand_holistic_mp'])
            
        elif zero_rate(Dict_2['keypoint'][1]['right_hand_mp']) < 0.25:
            actual_dict_2['right_hand_mp'] = deepcopy(Dict_2['keypoint'][1]['right_hand_mp'])
            actual_dict_2['hands'][1] = Dict_2['keypoint'][1]['hands'][1]
            
        elif zero_rate(Dict_2['keypoint'][1]['right_hand_holistic_mp']) < 0.25:
            actual_dict_2['right_hand_mp'] = deepcopy(Dict_2['keypoint'][1]['right_hand_holistic_mp'])
            actual_dict_2['hands'][1] = Dict_2['keypoint'][1]['hands'][1]
            
        elif zero_rate(Dict_2['keypoint'][2]['right_hand_mp']) < 0.25:
            actual_dict_2['right_hand_mp'] = deepcopy(Dict_2['keypoint'][2]['right_hand_mp'])
            actual_dict_2['hands'][1] = Dict_2['keypoint'][2]['hands'][1]
            
        elif zero_rate(Dict_2['keypoint'][2]['right_hand_holistic_mp']) < 0.25:
            actual_dict_2['right_hand_mp'] = deepcopy(Dict_2['keypoint'][2]['right_hand_holistic_mp'])
            actual_dict_2['hands'][1] = Dict_2['keypoint'][2]['hands'][1]

    return(actual_dict_1, actual_dict_2)


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


def load_even_pickles(folder_path):
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
            
        # Check if it's even
        if file_id % 2 != 0:
            raise ValueError(f"Non-even integer filename found: {file_id}")
            
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
#########final class#############
#################################

class MotionInterpolation:
    
    def __init__(self, path_out, support_path):

        self.path_out = path_out + 'raw/individual_pickles/'
        self.s_e_path = support_path + 'std_only.pickle'

        with open(self.s_e_path, 'rb') as handle:
            self.s_e_file = pickle.load(handle)
            
    def interpolate(self, s_file, e_file, num_itpl):

        self.path_s = self.path_out + s_file
        self.path_e = self.path_out + e_file

        self.start_id = int( s_file.split('.')[0] )
        self.itpl_id = self.start_id + 1
        
        with open(self.path_s, 'rb') as handle:
            Dict_1 = pickle.load(handle)
            
        with open(self.path_e, 'rb') as handle:
            Dict_2 = pickle.load(handle)
            
        #we use the first video size
        f_H = Dict_1['info']['H']
        f_W = Dict_1['info']['W']
        fps = Dict_1['info']['fps']

        #we want to select or combine mediapipe hands, 
        #when a hand is all zero, we replace the hand with that from nearby frame
        actual_1, actual_2 = replace_hand_and_build_actual_dict(Dict_1, Dict_2)
        
        #decide whether hands are below screen
        l_1_below = check_hand_below_screen(actual_1, 'left')
        r_1_below = check_hand_below_screen(actual_1, 'right')
        l_2_below = check_hand_below_screen(actual_2, 'left')
        r_2_below = check_hand_below_screen(actual_2, 'right')
        
        #interpolate mediapipe keypoints
        interpolation_mp = interpolate_mp(actual_1, actual_2,
                                       l_1_below, l_2_below, r_1_below, r_2_below,
                                       self.s_e_file, n_interp=num_itpl)

        #interpolate DWPose keypoints
        interpolation_DW = interpolate_DW(actual_1, actual_2,
                                          l_1_below, l_2_below, r_1_below, r_2_below,
                                          self.s_e_file, n_interp=num_itpl)
        
        if len(interpolation_mp) != len(interpolation_DW):
            raise ValueError('interpolation length not the same between mp and DW')

        #save to file
        Dict_ = defaultdict()
        vec_array_new = list()

        for i_new in range(len(interpolation_mp)):
            temp_dict_0 = dict()
            
            for key in interpolation_mp[i_new]:
                temp_dict_0[key] = deepcopy(interpolation_mp[i_new][key])
            
            for key in interpolation_DW[i_new]:
                temp_dict_0[key] = deepcopy(interpolation_DW[i_new][key])

            vec_array_new.append(temp_dict_0)
            del(temp_dict_0)

        Dict_['keypoint'] = vec_array_new

        Dict_['info'] = {'interpolation': True, 
                         'start_file': self.path_s, 'end_file': self.path_e, 'n_interpolation': num_itpl}
        
        with open(self.path_out + str(self.itpl_id) + '.pickle', 'wb') as handle:
            pickle.dump(Dict_, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def run(self, n_interpolation):
        data_list = load_even_pickles(self.path_out)
        for i in range( len(data_list)-1 ):
            self.interpolate(data_list[i], data_list[i+1], n_interpolation)





