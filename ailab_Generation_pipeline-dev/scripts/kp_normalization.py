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
from copy import deepcopy
from tqdm import tqdm
#from multiprocessing import Pool, Manager
from pathlib import Path

# Change to the model directory for DWPose
#sys.path.append('./ailab_DWPose_not_git/ControlNet-v1-1-nightly/')
#from annotator.dwpose import DWposeDetector_canlin_no_output_img

'''
One issue we should pay extra attention:
face_mp (from FaceMesh of mediapipe) is not stable. The zero face rate is nearly 40%. 
But the face_holistice_mp is okay, with zero rate less than 0.1% 
'''

##################################
mp_pose = mp.solutions.pose #pose model
mp_face_mesh = mp.solutions.face_mesh # FaceMesh model
mp_hands = mp.solutions.hands #hands model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

mp_holistic = mp.solutions.holistic  # Holistic model: used for hand distinguish/trim, etc.


#detect the landmarks from the image
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #color conversion
    image.flags.writeable = False                  #Image is no longer writeable
    results = model.process(image)                 #Make prediction
    image.flags.writeable = True                   #Image is writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # color conversion
    return image, results


#formly define the landmark (keypoint) extraction function
def extract_keypoints_holistic(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return(pose, face, lh, rh)


'''
When there's an unexpected handedness value for a single detected hand, 
we assume it to be the right hand and extract its keypoints.

If two hands are detected and either one or both have unexpected handedness values, 
we extract keypoints for both hands, assigning the first detected hand as right and the 
second as left.

If the handedness value is all expected. We will append the keypoints accordingly.
'''

def extract_keypoints(pose_results, face_mesh_results, hand_results):
    # Extracting Pose Landmarks
    pose = np.array([[res.x, res.y, res.z] for res in pose_results.pose_landmarks.landmark]).flatten() if pose_results.pose_landmarks else np.zeros(33 * 3)
    
    # Extracting Face Landmarks
    face = np.array([[res.x, res.y, res.z] for res in face_mesh_results.multi_face_landmarks[0].landmark]).flatten() if face_mesh_results.multi_face_landmarks else np.zeros(478 * 3)
    
    # Initialize empty hand keypoints
    right_hand = np.zeros(21 * 3)
    left_hand = np.zeros(21 * 3)

    # Check number of hands detected
    num_hands_detected = len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0

    valid_handedness_values = ['Right', 'Left']

    if num_hands_detected == 1:
        # Only one hand is detected, rely on handedness
        handedness = hand_results.multi_handedness[0].classification[0].label

        # Check for valid handedness value
        if handedness not in valid_handedness_values:
            # Handle unexpected handedness value here (e.g., log a warning, skip the frame, etc.)
            print(f"Warning: Unexpected handedness value '{handedness}'")
            right_hand = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_results.multi_hand_landmarks[0].landmark]).flatten()
            return pose, face, right_hand, left_hand
        
        if handedness == 'Right':
            right_hand = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_results.multi_hand_landmarks[0].landmark]).flatten()
        else:
            left_hand = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_results.multi_hand_landmarks[0].landmark]).flatten()
    
    elif num_hands_detected == 2:
        # Two hands are detected, first check handedness
        handedness_0 = hand_results.multi_handedness[0].classification[0].label
        handedness_1 = hand_results.multi_handedness[1].classification[0].label

        # Check for valid handedness values
        if handedness_0 not in valid_handedness_values or handedness_1 not in valid_handedness_values:
            print(f"Warning: Unexpected handedness values '{handedness_0}' and '{handedness_1}'")
            right_hand = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_results.multi_hand_landmarks[0].landmark]).flatten()
            left_hand = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_results.multi_hand_landmarks[1].landmark]).flatten()
            return pose, face, right_hand, left_hand
        
        # If both hands have different handedness
        if handedness_0 != handedness_1:
            if handedness_0 == 'Right':
                right_hand = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_results.multi_hand_landmarks[0].landmark]).flatten()
                left_hand = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_results.multi_hand_landmarks[1].landmark]).flatten()
            else:
                right_hand = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_results.multi_hand_landmarks[1].landmark]).flatten()
                left_hand = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_results.multi_hand_landmarks[0].landmark]).flatten()
        
        # If both hands are detected as the same handedness
        else:
            # Ignore handedness and assign the first as right hand and the second as left hand
            right_hand = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_results.multi_hand_landmarks[0].landmark]).flatten()
            left_hand = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_results.multi_hand_landmarks[1].landmark]).flatten()
    
    '''
    The handedness assume the image is mirrored, which is not true for Signing Savvy.
    As a result, we will just switch the left and right hand!
    '''
    temp = right_hand
    right_hand = left_hand
    left_hand = temp
    del(temp)
    
    return pose, face, right_hand, left_hand


def reshape_array(arr):
    if len(arr) % 3 != 0:
        raise ValueError("The length of the array must be a multiple of 3.")
    return arr.reshape(-1, 3)


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


#two supportive functions:
def euclidean_distance(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) ** 0.5


def dimensionwise_distance(hand1, hand2):
    """Computes the dimension-wise distance between two hands, ignoring z-values."""
    distances = [abs(a - b) for i, (a, b) in enumerate(zip(hand1, hand2)) if i % 3 != 2]
    return sum(distances)


'''We apply the strict L-2 distance here'''
def dimensionwise_distance_single_max(hand1, hand2):
    if len(hand1) % 3 != 0 or len(hand2) % 3 != 0:
        raise ValueError("Hand keypoints should be a multiple of 3")
    
    # Compute the L-2 distance for each pair of (xi, yi) points, 
    # considering zero distances if either of the keypoints is zero.
    distances = [
        0 if (hand1[i] == 0 and hand1[i+1] == 0) or (hand2[i] == 0 and hand2[i+1] == 0) 
        else ((hand1[i] - hand2[i])**2 + (hand1[i+1] - hand2[i+1])**2)**0.5 
        for i in range(0, len(hand1), 3)
    ]
    return max(distances)


'''we calculate the average-distance based on L2'''
def dimensionwise_distance_single_avg(hand1, hand2):
    
    if len(hand1) % 3 != 0 or len(hand2) % 3 != 0:
        raise ValueError("Hand keypoints should be a multiple of 3")
        
    # If either hand is mostly zeros, return zero
    if zero_rate(hand1) > 0.5 or zero_rate(hand2) > 0.5:
        return 0.
        
    # Compute the L-2 distance for each pair of (xi, yi) points.
    distances = [((hand1[i] - hand2[i])**2 + (hand1[i+1] - hand2[i+1])**2)**0.5 for i in range(0, len(hand1), 3)]
    
    # Calculate the average distance if there are any distances calculated
    if distances:
        return sum(distances) / len(distances)
    else:
        return 0.  # Return 0 if no distances to avoid division by zero


'''
We define overlap hand like this: The left and right hand (both hands) are not zeros, and
their elementwise distence is less than the threshold. Also, we in addition require that
overlapped hands is close to one wrist, while the other wrist is missing or outside the image.

In this case, we will delete one hand (change to zero): we will assume the left/right hand
assignment is correct, which is basically proved to be true on character sign videos.
Then, we will delete the hand from the far away pose wrist.

Note that the distance_threshold here is different from that in correct_handedness_temporal.
'''

#function to detect & remove one hand if both hands are overlapping
def handle_overlap_hand_new(pose_landmark, left_hand, right_hand):
    
    #pose_landmark = data_dict['pose']
    #left_hand = data_dict['left_hand']
    #right_hand = data_dict['right_hand']
    
    #only make sense to continue if both and are not zeros
    if any(value != 0 for value in left_hand) and any(value != 0 for value in right_hand):
    
        hand_d = dimensionwise_distance(left_hand, right_hand)

        #left wrist to right hand wrist, similar naming strategy below
        l_d_r = euclidean_distance([pose_landmark[15 * 3], pose_landmark[15 * 3 + 1]], 
                                   [right_hand[0], right_hand[1]])
        l_d_l = euclidean_distance([pose_landmark[15 * 3], pose_landmark[15 * 3 + 1]], 
                                   [left_hand[0], left_hand[1]])
        r_d_r = euclidean_distance([pose_landmark[16 * 3], pose_landmark[16 * 3 + 1]], 
                                   [right_hand[0], right_hand[1]])
        r_d_l = euclidean_distance([pose_landmark[16 * 3], pose_landmark[16 * 3 + 1]], 
                                   [left_hand[0], left_hand[1]])

        #0.4 is the threshold for dimensionwise distance, we have testify this value, should be fine
        # !!!!! 0.4 is different from the SL Generation code, where we use 0.6. We are more cautious here.
        #0.08 is the lower bound wrist distance, 0.3 is the upper bound
        #this says that: if hands are close and (hands are close to one wrist, away from the other wrist, or the other wrist is outside the image)
        if (hand_d <= 0.4) and ((r_d_r <= 0.08 and (l_d_r >= 0.3 or pose_landmark[15 * 3] >= 0.93 or pose_landmark[15 * 3 + 1] >= 0.93)) or (
                                 l_d_r <= 0.08 and (r_d_r >= 0.3 or pose_landmark[16 * 3] >= 0.93 or pose_landmark[16 * 3 + 1] >= 0.93)) or (
                                 r_d_l <= 0.08 and (l_d_l >= 0.3 or pose_landmark[15 * 3] >= 0.93 or pose_landmark[15 * 3 + 1] >= 0.93)) or (
                                 l_d_l <= 0.08 and (r_d_l >= 0.3 or pose_landmark[16 * 3] >= 0.93 or pose_landmark[16 * 3 + 1] >= 0.93))):

            #right wrist close: so left wrist should be far away or outside, so we remove left hand
            if r_d_r <= 0.08 or r_d_l <= 0.08:
                left_hand = np.zeros(21 * 3)

            #left wrist close: so right wrist should be far away or outside, so we remove right hand
            if l_d_r <= 0.08 or l_d_l <= 0.08:
                right_hand = np.zeros(21 * 3)
                
    return(left_hand, right_hand)


def count_invalid_dimensions(array_2):
    """
    Count the number of invalid dimensions in array_2 (DWPose).
    An invalid dimension is defined as having either x or y coordinates to be -1.
    
    !!!!! In the same function previously (in extract_kp_from_Kylie_data), we define
    invalid dimension as both x and y equals -1. This is not good:
    In SL_generation code, we force x and y to be -1 if either of them is -1.
    But without this procedure, ususally we cannot assume both x and y to be -1.
    Hence, we changed to either x or y coordinates to be -1.
    That says, the kp_extract_main_0 and 1.py has defult in extract_kp_from_Kylie_data!
    But luckly we do not use DWPose at all so doesn't matter.....
    
    Finally, instead of checking whether the value exactly match -1, we check < -0.9 instead

    Parameters:
    - array_2 (np.ndarray): 2D array of keypoints [x1, y1], [x2, y2], ..., [xn, yn]

    Returns:
    - int: Number of invalid dimensions
    """
    num_invalid = 0
    for dim in array_2:
        if dim[0] < -0.9 or dim[1] < -0.9:
            num_invalid += 1
            
        #print(dim)
    return num_invalid


def mp_DW_hand_dist(array_1, array_2):
    """
    Calculate the average and maximum Euclidean distances between corresponding keypoints 
    in array_1 and array_2 based on x and y coordinates, excluding keypoints in array_2
    
    !!!!! Same story, we changed checking [-1, -1] to check either x or y < 0.9.
    This is different from extract_kp_from_Kylie_data.

    Parameters:
    - array_1 (np.ndarray): 1D array of keypoints including z coordinates [x1, y1, z1, ..., xn, yn, zn]
    - array_2 (np.ndarray): 2D array of keypoints [[x1, y1], [x2, y2], ..., [xn, yn]]

    Returns:
    - float: average distance between valid keypoints
    - float: maximum distance between valid keypoints
    """
    # Reshape array_1 to extract x and y coordinates, ignoring z coordinates
    xy_array_1 = array_1.reshape(-1, 3)[:, :2]
    
    # Ensure array_2 is a numpy array (in case it isn't)
    array_2 = np.array(array_2)
    
    # Create a mask manually to filter out invalid keypoints from array_2
    valid_mask = []
    for point in array_2:
        if point[0] < -0.9 or point[1] < -0.9:
            valid_mask.append(False)
        else:
            valid_mask.append(True)
    
    # Convert list to numpy array for indexing
    valid_mask = np.array(valid_mask)
    
    # Apply mask to both arrays
    valid_xy_array_1 = xy_array_1[valid_mask]
    valid_array_2 = array_2[valid_mask]
    
    # Calculate Euclidean distances between valid keypoints
    distances = np.linalg.norm(valid_xy_array_1 - valid_array_2, axis=1)
    
    # Calculate average and maximum distance
    if distances.size == 0:  # Check if there are no valid points after masking
        return float(10), float(10)  # Return NaN if no valid distances to calculate
    else:
        average_distance = np.mean(distances)
        maximum_distance = np.max(distances)
    
    return(average_distance, maximum_distance)


'''
Given the mediapipe hand, we want to decide whether it is more 'left' or more 'right':
    We will calculate the distance between this hand and each DWpose hand, 1/distance will be the score
    We will calculate the distance between this hand wrist and each DWpose wrist, 1/distance is the score
    We add the score to decide its left score or right score, indicating how 'left' the given hand is, how right the given hand is.
'''
# this is the function to actually choose the most appropriate hand
def left_or_right_for_given_hand(given_hand, l_h_DW, r_h_DW, l_w_DW, r_w_DW):
    
    #the score on whether the given mediapipe hand is 'how left' or 'how right'
    l_score, r_score = 0., 0.
    
    #DWPose hand vote, must be not too much [-1,-1], in which case we trust it the most
    if count_invalid_dimensions(l_h_DW) <= 3: #left 
        ave_l, max_l = mp_DW_hand_dist(given_hand, l_h_DW)
        l_score += 1/(ave_l + 0.00000001)
    
    if count_invalid_dimensions(r_h_DW) <= 3: #right
        ave_r, max_r = mp_DW_hand_dist(given_hand, r_h_DW)
        r_score += 1/(ave_r + 0.00000001)


    #left pose wrist vote, must be non-zero and not -1 (valid) pose
    if l_w_DW[0] >= -0.9 and l_w_DW[1] >= -0.9 and l_w_DW[0] != 0 and l_w_DW[1] != 0:
        x1 = l_w_DW[0]
        y1 = l_w_DW[1]
        
        x2 = given_hand[0]
        y2 = given_hand[1]
        
        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        l_score += 0.4/(distance + 0.00000001) #0.4 is the weight we add to wrist decision
        
    #right pose wrist vote, must be non-zero (valid) pose
    if r_w_DW[0] >= -0.9 and r_w_DW[1] >= -0.9 and r_w_DW[0] != 0 and r_w_DW[1] != 0:
        x1 = r_w_DW[0]
        y1 = r_w_DW[1]
        
        x2 = given_hand[0]
        y2 = given_hand[1]
        
        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        r_score += 0.4/(distance + 0.00000001)

    return(l_score, r_score)


#function to finally decide left/right hand
def decide_l_r_hand(lh, rh, lh_DW, rh_DW, lw, rw):
    
    #calculate with DWPose only if the mediapipe hand is not zero
    if zero_rate(lh) < 0.2:
        #l_l is how 'left hand' the left hand looks like
        #r_l is how 'right hand' the left hand looks liks 
        l_l, r_l = left_or_right_for_given_hand(lh, lh_DW, rh_DW, lw, rw)
    else:
        l_l, r_l = 0., 0.
    
    #calculate with DWPose only if the mediapipe hand is not zero
    if zero_rate(rh) < 0.2:
        #similarly, l_r is how 'left hand' right hand looks like
        #r_r is how 'right hand' the right hand looks like
        l_r, r_r = left_or_right_for_given_hand(rh, lh_DW, rh_DW, lw, rw)
    else:
        l_r, r_r = 0., 0.
    
    List = [['l_l', l_l], ['r_l', r_l], ['l_r', l_r], ['r_r', r_r]]
        
    List = sorted(List, key=lambda x: x[1], reverse=True)
    top_sub_list = List[0] #top top_sub_list is the highest score hand
    top_str = top_sub_list[0]
    
    hand_should_be = top_str.split('_')[0]
    actual_hand = top_str.split('_')[1]
    
    #return: decided left hand, decided right hand
    #if reverted, we return 'Reverted', otherwise return 'Not_reverted'
    if hand_should_be == 'l' and actual_hand == 'l':
        return(lh, rh, 'Not_reverted')
    if hand_should_be == 'l' and actual_hand == 'r':
        return(rh, lh, 'Reverted')
    if hand_should_be == 'r' and actual_hand == 'l':
        return(rh, lh, 'Reverted')
    if hand_should_be == 'r' and actual_hand == 'r':
        return(lh, rh, 'Not_reverted')


def fix_scattered_keypoints(keypoints, bd_1=0.96, bd_2=0.86):
    """
    Fix scattered keypoints in a hand keypoint array.

    :param keypoints: A (21, 2) numpy array of hand keypoints.
    :return: A (21, 2) numpy array with scattered points replaced by [-1, -1].
    
    We detect scatterred points like this:
    if there exists keypoints with y >= bd_1;
    if there exists keypoints with y <= bd_2;
    but there is not keypoint with y in between;
    
    Then, we regard all the keypoints with y <= bd_2 scattered points, we change to [-1 -1]
    """
    # Set the other value to -1 if one is -1
    for i in range(len(keypoints)):
        if keypoints[i][0] == -1 or keypoints[i][1] == -1:
            keypoints[i] = np.asarray([-1, -1])

    # Check the specified conditions for scattered points
    has_low_keypoints = any(point[1] >= bd_1 for point in keypoints if point[1] != -1)
    has_high_keypoints = any(point[1] <= bd_2 for point in keypoints if point[1] != -1)
    has_no_mid_keypoints = not any(bd_2 < point[1] < bd_1 for point in keypoints if point[1] != -1)

    # If conditions are met, fix scattered points
    if has_low_keypoints and has_high_keypoints and has_no_mid_keypoints:
        for i in range(len(keypoints)):
            if keypoints[i][1] != -1 and keypoints[i][1] <= bd_2:
                keypoints[i] = np.asarray([-1, -1])

    return keypoints


#keypoint extraction from one image
def obtain_key_vector_from_image(image_path, model):
    """
    Process a single image to obtain keypoint vectors using Mediapipe and DWPose.

    Args:
        image (numpy.ndarray): The input image in BGR format (as read by OpenCV).
        model: The DWPose model for extracting keypoints.

    Returns:
        dict: A dictionary containing keypoint vectors and other relevant data.
    """

    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Check if the image was successfully loaded
    if image is None:
        raise ValueError(f"Failed to load image from path: {image_path}")
    
    # Get the height and width of the image
    H, W, _ = image.shape  # shape returns (height, width, channels)

    vec_dict = {}

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, \
         mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_func, \
         mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh_func, \
         mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands_func:

        ##### Mediapipe detections #####
        _, results = mediapipe_detection(image, holistic)
        _, pose_results = mediapipe_detection(image, pose_func)
        _, face_mesh_results = mediapipe_detection(image, face_mesh_func)
        _, hand_results = mediapipe_detection(image, hands_func)

        # Transfer keypoint results into arrays
        pose, face, rh, lh = extract_keypoints(pose_results, face_mesh_results, hand_results)

        # Obtain the keypoint array from the holistic model
        pose_ho, face_ho, lh_ho, rh_ho = extract_keypoints_holistic(results)

        # Deal with overlapped hands first
        lh, rh = handle_overlap_hand_new(pose, lh, rh)
        lh_ho, rh_ho = handle_overlap_hand_new(pose, lh_ho, rh_ho)

        # Use the DWPose model to extract keypoints
        kp_vectors = model(image)

        # Fix scattered keypoints for hands
        fixed_l = fix_scattered_keypoints(kp_vectors['hands'][0])
        fixed_r = fix_scattered_keypoints(kp_vectors['hands'][1])
        kp_vectors['hands'][0] = fixed_l
        kp_vectors['hands'][1] = fixed_r
        kp_vectors['bodies']['candidate'] = deepcopy(kp_vectors['bodies']['candidate'][:18, :])

        #### Left/Right hand correction of Mediapipe ####
        lh, rh, revert = decide_l_r_hand(lh, rh, kp_vectors['hands'][0],
                                         kp_vectors['hands'][1],
                                         pose[45:48], pose[48:51])

        lh_ho, rh_ho, revert_ho = decide_l_r_hand(lh_ho, rh_ho, kp_vectors['hands'][0],
                                                  kp_vectors['hands'][1],
                                                  pose[45:48], pose[48:51])

        #### Add Mediapipe data to the dictionary ####
        kp_vectors['pose_mp'] = pose
        kp_vectors['face_mp'] = face

        kp_vectors['pose_holistic_mp'] = pose_ho
        kp_vectors['face_holistic_mp'] = face_ho

        if zero_rate(lh) < 0.2:
            kp_vectors['left_hand_mp'] = lh
        else:
            kp_vectors['left_hand_mp'] = lh_ho

        if zero_rate(rh) < 0.2:
            kp_vectors['right_hand_mp'] = rh
        else:
            kp_vectors['right_hand_mp'] = rh_ho

        kp_vectors['left_hand_holistic_mp'] = lh_ho
        kp_vectors['right_hand_holistic_mp'] = rh_ho

        kp_vectors['hand_revert'] = revert
        kp_vectors['holistic_hand_revert'] = revert_ho

        vec_dict = kp_vectors

    return vec_dict, H, W


def obtain_key_vector(file_name, ID, model):
    
    #Dict = defaultdict(dict)
    vec_array = list()
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, \
         mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_func, \
         mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh_func, \
         mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands_func:

        vidcap = cv2.VideoCapture(file_name)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        number_of_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Get the video's original width and height
        frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))

        success, image = vidcap.read()
        count = 0

        pbar = tqdm(total=number_of_frames, desc=f'Processing video {ID}.mp4/mov')
        while success:

            #####Mediapipe detections############
            _, results = mediapipe_detection(image, holistic)
            _, pose_results = mediapipe_detection(image, pose_func)
            _, face_mesh_results = mediapipe_detection(image, face_mesh_func)
            _, hand_results = mediapipe_detection(image, hands_func)
            
            # Transfer keypoint results into arrays
            pose, face, rh, lh = extract_keypoints(pose_results, face_mesh_results, hand_results)
            
            #obtain the keypoint array from the holistic model
            pose_ho, face_ho, lh_ho, rh_ho = extract_keypoints_holistic(results)
            
            #deal with overlapped hands first
            lh, rh = handle_overlap_hand_new(pose, lh, rh)
            lh_ho, rh_ho = handle_overlap_hand_new(pose, lh_ho, rh_ho)
            
            kp_vectors = model(image)

            fixed_l = fix_scattered_keypoints(kp_vectors['hands'][0])
            fixed_r = fix_scattered_keypoints(kp_vectors['hands'][1])
            kp_vectors['hands'][0] = fixed_l
            kp_vectors['hands'][1] = fixed_r
            kp_vectors['bodies']['candidate'] = deepcopy(kp_vectors['bodies']['candidate'][:18, :])
            
            ####Left/Right hand correction of mediapipe#######
            lh, rh, revert = decide_l_r_hand(lh, rh, kp_vectors['hands'][0], 
                                                     kp_vectors['hands'][1], 
                                                     pose[45:48], pose[48:51])
            
            lh_ho, rh_ho, revert_ho = decide_l_r_hand(lh_ho, rh_ho, kp_vectors['hands'][0], 
                                                                    kp_vectors['hands'][1], 
                                                                    pose[45:48], pose[48:51])

            
            ####add mediapipe stuff###############
            kp_vectors['pose_mp'] = pose
            kp_vectors['face_mp'] = face
                        
            kp_vectors['pose_holistic_mp'] = pose_ho
            kp_vectors['face_holistic_mp'] = face_ho
            
            if zero_rate(lh) < 0.2:
                kp_vectors['left_hand_mp'] = lh
            else:
                kp_vectors['left_hand_mp'] = lh_ho
            
            if zero_rate(rh) < 0.2:
                kp_vectors['right_hand_mp'] = rh
            else:
                kp_vectors['right_hand_mp'] = rh_ho
            
            kp_vectors['left_hand_holistic_mp'] = lh_ho
            kp_vectors['right_hand_holistic_mp'] = rh_ho

            kp_vectors['hand_revert'] = revert
            kp_vectors['holistic_hand_revert'] = revert_ho

            vec_array.append(kp_vectors)

            success, image = vidcap.read()
            count += 1

            pbar.update(1)  # Update progress bar by 1
        pbar.close()

        vidcap.release()
        
    return vec_array, fps, number_of_frames, frame_height, frame_width


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
That is, the difference Dict['body']['candidate'][1][1] and 
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

The standard face length is also based on Lady Brenda Cartwright in character a:
../../data/SigningSavvy_Dict/new_2_key_vectors_indpt_h_adjust_DWPose/a/1-#-the-letter-a/2.pickle
We use the DWPose face, we take the average y-value beween node 1 and 17 (0 and 16 if 0-index); 
Then we take the average y-value between node 5 and 13 (4 and 12 in 0-index)
Then we calculate the absolute dist between the y-value
'''
def do_normalization(path, path_out_, ID, model, num_insert_interpolation, sty_std,
                     std_sholder=0.323, std_d_y=0.208, std_x=0.502, std_y=0.559, std_ear_x=0.159, std_face_length=0.0821):
    
    if sty_std is not None:
        std_sholder = sty_std[0]
        std_d_y = sty_std[1]
        std_x = sty_std[2]
        std_y = sty_std[3]
        std_ear_x = sty_std[4]
        std_face_length = sty_std[5]
    
    #extract keypoint
    #vec_array, fps, n_frames, f_H, f_W = obtain_key_vector(path, ID, model)
    
    #directly load keypoint
    with open(path, 'rb') as file:
        # Load the data from the pickle file
        vec_array = pickle.load(file)

    fps, n_frames, f_H, f_W = vec_array[0]['fps'], len(vec_array), vec_array[0]['H'], vec_array[0]['W']
    
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

    #only change face and hands now: if the face is too narrow or two wide, change it.
    #however, face length is not considered here !!
    #hands follow face re-size factor.
    max_d_ear = 0.
    for i in range(len(vec_array_4)):
        x1 = vec_array_4[i]['bodies']['candidate'][16][0]
        x2 = vec_array_4[i]['bodies']['candidate'][17][0]
        distance = abs(x1 - x2)
        max_d_ear = max(distance, max_d_ear)

    r_x_ear = std_ear_x/max_d_ear

    #we also need to record the nose point before resize, so that we can move back after resizing
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
        face_holistic_mp_mask = [j for j in range( int(len(vec_array_6[0]['face_holistic_mp'])/3) )]
        
        face_DW_mask = [j for j in range(68)]

        processed = process_keypoints_with_mask(vec_array_6[i], 
                                                {'pose_mp':[0,1,2,3,4,5,6,7,8,9,10, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
                                                 'bodies':[0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                                                 'faces':face_DW_mask,
                                                 'face_mp':face_mp_mask,
                                                 'face_holistic_mp': face_holistic_mp_mask,
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
        face_holistic_mp_mask = [j for j in range( int(len(vec_array_7[0]['face_holistic_mp'])/3) )]
        face_DW_mask = [j for j in range(68)]
        
        mv_x = l_position_list[i][0] - l_position_new_list[i][0]
        mv_y = l_position_list[i][1] - l_position_new_list[i][1]

        processed = process_keypoints_with_mask(vec_array_7[i], 
                                                {'pose_mp':[0,1,2,3,4,5,6,7,8,9,10, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
                                                 'bodies':[0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                                                 'faces':face_DW_mask,
                                                 'face_mp':face_mp_mask,
                                                 'face_holistic_mp': face_holistic_mp_mask,
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

    #resize right arm and right hand
    vec_array_9 = list()
    for i in range(len(vec_array_8)):

        face_mp_mask = [j for j in range( int(len(vec_array_8[0]['face_mp'])/3) )]
        face_holistic_mp_mask = [j for j in range( int(len(vec_array_8[0]['face_holistic_mp'])/3) )]
        face_DW_mask = [j for j in range(68)]

        processed = process_keypoints_with_mask(vec_array_8[i], 
                                                {'pose_mp':[0,1,2,3,4,5,6,7,8,9,10,11, 13, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
                                                 'bodies':[0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                                                 'faces':face_DW_mask,
                                                 'face_mp':face_mp_mask,
                                                 'face_holistic_mp': face_holistic_mp_mask,
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
        face_holistic_mp_mask = [j for j in range( int(len(vec_array_9[0]['face_holistic_mp'])/3) )]
        face_DW_mask = [j for j in range(68)]
        
        mv_x = r_position_list[i][0] - r_position_new_list[i][0]
        mv_y = r_position_list[i][1] - r_position_new_list[i][1]

        processed = process_keypoints_with_mask(vec_array_9[i], 
                                                {'pose_mp':[0,1,2,3,4,5,6,7,8,9,10,11, 13, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
                                                 'bodies':[0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                                                 'faces':face_DW_mask,
                                                 'face_mp':face_mp_mask,
                                                 'face_holistic_mp': face_holistic_mp_mask,
                                                 'left_hand_DW':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                                 'left_hand_mp':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                                 'left_hand_holistic_mp':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]},
                                                resize_height_rate=1, resize_width_rate=1,
                                                move_x=mv_x, move_y=mv_y)
        vec_array_10.append(processed)

    #change face length based on nose length
    #we first calculate nose length using both x and y value
    face_length = 0.
    for i in range(len(vec_array_10)):
        y_l_up = vec_array_10[i]['faces'][0][0][1]
        y_r_up = vec_array_10[i]['faces'][0][16][1]
        y_up = (y_l_up + y_r_up)/2 #average

        y_l_down = vec_array_10[i]['faces'][0][4][1]
        y_r_down = vec_array_10[i]['faces'][0][12][1]
        y_down = (y_l_down + y_r_down)/2 #average

        distance_face = abs(y_up - y_down)
        face_length = max(distance_face, face_length)

    r_face_length = std_face_length/face_length

    #again, need to record the nose point before resize, so that we can move back after resizing
    nose_x_list, nose_y_list = list(), list()
    for i in range(len(vec_array_10)):
        nose_x = vec_array_10[i]['bodies']['candidate'][0][0]
        nose_y = vec_array_10[i]['bodies']['candidate'][0][1]
        
        nose_x_list.append(nose_x)
        nose_y_list.append(nose_y)

    #resize mp and DW for ear (face) width
    vec_array_11 = list()
    for i in range(len(vec_array_10)):
        processed = process_keypoints_with_mask(vec_array_10[i], 
                                                {'pose_mp':[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
                                                 'bodies':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                                                 'left_hand_DW':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                                 'right_hand_DW':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                                 'left_hand_mp':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                                 'right_hand_mp':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                                 'left_hand_holistic_mp':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                                 'right_hand_holistic_mp':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]},
                                                resize_height_rate=r_face_length, resize_width_rate=1,
                                                move_x=0, move_y=0)
        vec_array_11.append(processed)

    #move back to original nose location
    nose_x_new_list, nose_y_new_list = list(), list()
    for i in range(len(vec_array_11)):
        nose_x_new = vec_array_11[i]['bodies']['candidate'][0][0]
        nose_y_new = vec_array_11[i]['bodies']['candidate'][0][1]

        nose_x_new_list.append(nose_x_new)
        nose_y_new_list.append(nose_y_new)

    vec_array_12 = list()
    for i in range(len(vec_array_11)):

        mv_x_new = nose_x_list[i] - nose_x_new_list[i]
        mv_y_new = nose_y_list[i] - nose_y_new_list[i]

        processed = process_keypoints_with_mask(vec_array_11[i], 
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
        vec_array_12.append(processed)

    #stop normalization, insert interpolation if required
    if num_insert_interpolation > 0:
        final = interpolate_dicts(vec_array_12, num_insert_interpolation)
    else:
        final = vec_array_12

    #save to file
    Dict_ = defaultdict()
    
    Dict_['keypoint'] = final

    Dict_['info'] = {'video_file_location': path,
                     'fps': fps, 'number_of_frames': n_frames, 'H': f_H, 'W': f_W}
    
    #by doing 2*ID, the name of the extracted kp will be 0.pickle, 2.pickle, ....
    #then, interpolation pickle will be like 1.pickle, 3.pickle, ....
    with open(path_out_ + str(2*ID) + '.pickle', 'wb') as handle:
        pickle.dump(Dict_, handle, protocol=pickle.HIGHEST_PROTOCOL)


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


def load_and_sort_pickles(path_in):
    """
    Load and sort pickle files by their integer filenames in ascending order.

    Args:
        path_in (str): Path to the directory containing pickle files.

    Returns:
        list: List of sorted pickle file paths as strings.
    """
    # Convert to Path object
    input_path = Path(path_in)

    # Get all files with .pkl extension
    pickle_files = list(input_path.glob('*.pkl'))

    # Dictionary to check for duplicates
    id_check = {}

    # List to store (id, path) pairs
    valid_pickles = []

    for pickle_path in pickle_files:
        # Get filename without extension
        name = pickle_path.stem

        # Check if filename is an integer
        try:
            pickle_id = int(name)
        except ValueError:
            raise ValueError(f"Non-integer filename found: {name}")

        # Check for duplicates
        if pickle_id in id_check:
            raise ValueError(f"Duplicate pickle ID found: {pickle_id}")

        id_check[pickle_id] = True
        valid_pickles.append((pickle_id, pickle_path))

    # Sort by ID
    valid_pickles.sort(key=lambda x: x[0])

    # Return only the paths in sorted order
    return [str(path) for _, path in valid_pickles]


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



##############################
#########the class############
class KeyPointNormalization:
    
    def __init__(self, path_in, path_out, style_img_path, model):
        
        self.path_in = path_in
        self.path_out = path_out
        self.model = model
        self.style_img_path = style_img_path
        
        # Initialize H and W as None by default
        self.H = None
        self.W = None
        
        self.sorted_pkl_files = load_and_sort_pickles(self.path_in)
        clean_and_create_folder(self.path_out)

        os.makedirs(self.path_out + 'raw/individual_pickles/', exist_ok=True)

        #update the std parameter according to the style image.
        if self.style_img_path is not None:
            sty_kp_data, H, W = obtain_key_vector_from_image(self.style_img_path, model)

            # Store H and W as instance attributes
            self.H = H
            self.W = W

            std_sholder = euclidean_distance(sty_kp_data['bodies']['candidate'][2], sty_kp_data['bodies']['candidate'][5])
            std_d_y = sty_kp_data['bodies']['candidate'][1][1] - (sty_kp_data['bodies']['candidate'][16][1] + sty_kp_data['bodies']['candidate'][17][1])/2.
            std_x = sty_kp_data['bodies']['candidate'][1][0]
            std_y = sty_kp_data['bodies']['candidate'][1][1]
            std_ear_x = abs(sty_kp_data['bodies']['candidate'][16][0] - sty_kp_data['bodies']['candidate'][17][0])

            y_l_up = sty_kp_data['faces'][0][0][1]
            y_r_up = sty_kp_data['faces'][0][16][1]
            y_up = (y_l_up + y_r_up)/2 #average

            y_l_down = sty_kp_data['faces'][0][4][1]
            y_r_down = sty_kp_data['faces'][0][12][1]
            y_down = (y_l_down + y_r_down)/2 #average

            std_face_len = abs(y_up - y_down)

            self.sty_std = (std_sholder, std_d_y, std_x, std_y, std_ear_x, std_face_len)

        else:
            self.sty_std = None

    def run(self, num_insert_interpolation=0):
        for i in range(len(self.sorted_pkl_files)):
            do_normalization(self.sorted_pkl_files[i], self.path_out + 'raw/individual_pickles/', i, self.model, num_insert_interpolation, self.sty_std)

    def get_image_dimensions(self):
            """
            Returns the height (H) and width (W) of the style image.
            """
            return self.H, self.W


