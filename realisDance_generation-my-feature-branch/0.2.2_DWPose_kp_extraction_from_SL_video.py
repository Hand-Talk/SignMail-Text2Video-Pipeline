import sys
import os
import pickle
import cv2
import matplotlib
import mediapipe as mp
import shutil
import multiprocessing
import numpy as np
import gc
import math
import onnxruntime as ort  # Assuming you're using ONNX Runtime
import argparse
import torch

from collections import Counter, defaultdict
from scipy.interpolate import CubicSpline
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count
import glob


# Change to the model directory for DWPose
sys.path.append('/home/ubuntu/realisDance_generation/ControlNet-v1-1-nightly/')
from annotator.dwpose import DWposeDetector_canlin_no_output_img



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


def convert_tensors_to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()  # Convert tensor to NumPy array
    elif isinstance(data, dict):
        return {k: convert_tensors_to_numpy(v) for k, v in data.items()}  # Recursively process dictionaries
    elif isinstance(data, list):
        return [convert_tensors_to_numpy(v) for v in data]  # Recursively process lists
    else:
        return data  # Return as-is for other types


def obtain_key_vector(file_name, model, in_root, root_folder, root_folder_kp_vid):
    
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

        #pbar = tqdm(total=number_of_frames, desc=f'Processing video {ID}')
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
            
            with torch.no_grad():
                kp_vectors = model(image)

            kp_vectors = convert_tensors_to_numpy(kp_vectors)

            fixed_l = fix_scattered_keypoints(kp_vectors['hands'][0])
            fixed_r = fix_scattered_keypoints(kp_vectors['hands'][1])
            kp_vectors['hands'][0] = fixed_l
            kp_vectors['hands'][1] = fixed_r
            
            # !!!!! We encounted error in 
            #    Y = candidate[index.astype(int), 0] * float(W)
            #    IndexError: index 19 is out of bounds for axis 0 with size 18
            #which is caused in below. So we comment out.
            
            #kp_vectors['bodies']['candidate'] = deepcopy(kp_vectors['bodies']['candidate'][:18, :])
            
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

            kp_vectors['H'] = frame_height
            kp_vectors['W'] = frame_width
            kp_vectors['fps'] = fps

            vec_array.append(kp_vectors)

            success, image = vidcap.read()
            count += 1

            #pbar.update(1)  # Update progress bar by 1
        
        #pbar.close()

        vidcap.release()

        # Replace 'test_test_2' with the new root folder
        new_path = file_name.replace(in_root, root_folder)

        # Change the file extension and name to 'keypoint.pkl'
        file_name = os.path.splitext(os.path.basename(new_path))[0]
        out_path = new_path.replace(file_name + '.mp4', file_name + '_keypoint.pkl')

        print(f"Saving keypoints to {out_path}")
        # Save the keypoints to a pickle file
        with open(out_path, 'wb') as handle:
            pickle.dump(vec_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #draw keypoints
        kp_vid_path = out_path.replace(root_folder, root_folder_kp_vid) #change to kp vid root
        kp_vid_path = kp_vid_path[:-4] + '.mp4' #change .pkl to .mp4

        image_list = []
        for i_ in range(len(vec_array)):
            pose_image = draw_pose(vec_array[i_], frame_height, frame_width)
            image_list.append(pose_image)
        
        #save the individual video
        out_temp = cv2.VideoWriter(kp_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        print(f"Saving video to {kp_vid_path}")
        print(f"out_temp: {out_temp}")

        for pose_image in image_list:
            out_temp.write(pose_image)
        out_temp.release()


'''function reduced to only extract DWPose. But I recommand to use the above one.'''
def obtain_key_vector_reduced(file_name, model, in_root, root_folder, root_folder_kp_vid):
    # List to store keypoint vectors
    vec_array = list()

    # Open the video file
    vidcap = cv2.VideoCapture(file_name)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    number_of_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get the video's original width and height
    frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))

    success, image = vidcap.read()
    count = 0

    # Move the model to GPU if available
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    while success:
        # Convert the image to a tensor and move it to the GPU
        #image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
        #image = torch.from_numpy(image).float().to(device)

        # Run the model on the image tensor
        with torch.no_grad():  # Disable gradient computation for inference
            kp_vectors = model(image)

        # Move the keypoints back to CPU for further processing
        kp_vectors = convert_tensors_to_numpy(kp_vectors)

        # Fix scattered keypoints
        #fixed_l = fix_scattered_keypoints(kp_vectors['hands'][0])
        #fixed_r = fix_scattered_keypoints(kp_vectors['hands'][1])
        #kp_vectors['hands'][0] = fixed_l
        #kp_vectors['hands'][1] = fixed_r
        #kp_vectors['bodies']['candidate'] = deepcopy(kp_vectors['bodies']['candidate'][:18, :])

        # Add metadata
        kp_vectors['H'] = frame_height
        kp_vectors['W'] = frame_width
        kp_vectors['fps'] = fps

        # Append the keypoints to the list
        vec_array.append(kp_vectors)

        # Read the next frame
        success, image = vidcap.read()
        count += 1

    vidcap.release()

    # Replace 'test_test_2' with the new root folder
    new_path = file_name.replace(in_root, root_folder)

    # Change the file extension and name to 'keypoint.pkl'
    out_path = new_path.replace('video.mp4', 'keypoint.pkl')

    # Save the keypoints to a pickle file
    with open(out_path, 'wb') as handle:
        pickle.dump(vec_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Draw keypoints
    #kp_vid_path = file_name.replace(in_root, root_folder_kp_vid)  # Change to kp vid root

    #image_list = []
    #for i_ in range(len(vec_array)):
    #    pose_image = draw_pose(vec_array[i_], frame_height, frame_width)
    #    image_list.append(pose_image)

    # Save the individual video
    #out_temp = cv2.VideoWriter(kp_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    #for pose_image in image_list:
    #    out_temp.write(pose_image)
    #out_temp.release()


def load_paths_from_txt(file_path):
    """
    Load video paths from a .txt file into a list.

    Args:
        file_path (str): Path to the .txt file.

    Returns:
        list: List of video paths.
    """
    with open(file_path, 'r') as f:
        video_paths = [line.strip() for line in f.readlines()]
    return video_paths


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



##########################
##########################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("--text_file", type=str, required=True, help="Path to the text file containing video paths")
    # args = parser.parse_args()

    # Define the source root directory containing videos
    source_root = "/opt/dlami/nvme/data/val/videos/"

    # Define the target root directory where keypoints will be saved
    target_root = "/opt/dlami/nvme/data/val/sign_kpt_data_MP/"
    target_root_kp_vid = "/opt/dlami/nvme/data/val/sign_video_data/"

    model = DWposeDetector_canlin_no_output_img()

    #process videos
    video_paths = video_files = glob.glob("/opt/dlami/nvme/data/val/videos/*.mp4")

    for path in tqdm(video_paths, desc='processing videos'):
        obtain_key_vector(path, model, source_root, target_root, target_root_kp_vid)
    


