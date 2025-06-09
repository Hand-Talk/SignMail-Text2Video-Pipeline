import sys
import os
#import time
import pickle
import cv2
#import matplotlib.pyplot as plt
import argparse
import mediapipe as mp
import shutil
#import multiprocessing
#import logging
import numpy as np
#import gc
#import traceback

from collections import Counter, defaultdict
from copy import deepcopy
from tqdm import tqdm
#from multiprocessing import Pool, Manager
from pathlib import Path

# Change to the model directory for DWPose - only if not already in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
dwpose_path = os.path.join(current_dir, 'ailab_DWPose_not_git', 'ControlNet-v1-1-nightly')
annotator_init = os.path.join(dwpose_path, 'annotator', '__init__.py')
dwpose_init = os.path.join(dwpose_path, 'annotator', 'dwpose', '__init__.py')

# Create __init__.py files if they don't exist
for init_file in [annotator_init, dwpose_init]:
    os.makedirs(os.path.dirname(init_file), exist_ok=True)
    if not os.path.exists(init_file):
        open(init_file, 'a').close()

# Only add to sys.path if not already there
if dwpose_path not in sys.path:
    sys.path.insert(0, dwpose_path)

try:
    from annotator.dwpose import DWposeDetector_canlin_no_output_img
except ImportError as e:
    print(f"Error importing DWposeDetector_canlin_no_output_img: {str(e)}")
    print("Contents of annotator directory:", os.listdir(os.path.join(dwpose_path, 'annotator')))
    print("Contents of dwpose directory:", os.listdir(os.path.join(dwpose_path, 'annotator', 'dwpose')))
    raise

from scripts.prepare_data import PrepareData
from scripts.kp_normalization import KeyPointNormalization
from scripts.motion_interpolation import MotionInterpolation
from scripts.draw_kp_img import CombineKeyPoint_and_DrawKeyPointVideo

def main(args):
    """
    Main function to process sign IDs and generate videos
    
    Args:
        args (dict): Dictionary containing:
            - sign_ids (list): List of sign IDs to process
            - num_interpolation (int): Number of frames for interpolation
            - num_insert_interpolation (int): Number of interpolations to insert
            - style_image_path (str): Path to style image (optional)
    """
    # Get the absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    meta_data_dir = os.path.join(current_dir, 'meta_data')
    input_keypoints_dir = os.path.join(current_dir, 'input_keypoints')
    results_dir = os.path.join(current_dir, 'results')
    
    # Create directories if they don't exist
    for directory in [meta_data_dir, input_keypoints_dir, results_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Initialize the model and other classes with absolute paths
    data_processor = PrepareData(
        credentials_path=os.path.join(meta_data_dir, 'credentials.txt'),
        sign_details_path=os.path.join(meta_data_dir, 'sign_details_dict.pkl')
    )
    
    # Process the videos with provided sign IDs
    print('prepare data')
    results = data_processor.run(args['sign_ids'])

    DW_model = DWposeDetector_canlin_no_output_img()
    kp_normalizer = KeyPointNormalization(
        input_keypoints_dir,
        results_dir,
        args['style_image_path'],
        DW_model
    )
    interpolator = MotionInterpolation(results_dir, meta_data_dir)

    # Run key point extraction
    print('implement keypoint extraction and normalization')
    kp_normalizer.run(num_insert_interpolation=args['num_insert_interpolation'])

    # Now, get the image dimensions
    H, W = kp_normalizer.get_image_dimensions()

    #do interpolation
    print('implement motion interpolation')
    n_frames = args['num_interpolation'] + 2
    interpolator.run(n_frames)

    #draw keypoint video and keep normalized array together
    print('generate videos and save keypoint vectors')
    finalizer = CombineKeyPoint_and_DrawKeyPointVideo(results_dir, H, W)
    finalizer.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run keypoint extraction and motion interpolation.")
    parser.add_argument('--sign_ids', nargs='+', required=True, help='List of sign IDs to process')    
    parser.add_argument('--num_interpolation', type=int, default=8, help='Number of frames for motion interpolation.')
    parser.add_argument('--num_insert_interpolation', type=int, default=0, help='Number of interpolations to be inserted into each frame.')
    parser.add_argument('--style_image_path', type=str, default=None, help='Path to the style image (optional).')

    args = parser.parse_args()
    # Convert namespace to dictionary
    args_dict = vars(args)
    main(args_dict)



