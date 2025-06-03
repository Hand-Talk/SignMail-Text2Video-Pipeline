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

# Change to the model directory for DWPose
sys.path.append('./ailab_DWPose_not_git/ControlNet-v1-1-nightly/')
from annotator.dwpose import DWposeDetector_canlin_no_output_img

from scripts.prepare_data import PrepareData
from scripts.kp_normalization import KeyPointNormalization
from scripts.motion_interpolation import MotionInterpolation
from scripts.draw_kp_img import CombineKeyPoint_and_DrawKeyPointVideo



##########################
##########################
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run keypoint extraction and motion interpolation.")
    parser.add_argument('--sign_ids', nargs='+', required=True, help='List of sign IDs to process')    
    parser.add_argument('--num_interpolation', type=int, default=8, help='Number of frames for motion interpolation.')
    parser.add_argument('--num_insert_interpolation', type=int, default=0, help='Number of interpolations to be inserted into each frame.')
    parser.add_argument('--style_image_path', type=str, default=None, help='Path to the style image (optional).')

    args = parser.parse_args()    

    # Initialize the model and other classes
    data_processor = PrepareData(credentials_path='./meta_data/credentials.txt',
                                 sign_details_path='./meta_data/sign_details_dict.pkl')
    
    # Process the videos with provided sign IDs
    print('prepare data')
    results = data_processor.run(args.sign_ids)

    DW_model = DWposeDetector_canlin_no_output_img()
    kp_normalizer = KeyPointNormalization('./input_keypoints/', './results/', args.style_image_path, DW_model)
    interpolator = MotionInterpolation('./results/', './meta_data/')

    # Run key point extraction
    print('implement keypoint extraction and normalization')
    kp_normalizer.run(num_insert_interpolation=args.num_insert_interpolation)

    # Now, get the image dimensions
    H, W = kp_normalizer.get_image_dimensions()

    #do interpolation
    print('implement motion interpolation')
    n_frames = args.num_interpolation + 2
    interpolator.run(n_frames)

    #draw keypoint video and keep normalized array together
    print('generate videos and save keypoint vectors')
    finalizer = CombineKeyPoint_and_DrawKeyPointVideo('./results/', H, W)
    finalizer.run()



