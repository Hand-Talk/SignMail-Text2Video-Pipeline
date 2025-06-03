import boto3
import os
import shutil
import cv2
import botocore
import pickle
import time
import configparser

from tqdm import tqdm


############################
### functions ##############
def download_and_process_sign_videos(sign_ids, session, sign_details_dict, 
                                    bucket_name="vsl-dataset",
                                    folder_path="HT_raw_sign_videos/sign_videos_all/",
                                    temp_dir="./temp/",
                                    output_dir="./input_videos/"):
    """
    For each sign ID:
    1. Download the video
    2. Extract frames between start and end frame
    3. Save extracted frames as a new video
    4. Delete original video
    
    Args:
        sign_ids (list): List of sign IDs
        session (boto3.Session): Boto3 session with valid credentials
        sign_details_dict (dict): Dictionary mapping sign IDs to details
        bucket_name (str): S3 bucket name
        folder_path (str): Path to the folder in S3 bucket
        temp_dir (str): Temporary directory for downloaded videos
        output_dir (str): Directory to save processed videos
    
    Returns:
        dict: Results of processing
    """
    # Create S3 client from the session
    s3_client = session.client('s3')
    
    # Clean up and create new directories
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # Track results
    results = {
        "processed": [],
        "missing_sign": [],
        "missing_video": [],
        "processing_errors": []
    }
    
    # Counter for output videos
    count = 0
    
    # Process each sign ID
    for sign_id in tqdm(sign_ids, desc="Processing sign videos"):
        try:
            # Remove the ¶ prefix if present
            if sign_id.startswith('¶'):
                sign_id = sign_id[1:]
            
            # Check if sign ID exists in the dictionary
            if sign_id not in sign_details_dict:
                print(f"Sign ID not found in dictionary: {sign_id}")
                results["missing_sign"].append(sign_id)
                continue
            
            # Get the details for this sign
            sign_details = sign_details_dict[sign_id]
            
            # Use the first entry if there are multiple
            segment_id, video_id, start_frame, end_frame = sign_details[0]
            
            #print(f"Processing sign {sign_id} with video {video_id}, frames {start_frame}-{end_frame}")
            
            # Download the video
            downloaded_path = None
            
            # Try CROPPED version first
            cropped_key = f"{folder_path}{video_id}__CROPPED_VIDEO.mp4"
            original_key = f"{folder_path}{video_id}__ORIGINAL_VIDEO.mp4"
            
            try:
                # Check if CROPPED version exists
                s3_client.head_object(Bucket=bucket_name, Key=cropped_key)
                downloaded_path = os.path.join(temp_dir, f"{video_id}__CROPPED_VIDEO.mp4")
                #print(f"Downloading {cropped_key} to {downloaded_path}")
                s3_client.download_file(bucket_name, cropped_key, downloaded_path)
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] != '404':
                    print(f"Error accessing CROPPED video: {e.response['Error']['Code']}")
                
                # Try ORIGINAL version
                try:
                    s3_client.head_object(Bucket=bucket_name, Key=original_key)
                    downloaded_path = os.path.join(temp_dir, f"{video_id}__ORIGINAL_VIDEO.mp4")
                    #print(f"Downloading {original_key} to {downloaded_path}")
                    s3_client.download_file(bucket_name, original_key, downloaded_path)
                except botocore.exceptions.ClientError as e:
                    print(f"Video not found: {video_id}")
                    results["missing_video"].append(video_id)
                    continue
            
            if not downloaded_path or not os.path.exists(downloaded_path):
                print(f"Failed to download video: {video_id}")
                results["missing_video"].append(video_id)
                continue
            
            # Process the video with OpenCV
            try:
                # Open the video
                cap = cv2.VideoCapture(downloaded_path)
                
                # Check if video opened successfully
                if not cap.isOpened():
                    print(f"Could not open video: {downloaded_path}")
                    results["processing_errors"].append({"sign_id": sign_id, "error": "Could not open video"})
                    continue
                
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Create output video writer
                output_path = os.path.join(output_dir, f"{count}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                # Extract frames between start and end frame
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Check if current frame is within the desired range
                    if start_frame <= frame_count <= end_frame:
                        out.write(frame)
                    
                    frame_count += 1
                    
                    # Stop once we've processed all needed frames
                    if frame_count > end_frame:
                        break
                
                # Release resources
                cap.release()
                out.release()
                
                #print(f"Saved processed video to {output_path}")
                results["processed"].append({"sign_id": sign_id, "output_path": output_path})
                
                # Increment counter
                count += 1
                
            except Exception as e:
                print(f"Error processing video: {str(e)}")
                results["processing_errors"].append({"sign_id": sign_id, "error": str(e)})
            
            # Delete the original downloaded video
            if downloaded_path and os.path.exists(downloaded_path):
                os.remove(downloaded_path)
                #print(f"Deleted original video: {downloaded_path}")
            
        except Exception as e:
            print(f"Error processing sign ID {sign_id}: {str(e)}")
            results["processing_errors"].append({"sign_id": sign_id, "error": str(e)})
    
    # Print summary
    #print("\nProcessing Summary:")
    #print(f"- Processed videos: {len(results['processed'])}")
    #print(f"- Missing signs: {len(results['missing_sign'])}")
    #print(f"- Missing videos: {len(results['missing_video'])}")
    #print(f"- Processing errors: {len(results['processing_errors'])}")
    
    shutil.rmtree(temp_dir)
    
    return results



#######################################
## updated function to directly use keypoints from s3
#One may want to refer to chat (!!!!!) Processing ASL Videos on May 30 2025 in Abacus AI for detialed discussion.
def download_and_process_keypoints(sign_ids, session, sign_details_dict,
                                   bucket_name="vsl-dataset",
                                   folder_path="HT_raw_sign_videos/Mediapipe_and_DWPose_kp_all/",
                                   temp_dir="./temp/",
                                   output_dir="./input_keypoints/"):
    """
    For each sign ID:
    1. Download the keypoint pickle file (prefer CROPPED over ORIGINAL)
    2. Extract keypoints between start and end frame
    3. Save extracted keypoints as a new pickle file
    4. Delete original pickle file

    Args:
        sign_ids (list): List of sign IDs
        session (boto3.Session): Boto3 session with valid credentials
        sign_details_dict (dict): Dictionary mapping sign IDs to details
        bucket_name (str): S3 bucket name
        folder_path (str): Path to the folder in S3 bucket
        temp_dir (str): Temporary directory for downloaded files
        output_dir (str): Directory to save processed keypoint files

    Returns:
        dict: Results of processing
    """
    # Create S3 client from the session
    s3_client = session.client('s3')

    # Clean up and create new directories
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Track results
    results = {
        "processed": [],
        "missing_sign": [],
        "missing_keypoint_file": [],
        "processing_errors": []
    }

    # Counter for output pickle files
    count = 0

    # Process each sign ID
    for sign_id in tqdm(sign_ids, desc="Processing keypoint files"):
        try:
            # Remove the ¶ prefix if present
            if sign_id.startswith('¶'):
                sign_id = sign_id[1:]

            # Check if sign ID exists in the dictionary
            if sign_id not in sign_details_dict:
                print(f"Sign ID not found in dictionary: {sign_id}")
                results["missing_sign"].append(sign_id)
                continue

            # Get the details for this sign
            sign_details = sign_details_dict[sign_id]

            # Use the first entry if there are multiple
            segment_id, video_id, start_frame, end_frame = sign_details[0]

            # Construct the keys for the pickle files
            cropped_key = f"{folder_path}{video_id}__CROPPED_VIDEO.pkl"
            original_key = f"{folder_path}{video_id}__ORIGINAL_VIDEO.pkl"
            downloaded_path = None

            # Try downloading the CROPPED version first
            try:
                s3_client.head_object(Bucket=bucket_name, Key=cropped_key)
                downloaded_path = os.path.join(temp_dir, f"{video_id}__CROPPED_VIDEO.pkl")
                s3_client.download_file(bucket_name, cropped_key, downloaded_path)
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] != '404':
                    print(f"Error accessing CROPPED keypoint file: {e.response['Error']['Code']}")

                # Try downloading the ORIGINAL version
                try:
                    s3_client.head_object(Bucket=bucket_name, Key=original_key)
                    downloaded_path = os.path.join(temp_dir, f"{video_id}__ORIGINAL_VIDEO.pkl")
                    s3_client.download_file(bucket_name, original_key, downloaded_path)
                except botocore.exceptions.ClientError as e:
                    if e.response['Error']['Code'] == '404':
                        print(f"Keypoint file not found: {video_id}")
                        results["missing_keypoint_file"].append(video_id)
                    else:
                        print(f"Error accessing ORIGINAL keypoint file: {e.response['Error']['Code']}")
                    continue

            # If no file was downloaded, skip this sign ID
            if not downloaded_path or not os.path.exists(downloaded_path):
                print(f"Failed to download keypoint file for video ID: {video_id}")
                results["missing_keypoint_file"].append(video_id)
                continue

            # Process the keypoint pickle file
            try:
                # Load the keypoints from the pickle file
                with open(downloaded_path, 'rb') as f:
                    keypoints = pickle.load(f)

                # Extract keypoints for the specified frames
                processed_keypoints = keypoints[start_frame:end_frame + 1]

                # Save the processed keypoints to a new pickle file
                output_path = os.path.join(output_dir, f"{count}.pkl")
                with open(output_path, 'wb') as f:
                    pickle.dump(processed_keypoints, f)

                # Track the processed sign ID
                results["processed"].append({"sign_id": sign_id, "output_path": output_path})

                # Increment the counter
                count += 1

            except Exception as e:
                print(f"Error processing keypoint file for sign ID {sign_id}: {str(e)}")
                results["processing_errors"].append({"sign_id": sign_id, "error": str(e)})

            # Delete the original downloaded pickle file
            if downloaded_path and os.path.exists(downloaded_path):
                os.remove(downloaded_path)

        except Exception as e:
            print(f"Error processing sign ID {sign_id}: {str(e)}")
            results["processing_errors"].append({"sign_id": sign_id, "error": str(e)})

    # Clean up temporary directory
    shutil.rmtree(temp_dir)

    return results




def load_aws_credentials(credentials_path):
    """Load AWS credentials from a file."""
    config = configparser.ConfigParser()

    # Read the file as a simple text file first
    with open(credentials_path, 'r') as file:
        lines = file.readlines()

    # Parse the credentials from the lines
    credentials = {}
    for line in lines:
        line = line.strip()
        if '=' in line:
            key, value = line.split('=', 1)
            if key == 'aws_access_key_id':
                credentials['aws_access_key_id'] = value
            elif key == 'aws_secret_access_key':
                credentials['aws_secret_access_key'] = value
            elif key == 'aws_session_token':
                credentials['aws_session_token'] = value

    return credentials


##################
class PrepareData:
    def __init__(self, credentials_path='../meta_data/credentials.txt',
                 sign_details_path='../meta_data/sign_details_dict.pkl'):
        self.credentials_path = credentials_path
        self.sign_details_path = sign_details_path

    def run(self, sign_ids):
        """Process sign videos using AWS credentials.

        Args:
            sign_ids (list): List of sign IDs to process

        Returns:
            The results from download_and_process_sign_videos
        """
        # Load sign details dictionary
        with open(self.sign_details_path, 'rb') as f:
            sign_details_dict = pickle.load(f)

        # Load credentials from file
        aws_credentials = load_aws_credentials(self.credentials_path)

        # Create a session with credentials from file
        session = boto3.Session(
            aws_access_key_id=aws_credentials['aws_access_key_id'],
            aws_secret_access_key=aws_credentials['aws_secret_access_key'],
            aws_session_token=aws_credentials['aws_session_token']
        )

        # Process the videos
        results = download_and_process_keypoints(sign_ids, session, sign_details_dict)

        return results


