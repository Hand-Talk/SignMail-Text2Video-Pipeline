import os
import math

# Function to find all .mp4 and .mov files in the source directory
def find_mov_files(directory):
    """
    Search for all .mp4 and .mov files in the given directory and its subdirectories.

    Args:
        directory (str): Path to the source directory.

    Returns:
        list: List of full paths to the video files.
    """
    mov_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.mp4') or file.endswith('.mov'):
                path = os.path.join(root, file)
                mov_paths.append(path)
    return mov_paths

# Function to create the target directory tree
def create_target_directories(video_paths, source_root, root_folder, root_folder_kp_vid):
    """
    Create the target directory tree for keypoint .pkl files and keypoint video .mp4 files.

    Args:
        video_paths (list): List of video file paths.
        root_folder (str): Target root folder for keypoint .pkl files.
        root_folder_kp_vid (str): Target root folder for keypoint video .mp4 files.
    """

    for input_path in video_paths:
        # Replace 'test_test_2' with the new root folder
        new_path = input_path.replace(source_root, root_folder)

        # Change the file extension and name to 'keypoint.pkl'
        out_path = new_path.replace('video.mp4', 'keypoint.pkl')

        # Create the necessary directories for keypoint .pkl files
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)

        # Create the necessary directories for keypoint video .mp4 files
        kp_vid_path = input_path.replace(source_root, root_folder_kp_vid)  # Change to kp vid root
        kp_vid_dir = os.path.dirname(kp_vid_path)
        os.makedirs(kp_vid_dir, exist_ok=True)

# Function to divide the video paths into chunks and save to .txt files
def save_paths_to_chunks(video_paths, num_chunks, output_dir):
    """
    Divide the video paths into chunks and save each chunk to a separate .txt file.

    Args:
        video_paths (list): List of video file paths.
        num_chunks (int): Number of chunks to divide the paths into.
        output_dir (str): Directory to save the .txt files.
    """
    # Calculate the size of each chunk
    chunk_size = math.ceil(len(video_paths) / num_chunks)

    # Divide the paths into chunks and save to .txt files
    for i in range(num_chunks):
        chunk = video_paths[i * chunk_size:(i + 1) * chunk_size]
        chunk_file = os.path.join(output_dir, f'paths_{i}.txt')
        with open(chunk_file, 'w') as f:
            for path in chunk:
                f.write(path + '\n')
        print(f"Saved chunk {i} to {chunk_file}")

# Main function
def main():
    # Define the source root directory containing videos
    source_root = "/opt/dlami/nvme/data/val/videos/"

    # Define the target root directory for keypoint .pkl files
    root_folder = "/opt/dlami/nvme/data/val/sign_kpt_data_MP"

    # Define the target root directory for keypoint video .mp4 files
    root_folder_kp_vid = "/opt/dlami/nvme/data/val/"

    # Define the output directory for the .txt files
    output_dir = "/opt/dlami/nvme/data/val/"
    os.makedirs(output_dir, exist_ok=True)

    # Number of chunks to divide the video paths into
    num_chunks = 30

    # Step 1: Find all video files in the source root directory
    video_paths = find_mov_files(source_root)
    video_paths.sort()
    print(f"Found {len(video_paths)} video files.")

    # Step 2: Create the target directory tree
    create_target_directories(video_paths, source_root, root_folder, root_folder_kp_vid)
    print("Target directory tree created.")

    # Step 3: Divide the video paths into chunks and save to .txt files
    save_paths_to_chunks(video_paths, num_chunks, output_dir)
    print("Video paths divided into chunks and saved to .txt files.")

if __name__ == "__main__":
    main()















