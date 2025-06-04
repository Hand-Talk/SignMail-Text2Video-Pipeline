import cv2
import os

# Path to the input video file
input_video_path = "A001_11011827_C018.mov"

# Path to the output folder
output_folder = "./frames_18/"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(input_video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

# Frame count
frame_number = 0

# Get the rotation flag (if available in metadata)
rotation_flag = cap.get(cv2.CAP_PROP_ORIENTATION_META) if hasattr(cv2, 'CAP_PROP_ORIENTATION_META') else None

# Read frames from the video
while True:
    ret, frame = cap.read()

    if not ret:
        # Exit loop when no more frames are available
        print("Finished extracting frames.")
        break

    # Correct orientation if necessary
    if rotation_flag in [90, 270]:  # Rotated 90 or 270 degrees
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE if rotation_flag == 90 else cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation_flag == 180 or rotation_flag is None:  # Upside-down or unknown metadata
        frame = cv2.flip(frame, -1)  # Flip both vertically and horizontally

    # Save the corrected frame
    frame_filename = os.path.join(output_folder, f"frame_{frame_number:04d}.png")
    cv2.imwrite(frame_filename, frame)

    frame_number += 1

# Release the video capture object
cap.release()
print(f"Extracted {frame_number} frames to {output_folder}")
