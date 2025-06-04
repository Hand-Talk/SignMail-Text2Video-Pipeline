import os
import json
import cv2
import numpy as np
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import torch
import torchvision
from torchvision.transforms import Compose, ToPILImage, Resize, ToTensor, Normalize
import lpips
import cv2
import numpy as np

def preprocess_real_frames(real_frames, target_size=(576, 768), sample_n_frames=16):
    """
    Resize and sample real frames to match the generated video properties.
    
    Args:
        real_frames (list): List of real video frames (numpy arrays).
        target_size (tuple): Target frame size (width, height).
        sample_n_frames (int): Number of frames to sample.

    Returns:
        list: Processed real frames.
    """
    # Resize frames
    resized_frames = [cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA) for frame in real_frames]
    
    # Sample `sample_n_frames` evenly from the available frames
    num_frames = len(resized_frames)
    if num_frames < sample_n_frames:
        print(f"Warning: Not enough frames. Using all {num_frames} frames.")
        return resized_frames  # Return all if not enough
    
    indices = np.linspace(0, num_frames - 1, sample_n_frames, dtype=int)
    sampled_frames = [resized_frames[i] for i in indices]
    
    return sampled_frames


# ---------------------------
# Helper functions
# ---------------------------
def read_video_frames(video_path, sample_rate=1):
    """Read video frames (RGB) with an optional sampling rate."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate == 0:
            # Convert BGR (OpenCV) to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        frame_idx += 1
    cap.release()
    return frames

def compute_ssim_psnr(frames1, frames2):
    """Compute average SSIM and PSNR over a list of frame pairs."""
    ssim_scores = []
    psnr_scores = []
    for f1, f2 in zip(frames1, frames2):
        ssim_val = ssim(f1, f2, channel_axis=2)
        psnr_val = psnr(f1, f2, data_range=255)
        ssim_scores.append(ssim_val)
        psnr_scores.append(psnr_val)
    return np.mean(ssim_scores) if ssim_scores else 0, np.mean(psnr_scores) if psnr_scores else 0

def preprocess_frame_for_lpips(frame, device):
    """Convert a numpy frame (H x W x 3) to a torch tensor in [-1, 1]."""
    tensor = torch.from_numpy(frame).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor * 2 - 1  # scale to [-1, 1]

def compute_lpips(frames1, frames2, lpips_model, device):
    """Compute average LPIPS score over frame pairs."""
    lpips_scores = []
    for f1, f2 in zip(frames1, frames2):
        img1 = preprocess_frame_for_lpips(f1, device)
        img2 = preprocess_frame_for_lpips(f2, device)
        with torch.no_grad():
            d = lpips_model(img1, img2)
        lpips_scores.append(d.item())
    return np.mean(lpips_scores) if lpips_scores else 0

def preprocess_clip(frames, transform, device):
    """
    Preprocess a list of frames (assumed to be of equal size) into a clip tensor.
    The output shape is (1, 3, T, H, W).
    """
    clip = [transform(frame) for frame in frames]
    clip = torch.stack(clip, dim=1)  # shape: (3, T, H, W)
    return clip.unsqueeze(0).to(device)

def extract_video_feature(frames, video_model, device, clip_length=16, transform=None):
    """
    Extract features from the video by splitting into non-overlapping clips.
    Returns the averaged feature vector for the video.
    """
    video_features = []
    num_frames = len(frames)
    if num_frames < clip_length:
        return None
    for i in range(0, num_frames - clip_length + 1, clip_length):
        clip_frames = frames[i:i + clip_length]
        clip_tensor = preprocess_clip(clip_frames, transform, device)
        with torch.no_grad():
            feat = video_model(clip_tensor)
        # Remove batch dimension and convert to numpy
        video_features.append(feat.squeeze(0).cpu().numpy())
    if video_features:
        return np.mean(video_features, axis=0)
    return None

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Compute the Frechet Distance between two Gaussians parameterized by (mu, sigma).
    """
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    # Numerical error might give slight imaginary component
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

# ---------------------------
# Main evaluation function
# ---------------------------
def main(generated_dir, real_dir, output_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Initialize LPIPS (using the 'alex' network)
    lpips_model = lpips.LPIPS(net='alex').to(device)

    # Initialize the video feature extractor using r3d_18.
    # Replace the final fc layer with Identity to get features.
    video_model = torchvision.models.video.s3d(pretrained=True)
    video_model.fc = torch.nn.Identity()
    video_model = video_model.to(device)
    video_model.eval()

    # Define transform for video clips (resize to 112x112 and normalize).
    transform = Compose([
        ToPILImage(),
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.43216, 0.394666, 0.37645],
                  std=[0.22803, 0.22145, 0.216989])
    ])

    # Get list of video files (supporting .mp4 and .avi)
    generated_files = sorted([
        os.path.join(generated_dir, f) for f in os.listdir(generated_dir)
        if f.lower().endswith(('.mp4', '.avi'))
    ])
    real_files = sorted([
        os.path.join(real_dir, f) for f in os.listdir(real_dir)
        if f.lower().endswith(('.mp4', '.avi'))
    ])
    # Match videos by basename
    gen_dict = {os.path.basename(f): f for f in generated_files}
    
    # gen_dict = {k.rsplit('_', 2)[0] + ".mp4": v for k, v in gen_dict.items()}

    real_dict = {os.path.basename(f): f for f in real_files}
    
    
    common_files = set(gen_dict.keys()).intersection(real_dict.keys())

    if not common_files:
        print("No matching video files found between the directories.")
        return

    ssim_list = []
    psnr_list = []
    lpips_list = []
    fvd_features_gen = []
    fvd_features_real = []

    for fname in common_files:
        print("Processing video:", fname)
        gen_path = gen_dict[fname]
        real_path = real_dict[fname]

        # Read video frames
        # gen_frames = read_video_frames(gen_path)

        gen_frames = preprocess_real_frames(read_video_frames(gen_path))
        real_frames = preprocess_real_frames(read_video_frames(real_path))

        # Compute SSIM and PSNR on a per-frame basis
        ssim_val, psnr_val = compute_ssim_psnr(gen_frames, real_frames)
        ssim_list.append(ssim_val)
        psnr_list.append(psnr_val)

        # Compute LPIPS per frame (averaged)
        lpips_val = compute_lpips(gen_frames, real_frames, lpips_model, device)
        lpips_list.append(lpips_val)

        # Extract video features for FVD (using non-overlapping 16-frame clips)
        feat_gen = extract_video_feature(gen_frames, video_model, device, clip_length=16, transform=transform)
        feat_real = extract_video_feature(real_frames, video_model, device, clip_length=16, transform=transform)
        if feat_gen is not None and feat_real is not None:
            fvd_features_gen.append(feat_gen)
            fvd_features_real.append(feat_real)

    # Compute average scores across all videos
    avg_ssim = float(np.mean(ssim_list)) if ssim_list else 0
    avg_psnr = float(np.mean(psnr_list)) if psnr_list else 0
    avg_lpips = float(np.mean(lpips_list)) if lpips_list else 0

    # Compute FVD over the collection of features
    if fvd_features_gen and fvd_features_real:
        fvd_features_gen = np.array(fvd_features_gen)
        fvd_features_real = np.array(fvd_features_real)
        mu_gen = np.mean(fvd_features_gen, axis=0)
        sigma_gen = np.cov(fvd_features_gen, rowvar=False)
        mu_real = np.mean(fvd_features_real, axis=0)
        sigma_real = np.cov(fvd_features_real, rowvar=False)
        fvd_score = calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
    else:
        fvd_score = None

    final_scores = {
        "SSIM": avg_ssim,
        "PSNR": avg_psnr,
        "LPIPS": avg_lpips,
        "FVD": fvd_score
    }

    # Save final scores to the specified output file
    with open(output_file, "w") as f:
        json.dump(final_scores, f, indent=4)
    print("Final scores saved to", output_file)

# ---------------------------
# Main entry point
# ---------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate video quality metrics (FVD, SSIM, LPIPS, PSNR)")
    parser.add_argument("--generated_dir", type=str, required=True,
                        help="Directory containing generated videos")
    parser.add_argument("--real_dir", type=str, required=True,
                        help="Directory containing ground truth videos")
    parser.add_argument("--output_file", type=str, default="/home/ubuntu/realisDance_generation/res_eval/final_scores.json",
                        help="File to save the final scores (JSON)")
    args = parser.parse_args()

    main(args.generated_dir, args.real_dir, args.output_file)
