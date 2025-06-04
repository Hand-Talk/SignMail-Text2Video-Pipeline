import os
import argparse
import torch
import torchvision
from omegaconf import OmegaConf
from transformers import AutoModel
from diffusers import AutoencoderKL, DDIMScheduler
from src.models.rd_unet import RealisDanceUnet
from src.pipelines.pipeline import RealisDancePipeline
from src.utils.util import save_videos_grid
import numpy as np
import pickle

def load_keypoints(keypoint_path):
    """Load keypoints from file and preprocess them."""
    print(f"Loading keypoints from: {keypoint_path}")
    
    if keypoint_path.endswith('.npy'):
        keypoints = np.load(keypoint_path, allow_pickle=True)
    elif keypoint_path.endswith('.pkl'):
        with open(keypoint_path, 'rb') as f:
            keypoints = pickle.load(f)
    else:
        raise ValueError(f"Unsupported keypoint file format: {keypoint_path}. Must be .npy or .pkl")
    
    print(f"Initial data type: {type(keypoints)}, shape: {keypoints.shape if hasattr(keypoints, 'shape') else 'no shape'}")
    
    # Handle object arrays by converting them to a regular numpy array
    if isinstance(keypoints, np.ndarray) and keypoints.dtype == np.dtype('object'):
        try:
            keypoints = np.stack(keypoints.flatten()).reshape(keypoints.shape + (-1,))
            print(f"After stacking: shape = {keypoints.shape}, dtype = {keypoints.dtype}")
        except ValueError as e:
            raise ValueError(f"Failed to convert object array to regular array. Arrays might have inconsistent shapes: {e}")
    
    # Convert list to numpy array if needed
    if isinstance(keypoints, list):
        keypoints = np.array(keypoints)
        print(f"After list conversion: shape = {keypoints.shape}, dtype = {keypoints.dtype}")
    
    # Ensure we have a float array
    if not keypoints.dtype.kind in ['f', 'i', 'u']:
        raise ValueError(f"Keypoints must be numeric type, got {keypoints.dtype}")
    
    keypoints = keypoints.astype(np.float32)
    
    # Convert to tensor
    keypoints = torch.from_numpy(keypoints)
    print(f"Tensor shape before dimension check: {keypoints.shape}")
    
    # Add batch dimension if needed
    if keypoints.dim() == 4:  # [C, T, H, W]
        keypoints = keypoints.unsqueeze(0)  # Add batch dim -> [B, C, T, H, W]
    elif keypoints.dim() == 3:  # [T, H, W]
        keypoints = keypoints.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims -> [B, C, T, H, W]
    
    print(f"Final tensor shape: {keypoints.shape}")
    return keypoints

def get_default_config():
    """Get default configuration for the model."""
    config = {
        "image_finetune": False,
        "unet_additional_kwargs": {
            "use_motion_module": True,
            "motion_module_resolutions": [1, 2, 4, 8],
            "unet_use_cross_frame_attention": False,
            "unet_use_temporal_attention": False,
            "motion_module_type": "Vanilla",
            "motion_module_kwargs": {
                "num_attention_heads": 8,
                "num_transformer_block": 1,
                "attention_block_types": ["Temporal_Self", "Temporal_Self"],
                "temporal_position_encoding": True,
                "temporal_position_encoding_max_len": 32,
                "temporal_attention_dim_div": 1,
                "zero_initialize": True
            }
        },
        "pose_guider_kwargs": {
            "pose_guider_type": "side_guider",
            "args": {
                "out_channels": [320, 320, 640, 1280, 1280]
            }
        },
        "clip_projector_kwargs": {
            "projector_type": "ff",
            "in_features": 1024,
            "out_features": 768
        },
        "zero_snr": True,
        "v_pred": True,
        "train_cfg": False,
        "fix_ref_t": True,
        "vae_slicing": True,
        "validation_kwargs": {
            "guidance_scale": 2
        }
    }
    return OmegaConf.create(config)

def generate_video(
    keypoint_path: str,
    output_path: str,
    pretrained_model_path: str,
    pretrained_clip_path: str,
    unet_checkpoint_path: str,
    reference_image_path: str,
    fps: int = 8,
    height: int = 768,  # Updated to match config
    width: int = 576,   # Updated to match config
    num_inference_steps: int = 50,
    guidance_scale: float = 2.0,  # Updated to match config
):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    image_encoder = AutoModel.from_pretrained(pretrained_clip_path)
    noise_scheduler = DDIMScheduler.from_pretrained(
        pretrained_model_path,
        subfolder="scheduler",
        rescale_betas_zero_snr=True,  # Enable Zero-SNR
        prediction_type="v_prediction"  # Enable v_pred
    )
    
    # Get default config
    config = get_default_config()
    
    # Initialize UNet
    unet = RealisDanceUnet(
        pretrained_model_path=pretrained_model_path,
        image_finetune=config.image_finetune,
        pose_guider_kwargs=config.pose_guider_kwargs,
        unet_additional_kwargs=config.unet_additional_kwargs,
        clip_projector_kwargs=config.clip_projector_kwargs,
        fix_ref_t=config.fix_ref_t
    )
    
    # Load checkpoint
    checkpoint = torch.load(unet_checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    unet.load_state_dict(new_state_dict, strict=False)
    
    # Move models to device
    vae = vae.to(device)
    image_encoder = image_encoder.to(device)
    unet = unet.to(device)
    
    # Set models to eval mode
    vae.eval()
    image_encoder.eval()
    unet.eval()
    
    # Enable VAE slicing if configured
    if config.vae_slicing:
        vae.enable_slicing()
    
    # Create pipeline
    pipeline = RealisDancePipeline(
        unet=unet,
        vae=vae,
        image_encoder=image_encoder,
        scheduler=noise_scheduler
    ).to(device)
    
    # Load and preprocess keypoints
    pose = load_keypoints(keypoint_path).to(device)
    
    # Load and preprocess reference image
    ref_image = torchvision.io.read_image(reference_image_path)
    ref_image = (ref_image.float() / 255.0) * 2 - 1  # Normalize to [-1, 1]
    ref_image = ref_image.unsqueeze(0).to(device)
    
    # Generate video
    with torch.no_grad():
        sample = pipeline(
            pose=pose,
            ref_image=ref_image,
            ref_image_clip=ref_image,  # Using same image for both
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).videos
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_videos_grid(sample.cpu(), output_path, fps=fps)
    print(f"Video saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keypoint-path", type=str, required=True, help="Path to keypoint file")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save output video")
    parser.add_argument("--reference-image", type=str, required=True, help="Path to reference image")
    parser.add_argument("--pretrained-model", type=str, required=True, help="Path to pretrained stable diffusion model")
    parser.add_argument("--pretrained-clip", type=str, required=True, help="Path to pretrained CLIP model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--fps", type=int, default=8, help="FPS of output video")
    parser.add_argument("--height", type=int, default=768, help="Height of output video")
    parser.add_argument("--width", type=int, default=576, help="Width of output video")
    parser.add_argument("--guidance-scale", type=float, default=2.0, help="Guidance scale for generation")
    args = parser.parse_args()

    generate_video(
        keypoint_path=args.keypoint_path,
        output_path=args.output_path,
        pretrained_model_path=args.pretrained_model,
        pretrained_clip_path=args.pretrained_clip,
        unet_checkpoint_path=args.checkpoint,
        reference_image_path=args.reference_image,
        fps=args.fps,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale
    )

if __name__ == "__main__":
    main() 