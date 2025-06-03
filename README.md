# Complete Text-to-Sign Language Video Pipeline

This project combines two pipelines to convert text input into sign language videos:
1. Text-to-SignID Pipeline: Converts English text to ASL (American Sign Language) SignIDs
2. Generation Pipeline: Converts SignIDs into rendered sign language videos

## Prerequisites

- Python 3.9 (Python 3.8-3.10 should work)
- OpenAI API key
- AWS credentials (for accessing sign language video dictionary)
- CUDA-capable GPU (recommended for better performance)

## Installation

1. Clone this repository and enter the directory:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create and activate a conda environment:
```bash
conda create -n signlang python=3.9
conda activate signlang
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key to the `.env` file:
     ```
     OPENAI_API_KEY=your-api-key-here
     ```

5. Set up AWS credentials:
   - Create file `ailab_Generation_pipeline-dev/meta_data/credentials.txt` with:
     ```
     aws_access_key_id=xxx
     aws_secret_access_key=xxx
     aws_session_token=xxx
     ```

6. Download DWPose checkpoints:
   - Download the following files:
     - dw-ll_ucoco_384.onnx ([baidu](https://pan.baidu.com/s/1nuBjw-KKSxD_BkpmwXUJiw?pwd=28d7) or [google](https://drive.google.com/file/d/12L8E2oAgZy4VACGSK9RaZBZrfgx7VTA2/view?usp=sharing))
     - yolox_l.onnx ([baidu](https://pan.baidu.com/s/1fpfIVpv5ypo4c1bUlzkMYQ?pwd=mjdn) or [google](https://drive.google.com/file/d/1w9pXC8tT0p9ndMN-CArp1__b2GbzewWI/view?usp=sharing))
   - Place them in: `ailab_Generation_pipeline-dev/ailab_DWPose_not_git/ControlNet-v1-1-nightly/annotator/ckpts/`

## Usage

You can use the pipeline in two ways:

1. Using the example script:
```bash
python complete_pipeline.py
```

2. Using the pipeline in your code:
```python
from complete_pipeline import process_text_to_video

# Convert text to sign language video
process_text_to_video(
    input_text="Hello! How are you?",
    output_video_path="./output_video.mp4",
    dictionary="rdp",  # Use 'rdp' or 'ht' dictionary
    num_interpolation=8,  # Number of frames to interpolate
    style_image_path=None  # Optional: path to style image
)
```

### Optional: Using a Style Image

You can provide a style image to normalize the keypoints according to specific character proportions:
1. Place your style image in `ailab_Generation_pipeline-dev/style_image/style.png`
2. Pass the path when calling the function:
```python
process_text_to_video(
    input_text="Hello!",
    style_image_path="./ailab_Generation_pipeline-dev/style_image/style.png"
)
```

## Output

The pipeline will generate:
1. A video file (default: `output_video.mp4`) containing the rendered sign language animation
2. Temporary files that are cleaned up after processing

## Troubleshooting

1. If you encounter CUDA/GPU errors:
   - Try installing PyTorch without CUDA support:
     ```bash
     pip install torch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1
     ```

2. If you get AWS credential errors:
   - Verify your credentials in `ailab_Generation_pipeline-dev/meta_data/credentials.txt`
   - Make sure the credentials are valid and not expired

3. For OpenAI API errors:
   - Check that your API key is correctly set in the `.env` file
   - Verify your API key has sufficient credits

## Notes

- Processing time varies depending on:
  - Length of input text
  - Number of interpolation frames
  - GPU availability
  - Network speed (for AWS video retrieval)
- The default dictionary is 'rdp' (Omnibridge's RDP sign dictionary)
- You can use 'ht' dictionary (Hand Talk) if needed, but RDP is recommended for better quality 