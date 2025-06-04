# Text-to-Sign Language Video Pipeline

This project converts text into sign language videos. It works in three steps:
1. Converts text to sign language IDs
2. Generates pose videos
3. Renders realistic videos (in development)

## Setup

You'll need:
- Python 3.9
- OpenAI API key
- AWS credentials
- A GPU (recommended)

Quick start:
```bash
# Clone and enter the repo
git clone [repository-url]
cd [repository-name]

# Set up Python environment
conda create -n signlang python=3.9
conda activate signlang

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file with your OpenAI key:
```
OPENAI_API_KEY=your-key-here
```

2. Set up AWS access:
Create `ailab_Generation_pipeline-dev/meta_data/credentials.txt`:
```
aws_access_key_id=xxx
aws_secret_access_key=xxx
aws_session_token=xxx
```

3. Download DWPose files:
- Get these files:
  - [dw-ll_ucoco_384.onnx](https://drive.google.com/file/d/12L8E2oAgZy4VACGSK9RaZBZrfgx7VTA2/view?usp=sharing)
  - [yolox_l.onnx](https://drive.google.com/file/d/1w9pXC8tT0p9ndMN-CArp1__b2GbzewWI/view?usp=sharing)
- Put them in: `ailab_Generation_pipeline-dev/ailab_DWPose_not_git/ControlNet-v1-1-nightly/annotator/ckpts/`

## How it Works

### 1. Text to SignID
Takes English text and finds the right signs using OpenAI's help.

### 2. Generation Pipeline
Makes pose-based videos from the signs. Uses DWPose to track movements.

### 3. Video Rendering
Turns pose videos into realistic ones. Still working on this part!

## Usage

Basic example:
```python
from complete_pipeline import process_text_to_video

process_text_to_video(
    input_text="Hello! How are you?",
    output_video_path="./output.mp4",
    dictionary="rdp"  # use 'rdp' or 'ht'
)
```

Want better-looking videos? Add a style image:
```python
process_text_to_video(
    input_text="Hello!",
    style_image_path="./style.png"
)
```

## Tips

- Processing time depends on text length and your GPU
- RDP dictionary usually gives better results than HT
- If you get GPU errors, try:
  ```bash
  pip install torch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1
  ```
- Check your AWS credentials if videos aren't loading
