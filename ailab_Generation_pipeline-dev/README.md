# ailab_Generation_pipeline

This is the official repository of our neural rendering pipeline for SignMail generation. Roughly speaking, given the sign IDs, the corresponding videos in the video dictionary will be located. Then, the pipeline will obtain these videos from the s3 bucket, extract their mediapipe and DWPose keypoints, implement keypoint interpolation and normalization based on the style image, and finally provide continuous keypoint vectors and keypoint video.

## Installation

To install our model, please follow these steps:

1. **Clone this repository**: One can use `git clone https://github.com/captioncall/ailab_Generation_pipeline.git`. Then, enter the root folder by `cd 
ailab_Generation_pipeline`. 
2. **Create the conda environment**: Do `conda create -n your_env python=3.9`. But in fact, we believe that python version ranges from 3.8 to 3.10 will work. Then activate it by `conda activate your_env`
3. **Install general dependencies**: Do `pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html`. However, one may encounter the error saying that pytorch/torchvision version not available. This is because some system (like MacOS) does not support cuxxx cuda toolkit. To avoid this error, one can just install without specifying cuda toolkit: `pip install torch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html`, although such installation may reduce the performance speed.
4. **Install DWPose extractor packages**: Forward to the mmpose folder by `cd ailab_DWPose_not_git/mmpose`, and then type `pip install -r requirements.txt` to install DWPose extractor related packages.
5. **(Optional) Install onnxruntime**: One may need to install onnxruntime by `pip install onnxruntime` in the conda environment.
6. **Install boto3**: One need to install boto3 by `pip install boto3` to access the AWS.
7. **Download checkpoints**: Download the checkpoint files: Please download dw-ll_ucoco_384.onnx ([baidu](https://pan.baidu.com/s/1nuBjw-KKSxD_BkpmwXUJiw?pwd=28d7), [google](https://drive.google.com/file/d/12L8E2oAgZy4VACGSK9RaZBZrfgx7VTA2/view?usp=sharing)) and Det model yolox_l.onnx ([baidu](https://pan.baidu.com/s/1fpfIVpv5ypo4c1bUlzkMYQ?pwd=mjdn), [google](https://drive.google.com/file/d/1w9pXC8tT0p9ndMN-CArp1__b2GbzewWI/view?usp=sharing)), then put them into ailab_DWPose_not_git/ControlNet-v1-1-nightly/annotator/ckpts/.

## Implementing

First, make sure you put the credentials.txt file under ./meta_data/ In this file, you should provide the AWS credentials in this format:
```
aws_access_key_id=xxx
aws_secret_access_key=xxx
aws_session_token=xxx
```

Then, optionally, you may put a style image into style_image folder, and name it as style.png. If a style image is given, the model will normalize all keypoint according to the character in the style image. That is, the keypoint of the human character in the style image will be extracted first. Then, parameters (shoulder distence, face width, ear-to-neck-end distence, etc) will be calculated based on the style image keypoints. Keypoints from all videos will then be normalized according to this specific parameters. Also, when drawing the keypoint videos, H and W will be set to the style image size. But if no style image is given, the model will use default parameters, and draw keypoint video using H, W = 1024.

After that, please locate at the ailab_Generation_pipeline folder in terminal, and then run `python main.py --sign_ids ¶id_1 ¶id_2 ...¶id_N --num_interpolation 8 --num_insert_interpolation 0 --style_image_path ./style_image/style.png` to start the process. Here, ¶id_1 ¶id_2 ...¶id_N are the sign IDs (¶ is the standard starting character); 8 is the number of frames you want to interpolate between each video. You can adjust it at will. Also, 0 is the number of frames you want to insert between each real gloss video frames. That is, somethings the keypoint videos are short. We insert linearly interpolated frames to extend the length. The default value is 0, meaning that no frame inserted. But you can use your own setting. We suggest to set it between 0 and 5. 

This will then download the sign videos from AWS s3 according to the sign ID, extract keypoints from each video, normalize the extracted keypoints, interpolate the movements between keypoints from two adjacent videos, and finally draw the keypoint video in results/out_video.mp4. You can also find the normalized and interpolated keypoints in results/normalized_keypoints.pickle. To be specific, one can load the pickle file by
```
with open('normalized_keypoints.pickle', 'rb') as handle:
    List = pickle.load(handle)
```

Each element in the List is a dicionary of standard DWPose keypoints structure. In addition, we also add mediapipe keypoints there. That says, each dictionary in the List holds the mediapipe and DWPose keypoints extracted from the corresponding video frame, or the keypoints interpolated based on two adjacent videos.
```
'bodies': {'candidate': array of shape (18,2), 'subset': array([[ 0., 1., 2., 3., 4., 5., 6., 7., 8., -1., -1., 11., -1., -1., 14., 15., 16., 17.]])},
'hands': array of shape (2, 21, 2), where the first (21, 2) array is left hand, the second is right hand.
'faces': array of shape (1, 68, 2), storing the standard 68 keypoints of human face.
'pose_mp': array of shape (33*3), storing the x/y/z values of the 33 mediapipe pose keypoints.
'face_mp': array of shape (478*3), storing the x/y/z values of the 478 mediapipe face keypoints.
'left_hand_mp' and 'right_hand_mp': array of shape (21*3), storing the x/y/z values of the 21 mediapipe left/right hand keypoints.
```

