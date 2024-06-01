# A Real-time video fall detection using human pose and LSTM neural network

## Introduction
This project is a video-based fall detection system that allows users to train and predict falls. It utilizes techiniques such as YOLO, kalman filter and LSTM neural network.

## Features
- Fall training: Users can train the system by providing labeled video data of falls.

- Fall prediction: The system can predict falls in based on either real-time video input or video files input.

## Setup and run
**Create an environment and install dependencies:**
1. User venv or whatever python environment management tool to create a new env, here is the sample of using venv:
- python -m venv cvproject
- source cvproject/bin/activate (for mac/linux)
- cvproject\Scripts\activate (for windows)
2. Install the required dependencies: `pip install -r requirements.txt`

**Traning the model**  
Notes: current codes are designed to train based on the NTU RGB+D datasets, which encoded the labels (action classes) in the file names. If you want to train your own videos, the codes need modified.
The whole NTU+B dataset could be downloaded via (request needed):
https://rose1.ntu.edu.sg/dataset/actionRecognition/

1. python train.py keypoint # to get the keypoints from all videos under /video folder
2. python train.py train to train the neural network


Predicting steps:  
1. Run the application: Execute the main script `py main.py` 
- there is a switch in the main function to predict the test videos or predict on video inputs
