# %%
# !pip install -U ultralytics
# !pip install -U torch 

# %%
# !pip install -U torchvision


# %%
# !pip show ultralytics

# %%


# %%
# !pip show torch

# %%
import ultralytics
# import ultralyticsplus
# import supervision as sv
# import pathlib
# import glob
import numpy as np
import torch
# import matplotlib.pyplot as plt
import PIL
from config import PITCH_MODEL
# import cv2
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# import pandas as pd
# import torchvision

# %%
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

# %%
model = ultralytics.YOLO(PITCH_MODEL)
# model = ultralytics.YOLO("/home/mohammad-amin/Desktop/footbal anaylsis/models/field-colab.pt" )



# %%
class keypoint:
    def __init__(self , x , y , confidence):
        self.x = x
        self.y = y
        self.confidence = confidence
    def __str__(self):
        return f"Keypoint(x={self.x}, y={self.y}, confidence={self.confidence})"

# %%
def pitch_detection(frame , model = model):
    # cap = cv2.VideoCapture(video_path)
    data_s = []
    # while True:
    #     ls = []
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
        
    results = model.predict(source=frame)[0]
    # key_points=  sv.KeyPoints.from_inference(results)
    # print(results)
    keypoints = results[0].keypoints
    # print(keypoints.conf.shap
    # print(config.vertices)
    fil = keypoints.conf[0] > 0.8
    reference_frame_points = np.array(keypoints.xy[0][fil])
    return reference_frame_points , keypoints , fil
        # break




