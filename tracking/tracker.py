# %%
import ultralytics
# import ultralyticsplus
import supervision as sv
# import pathlib
# import glob
# import numpy as np
import torch
# import matplotlib.pyplot as plt
# import PIL
import cv2
from config import PLAYER_MODEL
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# import pandas as pd
# import sys
import ultralytics.utils.loss
# %%
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

# %%


# Custom models trained on older versions look for DFLoss here
if not hasattr(ultralytics.utils.loss, 'DFLoss'):
    from ultralytics.utils.loss import v8DetectionLoss
    # In newer versions, DFLoss is an attribute or internal class inside the detection loss
    # If the exact class exists under a different submodule, we link it:
    try:
        from ultralytics.models.yolo.detect.val import DFLoss
        ultralytics.utils.loss.DFLoss = DFLoss
    except ImportError:
        # Fallback dummy class if it's only needed for structural unpickling
        class DFLoss:
            pass
        ultralytics.utils.loss.DFLoss = DFLoss


# %%
# print(torch.__version__)
class tracked_obj():
    def __init__(self ,location, conf , pred_label , predicted_name , crop_pic ,id):
        self.bounding_box = location
        self.confidence =  conf
        self.predicted_label = pred_label
        self.predicted_name  = predicted_name
        self.crop_pic = crop_pic
        self.id = id
    def __str__(self):
        return f"""
        {self.predicted_name}
        {self.predicted_label}
        {self.confidence}
        {self.crop_pic}
        
        
        """

# %%
target_width = 100
target_height = 200

# %%






class Tracker():
    def __init__(self):
        # self.model = ultralytics.YOLO("yolo26n.pt")
        # self.model = ultralytics.YOLO()
        # self.model = ultralytics.YOLO("./runs/dete")
        self.model = ultralytics.YOLO(PLAYER_MODEL)
        
        # self.model = ultralytics.YOLO("/home/mohammad-amin/Desktop/footbal anaylsis/models/yolo-football-player-detection.pt")
        self.tracked_objects = {}
        self.tracker =sv.ByteTrack(
            track_activation_threshold=0.5,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30
)
        # print(self.model)
        # self.model = ultralytics.YOLO("yolo-football-player-detection.pt")


           

        # self.model = ultralytics.YOLO.from_pretrained("uisikdag/yolo-v8-football-players-detection")
        # self.model = ultralyticsplus.YOLO("uisikdag/yolo-v8-football-players-detection" , )
        # self.model = ultralytics.YOLO("uisikdag/yolo-v8-football-players-detection")


        
    def tracking(self , frame):
        # tracker = sv.ByteTrack(track_thresh=0.35,match_thresh=.95 , track_bufffer = 30)

        ls = []
        # path = pathlib.Path(video_path)
        # print(c)
        # box_annotator = sv.BoundingBoxAnnotator()
        # detections = None
        # ca = cv.VideoCapture()
        classes = self.model.names
        
        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)
        for i in range(len(detections)):
            # if detections[i].tracker_id[0] not in self.tracked_objects.keys():
                # print(detections[i])
            x1 , y1 , x2 , y2 = detections[i].xyxy[0][0] ,detections[i].xyxy[0][1] , detections[i].xyxy[0][2] , detections[i].xyxy[0][3]
            x1,y1,x2,y2 = int(x1) , int(y1) , int(x2) , int(y2)
            z = int((y2-y1)/2)
            crop_pic = frame[y1:y1+z , x1:x2]
            crop_pic = cv2.resize(crop_pic, (target_width, target_height))
            new_o = tracked_obj(detections[i].xyxy , detections[i].confidence , detections[i].class_id[0] , detections[i].data['class_name'] ,crop_pic , detections[i].tracker_id[0] )
                # print(crop_pic)
            ls.append(new_o)
            self.tracked_objects[detections[i].tracker_id[0]] = new_o
        return ls , detections
        # print(frame)
        # print(last_frame)
        # print(detections)
        # print(len(detections))
        # print(detections.tracker_id)
        # for item in detections:
            # print(len(detections))
      
        # return self.tracked_objects
    def return_tracked_objs(self):
        return self.tracked_objects
# t = Tracker()
# trackkk = t.tracking("/home/mohammad-amin/Desktop/footbal anaylsis/datasets/simple.mp4")

# %%
# print(len(xxxx))
# lasttt
# len(trackkk)
# lasttt

# x= 0
# for value in trackkk.values():
#     print(value.bounding_box)
#     # value.bounding_box
#     plt.imshow(value.crop_pic)
#     break
#     x+=1
    # if x==120:
    #     break
    # break
# for val in trackkk.values():
#     print(val.predicted_name)
# print(f)
# for item in xxxx:
    # prin
# print(xxxx[0].class_id)

# # %%
# pca = PCA(n_components=2)


# # %%
# len(trackkk)

# # %%
# cluster_model = KMeans(n_clusters=2)
# data = [{"id":x , "confidence" : trackkk[x].confidence , "label":trackkk[x].predicted_label , "class_name": trackkk[x].predicted_name , "crop_image":trackkk[x].crop_pic} for x in trackkk.keys()]

# data_frame = pd.DataFrame(data)
# # eee = torch.Tensor(data_frame["crop_image"].tolist()).reshape(115 , -1)
# # cluster_model.fit_transform
# # new_data_frame = pd.DataFrame()
# # croped_players = list(map(lambda x : x.crop_pic , trackkk.values()))


# # %%
# data_frame["crop_image_reshaped"] = data_frame["crop_image"].apply(lambda img : img.flatten())


# # %%
# results = pca.fit_transform(data_frame["crop_image_reshaped"].tolist())

# # %%
# x_s = results[:,0]
# y_s = results[:,1]

# # %%
# plt.scatter(x_s , y_s)

# # %%
# cluster_model.fit(results)

# # %%
# cluster_model.labels_

# # %%
# data_frame['team_label'] = cluster_model.labels_

# # %%
# data_frame['team_label']

# %%
# plt.scatter(results[0][:,0] , results[0][:,1])

# %%
# results.shape

# %%
# cluster_model.fit_transform(data_frame["crop_image"].tolist())

# %%
# len(data_frame["crop_image_reshaped"].tolist()[1])

# %%
# print(data_frame["class_name"].tolist()[0])

# %%
# results = pca.fit_transform(data_frame["crop_image_reshaped"].tolist())
# results = pca.fit_transform(data_frame["crop_image_reshaped"].tolist())



