from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2



# from rembg import remove as remove_bg

# from transformers import AutoProcessor, SiglipVisionModel

# SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(DEVICE)
# EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
# image = cv2.imread("1.png")
# hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
# blur = cv2.GaussianBlur(hsv[:,:,0], (15, 15), 5)
# _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
# mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
# print(mask)
# plt.imshow(mask)
# plt.show()


# def background_removal(image= None):

#     img = cv2.imread('1.png')


#     # Read image
#     # img = cv2.imread('pills.jpg')
#     hh, ww = img.shape[:2]

#     # threshold on white
#     # Define lower and uppper limits
#     lower = np.array([200, 200, 200])
#     upper = np.array([255, 255, 255])

#     # Create mask to only select black
#     thresh = cv2.inRange(img, lower, upper)

#     # apply morphology
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
#     morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

#     # invert morp image
#     mask = 255 - morph

#     # apply mask to image
#     result = cv2.bitwise_and(img, img, mask=mask)
#     print(result)
#     cv2.imwrite('pills_result.png', result)

#     return result


# background_removal()







def assigning(trackkk):
    pca = PCA(n_components=2)
    cluster_model = KMeans(n_clusters=2)
    data = [{"id":x.id , "confidence" : x.confidence , "label":x.predicted_label , "class_name": x.predicted_name , "crop_image":x.crop_pic} for x in trackkk]

    data_frame = pd.DataFrame(data , columns=["id" , "confidence" , "label" , "class_name" , "crop_image"])
    # print(data_frame.columns)
    data_frame["crop_image_reshaped"] = data_frame["crop_image"].apply(lambda img : img.flatten())


    # images = [cv2.]
    # print(data_frame["crop_image_reshaped"].tolist())
    # for item in data_frame["crop_image_reshaped"].tolist():
    #     print(item)
    # X = X_2d = np.stack(your_ldata_frame["crop_image_reshaped"].tolist())s
    X = np.array(data_frame["crop_image_reshaped"].tolist())
    # print(X.shape)
    results = pca.fit_transform(X)
    x_s = results[:,0]
    y_s = results[:,1]
    # plt.scatter(x_s , y_s)
    cluster_model.fit(results)
    cluster_model.labels_
    data_frame['team_label'] = cluster_model.labels_
    return data_frame


#SEEMS TO BE WORKING BETTER.

# def my_assigning2(frame, players_detection): #pca 
#     features = []

#     for i, bbox in enumerate(players_detection.xyxy):
#         x1, y1, x2, y2 = map(int, bbox)

#         # upper half crop
#         h = y2 - y1 + 5
#         x = x2 - x1
#         x1 = x1 + x // 4
#         x2 = x2 - x // 4
#         crop = frame[y1:y1 + h // 2, x1:x2]

#         if crop.size == 0:
#             continue

#         crop = cv2.resize(crop, (30, 30))
#         # sv.plot_image(crop)

#         pixels = crop.reshape(-1, 3).astype(np.float32)

#         pca = PCA(n_components=1)
#         pca.fit(pixels)

#         projection = pca.transform(pixels).mean()

#         domcol = pca.mean_ + projection * pca.components_[0]
#         domcol = np.clip(domcol, 0, 255) 
#         features.append(domcol)

#     features = np.array(features)
#     kmeans = KMeans(n_clusters=2, random_state=42)
#     labels = kmeans.fit_predict(features)

#     # print(labels, 'labels')
#     # print(len(labels))

#     return labels



# Load image
# image = cv2.imread("1.png")

# Create a mask
# mask = np.zeros(image.shape[:2], np.uint8)

# # Define background and foreground models (needed by GrabCut)
# bgd_model = np.zeros((1, 65), np.float64)
# fgd_model = np.zeros((1, 65), np.float64)

# # Define a rectangle around the foreground object
# height, width = image.shape[:2]
# rect = (10, 10, width-20, height-20)  # adjust as needed

# # Apply GrabCut algorithm
# cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# # Modify mask: 0,2 -> background; 1,3 -> foreground
# mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# # Apply mask to the image
# result = image * mask2[:, :, np.newaxis]

# # Save or display result
# cv2.imwrite("output_no_bg.png", result)
# cv2.imshow("Removed Background", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#SEEMS TO BEWORKING BETTER.

def my_assigning2(frame, players_detection): #pca 
    features = []
    aa = {}
    all_pixels = []
    for i, bbox in enumerate(players_detection.xyxy):
        x1, y1, x2, y2 = map(int, bbox)

        x1, y1, x2, y2 = map(int, bbox)

    #     #broader frame? 
        x1 = max(0, x1 - 15)
        y1 = max(0, y1 - 15)
        x2 = min(frame.shape[1], x2 + 15)
        y2 = min(frame.shape[0], y2 + 15)

        crop = frame[y1:y2 , x1:x2]
       
        if crop.size == 0:
            continue

        image = cv2.resize(crop, (50, 100))
        mask = np.zeros(image.shape[:2], np.uint8)

# Define background and foreground models (needed by GrabCut)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

# Define a rectangle around the foreground object
        height, width = image.shape[:2]
        rect = (10, 10, width-20, height-20)  # adjust as needed

# Apply GrabCut algorithm
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# Modify mask: 0,2 -> background; 1,3 -> foreground
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Apply mask to the image   
        result = image * mask2[:, :, np.newaxis]
        # sv.plot_image(crop)
        # break
    #     # Remove background
    #     bg_removed = remove_bg(crop)
    #     # print(bg_removed.size,'rm')
    #     bg_removed=cv2.resize(bg_removed,(100,100))

    #     #now crop the upper part only?
    #     bg_removed = bg_removed[:50, :, :]
        # print(bg_removed.shape)
        # print(bg_removed)

        # sv.plot_image(bg_removed)

        # plt.show()
        # print("here")
        # break
        # sv.plot_image(crop)

        # pixels = result.reshape(-, 3).astype(np.float32)
        all_pixels.append(result)
    all_pixels = np.array(all_pixels)
    # print("all pixels shape",all_pixels.shape)
    pca = PCA(n_components=5)
    # pca.fit(all_pixels)  # Fit PCA on the first player's pixels (or you can concatenate all pixels)
    projection = pca.fit_transform(all_pixels.reshape(len(all_pixels), -1)) # Get the mean projection for each player
    # print(projection.shape)
    # domcol = pca.mean_ + projection * pca.components_[0]
    # domcol = np.clip(domcol, 0, 255) 
    # features.append(domcol)
    # print(projection)
    features = np.array(projection)
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(features)
    for i in range(len(players_detection)):
        aa[players_detection.tracker_id[i]] = labels[i]
    # print(all_pixels.shape)
    cs1 = all_pixels[labels]
    cs2 = all_pixels[~labels]
    # print(cs1[0].shape)
    cs1_center = cs1.mean(axis=0)
    cs2_center = cs2.mean(axis=0)
    # cs1_center = cs1_center
    cs1_c= cs1_center[50][25]
    cs2_c = cs2_center[50][25]
    # print(cs2_center[50][25])

    # cs1_c = cs1_center.mean(axis=0)
    # cs2_c = cs2_center.mean(axis=0)
    # print(cs1_c.shape)
    # cs1_center = cs1.mean(axis=0)
    # cs2_center = cs2.mean(axis=0)
    # cs1
    # cs1_c = cs1_center[2500]
    # cs2_c = cs2_center[2500]
    green = np.array([0,255,0])
    # print(cs1_c )
    # print(cs2_c)
    if np.abs(cs1_c - green).sum() < np.abs(cs2_c - green).sum():
        labels = np.array([0 if label == 0 else 1 for label in labels])
    else:
        labels = np.array([1 if label == 0 else 0 for label in labels])

    # for i in range(len(players_detection)):
        # crops = 
        # print(f"Player ID: {players_detection.tracker_id[i]}, Team Label: {labels[i]}")
    #! the clustering works fine. however the cluster Ids are not consistent over different frames. to fix this :
    # we use cluster centers. and find out which one is more green :
    # green - avg(blue + red)
    # then we assign 1 to green and 0 to white.
    # this method works fine for white vs green. not sure about other color team combinations :D (it probably wont work tho)

    # l1 = labels.copy()

    # features = np.array(features)
    # green_scores = features[:, 1] - ((features[:, 0] + features[:, 2]) / 2)
    # centers = kmeans.cluster_centers_

    # green_scores = centers[:, 1] - ((centers[:, 0] + centers[:, 2]) / 2)
    # green_cluster = np.argmax(green_scores)

    # labels = np.array([
    #     0 if label == green_cluster else 1
    #     for label in labels
    # ])
    # print
    # print(labels, 'labels')
    # print(len(labels))
    # print(sum(labels))
    # print(labels.count(0))
    # print(labels.count(1))

    # print(features, 'features')

    # print(labels, 'labels-*')
    # print(l1, 'labels1')
    # print(len(labels))
    return labels , aa


def my_assigning4(frame, players_detection): #pca 
    features = []
    aa = {}
    all_pixels = []
    for i, bbox in enumerate(players_detection.xyxy):
        x1, y1, x2, y2 = map(int, bbox)

        x1, y1, x2, y2 = map(int, bbox)

       #broader frame? 
        x1 = max(0, x1 - 15)
        y1 = max(0, y1 - 15)
        x2 = min(frame.shape[1], x2 + 15)
        y2 = min(frame.shape[0], y2 + 15)

        crop = frame[y1:y2 , x1:x2]
       
        if crop.size == 0:
            continue

        image = cv2.resize(crop, (100, 100))
        # sv.plot_image(image)
        mask = np.zeros(image.shape[:2], np.uint8)

# Define background and foreground models (needed by GrabCut)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

# Define a rectangle around the foreground object
        height, width = image.shape[:2]
        rect = (10, 10, width-20, height-20)  # adjust as needed

# Apply GrabCut algorithm
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# Modify mask: 0,2 -> background; 1,3 -> foreground
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Apply mask to the image   
        result = image * mask2[:, :, np.newaxis]

        result = result[:50, :, :]
        # sv.plot_image(result)
        # print(result.shape,'rm')

        mask = np.any(result != [0, 0, 0], axis=-1)
        avg_color = result[mask].mean(axis=0)


        swatch = np.zeros((80, 80, 3), dtype=np.uint8)
        swatch[:, :] = avg_color.astype(np.uint8)
        # plt.figure()
        # plt.title(f"Player {i} avg color {avg_color.astype(int)}")
        # plt.imshow(swatch)
        # plt.axis("off")
        # plt.show()


        features.append(avg_color)
    
    features = np.array(features)
    # print(features)
    features[np.isnan(features)] = 255

    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(features)

    #! the clustering works fine. however the cluster Ids are not consistent over different frames. to fix this :
    # we use cluster centers. and find out which one is more green :
    # green - avg(blue + red)
    # then we assign 1 to green and 0 to white.
    # this method works fine for white vs green. not sure about other color team combinations :D (it probably wont work tho)

    # l1 = labels.copy()
    features = np.array(features)
    green_scores = features[:, 1] - ((features[:, 0] + features[:, 2]) / 2)
    centers = kmeans.cluster_centers_

    green_scores = centers[:, 1] - ((centers[:, 0] + centers[:, 2]) / 2)
    green_cluster = np.argmax(green_scores)

    labels = np.array([
        0 if label == green_cluster else 1
        for label in labels
    ])

    
    # print(features, 'features')

    # print(labels, 'labels-*')
    # print(l1, 'labels1')
    # print(len(labels))
    return labels
