import numpy as np
import cv2
from skimage.metrics import structural_similarity
import os


def mse(img1, img2):
    h, w = img1.shape
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff ** 2)
    mse = err / (float(h * w))
    return mse


tof_path = "resources/depth_original"
prediction_path = "resources/predicted_adabins"

for file in os.listdir(tof_path):
    img1 = cv2.imread(os.path.join(tof_path, file), 0)
    img2 = cv2.imread(os.path.join(prediction_path, file), 0)
    score, diff = structural_similarity(img1, img2, full=True)  # Calculating SSSIM
    err = mse(img1, img2)  # Calculating Mean Square Error
    cv2.imshow('diff', diff)
    cv2.waitKey(0)
    print("Comapring predictions for " + file + " ,Mean Square Error:" + str(err) + " ,SSIM Score:" + str(score))