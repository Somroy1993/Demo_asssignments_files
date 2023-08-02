import numpy as np
import os
import cv2

path = "resources/rgb"
out_path = "resources/resized"

for image in os.listdir(path):
    img = cv2.imread(os.path.join(path, image))
    print(img.shape)
    img = np.pad(img, ((120,120), (230,230), (0,0)), 'constant', constant_values=255)
    print(img.shape)
    cv2.imwrite(os.path.join(out_path, image), img)
    # cv2.imshow('padded', img)
    # cv2.waitKey(0)