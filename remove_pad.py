import os
import cv2

path = "resources/depth_adabins"
out_path = "resources/predicted_adabins"

for file in os.listdir(path):
    img = cv2.imread(os.path.join(path, file))
    print(img.shape)
    img = img[120:360, 230:410, :]
    print(img.shape)
    cv2.imwrite(os.path.join(out_path,file), img)