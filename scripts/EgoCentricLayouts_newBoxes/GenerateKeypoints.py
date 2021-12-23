import cv2
from FileNameManager import filePathManager
import os
import numpy as np


def addGaussianNoise(img, y, x):
    h, w = img.shape

    # extent of gaussian noise centered at y, x
    extent = 1

    # generate noise
    for j in range(-extent, extent+1):
        for i in range(-extent, extent+1):
            noise = np.random.randint(0, 255)
            if 0 <= y + j < h and 0 <= x + i < w:
                img[y + j, x + i] = noise


def main():
    img = cv2.imread(filePathManager.datasetDumpDirectory + "img/000000.png")
    h, w = img.shape[:-1]

    keypoints_dir = filePathManager.datasetDumpDirectory + "Keypoints/"
    output_dir = filePathManager.keypointImagesPath

    # loop over all keypoint files
    for filename in os.listdir(keypoints_dir):
        print(filename)
        img = np.zeros((h, w))
        
        with open(keypoints_dir + filename) as f:        
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
            for line in lines:
                line = list(map(float, line.split(", ")))
                x = min(int(round(line[-2])), w-1)
                y = min(int(round(line[-1])), h-1)
                
                # set keypoint to white
                img[y, x] = 255
                # add gaussian noise around the keypoint
                addGaussianNoise(img, y, x)

            cv2.imwrite(output_dir + filename.split(".")[0] + ".png", img)


main()
