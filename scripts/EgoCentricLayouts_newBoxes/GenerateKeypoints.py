import cv2
from FileNameManager import filePathManager
import os
import numpy as np
import json


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

        # loop over each line and construct box_stacks dict
        box_stacks = {}
        stack_num = None
        for line in lines:
            line = line.split(", ")

            # coordinates
            try:
                line = list(map(float, line))
                if stack_num not in box_stacks:
                    box_stacks[stack_num] = []
                box_stacks[stack_num].append(line)
            # name of box
            except ValueError:
                stack_num = int(line[0].split(" ")[-3])

        # if (filename == "000000.txt"):
        #     with open('data.json', 'w+') as f:
        #         json.dump(box_stacks, f)

        # loop over each box stack
        for stack_num, box_points in box_stacks.items():
            # sort box keypoints in current stack by 3D y coordinate
            box_points.sort(key=lambda coords: coords[1])

            min_y = box_points[0][1]
            max_y = box_points[-1][1]

            # find 2D correspondences of points with max y and min y (in 3D)
            # use these as keypoints
            keypoints = [[point[-2], point[-1]]
                         for point in box_points if point[1] in [min_y, max_y]]

            # use all 2D correspondences as keypoints
            # keypoints = [[point[-2], point[-1]] for point in box_points]

            # loop over keypoints
            for keypoint in keypoints:
                x = min(int(round(keypoint[0])), w-1)
                y = min(int(round(keypoint[1])), h-1)

                # set keypoint to white
                img[y, x] = 255
                # add gaussian noise around the keypoint
                addGaussianNoise(img, y, x)

        cv2.imwrite(output_dir + filename.split(".")[0] + ".png", img)


main()
