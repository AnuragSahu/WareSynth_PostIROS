from math import floor
from turtle import screensize
import cv2
import numpy as np
from ConvexHullUtil import convexHull
import matplotlib.pyplot as plt

###########################################################################
# Define Configuration here
LAYOUT_SIZE_LEN = 480
LAYOUT_SIZE_WID = 270
PATH_TO_ANNOTATION_FILES = "/home/anurag/Research/Unity/data/Keypoints/"
PATH_TO_DUMP_LAYOUT = "/home/anurag/Research/Unity/data/keypointImages/"
###########################################################################

def populate_scene_info(path):
    scene_info = {}
    file = open(path)
    annotations = file.readlines()
    for line in annotations:
        # check if the line is a starting to indicate a new box
        if(line[:3] == "box"):
            # get the shelf number and rack number
            ann = line.split(" ")
            rack_name = ann[-1][:-1]
            shelf_name = ann[-2]
            key = rack_name+" "+shelf_name 
            if(key not in scene_info):
                scene_info[key] = []
        else:
            ann = line.split(", ")
            # print(ann)
            x, y = float(ann[-2]), float(ann[-1][:-1])
            scene_info[key].append([floor(x),floor(y)])
    return scene_info

def getEndPoints(scene_info):
    for key in scene_info:
        scene_info[key] = convexHull(scene_info[key])
    return scene_info

def make_layout(scene_info):
    layout = np.zeros((LAYOUT_SIZE_WID, LAYOUT_SIZE_LEN))
    for key in scene_info:
        contours = np.array(scene_info[key])
        cv2.fillPoly(layout, pts = [contours], color =(255,255,255))
    return layout

path = "/home/anurag/Research/WareSynth_PostIROS/scripts/KeyPointsLayouts/000000.txt"
scene_info = populate_scene_info(path)
scene_info = getEndPoints(scene_info)
layout = make_layout(scene_info)
cv2.imwrite("test.png", layout)