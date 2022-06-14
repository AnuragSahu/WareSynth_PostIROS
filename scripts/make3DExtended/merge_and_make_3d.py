import cv2
import os
import numpy as np
from utils import MERGE_LAYOUTS, MAKE_3D


FRONT_VIEW_FOLDER = "/Users/vampire/RESEARCH/data/debugOutputs/front"
TOP_VIEW_FOLDER = "/Users/vampire/RESEARCH/data/debugOutputs/top"
RGB_IMAGE_FOLDER = "./3D_Reconstruction/img/"

def write_in_file(path, boxes, shelves):
    f = open(path, "w")
    for box in boxes:
        x,y,z,length, width, height = box
        x = str(x)
        y = str(y)
        z = str(z)
        length = str(length)
        width = str(width)
        height = str(height)

        f.write("Box, " +  x + ", " + y + ", " + z + ", " + length + ", " + width + ", " + height + "\n")

    for box in shelves:
        x,y,z,length, width, height = box
        x = str(x)
        y = str(y)
        z = str(z)
        length = str(length)
        width = str(width)
        height = str(height)
        f.write("Shelf, " + x + ", " + y + ", " + z + ", " + length + ", " + width + ", " + height + "\n")


merged_top_view, merged_front_view = MERGE_LAYOUTS(FRONT_VIEW_FOLDER,
                                                    TOP_VIEW_FOLDER,
                                                    RGB_IMAGE_FOLDER).merge_layouts();

# print(merged_front_view.shape)

BB_3D_Boxes, BB_3D_Shelves = MAKE_3D(merged_top_view, merged_front_view).make_3D_BB();
write_in_file("./ann.txt", BB_3D_Boxes, BB_3D_Shelves)

# CREATE_VIZ_BLENDER(BB_3D_Boxes, BB_3D_Shelves).make_in_blender();