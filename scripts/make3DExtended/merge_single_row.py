import cv2
import os
import numpy as np
from utils import MERGE_LAYOUTS, MAKE_3D

FRONT_VIEW_FOLDER = "/home/pranjali/Documents/Post_RackLay/Datasets/singleCol/data/debugOutputs/top"
TOP_VIEW_FOLDER = "/home/pranjali/Documents/Post_RackLay/Datasets/singleCol/data/debugOutputs/front"
folder = "/home/pranjali/Documents/Post_RackLay/Datasets/singleCol/data/debugOutputs/"
frames_per_frame = 70
num_rows = 2
num_cols = 2

corridor_col = 800
corridor_row = 300

_, _, files = next(os.walk(folder))
file_count = int(len(files) / 6)
print(file_count)
curr_row = 0
curr_col = 0

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

for i in range(file_count):

    print(f"Generating annotation for {i}")
    # Bottom
    top_view = cv2.imread(TOP_VIEW_FOLDER + str(i).zfill(6) + "_0.png", 0)
    front_view = cv2.imread(FRONT_VIEW_FOLDER + str(i).zfill(6) + "_0.png", 0)
    BB_3D_Boxes_bottom, BB_3D_Shelves_bottom = MAKE_3D(top_view, front_view).make_3D_BB() 

    # Middle
    top_view = cv2.imread(TOP_VIEW_FOLDER + str(i).zfill(6) + "_1.png", 0)
    front_view = cv2.imread(FRONT_VIEW_FOLDER + str(i).zfill(6) + "_1.png", 0)
    BB_3D_Boxes_middle, BB_3D_Shelves_middle = MAKE_3D(top_view, front_view).make_3D_BB()
    for ii in range(len(BB_3D_Boxes_middle)):
        BB_3D_Boxes_middle[ii][1] -= BB_3D_Shelves_bottom[-1][4]

    for ii in range(len(BB_3D_Shelves_middle)):
        BB_3D_Shelves_middle[ii][1] -= BB_3D_Shelves_bottom[-1][4]  
 
    # Top
    top_view = cv2.imread(TOP_VIEW_FOLDER + str(i).zfill(6) + "_2.png", 0)
    front_view = cv2.imread(FRONT_VIEW_FOLDER + str(i).zfill(6) + "_2.png", 0)
    BB_3D_Boxes_top, BB_3D_Shelves_top = MAKE_3D(top_view, front_view).make_3D_BB()
    for ii in range(len(BB_3D_Boxes_top)):
        BB_3D_Boxes_top[ii][1] -= (BB_3D_Shelves_bottom[-1][4] + BB_3D_Shelves_middle[-1][4])

    for ii in range(len(BB_3D_Shelves_top)):
        BB_3D_Shelves_top[ii][1] -= (BB_3D_Shelves_bottom[-1][4] + BB_3D_Shelves_middle[-1][4])  
 

    write_in_file(f"./bottom_ann/ann_{str(i).zfill(6)}_bottom.txt", BB_3D_Boxes_bottom, BB_3D_Shelves_bottom)
    write_in_file(f"./middle_ann/ann_{str(i).zfill(6)}_middle.txt", BB_3D_Boxes_middle, BB_3D_Shelves_middle)
    write_in_file(f"./top_ann/ann_{str(i).zfill(6)}_top.txt", BB_3D_Boxes_top, BB_3D_Shelves_top)

# CREATE_VIZ_BLENDER(BB_3D_Boxes, BB_3D_Shelves).make_in_blender();