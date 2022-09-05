import cv2
import os
import numpy as np
from utils import MERGE_LAYOUTS, MAKE_3D

FRONT_VIEW_FOLDER = "/home/pranjali/Documents/Post_RackLay/data_new/debugOutputs/front"
TOP_VIEW_FOLDER = "/home/pranjali/Documents/Post_RackLay/data_new/debugOutputs/top"
folder = "/home/pranjali/Documents/Post_RackLay/data_new/debugOutputs/"
frames_per_seq = 70
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

Bottom_box_front_2d_BB = None
Bottom_box_top_2d_BB = None
Mid_box_front_2d_BB = None
Mid_box_top_2d_BB = None
Top_box_front_2d_BB = None
Top_box_top_2d_BB = None

for i in range(file_count):
    if i%frames_per_seq == 0:
        if curr_col < num_cols:
            curr_col += 1
            if i ==0 :
                curr_row += 1
        else :
            curr_col = 1
            curr_row += 1
    print(f"Generating annotation for {i}")

    # Bottom
    top_view = cv2.imread(TOP_VIEW_FOLDER +  str(i).zfill(6) + "_0.png", 0)
    front_view = cv2.imread(FRONT_VIEW_FOLDER +  str(i).zfill(6) + "_0.png", 0)
    BB_3D_Boxes_bottom, BB_3D_Shelves_bottom, Bottom_box_front_2d_BB, Bottom_box_top_2d_BB = MAKE_3D(top_view, front_view, Bottom_box_front_2d_BB, Bottom_box_top_2d_BB).make_3D_BB() 
    for ii in range(len(BB_3D_Boxes_bottom)):
        BB_3D_Boxes_bottom[ii][0] -= (curr_col - 1)*corridor_col
        BB_3D_Boxes_bottom[ii][1] -= (curr_row - 1)*corridor_row
    for ii in range(len(BB_3D_Shelves_bottom)): 
        BB_3D_Shelves_bottom[ii][0] -= (curr_col - 1)*corridor_col 
        BB_3D_Shelves_bottom[ii][1] -= (curr_row - 1)*corridor_row

    # Middle
    top_view = cv2.imread(TOP_VIEW_FOLDER +  str(i).zfill(6) + "_1.png", 0)
    front_view = cv2.imread(FRONT_VIEW_FOLDER +  str(i).zfill(6) + "_1.png", 0)
    BB_3D_Boxes_middle, BB_3D_Shelves_middle, Mid_box_front_2d_BB, Mid_box_top_2d_BB = MAKE_3D(top_view, front_view, Mid_box_front_2d_BB, Mid_box_top_2d_BB).make_3D_BB()
    for ii in range(len(BB_3D_Boxes_middle)):
        BB_3D_Boxes_middle[ii][2] -= BB_3D_Shelves_bottom[-1][-1]
        BB_3D_Boxes_middle[ii][0] -= (curr_col - 1)*corridor_col
        BB_3D_Boxes_middle[ii][1] -= (curr_row - 1)*corridor_row
    for ii in range(len(BB_3D_Shelves_middle)):
        BB_3D_Shelves_middle[ii][2] -= BB_3D_Shelves_bottom[-1][-1]  
        BB_3D_Shelves_middle[ii][0] -= (curr_col - 1)*corridor_col 
        BB_3D_Shelves_middle[ii][1] -= (curr_row - 1)*corridor_row  

    # Top
    top_view = cv2.imread(TOP_VIEW_FOLDER +  str(i).zfill(6) + "_2.png", 0)
    front_view = cv2.imread(FRONT_VIEW_FOLDER +  str(i).zfill(6) + "_2.png", 0)
    BB_3D_Boxes_top, BB_3D_Shelves_top, Top_box_front_2d_BB, Top_box_top_2d_BB = MAKE_3D(top_view, front_view, Top_box_front_2d_BB, Top_box_top_2d_BB).make_3D_BB()
    for ii in range(len(BB_3D_Boxes_top)):
        BB_3D_Boxes_top[ii][2] -= (BB_3D_Shelves_bottom[-1][-1] + BB_3D_Shelves_middle[-1][-1])
        BB_3D_Boxes_top[ii][0] -= (curr_col - 1)*corridor_col
        BB_3D_Boxes_top[ii][1] -= (curr_row - 1)*corridor_row
    for ii in range(len(BB_3D_Shelves_top)):
        BB_3D_Shelves_top[ii][2] -= (BB_3D_Shelves_bottom[-1][-1] + BB_3D_Shelves_middle[-1][-1])  
        BB_3D_Shelves_top[ii][0] -= (curr_col - 1)*corridor_col 
        BB_3D_Shelves_top[ii][1] -= (curr_row - 1)*corridor_row  

    write_in_file(f"./bottom_ann/ann_{curr_row}_{curr_col}_{str(i).zfill(6)}_bottom.txt", BB_3D_Boxes_bottom, BB_3D_Shelves_bottom)
    write_in_file(f"./middle_ann/ann_{curr_row}_{curr_col}_{str(i).zfill(6)}_middle.txt", BB_3D_Boxes_middle, BB_3D_Shelves_middle)
    write_in_file(f"./top_ann/ann_{curr_row}_{curr_col}_{str(i).zfill(6)}_top.txt", BB_3D_Boxes_top, BB_3D_Shelves_top)