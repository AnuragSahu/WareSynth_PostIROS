# utils.py

from dis import dis
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# from merge_and_make_3d import BB_3D_Boxes, BB_3D_Shelves

# ONE PIXEL MAPS TO HOW MUCH DISTANCE
PIXEL_TO_DISTANCE_MAPPING = int(512 / 80)

class MERGE_LAYOUTS:
    def __init__(self, FRONT_VIEW_FOLDER, TOP_VIEW_FOLDER, RGB_IMAGE_FOLDER):
        self.FRONT_VIEW_FOLDER = FRONT_VIEW_FOLDER
        self.TOP_VIEW_FOLDER = TOP_VIEW_FOLDER
        self.RGB_IMAGE_FOLDER = RGB_IMAGE_FOLDER
        self.mapping_rgb_layout = {}
        self.previous_dist = 0

    def getBoundingBoxes(self, img):
        img = img.astype(np.uint8)
        boundingBoxes = []
        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        for cntr in contours:
            x,y,w,h = cv2.boundingRect(cntr)
            boundingBoxes.append([x,y,w,h])
        
        # return boundingBoxes[0]
        boundingBoxes.sort()

        bb1 = boundingBoxes[0]
        bb2 = boundingBoxes[-1]

        # print(bb1, bb2)

        x = bb1[0]
        y = bb2[1]
        w = bb2[2] + (bb2[0] - bb1[0])
        h = bb1[3]
        return [x, y, w, h]

    def merge_layout(self, layout_1, layout_2, distance=0.1):

        PIXELS_MOVED = distance
        
        # Create a blank layout
        canvas = np.zeros((layout_1.shape[0], layout_1.shape[1]+ 10))
        
        # Fill canvas with layout 1
        canvas[:, :layout_1.shape[1]] = layout_1

        # get the boundind boxes
        shelfThresh = cv2.threshold(layout_2, 10, 155, cv2.THRESH_BINARY)[1]
        rackBB = self.getBoundingBoxes(shelfThresh)

        # print(rackBB)
        # put racks layout inside Bounding box to canvas
        canvas[rackBB[1]: rackBB[1]+rackBB[3], PIXELS_MOVED+rackBB[0]:PIXELS_MOVED+rackBB[0]+rackBB[2]] = layout_2[rackBB[1]: rackBB[1]+rackBB[3], rackBB[0]: rackBB[0]+rackBB[2]]        
        
        return canvas

    def compare_two_layouts(self, layout_1, layout_2):
        layout_1 = layout_1.astype(np.uint8)
        layout_2 = layout_2.astype(np.uint8)
        
        # Find the bounding box for layout 1
        thres_l1 = cv2.threshold(layout_1, 10, 155, cv2.THRESH_BINARY)[1]
        l1_BB = self.getBoundingBoxes(thres_l1)

        # thres_l2 = cv2.threshold(layout_2, 10, 155, cv2.THRESH_BINARY)[1]
        # l2_BB = self.getBoundingBoxes(thres_l2)

        # print(l1_BB, l2_BB)
        
        l1 = layout_1[l1_BB[1] : l1_BB[1] + l1_BB[3], l1_BB[0] : l1_BB[0] + l1_BB[2]]
        l2 = np.zeros((l1.shape[0], l1.shape[1]))
        # print(l1.shape, l2.shape)
        l2 = layout_2[l1_BB[1] : l1_BB[1] + l1_BB[3], l1_BB[0] : l1_BB[0] + l1_BB[2]]

        if(l1.shape != l2.shape):
            # print("Error Anne wala h")
            # print(l1.shape, l2.shape)
            l2_new = np.zeros(l1.shape).astype(np.uint8)
            l2_new[:, :l2.shape[1]] = l2
            # print(l2_new.shape)
            l2 = l2_new

        # print(l1.shape, l2.shape)
        
        # compare the two images
        errorL2 = cv2.norm( l1, l2, cv2.NORM_L2 )
        return errorL2

    def find_distance_moved(self, layout_1, layout_2):

        # shift image by one pixel
        min_error = 9999999
        min_error_Index = 1

        # Find the bounding box for layout 1
        thres_l1 = cv2.threshold(layout_1, 10, 155, cv2.THRESH_BINARY)[1]
        l1_BB = self.getBoundingBoxes(thres_l1)

        # thres_l2 = cv2.threshold(layout_2, 10, 155, cv2.THRESH_BINARY)[1]
        # l2_BB = self.getBoundingBoxes(thres_l2)

        for shift in range(self.previous_dist, 200):
            _, w = layout_2.shape
            img_shift_right = np.zeros(layout_2.shape)
            img_shift_right[:,shift:w] = layout_2[:,:w-shift]

            # Compare Layout 1 and Layout 2
            error = self.compare_two_layouts(layout_1, img_shift_right)
            if(error < min_error):
                min_error = error
                min_error_Index = shift
        
        self.previous_dist = min_error_Index

        return min_error_Index

    def merge_layouts(self):

        first_index = 0
        last_index = 49
        layout_front_1_path = self.FRONT_VIEW_FOLDER + "000"+str(first_index).zfill(3)+"_0.png"
        merged_front = cv2.imread(layout_front_1_path, 0)
        layout_top_1_path = self.TOP_VIEW_FOLDER + "000"+str(first_index).zfill(3)+"_0.png"
        merged_top = cv2.imread(layout_top_1_path, 0)

        for index in range(first_index +1, last_index+1):
            ## Fill the above code for faster processing
            ## This code from now will merge the layout for just the bottom most layout
            
            layout_front_2_path = self.FRONT_VIEW_FOLDER + "000"+str(index).zfill(3)+"_0.png"            
            layout_top_2_path = self.TOP_VIEW_FOLDER + "000"+str(index).zfill(3)+"_0.png"
            
            layout_front_2 = cv2.imread(layout_front_2_path, 0)
            layout_top_2 = cv2.imread(layout_top_2_path, 0)

            distance = self.find_distance_moved(merged_front, layout_front_2)
            print(index, "Distance", distance)
            merged_front = self.merge_layout(merged_front, layout_front_2, distance = distance)
            merged_top = self.merge_layout(merged_top, layout_top_2, distance = distance)

        return merged_top, merged_front

class MAKE_3D:
    def __init__(self, top_layouts, front_layouts, box_front_2d, box_top_2d):
        self.front_layouts = front_layouts
        self.top_layouts = top_layouts
        self.prev_box_front_2d = box_front_2d
        self.prev_box_top_2d = box_top_2d

    def getBoundingBoxes(self, img):
        img = img.astype(np.uint8)
        boundingBoxes = []
        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        for cntr in contours:
            x,y,w,h = cv2.boundingRect(cntr)
            boundingBoxes.append([x,y,w,h])

        boundingBoxes.sort()

        return boundingBoxes

    def calculate3DBB(self, topBBox, frontBBox):
        Boxes = []
        j = 0
        iter = max(len(topBBox), len(frontBBox))

        for i in range(iter):    
            length = min(topBBox[i][2], frontBBox[i][2])
            # width
            width = topBBox[i][3]
            # height
            height = frontBBox[i][3]
            # x is towards right
            # Can be averaged?
            x = int(length/2) + max(topBBox[i][0], frontBBox[i][0])
            # y is depth
            y = int(width/2) + topBBox[i][1]
            # z is up
            z = int(height/2) + frontBBox[i][1]
            Boxes.append([x, y, z, length, width, height])
        
        return Boxes

    def plot_BB_Lay(self, im, BB_list):
        id = 0
        for BB in BB_list:
            x,y,w,h = [int(i) for i in BB]
            
            id += 1
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(im, str(id),(x+1,y+h),0,0.5,(0,255,0))
        plt.imshow(im)
        plt.show()
        
    def optimPred(self, curr_front_2d_bb, curr_top_2d_bb, prev_front_2d, prev_top_2d):
        front_2d_bb, top_2d_bb = [], []
        lf = len(curr_front_2d_bb)
        lt = len(curr_top_2d_bb)
        i = 0
        j = 0
        while i < lf and j < lt:
            if abs(curr_front_2d_bb[i][0] - curr_top_2d_bb[j][0]) < 5:
                front_2d_bb.append(curr_front_2d_bb[i])
                top_2d_bb.append(curr_top_2d_bb[j])
                i+=1
                j+=1
            elif i < lf - 1 and j < lt - 1:
                if curr_front_2d_bb[i][0] < curr_top_2d_bb[j][0]:
                    if abs(prev_front_2d[i][0] - curr_front_2d_bb[i][0]) < 10 and abs(prev_front_2d[i][2] - curr_front_2d_bb[i][2]) < 5:
                        front_2d_bb.append(curr_front_2d_bb[i])
                        new = curr_front_2d_bb[i]
                        new[3] = prev_top_2d[i][3]  
                        top_2d_bb.append(new)
                        i+=1
                    elif abs(prev_front_2d[i+1][0] - curr_front_2d_bb[i][0]) < 10 and abs(prev_front_2d[i+1][2] - curr_front_2d_bb[i][2]) < 5:
                        front_2d_bb.append(curr_front_2d_bb[i])
                        new = curr_front_2d_bb[i]
                        new[3] = prev_top_2d[i+1][3]  
                        top_2d_bb.append(new)
                        i+=1
                    else:
                        i+=1
                else:
                    if abs(prev_top_2d[j][0] - curr_top_2d_bb[j][0]) < 10 and abs(prev_top_2d[j][2] - curr_top_2d_bb[j][2]) < 5:
                        top_2d_bb.append(curr_top_2d_bb[j])
                        new = curr_top_2d_bb[j]
                        new[3] = prev_front_2d[j][3]  
                        front_2d_bb.append(new)
                        j+=1
                    elif abs(prev_top_2d[j+1][0] - curr_top_2d_bb[j][0]) < 10 and abs(prev_top_2d[j+1][2] - curr_top_2d_bb[j][2]) < 5:
                        top_2d_bb.append(curr_top_2d_bb[j])
                        new = curr_top_2d_bb[j]
                        new[3] = prev_front_2d[j+1][3]  
                        front_2d_bb.append(new)
                        j+=1
                    else:
                        j+=1

        return front_2d_bb, top_2d_bb

    def make_3D_BB(self):

        # get layout only for boxes
        boxThresh_top = cv2.threshold(self.top_layouts,128,255,cv2.THRESH_BINARY)[1]
        boxThresh_front = cv2.threshold(self.front_layouts,128,255,cv2.THRESH_BINARY)[1]

        # get layout only for racks
        # get the boundind boxes
        shelfThresh_top = cv2.threshold(self.top_layouts, 10, 155, cv2.THRESH_BINARY)[1]
        shelfThresh_front = cv2.threshold(self.front_layouts, 10, 155, cv2.THRESH_BINARY)[1]

        # plt.imshow(boxThresh_top, cmap="gray")
        # plt.show()        

        # Get the 2D BB for boxes top view
        curr_box_top_2d_bb = self.getBoundingBoxes(boxThresh_top)
        
        # Get the 2D BB for boxes front view
        curr_box_front_2d_bb = self.getBoundingBoxes(boxThresh_front)

        # Get the 2D BB for racks top view
        shelf_top_2d_bb = self.getBoundingBoxes(shelfThresh_top)

        # Get the 2D BB for racks front view
        shelf_front_2d_bb = self.getBoundingBoxes(shelfThresh_front)

        # print("Front : ", curr_box_front_2d_bb, len(curr_box_front_2d_bb))
        # print("Top : ", curr_box_top_2d_bb, "\n")
        # Add here
        if self.prev_box_front_2d != None and self.prev_box_top_2d != None:
            box_front_2d_bb, box_top_2d_bb = self.optimPred(curr_box_front_2d_bb, curr_box_top_2d_bb, self.prev_box_front_2d, self.prev_box_top_2d)
        else:
            box_front_2d_bb, box_top_2d_bb = curr_box_front_2d_bb, curr_box_top_2d_bb
        # Uncomment to plot
        # self.plot_BB_Lay(self.top_layouts, box_top_2d_bb)
        # self.plot_BB_Lay(self.front_layouts, box_front_2d_bb)
        # print("\n", box_top_2d_bb, box_front_2d_bb)
        # Get the 3D BB for boxes
        boxes_3d_BB = self.calculate3DBB(box_top_2d_bb, box_front_2d_bb)

        # Get the 3D BB for Racks
        shelves_3d_BB = self.calculate3DBB(shelf_top_2d_bb, shelf_front_2d_bb)

        # print(boxes_3d_BB)

        # return these boxes and racks 3D BB
        return boxes_3d_BB, shelves_3d_BB, box_front_2d_bb, box_top_2d_bb