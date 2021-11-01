from numpy.lib.function_base import append
from FileNameManager import filePathManager
import Constants
import cv2
import numpy as np
from pprint import pprint
from PIL import Image, ImageDraw, ImageOps
from utils import chop_corners

class GenerateEgoCentricFrontLayout(object):
    def __init__(self):
        self.length = Constants.LENGTH
        self.width = Constants.WIDTH
        self.layout_size = Constants.LAYOUT_SIZE
        self.res = self.length / self.layout_size 
        self.DEBUG = Constants.DEBUG

    def writeLayout(self, ID, dump_path, shelf_and_boxes, min_shelf_number, max_shelf_number, aa, bb, cc, dd, ee, ff):
        # print(shelf_and_boxes)
        shelf_layouts = {}
        box_layouts = {}
        #interShelfDistance = self.annotations["intershelfDistance"] #self.getInterShelfDistance(min_shelf_number)
        for shelf_number in range(min_shelf_number, max_shelf_number+1):
            if shelf_number not in shelf_and_boxes:
                continue
            shelfs, boxes = shelf_and_boxes[shelf_number]
            shelf = self.getProminentShelfAnnotation(shelfs)
            layout = np.zeros(
                [int(self.length/self.res), 
                int(self.width/self.res)],
                dtype= np.uint8
            )
            layout_shelf = Image.fromarray(layout)
            for shelf in shelfs:
                interShelfDistance = float(shelf["object_dimensions"][1])#shelf["interShelfDistance"])
                centerX, centerY, _ = shelf["object_ego_location"]
                camera_rotation_z = shelf["camera_rotation"][1]
                layout_shelf, bottom_y = self.generateFrontalLayoutShelf(layout_shelf, shelf, centerX , centerY, interShelfDistance)
                layout_shelf = self.accountCameraRotation(layout_shelf, camera_rotation_z)
            shelf_layouts[shelf_number] = layout_shelf

            layout_box = self.generateFrontalLayoutBoxes(boxes, centerX, centerY)
            layout_box = self.accountCameraRotation(layout_box, camera_rotation_z)
            box_layouts[shelf_number] = layout_box

        self.write_layouts(shelf_layouts, box_layouts, interShelfDistance, ID, dump_path, aa, bb, cc, dd, ee, ff)

    def getProminentShelfAnnotation(self, shelves):
        mx = 0
        ret_an = None
        for shelf in shelves:
            if(shelf["object_dimensions"][0] > mx):
                ret_an = shelf
                mx = shelf["object_dimensions"][0]
        return ret_an

    def generateFrontalLayoutShelf(self, layout, annotation, img_x, img_y, obj_w):
        
        x,y,_ = annotation["object_ego_location"]
        center_x = int((float(x)) / self.res + self.width / (2*self.res))
        center_y = int((-float(y)) / self.res + self.length / (2*self.res))
        # center_y = int((float(img_y)-float(y)) / self.res + self.length / (2*self.res))
        orient = 0
        dimensions = annotation["object_dimensions"]
        # #print("FREE SPACE : ", obj_w)
        obj_w = int((float(dimensions[1]))/self.res)
        obj_l = int(float(dimensions[0])/self.res)
        rectangle = self.get_rect(center_x, center_y, obj_l, obj_w, orient)
        draw = ImageDraw.Draw(layout)
        draw.polygon([tuple(p) for p in rectangle], fill = 115)
        layout = layout.convert('L')
        bottom_y = center_y+int(obj_w/2)
        return layout, bottom_y

    def accountCameraRotation(self, layout, camera_rotation):
        # if(float(camera_rotation) < np.pi and float(camera_rotation) > -np.pi):
        #     layout = ImageOps.mirror(layout)
        return layout

    def generateFrontalLayoutBoxes(self, annotations, img_x, img_y):
        layout = np.zeros(
            [int(self.length/self.res), 
            int(self.width/self.res)],
            dtype= np.uint8
        )
        layout = Image.fromarray(layout)
        for annotation in annotations:
            x,y,_ = annotation["object_ego_location"]
            center_x = int((float(x)) / self.res + self.width / (2*self.res))
            center_y = int((-float(y)) / self.res + self.length / (2*self.res))
            orient = 0
            dimensions = annotation["object_dimensions"]
            # print("BOX",dimensions[2])
            obj_w = int(float(dimensions[1])/self.res)
            obj_l = int(float(dimensions[0])/self.res)
            rectangle = self.get_rect(center_x, center_y, obj_l, obj_w, orient)
            draw = ImageDraw.Draw(layout)
            draw.polygon([tuple(p) for p in rectangle], fill = 255)
            layout = layout.convert('L')
        return layout

    def get_rect(self, x, y, width, height, theta):
        rect = np.array([(-width / 2, -height / 2), (width / 2, -height / 2),
                         (width / 2, height / 2), (-width / 2, height / 2),
                         (-width / 2, -height / 2)])

        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])

        offset = np.array([x, y])
        transformed_rect = np.dot(rect, R) + offset
        return transformed_rect

    def getInterShelfDistance(self, min_shelf):
        min_shelf, _ = self.get_shelf_range()
        shelf_1, shelf_2 = min_shelf, min_shelf+1
        if(shelf_1 == None or shelf_2 == None):
            shelfHeightDifference = Constants.MAX_SHELF_DIFF_VAL
        else:
            bottomShelfAnnotation,_ = self.get_shelf_and_boxes(shelf_1)
            topShelfAnnotation,_ = self.get_shelf_and_boxes(shelf_2)
            heightOfBottomShelf = bottomShelfAnnotation["location"][2]
            heightOftopShelf = topShelfAnnotation["location"][2]
            shelfHeightDifference = abs(float(heightOftopShelf) - float(heightOfBottomShelf))
        return shelfHeightDifference


    def write_layouts(self, rack_layouts, box_layouts, shelfHeightDifference, ID, dump_path, aa, bb, cc, dd, ee, ff):
        final_layout_racks = []
        empty_npy = 0
        write_track = 0
        for shelf in range(Constants.MAX_SHELVES):
            if(shelf not in rack_layouts):
                # pixels = np.zeros((int(self.length/self.res), int(self.width/self.res)))
                empty_npy += 1
                continue
            else:
                pixels = list(rack_layouts[shelf].getdata())
                width, height = rack_layouts[shelf].size
                pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
                pixels = np.array(pixels)

                pixelsb = list(box_layouts[shelf].getdata())
                width, height = box_layouts[shelf].size
                pixelsb = [pixelsb[i * width:(i + 1) * width] for i in range(height)]
                

                for i in range(len(pixels)):
                    for j in range(len(pixels[i])):
                        if(pixelsb[i][j] != 255):
                            pixelsb[i][j] = pixels[i][j]
                pixels = np.array(pixelsb) 
                pixels = chop_corners(pixels)   

                perm_string = str(aa) + str(bb) + str(cc) + str(dd) + str(ee) + str(ff)
                perm_string = ""
                if(self.DEBUG):
                    cv2.imwrite(filePathManager.getDebugRackLayoutPath("front"+perm_string ,ID, write_track), pixels)
                    filePathManager.updateDebugImageNumber()
                final_layout_racks.append(pixels)
                write_track += 1

        empty_pixels = np.zeros((int(self.length/self.res), int(self.width/self.res)))
        for shelf in range(empty_npy):
            if(self.DEBUG):
                cv2.imwrite(filePathManager.getDebugRackLayoutPath("front",ID, write_track+shelf), empty_pixels)
                filePathManager.updateDebugImageNumber()
            final_layout_racks.append(empty_pixels)    
        final_layout_racks = np.array(final_layout_racks)

        file_path = dump_path +"front"+ ID[:-4] + ".npy"
        np.save(file_path, final_layout_racks)

        # final_layouts_boxes = []
        # for shelf in range(Constants.MAX_SHELVES):
        #     #boxes_img_data[0][i] = boxes_img_data[0][i].rotate(180)
        #     if(shelf >= len(box_layouts)):
        #         pixels = np.zeros((int(self.length/self.res), int(self.width/self.res)))
        #     else:
        #         pixels = list(box_layouts[shelf].getdata())
        #         width, height = box_layouts[shelf].size
        #         pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
        #         pixels = np.array(pixels)
        #     if(self.DEBUG):
        #         cv2.imwrite(filePathManager.getDebugRackLayoutPath("frontBox",ID, shelf), pixels)
        #         filePathManager.updateDebugImageNumber()
        #     final_layouts_boxes.append(pixels)
        # final_layouts_boxes = np.array(final_layouts_boxes)
        # file_path = dump_path +"frontBox"+ ID[:-4]+ ".npy"
        # np.save(file_path,final_layouts_boxes)
    
        np.save(dump_path +"height"+ ID[:-4] + ".npy",shelfHeightDifference)

generateEgoCentricFrontLayout = GenerateEgoCentricFrontLayout()
