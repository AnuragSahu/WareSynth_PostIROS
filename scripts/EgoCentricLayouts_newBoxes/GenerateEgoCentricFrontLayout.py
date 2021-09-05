from numpy.lib.function_base import append
from FileNameManager import filePathManager
import Constants
import cv2
import numpy as np
from pprint import pprint
from PIL import Image, ImageDraw, ImageOps

class GenerateEgoCentricFrontLayout(object):
    def __init__(self):
        self.length = Constants.LENGTH
        self.width = Constants.WIDTH
        self.layout_size = Constants.LAYOUT_SIZE
        self.res = self.length / self.layout_size
        self.DEBUG = True

    def writeLayout(self, ID, dump_path, shelf_and_boxes, min_shelf_number, max_shelf_number, aa, bb, cc, dd, ee, ff):
        # pprint(shelf_and_boxes)
        shelf_layouts = {}
        box_layouts = {}
        #interShelfDistance = self.annotations["intershelfDistance"] #self.getInterShelfDistance(min_shelf_number)
        for shelf_number in range(min_shelf_number, max_shelf_number+1):
            if shelf_number not in shelf_and_boxes:
                continue
            shelf, boxes = shelf_and_boxes[shelf_number]
            interShelfDistance = float(shelf["object_dimensions"][1])#shelf["interShelfDistance"])
            centerX, centerY, _ = shelf["object_ego_location"]
            camera_rotation_z = shelf["camera_rotation"][1]
            layout_shelf, bottom_y = self.generateFrontalLayoutShelf(shelf, centerX , centerY, interShelfDistance)
            layout_shelf = self.accountCameraRotation(layout_shelf, camera_rotation_z)
            shelf_layouts[shelf_number] = layout_shelf

            layout_box = self.generateFrontalLayoutBoxes(boxes, centerX, centerY, bottom_y)
            layout_box = self.accountCameraRotation(layout_box, camera_rotation_z)
            box_layouts[shelf_number] = layout_box

        self.write_layouts(shelf_layouts, box_layouts, interShelfDistance, ID, dump_path, aa, bb, cc, dd, ee, ff)

    def generateFrontalLayoutShelf(self, annotation, img_x, img_y, obj_w):
        layout = np.zeros(
            (int(self.length/self.res), 
            int(self.width/self.res))
        )
        layout = Image.fromarray(layout)
        x,y,_ = annotation["object_ego_location"]
        center_x = int((float(x)) / self.res + self.width / (2*self.res))
        center_y = int((-float(y)) / self.res + self.length / (2*self.res))
        # center_y = int((float(img_y)-float(y)) / self.res + self.length / (2*self.res))
        orient = 0
        dimensions = annotation["object_dimensions"]
        # print("FREE SPACE : ", obj_w)
        obj_w = int((float(obj_w))/self.res)
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

    def generateFrontalLayoutBoxes(self, annotations, img_x, img_y, bottom_y):
        layout = np.zeros(
            (int(self.length/self.res), 
            int(self.width/self.res))
        )
        layout = Image.fromarray(layout)
        stacked_list = []
        for annotation in annotations:
            x,y,_ = annotation["object_ego_location"]
            center_x = int((float(x)) / self.res + self.width / (2*self.res))
            center_y = int((-float(y)) / self.res + self.length / (2*self.res))
            orient = 0
            dimensions = annotation["object_dimensions"]
            # print("BOX",dimensions[2])
            obj_w = int(float(dimensions[1])/self.res)
            obj_l = int(float(dimensions[0])/self.res)
            print(annotation["object_name"], center_x, center_y, obj_w, obj_l)
            if (len(stacked_list) == 0):
                stacked_list.append([[center_x, center_y, obj_l, obj_w]])
            # print(stacked_list)
            
            appended = False
            for box in stacked_list:
                cx,cy,ox,oy = box[0]
                if(center_x < cx + int(ox/2) and center_x > cx - int(ox/2)):
                    box.append([cx, center_y, obj_l, obj_w])
                    # print("stacking detected")
                    appended = True
                    break
            if(not appended):
                stacked_list.append([[center_x, center_y, obj_l, obj_w]])
        if(len(stacked_list) == 0):
            # No boxes in scene return layout
            return layout
        stacked_list[0].pop(0)
        max = 0
        for i in range(len(stacked_list)):
            if(max < len(stacked_list[i])):
                max = len(stacked_list[i])

        # Get bottom most box
        for stacked_boxes in stacked_list:
            maxBox_y = stacked_boxes[0][1]
            maxBoxIndex = 0
            for i in range(len(stacked_boxes)):
                if(maxBox_y < stacked_boxes[i][1]):
                    maxBox_y = stacked_boxes[i][1]
                    maxBoxIndex = i

            # Now we know the bottom most box
            # adjust the y for bottom most box
            # print(maxBox_y, maxBoxIndex)
            prev_y = stacked_boxes[maxBoxIndex][1]
            stacked_boxes[maxBoxIndex][1] = bottom_y - int(stacked_boxes[maxBoxIndex][3]/2)
            current_y = stacked_boxes[maxBoxIndex][1]
            change = current_y - prev_y
            for i in range(len(stacked_boxes)):
                if(i != maxBoxIndex):
                    stacked_boxes[i][1] += change
            # print(stacked_boxes)

        # print("Done ")
        for stacked_boxes in stacked_list:
            # x,y,_ = annotation["object_ego_location"]
            # center_x = int((float(x)) / self.res + self.width / (2*self.res))
            # center_y = int((-float(y)) / self.res + self.length / (2*self.res))
            # orient = 0
            # dimensions = annotation["object_dimensions"]
            # obj_w = int(float(dimensions[1])/self.res)
            # obj_l = int(float(dimensions[0])/self.res)

            # -----------------
            for box in stacked_boxes:
                center_x, center_y, obj_l, obj_w = box
            # -----------------

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

                perm_string = str(aa) + str(bb) + str(cc) + str(dd) + str(ee) + str(ff)
                perm_string = ""
                if(self.DEBUG):
                    cv2.imwrite(filePathManager.getDebugRackLayoutPath("front "+perm_string ,ID, write_track), pixels)
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
