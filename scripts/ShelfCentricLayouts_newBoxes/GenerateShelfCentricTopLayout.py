import Constants
import numpy as np
import cv2
import mathutils
from PIL import Image, ImageDraw
import Constants
from FileNameManager import filePathManager
# from preProcessing.FillRackGaps import fillRackGaps

class GenerateShelfCentricTopLayout(object):
    def __init__(self):
        self.length = Constants.LENGTH
        self.width = Constants.WIDTH
        self.layout_size = Constants.LAYOUT_SIZE
        self.res = self.length / self.layout_size
        self.DEBUG = Constants.DEBUG
        self.annotations = {}
        self.scale = 1
    
    def eul2rot(self, theta) :
        theta = [float(theta[0]), float(theta[1]), float(theta[2])]
        R = np.array([[np.cos(theta[1])*np.cos(theta[2]),np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]), np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                      [np.sin(theta[2])*np.cos(theta[1]),np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]), np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                      [-np.sin(theta[1]),np.sin(theta[0])*np.cos(theta[1]),np.cos(theta[0])*np.cos(theta[1])]])

        pSet = 6
        R = np.array([[round(R[0,0],pSet), round(R[0,1], pSet), round(R[0,2],pSet)],
                      [round(R[1,0],pSet), round(R[1,1], pSet), round(R[1,2],pSet)],
                      [round(R[2,0],pSet), round(R[2,1], pSet), round(R[2,2],pSet)]])

        return R

    def get_3x4_RT(self, loc, rot):
        # bcam stands for blender camera
        R_bcam2cv = mathutils.Matrix(
            ((1, 0,  0),
            (0, 1, 0),
            (0, 0, 1)))
        R_bcam2cv = np.array(R_bcam2cv)
        # Transpose since the rotation is object rotation, 
        # and we want coordinate rotation
        # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
        # T_world2bcam = -1*R_world2bcam * location
        #
        # Use matrix_world instead to account for all constraints
        R_world2bcam = self.eul2rot(rot)
        R_world2bcam = R_world2bcam.T
        
        location = np.array([float(loc[0]), float(loc[1]), float(loc[2])])
        #print(loc)
        #print(rot)
        #rotation = mathutils.Euler((float(rot[0]), float(rot[1]), float(rot[2])))
        #R_world2bcam = rotation.to_matrix().transposed()
        #R_world2bcam = np.array(R_world2bcam)
        #print("R World Matrix : ", np.array(R_world2bcam))
        # Convert camera location to translation vector used in coordinate changes

        #print(R_world2bcam)
        #print(location)
        T_world2bcam = -1*R_world2bcam @ location

        #print("T_world2bcam : ",T_world2bcam)
        # Use location from matrix_world to account for constraints:     
        #T_world2bcam = -1*R_world2bcam * location

        # Build the coordinate transform matrix from world to computer vision camera
        R_world2cv = R_bcam2cv @ R_world2bcam
        T_world2cv = R_bcam2cv @ T_world2bcam
        # put into 3x4 matrix
        RT = np.array([[R_world2cv[0][0], R_world2cv[0][1], R_world2cv[0][2] , T_world2cv[0]],
                       [R_world2cv[1][0], R_world2cv[1][1], R_world2cv[1][2] , T_world2cv[1]],
                       [R_world2cv[2][0], R_world2cv[2][1], R_world2cv[2][2] , T_world2cv[2]]
        ])
        return np.array(RT)

    def get_locations(self, obj_loc, obj_rot, cam_loc, cam_rot):
        RT = self.get_3x4_RT(cam_loc, cam_rot)
        extra_vec = np.array([0,0,0,1])
        RT = np.vstack((RT, extra_vec))

        locations = np.array([
            float(obj_loc[0]),
            float(obj_loc[1]),
            float(obj_loc[2]),
            1
        ])

        #print("RT : ",RT)
        #print("loc : ",locations)
        locations = RT @ locations
        locations /= locations[3]
        #print(obj_loc,cam_loc)
        locations = locations[:3]
        #locations = [locations[2],locations[1]+float(obj_dim[2])/2,locations[0]]
        #locations = [locations[2],locations[1],locations[0]]
        #locations = [str(i) for i in locations]
        return locations

    # def get_locations(self, locations):
    #     x = locations[0]
    #     y = locations[1]
    #     z = locations[2]
    #     return [x,y,z]

    def get_rect(self, x, y, width, height, theta):
        rect = np.array([(-width / 2, -height / 2), (width / 2, -height / 2),
                         (width / 2, height / 2), (-width / 2, height / 2),
                         (-width / 2, -height / 2)])

        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])

        offset = np.array([x, y])
        transformed_rect = np.dot(rect, R) + offset

        return transformed_rect

    def writeLayout(self, ID, dump_path, shelf_and_boxes, min_shelf_number, max_shelf_number):
        shelf_layouts = {}
        box_layouts = {}


        for shelf_number in range(min_shelf_number, max_shelf_number+1):
            if shelf_number not in shelf_and_boxes:
                continue
            shelfs, boxes = shelf_and_boxes[shelf_number]
            shelfs, boxes = self.calculateCenter(shelfs, boxes)
            shelf_layouts[shelf_number], center_of_shelf = self.getShelfLayout(shelfs)
            box_layouts[shelf_number] = self.getBoxesLayouts(boxes, center_of_shelf)
            
        self.write_layouts(shelf_layouts, box_layouts, ID, dump_path)
        # self.saveNpyFiles(topEgoLayouts, ID, dump_path)
    
    def calculateCenter(self, shelf, boxes):
        center_x, center_y = shelf["object_location"][0], shelf["object_location"][2]
        shelf["center"][:2] = [0,0]
        for box in boxes:
            box_center_x, box_center_y = box["object_location"][0], box["object_location"][2]
            box["center"] = [float(box_center_x) - float(center_x), float(box_center_y)-float(center_y)]
        return [shelf, boxes]

    def saveNpyFiles(self, topEgoLayouts, ID, dump_path):
        final_layout_racks = []
        for shelf in range(Constants.MAX_SHELVES):
            if(shelf >= len(topEgoLayouts)):
                pixels = np.zeros((int(self.length/self.res), int(self.width/self.res)))
            else:
                pixels = list(topEgoLayouts[shelf].getdata())
                width, height = topEgoLayouts[shelf].size
                pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
    
            pixels = np.array(pixels)    

            if(self.DEBUG):
                cv2.imwrite(filePathManager.getDebugRackLayoutPath("top",ID, shelf), pixels)
                filePathManager.updateDebugImageNumber()
            final_layout_racks.append(pixels)
        final_layout_racks = np.array(final_layout_racks)
        file_path = dump_path +"top"+ ID[:-4] + ".npy"
        np.save(file_path, final_layout_racks)

    
    def get_shelf_range(self):
        min_shelf = 99999999
        max_shelf = 0
        for annotation in self.annotations.values():
            if(annotation["shelf_number"] < min_shelf):
                min_shelf = annotation["shelf_number"]
            if(annotation["shelf_number"] > max_shelf):
                max_shelf = annotation["shelf_number"]
        return [min_shelf, max_shelf]
    
    def get_shelf_and_boxes(self, shelfNumber):
        shelfs = []
        boxes = []
        for annotation in self.annotations.values():
            if(annotation["shelf_number"] == shelfNumber):
                if(annotation["object_type"] == "Shelf"):
                    shelfs.append(annotation)
                elif(annotation["object_type"] == "Box"):
                    boxes.append(annotation)
        return [shelfs,boxes]

    def getShelfLayout(self, shelf):
        layout = np.zeros(
            (int(self.length/self.res), 
            int(self.width/self.res))
        )
        layout = Image.fromarray(layout)
        layout, centers =  self.getOneLayout(shelf,layout, 115)
        return self.accountCameraRotation(shelf["camera_rotation"], layout), centers

    def getBoxesLayouts(self, boxes, center_of_shelf):
        layout = np.zeros(
            (int(self.length/self.res), 
            int(self.width/self.res))
        )
        layout = Image.fromarray(layout)
        camera_layout = None
        for box in boxes:
            camera_layout = box["camera_rotation"]
            layout = self.getOneBoxLayout(box, layout, 255, center_of_shelf)
        if(camera_layout != None): # rotate only if there is/are some boxes in the shelf
            layout = self.accountCameraRotation(camera_layout, layout)
        return layout

    def accountCameraRotation(self,camera_rotation, layout):
        # layout = layout.rotate(float(camera_rotation[2]) * 180 / np.pi)
        return layout
    
    def getOneBoxLayout(self,annotation, layout, fill, center_of_shelf):
        # print(center_of_shelf)
        x,y = annotation["center"][0], annotation["center"][1]
        center_x = int(float(x) / self.res + self.width / (2*self.res))
        center_y = int(float(y) / self.res + self.length / (2*self.res))
        # center_x = center_of_shelf[0] - self.scale*(center_of_shelf[0]-center_x)
        # center_y = center_of_shelf[1] - self.scale*(center_of_shelf[1]-center_y)

        orient = 0 #float(annotation["ego_rotation_y"])
        dimensions = annotation["object_dimensions"]
        obj_w = int(float(dimensions[2])/self.res)*self.scale
        obj_l = int(float(dimensions[0])/self.res)*self.scale
        rectangle = self.get_rect(center_x, int(self.length/self.res) -center_y, obj_l, obj_w, orient)
        draw = ImageDraw.Draw(layout)
        draw.polygon([tuple(p) for p in rectangle], fill=fill)
        layout = layout.convert('L')
        return layout

    def getOneLayout(self,annotation, layout, fill):
        x,y = annotation["center"][0], annotation["center"][1]
        center_x = int(float(x) / self.res + self.width / (2*self.res))
        center_y = int(float(y) / self.res + self.length / (2*self.res))
        
        orient = 0 #float(annotation["ego_rotation_y"])
        dimensions = annotation["object_dimensions"]
        obj_w = int(float(dimensions[2])/self.res)*self.scale
        obj_l = int(float(dimensions[0])/self.res)*self.scale
        rectangle = self.get_rect(center_x, int(self.length/self.res) -center_y, obj_l, obj_w, orient)
        draw = ImageDraw.Draw(layout)
        draw.polygon([tuple(p) for p in rectangle], fill=fill)
        layout = layout.convert('L')
        return layout, (center_x, center_y)

    def get_rect(self, x, y, width, height, theta):
        rect = np.array([(-width / 2, -height / 2), (width / 2, -height / 2),
                         (width / 2, height / 2), (-width / 2, height / 2),
                         (-width / 2, -height / 2)])

        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])

        offset = np.array([x, y])
        transformed_rect = np.dot(rect, R) + offset
        return transformed_rect

    def write_layouts(self, rack_layouts, box_layouts, ID, dump_path):
        # print(rack_layouts)
        final_layout_racks = []
        empty_npy = 0
        write_track = 0
        for shelf in range(Constants.MAX_SHELVES):
            #print(rack_layouts)
            if(shelf not in rack_layouts):
                # pixels = np.zeros((int(self.length/self.res), int(self.width/self.res)))
                empty_npy += 1
                continue
            else:
                pixels = list(rack_layouts[shelf].getdata())
                width, height = rack_layouts[shelf].size
                pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
                
                pixelsb = list(box_layouts[shelf].getdata())
                width, height = box_layouts[shelf].size
                pixelsb = [pixelsb[i * width:(i + 1) * width] for i in range(height)]
                

                for i in range(len(pixels)):
                    for j in range(len(pixels[i])):
                        if(pixelsb[i][j] != 255):
                            pixelsb[i][j] = pixels[i][j]
                pixels = np.array(pixelsb)    

            if(self.DEBUG):
                cv2.imwrite(filePathManager.getDebugRackLayoutPath("top",ID, write_track), pixels)
                filePathManager.updateDebugImageNumber()
            final_layout_racks.append(pixels)
            write_track += 1
        empty_pixels = np.zeros((int(self.length/self.res), int(self.width/self.res)))
        for shelf in range(empty_npy):
            if(self.DEBUG):
                cv2.imwrite(filePathManager.getDebugRackLayoutPath("top",ID, write_track+shelf), empty_pixels)
                filePathManager.updateDebugImageNumber()
            final_layout_racks.append(empty_pixels)
    
        final_layout_racks = np.array(final_layout_racks)
        file_path = dump_path +"top"+ ID[:-4] + ".npy"
        np.save(file_path, final_layout_racks)

generateShelfCentricTopLayout = GenerateShelfCentricTopLayout()
