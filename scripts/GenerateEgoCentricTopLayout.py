import Constants
import numpy as np
import cv2
from PIL import Image, ImageDraw
from FileNameManager import filePathManager

class GenerateEgoCentricTopLayout(object):
    def __init__(self):
        self.length = Constants.LENGTH
        self.width = Constants.WIDTH
        self.layout_size = Constants.LAYOUT_SIZE
        self.res = self.length / self.layout_size
        self.DEBUG = True
        self.annotations = {}

    def get_locations(self, locations):
        x = locations[1]
        y = locations[2]
        z = locations[2]
        return [x,y,z]

    def get_rect(self, x, y, width, height, theta):
        rect = np.array([(-width / 2, -height / 2), (width / 2, -height / 2),
                         (width / 2, height / 2), (-width / 2, height / 2),
                         (-width / 2, -height / 2)])

        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])

        offset = np.array([x, y])
        transformed_rect = np.dot(rect, R) + offset

        return transformed_rect

    def generate_layout_rack(self, img_data, label,locations, dimentions, rotation_y, rack_number):
        
        #print(label,locations, dimentions, rotation_y, rack_number)
        imgs = img_data
        res = self. res
        length = self.length
        width = self.width

        #print(locations)
        center_x = int(float(locations[0]) / res + width / (2*res))
        # center_x = 256
        # center_y = int(float(locations[1]) / res + length / (2*res))
        center_y = int(float(locations[1]) / res)# + length / (2*res))
        # center_y = 256

        orient = -1 * float(rotation_y)

        obj_w = int(float(dimentions[1])/res)
        obj_l = int(float(dimentions[0])/res)

        # rectangle = get_rect(center_x, center_y, obj_l, obj_w, orient)
        rectangle = self.get_rect(center_x, int(length/res) - center_y, obj_l, obj_w, orient)

        draw = ImageDraw.Draw(imgs)

        if (label == "Shelf"):
            draw.polygon([tuple(p) for p in rectangle], fill=155)

        imgs = imgs.convert('L')
        return imgs

    def generate_layout_Box(self, img_data, label,locations, dimentions, rotation_y, rack_number):
        
        #print(label,locations, dimentions, rotation_y, rack_number)
        imgs = img_data
        res = self. res
        length = self.length
        width = self.width

        #print(locations)
        center_x = int(float(locations[0]) / res + width / (2*res))
        # center_x = 256
        # center_y = int(float(locations[1]) / res + length / (2*res))
        center_y = int(float(locations[1]) / res)# + length / (2*res))
        # center_y = 256

        orient = -1 * float(rotation_y)

        obj_w = int(float(dimentions[1])/res)
        obj_l = int(float(dimentions[0])/res)

        rectangle = self.get_rect(center_x, int(length/res) - center_y, obj_l, obj_w, orient)

        draw = ImageDraw.Draw(imgs)

        draw.polygon([tuple(p) for p in rectangle], fill=255)

        imgs = imgs.convert('L')
        return imgs

    def writeLayout(self, annotations,  ID, dump_path):
        self.annotations = annotations
        min_shelf_number, max_shelf_number = self.get_shelf_range()

        topEgoLayouts = []

        for shelf_number in range(min_shelf_number, max_shelf_number+1):
            shelf, boxes = self.get_shelf_and_boxes(shelf_number)
            #print(len(shelf))
            
            # Get the layout of the shelf
            layout = np.zeros(
                (int(self.length/self.res), 
                int(self.width/self.res))
            )
            layout = Image.fromarray(layout)

            shelf["object_ego_location"] = self.get_locations(shelf["object_ego_location"])

            shelf_images_data = self.generate_layout_rack(layout, 
                                                          shelf["object_type"], 
                                                          shelf["object_ego_location"],
                                                          shelf["object_dimensions"],
                                                          shelf["ego_rotation_y"],
                                                          shelf["shelf_number"])
            for box in boxes:
                shelf_images_data = self.generate_layout_Box(shelf_images_data, 
                                                          box["object_type"], 
                                                          self.get_locations(box["object_ego_location"]),
                                                          box["object_dimensions"],
                                                          box["ego_rotation_y"],
                                                          box["shelf_number"])
                                                      
            if(shelf["camera_rotation"][2] != 4.71238899230957):
                shelf_images_data = shelf_images_data.transpose(Image.FLIP_LEFT_RIGHT)
            
            topEgoLayouts.append(shelf_images_data)
            #shelf_images_data.save("layout_"+str(ID)+"_"+str(shelf_number)+".jpg")
        # self.write_layouts(shelf_layouts, box_layouts, ID, dump_path)
        self.saveNpyFiles(topEgoLayouts, ID, dump_path)

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
        shelf = None
        boxes = []
        for annotation in self.annotations.values():
            if(annotation["shelf_number"] == shelfNumber):
                if(annotation["object_type"] == "Shelf"):
                    shelf = annotation
                elif(annotation["object_type"] == "Box"):
                    boxes.append(annotation)
        return [shelf,boxes]

    def calculateCenter(self, shelf, boxes):
        center_x, center_y = shelf["object_location"][:2]
        shelf["center"][:2] = [0,0]
        for box in boxes:
            box_center_x, box_center_y = box["object_location"][:2]
            box["center"] = [float(box_center_x) - float(center_x), float(box_center_y)-float(center_y)]
        return [shelf, boxes]

    def getShelfLayout(self, shelf):
        layout = np.zeros(
            (int(self.length/self.res), 
            int(self.width/self.res))
        )
        layout = Image.fromarray(layout)
        layout =  self.getOneLayout(shelf,layout, 115)
        return self.accountCameraRotation(shelf["camera_rotation"], layout)

    def getBoxesLayouts(self, boxes):
        layout = np.zeros(
            (int(self.length/self.res), 
            int(self.width/self.res))
        )
        layout = Image.fromarray(layout)
        camera_layout = None
        for box in boxes:
            camera_layout = box["camera_rotation"]
            layout = self.getOneLayout(box, layout, 255)
        if(camera_layout != None): # rotate only if there is/are some boxes in the shelf
            layout = self.accountCameraRotation(camera_layout, layout)
        return layout

    def accountCameraRotation(self,camera_rotation, layout):
        layout = layout.rotate(float(camera_rotation[2]) * 180 / np.pi)
        return layout
    
    def getOneLayout(self,annotation, layout, fill):
        x,y = annotation["center"]
        center_x = int(float(x) / self.res + self.width / (2*self.res))
        center_y = int(float(y) / self.res + self.length / (2*self.res))
        orient = float(annotation["rotation_y"])
        dimensions = annotation["object_dimensions"]
        obj_w = int(float(dimensions[1])/self.res)
        obj_l = int(float(dimensions[0])/self.res)
        rectangle = self.get_rect(center_x, center_y, obj_l, obj_w, orient)
        draw = ImageDraw.Draw(layout)
        draw.polygon([tuple(p) for p in rectangle], fill=fill)
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

    def write_layouts(self, rack_layouts, box_layouts, ID, dump_path):
        final_layout_racks = []
        for shelf in range(Constants.MAX_SHELVES):
            #print(rack_layouts)
            if(shelf >= len(rack_layouts)):
                pixels = np.zeros((int(self.length/self.res), int(self.width/self.res)))
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
                cv2.imwrite(filePathManager.getDebugRackLayoutPath("top",ID, shelf), pixels)
                filePathManager.updateDebugImageNumber()
            final_layout_racks.append(pixels)
        final_layout_racks = np.array(final_layout_racks)
        file_path = dump_path +"top"+ ID[:-4] + ".npy"
        np.save(file_path, final_layout_racks)

generateEgoCentricTopLayout = GenerateEgoCentricTopLayout()