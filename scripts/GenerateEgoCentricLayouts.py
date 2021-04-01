from os.path import join
from glob import glob
import Constants
import numpy as np
import mathutils
import math
from FileNameManager import filePathManager
from GenerateEgoCentricTopLayout import generateEgoCentricTopLayout
from GenerateFrontalLayout import generateFrontalLayout

class GenerateLayouts(object):
    def __init__(self):
        self.annotations = {}

    def get_3x4_RT(self, loc, rot):
        # bcam stands for blender camera
        R_bcam2cv = Matrix(
            ((1, 0,  0),
            (0, -1, 0),
            (0, 0, -1)))

        # Transpose since the rotation is object rotation, 
        # and we want coordinate rotation
        # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
        # T_world2bcam = -1*R_world2bcam * location
        #
        # Use matrix_world instead to account for all constraints
        location = loc
        rotation = mathutils.Euler((rot[0], rot[1], rot[2]), 'XYZ')
        R_world2bcam = rotation.to_matrix().transposed()

        # Convert camera location to translation vector used in coordinate changes
        T_world2bcam = -1*R_world2bcam @ location
        # Use location from matrix_world to account for constraints:     
        #T_world2bcam = -1*R_world2bcam * location

        # Build the coordinate transform matrix from world to computer vision camera
        R_world2cv = R_bcam2cv @ R_world2bcam
        T_world2cv = R_bcam2cv @ T_world2bcam

        # put into 3x4 matrix
        RT = math.Matrix((
            R_world2cv[0][:] + (T_world2cv[0],),
            R_world2cv[1][:] + (T_world2cv[1],),
            R_world2cv[2][:] + (T_world2cv[2],)
            ))
        return np.array(RT)

    def get_locations(self, obj_loc, obj_rot, obj_dim, cam_loc, cam_rot):
        RT = self.get_3x4_RT(cam_loc, cam_rot)
        extra_vec = np.array([0,0,0,1])
        RT = np.vstack((RT, extra_vec))
        locations = np.array([
            obj_loc[0],
            obj_loc[2],
            obj_loc[1],
            1
        ])

        locations = RT @ locations
        locations /= locations[3]
        locations = locations[:3]
        locations = [locations[0],locations[1]+obj_dim[2]/2,locations[2]]
        locations = [str(i) for i in locations]
        return locations

    def read_annotations(self, annotationsPath, dump_path):
        for file in glob(join(annotationsPath, '*.txt')):
            ID = file.split("/")[-1]
            print("For ID : ",ID)
            f = open(file, "r")
            annotationLines = f.readlines()
            annotationID = 0
            self.max_shelf_number = 0
            self.annotations = {}
            for annotationLine in annotationLines:
                labels = annotationLine.split(", ")
                object_type = labels[0]
                shelf_number = int(labels[1])
                object_location = labels[2:5]
                object_orientation = labels[5:8]
                rotation_y = labels[7]
                object_dimensions = labels[8:11]
                object_scale = labels[11:14]
                camera_location = labels[14:17]
                camera_rotation = labels[17:20]
                interShelfDistance = labels[23]

                objectEgoCentricLocation = self.get_locations(object_location, object_orientation, object_dimensions,
                                                                camera_location, camera_rotation)
                objectEgoCentricRotation_y = np.pi/2-object_orientation[2]

                self.annotations[annotationID] = {
                    "object_type" : object_type,
                    "shelf_number" : shelf_number,
                    "object_ego_location" : objectEgoCentricLocation,
                    "object_orientation" : object_orientation,
                    "ego_rotation_y" : objectEgoCentricRotation_y,
                    "object_scale" : object_scale,
                    "object_dimensions" : object_dimensions,
                    "camera_location" : camera_location,
                    "camera_rotation" : camera_rotation,
                    "center" : [0,0],
                    "interShelfDistance" : interShelfDistance
                }
                if(shelf_number > self.max_shelf_number):
                    self.max_shelf_number = shelf_number
                annotationID += 1
    
            generateEgoCentricTopLayout.writeLayout(self.annotations, ID, dump_path)
            #generateFrontalLayout.writeLayout(self.annotations, ID, dump_path)
            
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

    def getInterShelfDistance(self):
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
            print(shelfHeightDifference)
        return shelfHeightDifference



    def get_shelf_range(self):
        min_shelf = 99999999
        max_shelf = 0
        for annotation in self.annotations.values():
            if(annotation["shelf_number"] < min_shelf):
                min_shelf = annotation["shelf_number"]
            if(annotation["shelf_number"] > max_shelf):
                max_shelf = annotation["shelf_number"]

        return [min_shelf, max_shelf]

if __name__ == "__main__":
    generatelayouts = GenerateLayouts()
    generatelayouts.read_annotations(
        filePathManager.anuragAnnotationsLabelsPath,
        filePathManager.anuragRGBImagesPath
    )
    print("Generated Layouts at : ",filePathManager.anuragRGBImagesPath)
