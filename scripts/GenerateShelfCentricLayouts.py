from os.path import join
from glob import glob
import Constants
import numpy as np
import mathutils
import math
from FileNameManager import filePathManager
from GenerateShelfCentricTopLayout import generateShelfCentricTopLayout
from GenerateShelfCentricFrontLayout import generateShelfCentricFrontLayout
from GenerateFrontalLayout import generateFrontalLayout

from GenerateTopLayout import generateTopLayout
from GenerateFrontalLayout import generateFrontalLayout

class GenerateLayouts(object):
    def __init__(self):
        self.annotations = {}
        self.dimensions_map = {}
        with open(filePathManager.datasetDumpDirectory+"dimensions.txt") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                elem  = line.split(", ")

                self.dimensions_map[elem[0]] =  list(map(float, elem[1:]))
        # print(self.dimensions_map)

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

    def read_annotations(self, annotationsPath, dump_path):
        for file in glob(join(annotationsPath, '*.txt')):
            print("For File : ", file)
            ID = file.split("/")[-1]
            f = open(file, "r")
            annotationLines = f.readlines()
            rack_in_focus = annotationLines[0].strip('\n')
            annotationLines = annotationLines[1:]
            annotationID = 0
            self.max_shelf_number = 0
            self.annotations = {}
            # box_count = 0
            for annotationLine in annotationLines:
                annotationLine = annotationLine.strip('\n')
                labels = annotationLine.split(", ")
                object_type = labels[0]

                if labels[0][0] == 'S':
                    object_type = "Shelf"
                    object_dimensions = self.dimensions_map["Shelf"]
                else:
                    # box_count += 1
                    object_type = "Box"
                    object_dimensions = self.dimensions_map[labels[0]]

                shelf_number = int(labels[2].split('_')[-1])

                object_location = labels[3:6]
                object_orientation = labels[6:9]
                object_scale = labels[9:12]
                camera_location = labels[12:15]
                camera_rotation = labels[15:18]
                camera_rotation = [float(i)*np.pi for i in camera_rotation]

                cutting_plane_limits = {}

                for i in range(18, 18+30, 10):
                    one_plane = labels[i:i+10]
                    # print(one_plane)
                    #can parse here
                    cutting_plane_limits[one_plane[0]] = one_plane[1:]

                interShelfDistance = self.dimensions_map["Shelf"][1]
                
                
                object_location = [float(i) for i in object_location]
                object_location[1] += object_dimensions[1]/2
                
                # print("Object Location : ",object_location)
                # print("Camera Location : ",camera_location)
                objectEgoCentricLocation = self.get_locations(object_location, object_orientation,
                                                                    camera_location, camera_rotation)

                # print("objectEgoCentricLocation : ", objectEgoCentricLocation)                                                                    

                objectEgoCentricRotation_y = 0#np.pi/2-float(object_orientation[2])

                object_dimensions = [float(i) for i in object_dimensions]
                object_scale = [float(i) for i in object_scale]
                camera_location = [float(i) for i in camera_location]
                
                
                if(rack_in_focus == labels[1] or not Constants.RACK_IN_FOCUS):
                    # if(object_type == "BOX"):
                    #     box_count = box_count + 1
                    self.annotations[annotationID] = {
                        "object_type" : object_type,
                        "shelf_number" : shelf_number,
                        "object_location" : object_location,
                        "object_ego_location" : objectEgoCentricLocation,
                        "object_orientation" : object_orientation,
                        "rotation_y" : objectEgoCentricRotation_y,
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

            # print(box_count)
            # generateShelfCentricTopLayout.writeLayout(self.annotations, ID, dump_path)
            # generateShelfCentricFrontLayout.writeLayout(self.annotations, ID, dump_path)
            
            generateTopLayout.writeLayout(self.annotations, ID, dump_path)
            generateFrontalLayout.writeLayout(self.annotations, ID, dump_path)
            
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
        filePathManager.anuragEgoCentricLayouts
    )
    print("Generated Layouts at : ",filePathManager.anuragEgoCentricLayouts)
