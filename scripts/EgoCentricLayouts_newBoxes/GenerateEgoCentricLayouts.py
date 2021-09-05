import copy
from itertools import permutations
from os.path import join
from glob import glob
from pprint import pprint
import Constants
import numpy as np
import mathutils
import math
from FileNameManager import filePathManager
from GenerateEgoCentricTopLayout import generateEgoCentricTopLayout
from GenerateEgoCentricFrontLayout import generateEgoCentricFrontLayout
# from GenerateFrontalLayout import generateFrontalLayout
import threading

class GenerateLayouts(object):
    def __init__(self):
        self.annotations = {}
        self.dimensions_map = {}
        self.num_threads = 1
        self.count = 0
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

    def get_percentage_visible(self, cutting_planes_visible):
        percents_x = []
        percents_y = []
        percents_z = []

        # bounds_vis[1] = minX[1];
		# bounds_vis[2] = maxX[1];
		# bounds_vis[3] = minY[1];
		# bounds_vis[4] = maxY[1];
		# bounds_vis[5] = minZ[1];
		# bounds_vis[6] = maxZ[1];
		
		# bounds_vis[7] = minX[0];
		# bounds_vis[8] = maxX[0];
		# bounds_vis[9] = minY[0];
		# bounds_vis[10] = maxY[0];
		# bounds_vis[11] = minZ[0];
		# bounds_vis[12] = maxZ[0];
		
		# //// Debug.Log(go.name +" "+ z +" X PERCENTAGE IS "+ (maxX[1] - minX[1])/(maxX[0] - minX[0]) +"    Y PERCENTAGE IS"+ (maxY[1] - minY[1])/(maxY[0] - minY[0]));
		
        # do right and left things here
        planes_viz = 0
        for plane in cutting_planes_visible:
            bounds_vis = cutting_planes_visible[plane]
            #print(bounds_vis)
            if bounds_vis[0] == 0:
                percents_x.append(0)
                percents_y.append(0)
            else:
                percents_x.append( (bounds_vis[2] - bounds_vis[1])/(bounds_vis[8] - bounds_vis[7]) )
                percents_y.append( (bounds_vis[4] - bounds_vis[3])/(bounds_vis[10] - bounds_vis[9]) )
                planes_viz += 1

        return min(percents_x), min(percents_y), planes_viz*0.5  


    def get_locations(self, obj_loc, cam_loc, cam_rot):
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

    def read_annotations(self, annotationsPath, dump_path, aa, bb, cc, dd, ee, ff):
        files_split = {}

        for i in range(self.num_threads):
            files_split[i] = []

        c = 0
        for file in glob(join(annotationsPath, '*.txt')):
            files_split[c%self.num_threads].append(file)
            c += 1

        threads = []

        for i in range(self.num_threads):
            t = threading.Thread(target=self.generate_layout_from_file, args=(files_split[i], dump_path, aa, bb, cc, dd, ee, ff))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

    def generate_layout_from_file(self, files, dump_path, aa, bb, cc, dd, ee, ff):
        for file in files:
            print("For file no. %d"%(self.count))
            self.count += 1
            if self.count == 10:
                return
            ID = file.split("/")[-1]
            # if ID == "000000.txt":
            #     break
            f = open(file, "r")
            annotationLines = f.readlines()
            rack_in_focus = annotationLines[0].strip('\n')
            annotationLines = annotationLines[1:]
            annotationID = 0
            self.max_shelf_number = 0
            curr_annotations = {}
            stack_list = {}
            for annotationLine in annotationLines:
                annotationLine = annotationLine.strip('\n')
                labels = annotationLine.split(", ")
                object_type = labels[0].split(" stack ")[0]

                cutting_plane_limits = {}
                # cplns = []    
                for i in range(18, 18+3*14, 14):
                    one_plane = labels[i:i+14]
                    #print(one_plane)
                    #can parse here
                    cutting_plane_limits[one_plane[0]] = list(map(float, one_plane[1:]))

                percent_visible_x, percent_visible_y, _ = self.get_percentage_visible(cutting_plane_limits)
                if  percent_visible_x == 0 or percent_visible_y < 0.50: #or whatever threshold
                    continue
                
                object_location = list(map(float,labels[3:6]))
                # object_orientation = labels[6:9]
                object_scale = list(map(float,labels[9:12]))
                camera_location = list(map(float,labels[12:15]))
                camera_rotation = labels[15:18]
                camera_rotation = [float(i)*np.pi for i in camera_rotation]

                interShelfDistance = self.dimensions_map["Shelf"][1]
                
                
                

                if labels[0][0] == 'S':
                    object_type = "Shelf"
                    object_dimensions_old = self.dimensions_map["Shelf"]
                else:
                    object_type = "Box"
                    # object_dimensions = self.dimensions_map[labels[0].split(" stack ")[0]]
                    object_dimensions_old = self.dimensions_map[labels[0].split(" stack ")[0]]
                    print(labels[0])
                    stack_num = labels[0].split(" stack ")[1]
                    # print(object_dimensions_old)
                    # print(object_scale)
                    if stack_num not in stack_list:
                        stack_list[stack_num] = object_scale
                    else:
                        object_scale = stack_list[stack_num]
                
                object_dimensions = [100, 100, 100]
                object_dimensions[0] = object_dimensions_old[aa]*object_scale[ff]
                object_dimensions[1] = object_dimensions_old[bb]*object_scale[ee]
                object_dimensions[2] = object_dimensions_old[cc]*object_scale[dd]

                
                if labels[0] == "Shelf_0":
                    object_dimensions[1] *= (0.9895935 - 0.02)/ (0.9895935 + 1.35)
                    # print(object_dimensions)
                    
                shelf_number = int(labels[2].split('_')[-1])

                object_location = [float(i) for i in object_location]
                object_location[1] += object_dimensions[1]/2
                
                # print("Object Location : ",object_location)
                # print("Camera Location : ",camera_location)
                objectEgoCentricLocation = self.get_locations(object_location, 
                                                                    camera_location, camera_rotation)

                # print("objectEgoCentricLocation : ", objectEgoCentricLocation)                                                                    

                objectEgoCentricRotation_y = 0 #np.pi/2-float(object_orientation[2])

                object_scale = [float(i) for i in object_scale]
                camera_location = [float(i) for i in camera_location]
                
                print(object_dimensions)

                if(rack_in_focus == labels[1] or not Constants.RACK_IN_FOCUS):
                    curr_annotations[annotationID] = {
                        "object_name" : labels[0],
                        "object_type" : object_type,
                        "shelf_number" : shelf_number,
                        "object_location" : object_location,
                        "object_ego_location" : objectEgoCentricLocation,
                        "ego_rotation_y" : objectEgoCentricRotation_y,
                        "object_dimensions" : copy.deepcopy(object_dimensions),
                        "camera_location" : camera_location,
                        "camera_rotation" : camera_rotation,
                        "center" : [0,0],
                        "interShelfDistance" : interShelfDistance
                    }
                # print(curr_annotations[annotationID])
                # print()
                # if(shelf_number > self.max_shelf_number):
                    # self.max_shelf_number = shelf_number
                annotationID += 1
    
            shelfs_and_boxes = {}
            min_shelf_number, max_shelf_number = self.get_shelf_range(curr_annotations)
            for shelf_number in range(min_shelf_number, max_shelf_number+1):
                shelf_and_box_val = self.get_shelf_and_boxes(shelf_number, curr_annotations)
                if shelf_and_box_val[0] != None: # if the shelf is not visible then do not generate the box
                    shelfs_and_boxes[shelf_number] = shelf_and_box_val
            # pprint(shelfs_and_boxes)
            generateEgoCentricTopLayout.writeLayout(ID, dump_path, shelfs_and_boxes, min_shelf_number, max_shelf_number)
            generateEgoCentricFrontLayout.writeLayout(ID, dump_path, shelfs_and_boxes, min_shelf_number, max_shelf_number,
            aa, bb, cc, dd, ee, ff)
            # return

    def get_shelf_and_boxes(self, shelfNumber, curr_annotations):
        shelf = None
        boxes = []
        for annotation in curr_annotations.values():
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



    def get_shelf_range(self, curr_annotations):
        min_shelf = 99999999
        max_shelf = 0
        for annotation in curr_annotations.values():
            if(annotation["shelf_number"] < min_shelf):
                min_shelf = annotation["shelf_number"]
            if(annotation["shelf_number"] > max_shelf):
                max_shelf = annotation["shelf_number"]

        return [min_shelf, max_shelf]

if __name__ == "__main__":
    generatelayouts = GenerateLayouts()
    perm_listl = list(permutations(range(3)))
    # for aa, bb, cc in perm_listl:
    #     for dd, ee, ff in perm_listl:
    #         print(aa, bb, cc, dd, ee, ff)

    generatelayouts.read_annotations(
        filePathManager.anuragAnnotationsLabelsPath,
        filePathManager.anuragEgoCentricLayouts,
        0, 1, 2, 1, 2, 0
    )
        #     break
        # break
    print("Generated Layouts at : ",filePathManager.anuragEgoCentricLayouts)
