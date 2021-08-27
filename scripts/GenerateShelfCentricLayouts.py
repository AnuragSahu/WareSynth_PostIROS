from ast import dump
from os.path import join
from glob import glob
import Constants
import numpy as np
import mathutils
import math
from FileNameManager import filePathManager
# from GenerateShelfCentricTopLayout import generateShelfCentricTopLayout
# from GenerateShelfCentricFrontLayout import generateShelfCentricFrontLayout
from GenerateFrontalLayout import generateFrontalLayout

from GenerateTopLayout import generateTopLayout
from GenerateFrontalLayout import generateFrontalLayout
import threading

class GenerateLayouts(object):
    def __init__(self):
        self.dimensions_map = {}
        self.count = 0
        with open(filePathManager.datasetDumpDirectory+"dimensions.txt") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                elem  = line.split(", ")
        
                self.dimensions_map[elem[0]] =  list(map(float, elem[1:]))
        self.num_threads = 10

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

    def get_percentage_visible(self, cutting_planes_visible):
        percents_x = []
        percents_y = []

        # bounds_vis[0] = side_tracker;
		
		# bounds_vis[1] = minX[1];
		# bounds_vis[2] = maxX[1];
		# bounds_vis[3] = minY[1];
		# bounds_vis[4] = maxY[1];
		
		# bounds_vis[5] = minX[0];
		# bounds_vis[6] = maxX[0];
		# bounds_vis[7] = minY[0];
		# bounds_vis[8] = maxY[0];
        # Debug.Log(go.name +" "+ z +" X PERCENTAGE IS "+ (maxX[1] - minX[1])/(maxX[0] - minX[0]) +"    Y PERCENTAGE IS"+ (maxY[1] - minY[1])/(maxY[0] - minY[0]));
		
        # do right and left things here
        for plane in cutting_planes_visible:
            bounds_vis = cutting_planes_visible[plane]

            if bounds_vis[0] == 0:
                percents_x.append(0)
                percents_y.append(0)
            else:
                percents_x.append( (bounds_vis[2] - bounds_vis[1])/(bounds_vis[6] - bounds_vis[5]) )
                percents_y.append( (bounds_vis[4] - bounds_vis[3])/(bounds_vis[8] - bounds_vis[7]) )

        return min(percents_x), min(percents_y)  

    def read_annotations(self, annotationsPath, dump_path):
        files_split = {}

        for i in range(self.num_threads):
            files_split[i] = []

        c = 0
        for file in glob(join(annotationsPath, '*.txt')):
            files_split[c%self.num_threads].append(file)
            c += 1

        threads = []

        for i in range(self.num_threads):
            t = threading.Thread(target=self.generate_layout_from_file, args=(files_split[i], dump_path))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

    def generate_layout_from_file(self, files, dump_path):
        for file in files:
            print("For file no. %d"%(self.count))
            self.count += 1
            ID = file.split("/")[-1]
            f = open(file, "r")
            annotationLines = f.readlines()
            rack_in_focus = annotationLines[0].strip('\n')
            annotationLines = annotationLines[1:]
            annotationID = 0
            self.max_shelf_number = 0
            curr_annotations = {}
            # box_count = 0
            for annotationLine in annotationLines:
                annotationLine = annotationLine.strip('\n')
                labels = annotationLine.split(", ")

                cutting_plane_limits = {}

                for i in range(18, 18+30, 10):
                    one_plane = labels[i:i+10]
                    # print(one_plane)
                    #can parse here
                    cutting_plane_limits[one_plane[0]] = list(map(float, one_plane[1:]))

                percent_visible_x, percent_visible_y = self.get_percentage_visible(cutting_plane_limits)
                if  percent_visible_x == 0 or percent_visible_y < 0.20: #or whatever threshold
                    continue
                
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

                interShelfDistance = self.dimensions_map["Shelf"][1]
                
                
                object_location = [float(i) for i in object_location]
                object_location[1] += object_dimensions[1]/2
                
               
                objectEgoCentricRotation_y = 0

                object_dimensions = [float(i) for i in object_dimensions]
                object_scale = [float(i) for i in object_scale]
                camera_location = [float(i) for i in camera_location]
                
                
                if(rack_in_focus == labels[1] or not Constants.RACK_IN_FOCUS):
                    curr_annotations[annotationID] = {
                        "object_type" : object_type,
                        "shelf_number" : shelf_number,
                        "object_location" : object_location,
                        "object_orientation" : object_orientation,
                        "rotation_y" : objectEgoCentricRotation_y,
                        "object_scale" : object_scale,
                        "object_dimensions" : object_dimensions,
                        "camera_location" : camera_location,
                        "camera_rotation" : camera_rotation,
                        "center" : [0,0],
                        "interShelfDistance" : interShelfDistance
                    }

                # if(shelf_number > self.max_shelf_number):
                #     self.max_shelf_number = shelf_number
                annotationID += 1

            shelfs_and_boxes = {}
            min_shelf_number, max_shelf_number = self.get_shelf_range(curr_annotations)
            for shelf_number in range(min_shelf_number, max_shelf_number+1):
                shelf_and_box_val = self.get_shelf_and_boxes(shelf_number, curr_annotations)
                if shelf_and_box_val[0] != None: # if the shelf is not visible then do not generate the box
                    shelfs_and_boxes[shelf_number] = shelf_and_box_val

            generateTopLayout.writeLayout(ID, dump_path, shelfs_and_boxes, min_shelf_number, max_shelf_number)
            generateFrontalLayout.writeLayout(ID, dump_path, shelfs_and_boxes, min_shelf_number, max_shelf_number)
            
    def get_shelf_and_boxes(self, shelfNumber, curr_annotations):
        shelf = None
        boxes = []
        for annotation in curr_annotations.values():
            if(annotation["shelf_number"] == shelfNumber):
                if(annotation["object_type"] == "Shelf"):
                    shelf = annotation
                elif(annotation["object_type"] == "Box"):
                    boxes.append(annotation)
        return (shelf,boxes)

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
        max_shelf = -1
        for annotation in curr_annotations.values():
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
