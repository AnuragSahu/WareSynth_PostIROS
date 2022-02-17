import cv2
from os import truncate
from os.path import join
from glob import glob
import Constants
import numpy as np
import mathutils
import math
import shutil
from FileNameManager import filePathManager

class GenerateKITTIAnnotations(object):
    def __init__(self):
        self.annotations = {}
        self.dimensions_map = {}
        self.transform_mat = np.array([[1, 0, 0],[0, -1, 0], [0, 0, 1]])

        with open(filePathManager.datasetDumpDirectory+"dimensions.txt") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                elem  = line.split(", ")
                if elem[0] == "Shelf_1":
                    elem[0] = "Shelf"
                self.dimensions_map[elem[0]] =  self.transform_point(list(map(float, elem[1:])), [0, 0, 0])
        # ####print(self.dimensions_map)

    def transform_point(self, point, camera_position):
        point[0] -= camera_position[0]
        point[1] -= camera_position[1]
        point[2] -= camera_position[2]
        # print("Input:", point)
        point = np.array(point).reshape((3,1))
        a = (self.transform_mat@point).T.tolist()[0]
        # print("Output:", a)
        return a

    def get_KRT(self, ID, camera_location):
        P = self.get_P(ID, camera_location)

        ##print("og proj",P)
    
        KR = P[:, :3]
        negKRinverse = -np.linalg.inv(KR)
        negKRC = P[:, 3]
        C = np.dot(-negKRinverse, negKRC).reshape(3,1)
        Rtranspose, KbarInverse = np.linalg.qr(negKRinverse)
        R = Rtranspose.T
        Kbar = np.linalg.inv(KbarInverse)
        K = Kbar / Kbar[2, 2]
        
        R_z_pi = np.array([[-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]])
        K = K@R_z_pi
        R = R_z_pi@R 
        RT = np.hstack((R,C))
                
        return P, K, RT

    def get_P(self, ID, camera_position):
        file = filePathManager.datasetDumpDirectory+"Correspondences/" + str(ID).zfill(6) + ".txt"
        
        f = open(file, "r")
        line = f.readline().strip("\n")
        
        worldCoords = []
        imageCoords = []
        while line:
            pts = [float(a) for a in line.split(", ")]
            line = f.readline().strip("\n")
            
            worldCoords.append(self.transform_point(pts[0:3], camera_position))
            imageCoords.append(pts[-2:])
        
        worldCoords = np.array(worldCoords)
        imageCoords = np.array(imageCoords)
        return self.DLT(worldCoords, imageCoords)

        
    def DLT(self, World, Image):
        M = np.zeros((2 * len(Image), 12))
        # Buiding M matrix
        for i in range(len(Image)):
            M[2 * i][11] = Image[i][0]
            M[2 * i + 1][11] = Image[i][1]
            M[2 * i][3] = M[2 * i + 1][7] = -1
            for j in range(4):
                M[2 * i + 1][j] = M[2 * i][4 + j] = 0
            for j in range(3):
                M[2 * i][j] = -World[i][j]
                M[2 * i + 1][4 + j] = -World[i][j]
                M[2 * i][8 + j] = Image[i][0] * World[i][j]
                M[2 * i + 1][8 + j] = Image[i][1] * World[i][j]

        # SVD to find U, S, V matrices
        U, S, V = np.linalg.svd(M)
        # Finding projection matrix
        P = V[11].reshape((3, 4))
        return P
        
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
        #####print(loc)
        #####print(rot)
        #rotation = mathutils.Euler((float(rot[0]), float(rot[1]), float(rot[2])))
        #R_world2bcam = rotation.to_matrix().transposed()
        #R_world2bcam = np.array(R_world2bcam)
        #####print("R World Matrix : ", np.array(R_world2bcam))
        # Convert camera location to translation vector used in coordinate changes

        #####print(R_world2bcam)
        #####print(location)
        T_world2bcam = -1*R_world2bcam @ location

        #####print("T_world2bcam : ",T_world2bcam)
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
        
    def get_bbs(self, cutting_planes_visible, object_location, object_dimensions, object_scale, camera_rotation, camera_location, P, K, RT, is_shelf):
        # # # #print("coord", self.world_to_image(object_location, P))
        max_x = object_location[0] + object_dimensions[0]*object_scale[0]/2
        min_x = object_location[0] - object_dimensions[0]*object_scale[0]/2


        max_y = object_location[1] 
        min_y = object_location[1] - object_dimensions[1]*object_scale[1]
        
        
        max_z = object_location[2] + object_dimensions[2]*object_scale[2]/2
        min_z = object_location[2] - object_dimensions[2]*object_scale[2]/2
        
        if is_shelf:
            min_y = object_location[1]
            max_y = min_y + 0.01
            return self.max_min_to_kitti(max_x, max_y, max_z, min_x, min_y, min_z, P)
        else:
            return {
                "max_x": max_x,
                "min_x": min_x,
                "max_y": max_y,
                "min_y": min_y,
                "max_z": max_z,
                "min_z": min_z
            }

    def max_min_to_kitti(self, max_x, max_y, max_z, min_x, min_y, min_z, P):      
        # IN THE WORLD FRAME
        bbox_corners = [
            (max_x, max_y, max_z),
            (min_x, max_y, max_z),
            (max_x, min_y, max_z),
            (min_x, min_y, max_z),
            (max_x, max_y, min_z),
            (min_x, max_y, min_z),
            (max_x, min_y, min_z),
            (min_x, min_y, min_z),
        ]

        # #print(bbox_corners)

        bbox_xmin = 100000
        bbox_ymin = 100000
        bbox_xmax = -1
        bbox_ymax = -1


        three_d_bbox_center = [float(max_x+min_x)/2, float(max_y+min_y)/2, float(max_z+min_z)/2]

        for point in bbox_corners:
            x, y = self.world_to_image(point, P)
            bbox_xmin = min(x, bbox_xmin)            
            bbox_ymin = min(y, bbox_ymin)            
            bbox_xmax = max(x, bbox_xmax)            
            bbox_ymax = max(y, bbox_ymax)            
        
        loc_x, loc_y, loc_z = three_d_bbox_center
        
        kitti_stuff = {
            "dim_height" : float(max_y-min_y),
            "dim_width" : float(max_z-min_z),
            "dim_length" : float(max_x - min_x),
            "loc_x" : loc_x,
            "loc_y" : loc_y,
            "loc_z" : loc_z,
            "bbox_xmin" : bbox_xmin,
            "bbox_ymin" : bbox_ymin,
            "bbox_xmax" : bbox_xmax,
            "bbox_ymax" : bbox_ymax,
        }

        return kitti_stuff

    def max_min_to_kitti_box(self, max_x, max_y, max_z, min_x, min_y, min_z, P):      
        # IN THE WORLD FRAME
        bbox_corners = [
            (max_x, max_y, max_z),
            (min_x, max_y, max_z),
            (max_x, min_y, max_z),
            (min_x, min_y, max_z),
            (max_x, max_y, min_z),
            (min_x, max_y, min_z),
            (max_x, min_y, min_z),
            (min_x, min_y, min_z),
        ]

        # #print(bbox_corners)

        bbox_xmin = 100000
        bbox_ymin = 100000
        bbox_xmax = -1
        bbox_ymax = -1


        three_d_bbox_center = [float(max_x+min_x)/2, float(max_y+min_y)/2, float(max_z+min_z)/2]

        for point in bbox_corners:
            x, y = self.world_to_image(point, P)
            bbox_xmin = min(x, bbox_xmin)            
            bbox_ymin = min(y, bbox_ymin)            
            bbox_xmax = max(x, bbox_xmax)            
            bbox_ymax = max(y, bbox_ymax)            
        
        loc_x, loc_y, loc_z = three_d_bbox_center
        
        kitti_stuff = {
            "dim_height" : float(max_y-min_y),
            "dim_width" : float(max_z-min_z),
            "dim_length" : float(max_x - min_x),
            "loc_x" : loc_x,
            "loc_y" : loc_y + float(max_y-min_y)/2,
            "loc_z" : loc_z,
            "bbox_xmin" : bbox_xmin,
            "bbox_ymin" : bbox_ymin,
            "bbox_xmax" : bbox_xmax,
            "bbox_ymax" : bbox_ymax,
        }

        return kitti_stuff


    #TODO: use this function to convert a point [x, y, z] to image coordinates [x', y']
    def world_to_image(self, point, P):
        point = [float(i) for i in point]
        point.append(1)
        res = P@point
        res[0] = res[0]/res[2]
        res[1] = res[1]/res[2]
        return res[0], res[1]

    def world_to_camera_frame(self, obj_loc, RT):

        locations = np.array([
            float(obj_loc[0]),
            float(obj_loc[1]),
            float(obj_loc[2]),
            1
        ])
        
        locations = RT @ locations
        # ####print(locations)
        return locations

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


    def build_string(self, types, truncated, occulded, alpha, bbox, dimensions, locations, rotation_y):
        empty_space = " "
        bbox = [str(i) for i in bbox]
        locations = [str(i) for i in locations]
        dimensions = [str(i) for i in dimensions]
        bbox_str = empty_space.join(bbox)
        loca_str = empty_space.join(locations)
        dim_str = empty_space.join(dimensions)
        lst = [str(types), str(truncated), str(occulded), str(alpha), bbox_str, dim_str, loca_str, str(rotation_y)]
        return empty_space.join(lst)

    def merge_boxes_and_to_file(self, boxes_kittis, f_label, P):
        max_x, max_y, max_z = -10000000, -1000000, -10000000
        min_x, min_y, min_z = 10000000, 1000000, 10000000
        for box_kitti in boxes_kittis:
            # print(box_kitti)
            max_x = max(max_x, box_kitti["max_x"])    
            max_y = max(max_y, box_kitti["max_y"])    
            max_z = max(max_z, box_kitti["max_z"])    
            
            min_x = min(min_x, box_kitti["min_x"])    
            min_y = min(min_y, box_kitti["min_y"])    
            min_z = min(min_z, box_kitti["min_z"])    

        final_kitti = self.max_min_to_kitti_box(max_x, max_y, max_z, min_x, min_y, min_z, P)
        self.kitti_obj_to_file(final_kitti, "Box", f_label)

    def kitti_obj_to_file(self, kitti_stuff, object_type, f_label):
        kitti_stuff["label"] = object_type
        kitti_stuff["truncated"] = 0
        kitti_stuff["occulded"] = 0
        kitti_stuff["alpha"] = 0
        kitti_stuff["rotation_y"] = 0

        name = object_type
        name += " " + str(kitti_stuff["truncated"])
        name += " " + str(kitti_stuff["occulded"])
        name += " " + str(kitti_stuff["alpha"])
        name += " " + str(kitti_stuff["bbox_xmin"])
        name += " " + str(kitti_stuff["bbox_ymin"])
        name += " " + str(kitti_stuff["bbox_xmax"])
        name += " " + str(kitti_stuff["bbox_ymax"])
        name += " " + str(kitti_stuff["dim_height"])
        name += " " + str(kitti_stuff["dim_width"])
        name += " " + str(kitti_stuff["dim_length"])
        name += " " + str(kitti_stuff["loc_x"])
        name += " " + str(kitti_stuff["loc_y"])
        name += " " + str(kitti_stuff["loc_z"])
        name += " " + str(kitti_stuff["rotation_y"])

        f_label.write(name+"\n") 


    def convert_to_KITTI(self, annotationsPath, dump_path):
        # my_stack = 0
        for file in glob(join(annotationsPath, '*.txt')):
            print("For File : ", file)

            ID = file.split("/")[-1]
            ID = int(ID.split(".")[0])

            # if ID != 0:
            #     continue
            # copy RGB image
            # shutil.copyfile(filePathManager.anuragRGBImagesPath+str(ID).zfill(6)+".png", filePathManager.kittiImagePath+str(ID).zfill(6)+".png")

            # copy depth image 

            f = open(file, "r")
            annotationLines = f.readlines()
            rack_in_focus = annotationLines[0].strip('\n')
            annotationLines = annotationLines[1:]
            annotationID = 0
            self.max_shelf_number = 0
            curr_annotations = {}
            P, K, RT = None, None, None

            boxes_kittis = {}
            shelfs_kittis = []
            shelfs_to_include = []

            f_label = open(filePathManager.datasetDumpDirectory+"label/" + str(ID).zfill(6) + ".txt",'w')
            for annotationLine in annotationLines:
                annotationLine = annotationLine.strip('\n')
                labels = annotationLine.split(", ")
                object_type = labels[0]

                cutting_plane_limits = {}
                # cplns = []    
                for i in range(18, 18+3*14, 14):
                    one_plane = labels[i:i+14]
                    #print(one_plane)
                    #can parse here
                    cutting_plane_limits[one_plane[0]] = list(map(float, one_plane[1:]))

                #print("\n", labels[0])

                percent_visible_x, percent_visible_y, _ = self.get_percentage_visible(cutting_plane_limits)
                #print(percent_visible_x, percent_visible_y)
                if percent_visible_x == 0 or percent_visible_y < 0.50: #or whatever threshold
                    continue
                #print(labels[0], "included")
                    

                shelf_number = int(labels[2].split('_')[-1])

                camera_location = list(map(float,labels[12:15]))
                camera_rotation = labels[15:18]
                camera_rotation = [float(i)*np.pi for i in camera_rotation]

                object_location = self.transform_point(list(map(float,labels[3:6])), camera_location)
                # object_orientation = labels[6:9]
                object_scale = self.transform_point(list(map(float,labels[9:12])), [0, 0, 0])
                
                
                try:
                    if P == None:
                        P, K, RT = self.get_KRT(ID, camera_location)
                except:
                    pass

                if labels[0][0] == 'S':
                    object_dimensions = self.dimensions_map["Shelf"]
                    kitti_stuff = self.get_bbs(cutting_plane_limits, object_location, object_dimensions, object_scale,
                                                camera_rotation, camera_location, P, K, RT, is_shelf=True)
                    shelfs_kittis.append(kitti_stuff)
                    shelfs_to_include.append(shelf_number)
                else:
                    box_name, stack_group = labels[0].split(" stack ") 
                    # stack_group = my_stack
                    # my_stack += 1
                    object_dimensions = self.dimensions_map[box_name]
                    kitti_stuff = self.get_bbs(cutting_plane_limits, object_location, object_dimensions, object_scale, 
                                                camera_rotation, camera_location, P, K, RT, is_shelf=False)
                    kitti_stuff["shelf_number"] = shelf_number

                    if stack_group not in boxes_kittis:
                       boxes_kittis[stack_group] = []
              
                    boxes_kittis[stack_group].append(kitti_stuff)
              
            #LABEL FILE
            for kitti_stuff in shelfs_kittis:
                self.kitti_obj_to_file(kitti_stuff, "Shelf", f_label)

            for key in boxes_kittis.keys():
                if boxes_kittis[key][0]["shelf_number"] in shelfs_to_include:
                    self.merge_boxes_and_to_file(boxes_kittis[key], f_label, P)
                           
            #CALIB FILE 
            str_2 = "P2: "
            # K@self.get_3x4_RT(camera_location,camera_rotation)
            for i in P:
                for j in i:
                    str_2 = str_2 + str(j) + " "

            str_0 =  "P0: 0 0 0 0 0 0 0 0 0 0 0 0"
            str_1 =  "P1: 0 0 0 0 0 0 0 0 0 0 0 0"
            # str_2 =  "P2: 0 0 0 0 0 0 0 0 0 0 0 0"
            str_3 =  "P3: 0 0 0 0 0 0 0 0 0 0 0 0"
            str_R =  "R0_rect: 1 0 0 0 1 0 0 0 1"
            str_T = "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0"
            str_I = "Tr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0"

            with open(filePathManager.datasetDumpDirectory+"calib/" +str(ID).zfill(6)+".txt", 'w') as f:
                f.write("%s\n" % str_0)
                f.write("%s\n" % str_1)
                f.write("%s\n" % str_2)
                f.write("%s\n" % str_3)
                f.write("%s\n" % str_R)
                f.write("%s\n" % str_T) 
                f.write("%s\n" % str_I)


            #VELODYNE POINTS
            depth = cv2.imread(filePathManager.datasetDumpDirectory+"depth/" +str(ID).zfill(6)+".png", 0)
            depth = np.flipud(depth)
            # depth = np.load("/scratch/warehouse/training_5000/depth/"+str(ID).zfill(6)+".npy")
            
            R = self.eul2rot(camera_rotation)
            T = np.array(camera_location).reshape((3,1))
            R_transpose = R.T
            neg_trans = -R_transpose@T
            pos_trans = R@T
            # ##print(R_transpose.shape, neg_trans.shape)
            # trans_mat = np.hstack((R_transpose, neg_trans))
            trans_mat = np.hstack((R, pos_trans))
            # trans_mat = np.linalg.inv(trans_mat)
            # ##print(R, T, trans_mat)

            h,w = depth.shape
            cam_points = np.zeros((h * w, 4))
            min_val = np.min(depth)
            max_val = np.max(depth)

            i = 0
            for v in range(h):
                for u in range(w):

                    # z = ((depth[v, u] - min_val)/(max_val - min_val))*4.99/2 + 0.01
                    z = 19.99 * (depth[v, u]/255) + 0.01
                    x = 1*(u - K[0, 2]) * z / K[0, 0]
                    y = 1*(v - K[1, 2]) * z / K[1, 1]

                    tmp = list(trans_mat@np.array([x, y, z, 1]))
                    tmp = self.transform_point(tmp, camera_location)
                    # tmp = [x, y, z]
                    tmp.append(1)
                    cam_points[i] = tmp 
                    i += 1

                    # ##print(u,v)
                    # l = K@np.array([x, y, z])
                    # ##print(l[0]/l[2], l[1]/l[2])
                    # ##print()


            # h,w = depth.shape
            # ##print(w, K[0, 2], K[0, 0])
            # ##print(h, K[1, 2], K[1, 1])
            # cam_points = np.zeros((h * w, 4))

            # i = 0
            # y_cnts = {}
            # for v in range(h):
            #     for u in range(w):

            #         # z = 4.99 * (depth[v, u]/255) + 0.01
            #         # x = -1*(u - K[1, 2]) * z / K[1, 1]
            #         # y = -1*(v - K[0, 2]) * z / K[0, 0]
            #         # # y = v*z/K[1, 1]
            #         # cam_points[i] =[x,y,z,1]


                    # z = 4.99 * (depth[v, u]/255) + 0.01
                    # x = (u - K[0, 2]) * z / K[0, 0]
                    # y = (v - K[1, 2]) * z / K[1, 1]
                    # cam_points[i] =[x,y,z,1]

            #         # ##print()
                    # x = (u - K[0, 2]) * depth[v, u] / K[0, 0]
                    # y = (v - K[1, 2]) * depth[v, u] / K[1, 1]
                    # z = depth[v, u]
                    # cam_points[i] =[x,y,z,1]
            #         k = z
            #         if k in y_cnts:
            #             y_cnts[k] += 1
            #         else:
            #             y_cnts[k] = 1

            #         i += 1
            # # ##print(y_cnts)
            cam_points.astype('float32').tofile(filePathManager.datasetDumpDirectory+"velodyne/"+str(ID).zfill(6)+".bin")


if __name__ == "__main__":
    generateKITTIAnnotations = GenerateKITTIAnnotations()

    generateKITTIAnnotations.convert_to_KITTI(
        filePathManager.anuragAnnotationsLabelsPath,
        filePathManager.anuragEgoCentricLayouts
    )
    ####print("Generated Layouts at : ",filePathManager.anuragEgoCentricLayouts)