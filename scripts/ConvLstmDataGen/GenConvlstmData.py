import os
import numpy as np

from GenKeyPointLayouts import make_layout
from FileNameManager import filePathManager
from FindVisibleRackCenters import findVisibleRackCenterByPath

def TransformToCenter(center, world_points):
    transformedPoints = []
    for hull in world_points:
        hull_temp = []
        for point in hull:
            new_x = point[0] - center[0]
            new_y = point[1] - center[1]
            new_z = point[2] - center[2]
            hull_temp.append([new_x, new_y, new_z])
        transformedPoints.append(hull_temp)
    return transformedPoints

# Read all the annottation files
ObjectAnnotation_dir = filePathManager.datasetDumpDirectory + "Annotations/"
KeyPointAnnotation_dir = filePathManager.datasetDumpDirectory + "Keypoints/"



# loop over all annotation files
lists = os.listdir(ObjectAnnotation_dir)
lists.sort()
print(lists)
banned = ["000006.txt", "000007.txt", "000027.txt", "000028.txt", "000029.txt"]
bn = []
for filename in lists:
    if(filename not in banned): # TO FIX
        ObjectAnnotationFilePath = ObjectAnnotation_dir + filename
        pointAnnotationFilePath = KeyPointAnnotation_dir + filename
        center_3D_point = findVisibleRackCenterByPath(ObjectAnnotationFilePath)
        layout, world_points = make_layout(pointAnnotationFilePath)
        transformedCoordinates = TransformToCenter(center_3D_point, world_points)
        
        # save the file
        fName = filePathManager.anurag3DKeyPointsPath + filename.split(".")[0] + ".npy"
        np.save(fName, transformedCoordinates)

        # print(filename, transformedCoordinates)