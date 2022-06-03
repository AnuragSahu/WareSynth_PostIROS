from FileNameManager import filePathManager
import os
import numpy as np
import math


def findVisibleRackCenters():
    annotations_dir = filePathManager.datasetDumpDirectory + "Annotations/"

    out = []

    # loop over all annotation files
    for filename in os.listdir(annotations_dir):
        print(filename)

        minX = math.inf
        maxX = -math.inf
        minY = math.inf
        maxY = -math.inf
        z = None

        with open(annotations_dir + filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]

        for line in lines:
            line = line.split(", ")
            if len(line) == 1 or line[2][:5] != "Shelf":
                continue

            # shelf not visible
            idx = 19
            if int(line[idx]) == 0:
                continue

            # get dimensions of shelf
            minX = min(minX, float(line[idx+1]))
            maxX = max(maxX, float(line[idx+2]))
            minY = min(minY, float(line[idx+3]))
            maxY = max(maxY, float(line[idx+4]))

            # get z of shelf/rack (same for all)
            if z is None:
                z = float(line[5])

        x = minX + (maxX - minX) / 2
        y = minY + (maxY - minY) / 2

        out.append([x, y, z])

    return out


findVisibleRackCenters()
