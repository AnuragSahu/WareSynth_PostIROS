from os import listdir, makedirs
from os.path import isfile, join, isdir
from pathlib import Path
from shutil import copyfile, move
import random

class MergeDatasets(object):
    def __init__(self):
        self.images = "anuragAnnotations/images/"
        self.labels = "anuragAnnotations/labels/"
        self.topLayouts = "anuragAnnotations/topLayouts/"
        self.depth = "anuragAnnotations/depth/"
        self.IDNumber = 0
        self.datasetNumber = 1

    def getAllTheIDs(self, imageIDs):
        path = imageIDs
        allImagesID = [f for f in listdir(path) if isfile(join(path, f))]
        allImagesID = [int(f[:-4]) for f in allImagesID]
        allImagesID.sort()
        return allImagesID

    def getAllDatasets(self):
        dataDirs = [name for name in listdir(".") if isdir(name) and "datasets_" in name]
        dataDirs.sort()
        dataDirsID = [int(f[9:]) for f in dataDirs]
        return dataDirsID

    def formatFileName(self, ID, extension):
        fileName = str(ID).zfill(6) + "." + extension
        return fileName

    def formatDatasetName(self, ID):
        datasetName = "datasets_" + str(ID)
        return datasetName

    def makeDirs(self):
        makedirs("./datasets/" + self.images, exist_ok = True)
        makedirs("./datasets/" + self.labels, exist_ok = True)
        makedirs("./datasets/" + self.topLayouts, exist_ok = True)
        makedirs("./datasets/" + self.depth, exist_ok = True)
        return 

    def MergeDatasets(self):
        # make Dataset folder
        self.makeDirs()
        datasetIDs = self.getAllDatasets()
        for DatasetID in datasetIDs:
            imageIDs = "./" + self.formatDatasetName(DatasetID) + "/" + self.images
            labelsIDs = "./" + self.formatDatasetName(DatasetID) + "/" + self.labels
            depthIDs = "./" + self.formatDatasetName(DatasetID) + "/" + self.depth
            topLayoutsIDs = "./" + self.formatDatasetName(DatasetID) + "/" + self.topLayouts

            imageIDs_next = "./datasets/" + self.images
            labelsIDs_next = "./datasets/" + self.labels
            depthIDs_next = "./datasets/" + self.depth
            topLayoutsIDs_next = "./datasets/" + self.topLayouts

            IDs = self.getAllTheIDs(imageIDs)
            for ID in IDs:
                imagePath_current = imageIDs + self.formatFileName(ID, "png")
                depthPath_current = depthIDs + self.formatFileName(ID, "png")
                topLayoutsPath_current = topLayoutsIDs + "top" + self.formatFileName(ID, "npy")
                labelsPath_current = labelsIDs + self.formatFileName(ID, "txt")

                imagePath_next = imageIDs_next + self.formatFileName(self.IDNumber, "png")
                depthPath_next = depthIDs_next + self.formatFileName(self.IDNumber, "png")
                topLayoutsPath_next = topLayoutsIDs_next +"top"+ self.formatFileName(self.IDNumber, "npy")
                labelsPath_next = labelsIDs_next + self.formatFileName(self.IDNumber, "txt")
                
                move(imagePath_current, imagePath_next)
                #move(depthPath_current, depthPath_next)
                move(topLayoutsPath_current, topLayoutsPath_next)
                move(labelsPath_current, labelsPath_next)

                self.IDNumber += 1            

if __name__ == "__main__":
    mergeDatasets = MergeDatasets()
    mergeDatasets.MergeDatasets()
