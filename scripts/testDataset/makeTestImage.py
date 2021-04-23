import cv2
import os
from PIL import Image
from matplotlib import pyplot as plt

class MakeTestImage(object):
    def __init__(self):
        self.testImagePath = "../test_images/"
        self.topViewpath = "../test_images/topview/"

    def getOutputImages(self):
        # get all the images in folder
        filelist = [ file for file in os.listdir(self.topViewpath) if file.endswith(".png")]
        return filelist

    def getRGBImages(self):
        filelist = [ file for file in os.listdir(self.testImagePath) if file.endswith(".png")]
        return filelist
    
    def getMappedImages(self):
        topViewImages = self.getOutputImages()
        RGB_to_OUTPUT = {}
        for RGB_ID in self.getRGBImages():
            outputs = []
            for topViewImage in topViewImages:
                if(RGB_ID[:6] == topViewImage[:6]):
                    outputs.append(topViewImage)
            outputs.sort()
            RGB_to_OUTPUT[RGB_ID] = outputs

        return RGB_to_OUTPUT

    def makeImage(self):
        mappedImagesOutput = self.getMappedImages()
        for RGB_image_path in mappedImagesOutput:
            RGB_image = cv2.imread(self.testImagePath+RGB_image_path)
            topViewImagePaths = mappedImagesOutput[RGB_image_path]
            topViewImagePaths = [self.topViewpath+i for i in topViewImagePaths]
            self.create_collage(800, 1435, topViewImagePaths, RGB_image_path[:6])
            
            
    def create_collage(self, width, height, listofimages, ID):
        cols = 1
        rows = 4
        thumbnail_width = width//cols
        thumbnail_height = height//rows - 4
        size = thumbnail_width, thumbnail_height
        new_im = Image.new('RGB', (width, height), color = (255, 255, 255))
        ims = []
        for p in listofimages:
            im = Image.open(p)
            im.thumbnail(size)
            ims.append(im)
        i = 0
        x = 0
        y = 0
        for col in range(cols):
            for row in range(rows):
                #print(i, x, y)
                new_im.paste(ims[i], (x, y))
                i += 1
                y += thumbnail_height +1
            x += thumbnail_width
            y = 0

        new_im.save("Collage"+str(ID)+".jpg")

makeTestImage = MakeTestImage()
if __name__ == "__main__":
    print(makeTestImage.makeImage())