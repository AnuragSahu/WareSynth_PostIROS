import cv2
import numpy as np
import Constants
from preProcessing.FillRackGaps import fillRackGaps

def showImage(img):
    cv2.imshow('ImageWindow', img)
    cv2.waitKey()
    return

img = cv2.imread("./readmeAsset/top000000_2.png")
opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

#img = cv2.UMat(img)
finalImage = fillRackGaps.process(opencvImage, Constants.GAP)
showImage(finalImage)