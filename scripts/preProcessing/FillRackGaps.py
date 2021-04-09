import cv2
from PIL import Image
import numpy as np

class FillRackGaps(object):
    def PIL_to_cv2(self, pilImage):
        pilImage = pilImage.convert('RGB')
        cv2Image = cv2.cvtColor(np.array(pilImage), cv2.COLOR_RGB2BGR)
        return cv2Image

    def CV2_to_PIL(self, cv2Image):
        cv2_im = cv2.cvtColor(cv2Image,cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        return pil_im

    def fillRackGap(self, img, boundingBoxes, gapFill):
        for boundingBox in boundingBoxes:
            WIDTH_PAD = 3
            x1,y1,l,w = boundingBox
            x1 = x1 - WIDTH_PAD
            l = l + 2*WIDTH_PAD
            for j in range(max(x1,0), min(x1+l,img.shape[0])):
                for i in range(max(y1,0), min(y1+w, img.shape[1])):
                    if(img[i][j][0] == 0 and img[i][j][1] == 0 and img[i][j][2] == 0):
                        img[i][j][:] = gapFill
        return img

    def getInterRackSpaces(self, rackBB):
        if(len(rackBB) < 2):
            return [False,"Only one or no rack"]
        else:
            rackBB.sort()
            spaces = []
            for i in range(len(rackBB)-1):
                BB = rackBB[i]
                nextBoundingBox = rackBB[i+1]
                rackGap = [BB[0]+BB[2],
                            BB[1],
                            nextBoundingBox[0]-(BB[0]+BB[2]),
                            BB[3]]

                spaces.append(rackGap)
            return spaces


    def getBoundingBoxes(self, img):
        boundingBoxes = []
        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        for cntr in contours:
            x,y,w,h = cv2.boundingRect(cntr)
            boundingBoxes.append([x,y,w,h])
        return boundingBoxes

    def rackBBs(self, img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        shelfThresh = cv2.threshold(gray,10,155,cv2.THRESH_BINARY)[1]
        rackBB = self.getBoundingBoxes(shelfThresh)
        return rackBB

    def process(self, img, gapFill):
        img = self.PIL_to_cv2(img)
        rackBB = self.rackBBs(img)
        rackGap = self.getInterRackSpaces(rackBB)
        preProcessedImage = self.fillRackGap(img, rackGap, gapFill)
        preProcessedImage = self.CV2_to_PIL(preProcessedImage)
        return preProcessedImage

fillRackGaps = FillRackGaps()