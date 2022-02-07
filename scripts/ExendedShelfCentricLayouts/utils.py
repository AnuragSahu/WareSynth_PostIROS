import cv2
import scipy.misc
import numpy as np
# import matplotlib.pyplot as plt
def getZommedLayout(layout):
    layout = cv2.resize(layout, None, fx = 10/8, fy = 10/8, interpolation = cv2.INTER_NEAREST)
    l,h = layout.shape[0], layout.shape[1]
    center_x, center_y = int(l/2), int(h/2)
    layout = layout[center_x - int(512/2):center_x + int(512/2), center_y - int(512/2):center_y + int(512/2) ]
    # print(layout.shape)
    return layout

def centerAlignImage(img):
    img.astype('int8')
    original_layout_width, original_layout_height = img.shape
    center_x = int(original_layout_width/2)
    center_y = int(original_layout_height/2)
    # get all the extreme points in layout
    ret, img1 = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY_INV)
    min_x = 9999
    min_y = 9999
    max_x = 0
    max_y = 0
    # cv2.imwrite("here.png", img)
    contours,_ = cv2.findContours(img.copy(), 1, 1) # not copying here will throw an error
    
    # no layout is present in the image
    if(len(contours) == 0):
        return img
    for i in range(len(contours)):
        rect = cv2.minAreaRect(contours[i]) # basically you can feed this rect into your classifier
        (x,y),(w,h), a = rect # a - angle

        box = cv2.boxPoints(rect)
        box = np.int0(box) #turn into ints
        min_x = max(min(min(box.T[0]), min_x), 0)
        min_y = max(min(min(box.T[1]), min_y), 0)
        max_x = max(max(box.T[0]) + 2, max_x)
        max_y = max(max(box.T[1]) + 2, max_y)
    
    # print(min_x, min_y, max_x, max_y)
    
    # copy this part of layout 
    layout = img[min_y : max_y, min_x : max_x].copy()
    
    black_pixels = np.where((layout[:, :] == 0))

    # set those pixels to white
    layout[black_pixels] = 115
    # cv2.imwrite("Here.png", layout)
    # print(min_x, max_x, min_y, max_y)
    
    # make the layout as black
    img[min_y : max_y, min_x : max_x] = np.zeros((max_y - min_y, max_x - min_x))
    
    # paste the layout at center
    layout_width, layout_height = max_y-min_y, max_x-min_x
    img[center_y - int(layout_width/2) : center_y - int(layout_width/2) + (max_y-min_y),
        center_x - int(layout_height/2) : center_x - int(layout_height/2) + (max_x-min_x)] = layout
    
    # return the new centered layouts
    return getZommedLayout(img)

def chop_corners(img_OG):
    # create a copy
    height, width = img_OG.shape
    img = np.zeros((height,width,3), np.uint8)
    img[:, :, 0] = img_OG.copy()
    img[:, :, 1] = img_OG.copy()
    img[:, :, 2] = img_OG.copy()
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img[img==255] = 0
    # print(img_OG.shape)
    # get the bbox for shelf
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    if(len(contours) == 0):
        return img_OG
        
    result = img.copy()
    bbox_x = []
    bbox_y = []
    bbox_h = []
    bbox_w = []
    
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        # print(x,y,w,h)
        bbox_x.append(x)
        bbox_y.append(y)
        bbox_w.append(w)
        bbox_h.append(h)

    x = y = w = h = 0
    # find the lowest x
    x = min(bbox_x)
    
    # find the lowest y
    y = min(bbox_y)
    
    # find the width
    # get the right most x
    for a in range(len(bbox_x)):
        bbox_x[a] += bbox_w[a]
        
    # get the maximum x
    x_max = max(bbox_x)
    
    # get the final width
    w = x_max - x
    
    # similarly get y
    for a in range(len(bbox_y)):
        bbox_y[a] += bbox_h[a]
        
    # get the maximum x
    y_max = max(bbox_y)
    
    # get the final width
    h = y_max - y
    
    rack_bbox = [x, y, w, h]
    
    # now make the region outside this region as black
    for row in range(len(img)):
        for col in range(len(img[row])):
            if(row < rack_bbox[0]+rack_bbox[2] and row > rack_bbox[0] and
               col < rack_bbox[1]+rack_bbox[3] and col > rack_bbox[1]):
                # do nothing
                pass
            else:
                # make it black
                img_OG[col, row] = 0
    
    return img_OG