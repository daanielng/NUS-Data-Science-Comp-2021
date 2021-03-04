# -*- coding: utf-8 -*-
"""Counting Gold chips

Original file is located at
    https://colab.research.google.com/drive/1ZNEzuHzPNQxP-TtBJEE3-1eUrmGlhHHK
"""

import os
import re
import cv2
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt

# declare path for images
col_frames = os.listdir("/content/drive/MyDrive/NUS/Data Science Competition 2021/Working Set/train_images") #list containing frame names
col_frames.sort(key = lambda f: int(re.sub('\D','',f))) #sort ascending order of frames

# create empty list to store images
col_images = []

for frame in col_frames:
  #read the frames
  temp_img = cv2.imread("/content/drive/MyDrive/NUS/Data Science Competition 2021/Working Set/train_images/" + frame)
  #append the temp image into the list of images
  col_images.append(temp_img)

# take a look at first item in col_images
col_images[0] #output is a series of values, which is an image, stored in matrices

# look at the shape of the image VERY IMPT - must have same size & dimension
# note: no. of color channels: RGB:3, Grayscale:1
col_images[0].shape # (height, width, no. of channels) -it is a color image thus the image we have is a 3-channeled image

# plot the 1st and 2nd frames - just an example
num = 0

for frame in [num, num+1]:
  #imshow = image show, cvtColor = changes the color channel of image, cv2.COLOR_BGR2RGB = changing color from BGR to RGB
  plt.imshow(cv2.cvtColor(col_images[frame], cv2.COLOR_BGR2RGB)) #changing the color to a suitable color for matplotlib
  #naming the image
  plt.title("frame: " + str(frame))

  #show image
  plt.show()

# resize images
frame_resized = cv2.resize(col_images[0],(400,600)) #(width, height)
plt.imshow(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HLS_FULL))
plt.show()

def rescaleFrame(frame, scale=0.75):
  width = int(frame.shape[1]*scale)
  height = int(frame.shape[0]*scale)
  dimensions = (width,height)

  return cv.resize(frame,dimensions)

edited_frame = cv2.cvtColor(col_images[num], cv2.COLOR_BGR2HSV_FULL)
ret, thresh = cv2.threshold(edited_frame, 30, 255, cv2.THRESH_BINARY)

plt.imshow(thresh, cmap='gray')
plt.show()

kernel = np.ones((4,4), np.uint8)
dilated = cv2.dilate(thresh, kernel, iterations=1)

plt.imshow(dilated, cmap=None)
plt.show()

chip_edges = cv2.Canny(dilated,20,30)
plt.imshow(chip_edges, cmap='gray')

contours, hierarchy = cv2.findContours(chip_edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
temp_copy = col_images[num].copy()

cv2.drawContours(temp_copy, contours, -1, (127,200,0),2)
plt.imshow(temp_copy)
plt.show()

# declare list of valid contours
valid_cntrs = []

for i, cntr in enumerate(contours):
  x,y,width,height = cv2.boundingRect(cntr) #draw a rectangle based on contour (x-,y-coords)
  if (cv2.contourArea(cntr)>= 10): #checking for its position and area within certain size
    valid_cntrs.append(cntr)

# check length of list
len(valid_cntrs)

