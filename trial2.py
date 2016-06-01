#proper limits for canny detection and small kernel for dilation provide thick edges

import cv2
import numpy as np

img = cv2.imread('flower.jpeg')
img = cv2.medianBlur(img, 5)

#edge detection
edges = cv2.Canny(img, 20, 200)

#opening to remove noise
kernel = np.ones((2,2), np.uint8)
#opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel) did not work
dilate = cv2.dilate(edges, kernel, iterations = 3)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
#cv2.namedWindow('opening', cv2.WINDOW_NORMAL)
cv2.namedWindow('dilate', cv2.WINDOW_NORMAL)

cv2.imshow('image', img)
cv2.imshow('edges', edges)
#cv2.imshow('opening', opening)
cv2.imshow('dilate', dilate)

cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()