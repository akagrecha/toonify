import cv2
import numpy as np

img = cv2.imread('flower.jpeg')
img = cv2.medianBlur(img, 5)

edges = cv2.Canny(img, 20, 200)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.namedWindow('edges', cv2.WINDOW_NORMAL)

cv2.imshow('image', img)
cv2.imshow('edges', edges)

cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()