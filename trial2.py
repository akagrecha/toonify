#proper limits for canny detection and small kernel for dilation provide thick edges
#thick edges not good, reverted to edges by 
#limits changed to make it suitable to cartoonify portraits
#tried adding bilateral filter but color quantization gives great results
import cv2
import numpy as np

img = cv2.imread('messi2.jpg')
imgCp = img.copy()
img = cv2.medianBlur(img, 5)

#bilateral filter
#filtered = cv2.bilateralFilter(imgCp, 5, 150, 150)

#color quantization(simply copied)
Z = imgCp.reshape((-1,3))
Z = np.float32(Z)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8
ret,label,center=cv2.kmeans(Z,K, criteria,10,cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))


#edge detection
edges = cv2.Canny(img, 90, 250)

#opening to remove noise
kernel = np.ones((1,1), np.uint8)
#opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel) did not work
dilate = cv2.dilate(edges, kernel, iterations = 3)

#negative of dilate
dNeg = cv2.bitwise_not(dilate)
# final edges
eFinal = cv2.cvtColor(dNeg, cv2.COLOR_GRAY2BGR)

#final image
result = cv2.bitwise_and(res2, eFinal)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
#cv2.namedWindow('opening', cv2.WINDOW_NORMAL)
cv2.namedWindow('dilate', cv2.WINDOW_NORMAL)
#cv2.namedWindow('filtered', cv2.WINDOW_NORMAL)
cv2.namedWindow('eFinal', cv2.WINDOW_NORMAL)
cv2.namedWindow('result', cv2.WINDOW_NORMAL)

cv2.imshow('image', img)
cv2.imshow('edges', edges)
#cv2.imshow('opening', opening)
cv2.imshow('dilate', dilate)
#cv2.imshow('filtered', filtered)
cv2.imshow('final edges', eFinal)
cv2.imshow('result', result)

cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()