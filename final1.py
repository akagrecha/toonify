import cv2
import numpy as np

img = cv2.imread('anmol.jpg')
imgCp = img.copy()

#applying median blur to image to reduce the number of edges detected
img = cv2.medianBlur(img, 5)

#edge detection and improvement
edges = cv2.Canny(img, 90, 250)
edges = cv2.bitwise_not(edges)
edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


#applying bilateral filter to reduce the details
imgCp = cv2.bilateralFilter(imgCp, 5, 150, 150)

#color quantization
Z = imgCp.reshape((-1,3))
Z = np.float32(Z)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8
ret,label,center=cv2.kmeans(Z,K, criteria,10,cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

#final image
result = cv2.bitwise_and(res2, edges)

cv2.namedWindow('result', cv2.WINDOW_NORMAL)
cv2.imshow('result', result)
cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()