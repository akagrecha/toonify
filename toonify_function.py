import cv2
import numpy as np


def toonify(image):

    """
    toonify converts an image into a cartoonized version of it. The image looks like it has been sketched.
    :param image:
    image is the input for the function
    :return:
    the function returns a tooned version of the image
    """

    if image.size is None :
        return "image does not exist"

    else:
        image_copy = image.copy()

        img = cv2.medianBlur(image, 5)

        # edge detection and improvement
        edges = cv2.Canny(img, 90, 250)
        edges = cv2.bitwise_not(edges)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # applying bilateral filter to reduce the details
        image_copy = cv2.bilateralFilter(image_copy, 5, 150, 150)

        # color quantization
        z = image_copy.reshape((-1, 3))
        z = np.float32(z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 8
        ret, label, center = cv2.kmeans(z, k, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape(img.shape)

        # final image
        result = cv2.bitwise_and(res2, edges)

        return result

img = cv2.imread('portrait2.jpg')

cv2.namedWindow('result', cv2.WINDOW_NORMAL)
cv2.imshow('result', toonify(img))
cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()

