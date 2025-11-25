import cv2
import numpy as np

"""
Tune thresholds – The 100, 100 in cv2.Canny are not magic; adjust until object outlines appear but noise disappears.

Kernel size – Odd numbers only (3, 5, 7…). Bigger kernels blur/dilate more; start small.

Order matters – Blur before Canny; dilate after Canny. Swapping often breaks results.

"""
# unint8 is an unsigned 8-bit integer, which means it can hold values from 0 to 255
kernel = np.ones((5, 5), np.uint8)  # kernel is a matrix of ones

path = 'resources/lena.png'
img = cv2.imread(path)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Applying Gaussian Blur kernel can be of any size, but it should be odd
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
imgCanny = cv2.Canny(imgBlur, 100, 100)
# we will itereate the kernel 1 time
# iteration it more times will increase the thickness of the edges
imgDilate = cv2.dilate(imgCanny, (7, 7), iterations=3)
imgErode = cv2.erode(imgDilate, kernel, iterations=2)

cv2.imshow('Original', img)
cv2.imshow('Gray', imgGray)
cv2.imshow('Blur', imgBlur)
cv2.imshow('Canny', imgCanny)
cv2.imshow('Dilate', imgDilate)
cv2.imshow('Erode', imgErode)
# use 0 for infinite wait
cv2.waitKey(0)
cv2.destroyAllWindows()
