import cv2
import numpy as np

img = cv2.imread('resources/cards.png')
# pts1 is the points of the image we want to warp
pts1 = np.float32([[370, 352], [571, 340], [402, 657], [639, 639]])
# pts2 is the points of the image we want to warp to
# below is the width and height of a regular card
width, height = 250, 350
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

for i in range(len(pts1)):
    cv2.circle(img, (int(pts1[i][0]), int(pts1[i][1])), 5, (0, 255, 0), cv2.FILLED)
# This is the transformation matrix that will be used to warp the image
matrix = cv2.getPerspectiveTransform(pts1, pts2)
# This will warp the image to the new perspective
output = cv2.warpPerspective(img, matrix, (width, height))
cv2.imshow('Original Image', img)
cv2.imshow('Warped Image', output)
cv2.waitKey(0)
