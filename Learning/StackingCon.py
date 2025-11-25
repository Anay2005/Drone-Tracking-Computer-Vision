import cv2
import numpy as np

img1 = cv2.imread("resources/lena.png")
img2 = cv2.imread("resources/land.jpg")

print(img1.shape)
print(img2.shape)

h,w = 300, 400

img1 = cv2.resize(img1, (h, w))
img2 = cv2.resize(img2, (h, w))

# For horizontal stacking, the images must have the same height
# For vertical stacking, the images must have the same width
hor = np.hstack((img1, img2))  # horizontal stacking
ver = np.vstack((img1, img2))  # vertical stacking

cv2.imshow("Horizontal Stacking", hor)
cv2.imshow("Vertical Stacking", ver)


cv2.waitKey(0)

