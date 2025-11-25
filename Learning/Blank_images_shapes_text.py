import cv2
import numpy as np

# create a blank image
img = np.zeros((512, 512, 3), np.uint8)

print(img)

# This will create a blue image(BGR) with B=255, G=0, R=0, always remember that image is a matrix
# img[:] = 255,0,0

# First arguement is the image we want to draw on
# second argument is the starting coordinate of the line, third argument is the ending coordinate of the line
# fourth argument is the color of the line in BGR format, fifth argument is the thickness of the line
# end result is the diagonal line from the top left corner to the bottom right corner
cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 0), 5)  # BGR
# Argument meaning is similar to the line function 
cv2.rectangle(img, (0, 0), (25, 35), (0, 0, 255), 2)  # BGR
# 2nd argument is the center of the circle, third argument is the radius of the circle
# fourth argument is the color of the circle in BGR format, fifth argument is the thickness of the circle
cv2.circle(img, (400, 50), 30, (255, 0, 0), cv2.FILLED)  # BGR
# Third argument is the starting position of the text, fourth argument is the font of the text
# fifth argument is the scale of the text, sixth argument is the color of the text in BGR format
# seventh argument is the thickness of the text
cv2.putText(img, "OpenCV", (300, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)  # BGR

cv2.imshow('image', img)

cv2.waitKey(0)
