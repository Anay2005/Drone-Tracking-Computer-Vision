import cv2
import numpy as np

framewidth = 640
frameheight = 480
cap = cv2.VideoCapture(0)
cap.set(3, framewidth)
cap.set(4, frameheight)

def empty(a):
    pass

cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 640, 240)

cv2.createTrackbar("HUE Min", "HSV", 0, 179, empty)
cv2.createTrackbar("HUE Max", "HSV", 179, 179, empty)
cv2.createTrackbar("SAT Min", "HSV", 0, 255, empty)
cv2.createTrackbar("SAT Max", "HSV", 255, 255, empty)
cv2.createTrackbar("VAL Min", "HSV", 0, 255, empty)
cv2.createTrackbar("VAL Max", "HSV", 255, 255, empty)

while True:
    _, img = cap.read()
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hmin = cv2.getTrackbarPos("HUE Min", "HSV")
    hmax = cv2.getTrackbarPos("HUE Max", "HSV")
    smin = cv2.getTrackbarPos("SAT Min", "HSV")
    smax = cv2.getTrackbarPos("SAT Max", "HSV")
    vmin = cv2.getTrackbarPos("VAL Min", "HSV")
    vmax = cv2.getTrackbarPos("VAL Max", "HSV")

    lower = np.array([hmin, smin, vmin])
    upper = np.array([hmax, smax, vmax])
    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask=mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    hstack = np.hstack((img, mask, imgResult))
    cv2.imshow("Stacked Images", hstack)
   
    cv2.imshow("Original", img)
    #cv2.imshow("HSV", imgHSV)
    #cv2.imshow("Mask", mask)
    cv2.imshow("Result", imgResult)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break