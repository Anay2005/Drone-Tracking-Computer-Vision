import cv2
import numpy as np
import sys
import utils

frame_width = 5000
frame_height = 1080
# Open the video file or 0 for webcam
cap = cv2.VideoCapture(0)
cap.set(3, frame_width) 
cap.set(4, frame_height)  

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
# We create trackbars to adjust the parameters for Canny edge detection
cv2.createTrackbar("Threshold1", "Parameters", 150, 255, lambda x: None)
cv2.createTrackbar("Threshold2", "Parameters", 255, 255, lambda x: None)
cv2.createTrackbar("Area", "Parameters", 5000, 30000, lambda x: None)
# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    sys.exit(1)

def getContours(img, imgContour):

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        # if the area is less than 500, we will not consider it
        if area > areaMin:
            # 7 is the width of the boundary
            cv2.drawContours(imgContour, contours, -1, (255, 0, 255), 2)
            peri = cv2.arcLength(cnt, True)
            # we will approximate the contour to a polygon
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # we will draw the polygon on the image
            x,y,w,h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(imgContour, "Points: " + str(len(approx)), (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(imgContour, "Area: " + str(int(area)), (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

           

while True:
    # Read a frame from the video
    success, frame = cap.read()
    imgCountour = frame.copy()
    # helps to resize the frame to any resolution we want
    cv2.resize(frame, (frame_width, frame_height))
    imgBlur = cv2.GaussianBlur(frame, (7, 7), 0)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")

    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
    kernel = np.ones((5, 5), np.uint8)  # kernel is a matrix of ones
    imgDilate = cv2.dilate(imgCanny, kernel, iterations=1)
    getContours(imgDilate, imgCountour)
    imgStack = utils.stackImages(0.8, ([frame, imgGray, imgCanny], [imgDilate, imgCountour, imgCountour]))


    # If the frame was not grabbed, we have reached the end of the video
    if not success:
        break

    # Display the frame in a window
    cv2.imshow("Video", imgStack)

    # Wait for 1 ms and check if 'q' key is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break