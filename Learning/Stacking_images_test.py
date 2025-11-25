import cv2
import numpy as np
import utils
import sys

"""
Tune thresholds – The 100, 100 in cv2.Canny are not magic; adjust until object outlines appear but noise disappears.

Kernel size – Odd numbers only (3, 5, 7…). Bigger kernels blur/dilate more; start small.

Order matters – Blur before Canny; dilate after Canny. Swapping often breaks results.

"""

frame_width = 5000
frame_height = 1080
# Open the video file or 0 for webcam
cap = cv2.VideoCapture(0)
# cap.set(3, frame_width) 
# cap.set(4, frame_height)  
# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    sys.exit(1)


while True:
    # Read a frame from the video
    success, img = cap.read()
    # helps to resize the frame to any resolution we want
    cv2.resize(img, (frame_width, frame_height))

    # If the frame was not grabbed, we have reached the end of the video
    if not success:
        break

    # Display the frame in a window
    cv2.imshow("Video", img)

    # unint8 is an unsigned 8-bit integer, which means it can hold values from 0 to 255
    kernel = np.ones((5, 5), np.uint8)  # kernel is a matrix of ones


    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Applying Gaussian Blur kernel can be of any size, but it should be odd
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
    imgCanny = cv2.Canny(imgBlur, 100, 100)
    # we will itereate the kernel 1 time
    # iteration it more times will increase the thickness of the edges
    imgDilate = cv2.dilate(imgCanny, (7, 7), iterations=3)
    imgErode = cv2.erode(imgDilate, kernel, iterations=2)
    StackedImages = utils.stackImages(0.5, ([img, imgGray, imgBlur], [imgCanny, imgDilate, imgErode]))
    cv2.imshow('Stacked Images', StackedImages)
    # cv2.imshow('Original', img)
    # cv2.imshow('Gray', imgGray)
    # cv2.imshow('Blur', imgBlur)
    # cv2.imshow('Canny', imgCanny)
    # cv2.imshow('Dilate', imgDilate)
    # cv2.imshow('Erode', imgErode)
    # use 0 for infinite wait
    # Wait for 1 ms and check if 'q' key is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




