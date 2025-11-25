import cv2
import numpy as np

img = cv2.imread('resources/cards.png')
def mousePoints(event, x, y, flags, param):
    """
    Mouse callback function to handle mouse events.
    
    :param event: Mouse event type
    :param x: X coordinate of the mouse event
    :param y: Y coordinate of the mouse event
    :param flags: Additional flags for the event
    :param param: Additional parameters for the callback
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        cv2.circle(img, (x, y), 20, (255, 0, 0), cv2.FILLED)
        # refresh the window with the updated image
        cv2.imshow('Original Image', img)
cv2.imshow('Original Image', img)
cv2.setMouseCallback('Original Image', mousePoints)

cv2.waitKey(0)