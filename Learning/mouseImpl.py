import cv2
import numpy as np

circles = np.zeros((4, 2), np.int32)
counter = 0
def mousePoints(event, x, y, flags, param):
    """
    Mouse callback function to handle mouse events.
    
    :param event: Mouse event type
    :param x: X coordinate of the mouse event
    :param y: Y coordinate of the mouse event
    :param flags: Additional flags for the event
    :param param: Additional parameters for the callback
    """
    global counter
    global circles
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        circles[counter] = x, y
        counter += 1
img = cv2.imread('resources/cards.png')

while True:
    if counter == 4:
        
        # pts1 is the points of the image we want to warp
        pts1 = np.float32([[370, 352], [571, 340], [402, 657], [639, 639]])
        # pts2 is the points of the image we want to warp to
        # below is the width and height of a regular card
        width, height = 250, 350
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

        # This is the transformation matrix that will be used to warp the image
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        # This will warp the image to the new perspective
        output = cv2.warpPerspective(img, matrix, (width, height))
        cv2.imshow('Warped Image', output)
    for i in range(4):
        cv2.circle(img, (int(circles[i][0]), int(circles[i][1])), 5, (0, 255, 0), cv2.FILLED)
    cv2.imshow('Original Image', img)
    cv2.setMouseCallback('Original Image', mousePoints)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
