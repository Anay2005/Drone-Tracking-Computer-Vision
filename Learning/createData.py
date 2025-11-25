"""
Script to use webcam to collect images for training a model.

"""

import os
import cv2

myPath = 'data/images'
cameraNo = 0
cameraBrightness = 190
# save every ith frame to avoid repetition
moduleVal = 10
# smaller value means more blurry images
minBlur = 200
# Images saved color or Gray
grayImage = False
# save data flag
saveData = True
# Image display flag
showImage = True
imgWidth = 180
imgHeight = 120

global countFolder
cap = cv2.VideoCapture(cameraNo)
cap.set(3, imgWidth)
cap.set(4, imgHeight)
cap.set(10, cameraBrightness)

count = 0
countSave = 0

def saveDataFunc():
    global countFolder
    countFolder = 0
    
    while os.path.exists(myPath + str(countFolder)):
        countFolder += 1
    os.makedirs(myPath + str(countFolder))

if saveData:
    saveDataFunc()
    print("Saving data to: " + myPath + str(countFolder))


while True:
    success, img = cap.read()

    if grayImage:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if saveData:
        blur = cv2.Laplacian(img, cv2.CV_64F).var()
        # save every ith frame
        if count % moduleVal == 0 and blur > minBlur:
            countSave += 1
            cv2.imwrite(myPath + str(countFolder) + '/' + str(countSave) + ' .png', img)
            print("Saved image: " + myPath + str(countFolder) + '/' + str(countSave) + '.jpg')

        count += 1
    if showImage:
        cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()