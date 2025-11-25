import cv2

path = "resources/road.jpg"

img = cv2.imread(path)

# .shape returns the dimensions of the image(resolution)
# remember that the first index is the y-axis pixels(height) and the second index is the x-axis pixels(width)
# tghe third index is the number of channels BGR(Blue, Green, Red)
print("Original Dimensions : ", img.shape)

# resize the image to 1000x500 pixels
width, height = 1000, 500
# Remember that resize takes the width first and then the height
imgResize = cv2.resize(img, (width, height))
print("Resized Dimensions : ", imgResize.shape)



# We can crop the image using slicing, suppose we want to detect lanes in the road and most of the other part of the image is not useful
# remember that the first index is the y-axis and the second index is the x-axis and that image is a matrix
# Recall the convention for openCV which is Rightward x = +ve and Downward y = +ve
# use the last 210 pixels(from bottom because of the convention we use -ve sign) of the image for the y-axis(height) and all the pixels for the x-axis(width)
imgCropped = imgResize[-210:, :]  # y1:y2, x1:x2

imgCroppedResize = cv2.resize(imgCropped, (img.shape[1], img.shape[0]))
print("Cropped Dimensions : ", imgCropped.shape)

# Dispaly the images
cv2.imshow("Original Image", img)
cv2.imshow("Resized Image", imgResize)
cv2.imshow("Cropped Image", imgCropped)
cv2.imshow("Cropped Resized Image", imgCroppedResize)
# use 0 for infinite wait
cv2.waitKey(0)