import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam




images = []
classNo = []
path = 'mnist_png/training'
mylist = os.listdir(path)
testRatio = 0.2
validationRatio = 0.2
imageDimesions = (28,28,3)
print(mylist)

number_of_classes = len(mylist) - 1
print("Total Classes Detected: ", number_of_classes)

print("Importing Classes...")
# Basically our directory mnist.png/training contains 10 folders, each folder contains images of the same digit hence a double loop is used
# x is the class number
for x in range(number_of_classes):
    myPicList = os.listdir(path + '/' + str(x))
    for y in myPicList:
        curImg = cv2.imread(path + '/' + str(x) + '/' + str(y))
        curImg = cv2.resize(curImg, (imageDimesions[0], imageDimesions[1]))
        images.append(curImg)
        classNo.append(x)

    print(x, end=' ')
    
print(" ")

images = np.array(images)
classNo = np.array(classNo)

# expect to see (no.of images, y-pixels, x-pixels, 3)
print(images.shape)
print(classNo.shape)

# Split the data into training and testing sets
# Always remember 'y' is the label(This what the answer is or output of a function) and 'x' is the data(think of this as the input of a function that gives y)
# ClassNo is the label and images is the data
# test_size=0.2 means 20% of the data will be used for testing and 80% for training
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=validationRatio)

print(y_train)

print("Training data shape: ", X_train.shape)
print("Testing data shape: ", X_test.shape)
print("Validation data shape: ", X_valid.shape)

numSamples = []
for x in range(number_of_classes):
    # np.where returns a tuple of arrays, one for each dimension of the input array in this case, it returns the indices of the elements that are equal to x
    # as a 1D tuple and we use [0] to get the first element of the tuple returned by np.where
    # print("Class ", x, " has ", len(np.where(y_train == x)[0]), " images")
    numSamples.append(len(np.where(y_train == x)[0]))

def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # improving the contrast of the image
    img = cv2.equalizeHist(img)
    img = img / 255
    return img
plt.figure(figsize=(10, 5))
plt.bar(range(number_of_classes), numSamples, color='blue')
plt.title("No.of images for each class")
plt.xlabel("Class ID")
plt.ylabel("Number of images")
plt.show()

# map is a built-in function in Python that applies a given function to all items in an iterable (like a list) and returns a map object (which is an iterator)
# The map object is then converted to a list
X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_valid = np.array(list(map(preProcessing, X_valid)))

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], X_valid.shape[2], 1)

dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, rotation_range=10, zoom_range=0.1, shear_range=0.1, fill_mode='nearest')
dataGen.fit(X_train)

y_train = to_categorical(y_train, num_classes=number_of_classes)
y_test = to_categorical(y_test, num_classes=number_of_classes)
y_valid = to_categorical(y_valid, num_classes=number_of_classes)


number_of_classes = 10               # digits 0-9
...
def myModel():
    model = Sequential([
        Conv2D(60, (5,5), activation='relu',
               input_shape=(28, 28, 1)),
        Conv2D(60, (5,5), activation='relu'),
        MaxPooling2D((2,2)),

        Conv2D(30, (3,3), activation='relu'),   # 60//2 = 30
        Conv2D(30, (3,3), activation='relu'),
        MaxPooling2D((2,2)),

        Dropout(0.5),
        Flatten(),
        Dense(500, activation='relu'),
        Dropout(0.5),
        Dense(number_of_classes, activation='softmax')   # ← 10 neurons
    ])

    opt = Adam(learning_rate=1e-3)        # “lr” is deprecated :contentReference[oaicite:2]{index=2}
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',       # typo fixed
                  metrics=['accuracy'])
    return model

# def myModel():
#     noOfFilters = 60
#     sizeOfFilter1 = (5, 5)
#     sizeOfFilter2 = (3, 3)
#     sizeOfPool = (2,2)
#     noOfNode = 500

#     model = Sequential()

#     # define convolutional layer
#     model.add(Conv2D(noOfFilters, sizeOfFilter1, input_shape=(imageDimesions[0], imageDimesions[1], 1), activation='relu'))
#     model.add(Conv2D(noOfFilters, sizeOfFilter1, activation='relu'))
#     model.add(MaxPooling2D(pool_size = sizeOfPool))
#     model.add(Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu'))
#     model.add(Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu'))
#     model.add(MaxPooling2D(pool_size = sizeOfPool))
#     model.add(Dropout(0.5))
#     model.add(Flatten())
#     model.add(Dense(noOfNode, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(noOfNode, activation='softmax'))
#     model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics = ['accuracy'])

#     return model

model = myModel()
print(model.summary())

batchSizeVal = 50
epochVals = 10
stepsPerEpoch = 2000

history = model.fit(dataGen.flow(X_train, y_train, batch_size=batchSizeVal), steps_per_epoch=stepsPerEpoch, epochs=epochVals, 
                    validation_data=(X_valid, y_valid), shuffle=1)