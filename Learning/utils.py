import cv2
import numpy as np

def stackImages(scale, imgArray):
    """
    Scales and stacks images in a grid.
    
    :param scale: float scaling factor for all images
    :param imgArray: 1D list of images or 2D list (list of lists) in sorter words: 2D array of images
    :return: single image composed of the input images stacked accordingly
    """
    # Check if imgArray is 2D (grid) or 1D (single row)
    rows_available = isinstance(imgArray[0], list)
    # Number of rows and columns
    rows = len(imgArray)
    cols = len(imgArray[0]) if rows_available else len(imgArray)
    
    # Get dimensions of first image
    first_img = imgArray[0][0] if rows_available else imgArray[0]
    height, width = first_img.shape[:2]
    
    # Prepare each image: resize and convert grayscale to BGR
    def prepare(img):
        # Resize to match first image, then scale
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

        # With (0,0) you’re essentially saying, “I don’t want to give you a hard target size—just use my fx/fy factors to compute it.
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if len(img.shape) == 2:  # grayscale to BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img
    
    # Stack
    if rows_available:
        # Process grid
        grid = []
        for row in imgArray:
            grid.append([prepare(img) for img in row])
        # Horizontal stacks for each row
        hor_stacks = [np.hstack(row) for row in grid]
        # Vertical stack of rows
        imgStack = np.vstack(hor_stacks)
    else:
        # Single row
        processed = [prepare(img) for img in imgArray]
        imgStack = np.hstack(processed)
    
    return imgStack



