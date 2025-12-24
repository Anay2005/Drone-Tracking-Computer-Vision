# OpenCV & Machine Learning Fundamentals

This directory contains foundational scripts and exercises used to learn Computer Vision concepts before applying them to the Drone Project. It covers image processing basics, geometric transformations, and a lightweight Convolutional Neural Network (CNN) for Optical Character Recognition (OCR).

## 📂 File Overview

### 1. Image Processing Basics
* **`Reading_images_video_webcam.py`**: Loading media from disk and accessing the live webcam feed.
* **`Basic_functions.py`**: Essential operations like Grayscale conversion, Gaussian Blur, Canny Edge Detection, and Dilation/Erosion.
* **`resizing_crop.py`**: manipulating image dimensions and regions of interest (ROI).
* **`Blank_images_shapes_text.py`**: Drawing geometry (circles, rectangles, lines) and putting text on frames.

### 2. Advanced Vision Concepts
* **`warpPerspective.py`**: Implementing "Bird's Eye View" transformations (homography) to flatten documents or cards.
* **`contoursDetection.py`**: Detecting object boundaries, shapes, and hierarchy retrieval.
* **`colorDetection.py`**: Converting to HSV color space and creating masks to isolate specific colors.
* **`StackingCon.py` / `Stacking_images_test.py`**: Utility to stack multiple image windows into a single UI for easier debugging.

### 3. Machine Learning
* **`OCR_CNN_Training.py`**: Implementation of a Convolutional Neural Network using Keras/TensorFlow to recognize handwritten digits (MNIST dataset).
* **`createData.py`**: Preprocessing and formatting raw image data for model training.

---

### 🛠️ Usage
Each script can be run independently to demonstrate a specific concept:
```bash
python contoursDetection.py