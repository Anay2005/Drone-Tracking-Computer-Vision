# Computer Vision & Autonomous Drone Tracking

This repository documents the journey from fundamental Computer Vision concepts to the development of a fully autonomous drone tracking system. It contains both the core flight control software and the educational modules used to build the necessary visual perception skills.

---

## 📂 Directory Structure

### 1. `Drone Project/` (Autonomous Flight Core)
This directory contains the production-grade code for the autonomous system, featuring:
* **Visual Servoing:** Translates 2D camera data into 3D velocity commands.
* **State Estimation:** Uses a **Kalman Filter** to predict target velocity and smooth out noisy detection data.
* **Robust Control:** Custom **Enhanced PID** controller with Anti-Windup and Rate Limiting.
* **Autonomous Recovery:** Implements a spiral search pattern to re-acquire targets if tracking is lost.

### 2. `Learning/` (OpenCV & ML Fundamentals)
This directory contains foundational scripts and exercises covering:
* **Image Processing:** Grayscale, Gaussian Blur, Edge Detection (Canny), and Morphological operations.
* **Geometric Transformations:** Homography and "Bird's Eye View" warping.
* **Machine Learning:** A Convolutional Neural Network (CNN) trained on the MNIST dataset for Optical Character Recognition (OCR).

---

## 🚁 Deep Dive: Drone Tracking Physics (`/Drone Project`)

The drone tracking system achieves autonomous behavior through three key control theories:

### A. Visual Servoing & Error Normalization
The drone steers based on **Visual Error** ($e$) derived from the camera frame, normalized to $[-1, 1]$:
$$e_{lateral} = \frac{x_{target} - x_{center}}{width / 2}$$
* **$e < 0$:** Target is left $\rightarrow$ Strafe Left.
* **$e > 0$:** Target is right $\rightarrow$ Strafe Right.

### B. Depth Estimation (Pinhole Model)
Since a mono-camera lacks depth information, distance is estimated using the **Pinhole Camera Model** and known target dimensions:
$$d = \frac{W_{known} \cdot f_{focal}}{W_{pixels}}$$

### C. The PID Control Loop
To convert error into smooth motor commands, a **Proportional-Integral-Derivative (PID)** controller is used:
$$u(t) = K_p e(t) + K_i \int e(t) dt + K_d \frac{de(t)}{dt}$$
* **$P$:** Reacts to current error (Speed).
* **$I$:** Corrects steady-state lag (Accuracy).
* **$D$:** Predicts future error to dampen oscillations (Stability).

### D. Kalman Filtering
A Linear Kalman Filter estimates the true state of the target $[x, y, z, v_x, v_y, v_z]^T$. This allows the drone to **coast** smoothly even if the visual detector misses a few frames or is noisy.

---

## 📚 Deep Dive: Learning Modules (`/Learning`)

Before building the drone, these scripts were used to master the basics:

* **`Reading_images_video_webcam.py`**: Handling media streams.
* **`Basic_functions.py`**: Implementation of essential kernels (Gaussian, Canny).
* **`warpPerspective.py`**: Implementing geometric transforms for document scanning.
* **`contoursDetection.py`**: Object boundary detection and shape analysis.
* **`colorDetection.py`**: HSV color space masking.
* **`OCR_CNN_Training.py`**: Training a Keras/TensorFlow model to read handwritten digits.

---

## 🛠️ Installation & Usage

### Dependencies
```bash
pip install airsim opencv-contrib-python filterpy numpy matplotlib pandas tensorflow