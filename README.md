# Computer Vision & Autonomous Drone Tracking

This repository documents the journey from fundamental Computer Vision concepts to the development of a fully autonomous drone tracking system. It contains the core flight control software, advanced vision utilities, and educational modules used to build the necessary visual perception skills.

---

## 📂 Directory Structure

### 1. `Drone Project/` (Autonomous Flight Core)
This directory contains the production-grade code for the autonomous system:
* **`track_PID.py`**: The main autonomous flight script using PID and Kalman logic.
* **`img_translation_detection_*.py`**: Advanced algorithms for detecting precise pixel shifts between frames (Digital Image Stabilization logic).

### 2. `Learning/` (OpenCV & ML Fundamentals)
Foundational scripts and exercises covering:
* **Image Processing:** Grayscale, Gaussian Blur, Edge Detection (Canny), and Morphological operations.
* **Geometric Transformations:** Homography and "Bird's Eye View" warping.
* **Machine Learning:** A CNN trained on the MNIST dataset for Optical Character Recognition (OCR).

---

## 🚁 Deep Dive: Drone Tracking Physics (`/Drone Project`)

The drone tracking system achieves autonomous behavior through three key control theories:

### A. Visual Servoing & Error Normalization
The drone steers based on **Visual Error** ($e$) derived from the camera frame, normalized to $[-1, 1]$:
$$e_{lateral} = \frac{x_{target} - x_{center}}{width / 2}$$
* **$e < 0$:** Target is left $\rightarrow$ Strafe Left.
* **$e > 0$:** Target is right $\rightarrow$ Strafe Right.

### B. The PID Control Loop
To convert error into smooth motor commands, a **Proportional-Integral-Derivative (PID)** controller is used:
$$u(t) = K_p e(t) + K_i \int e(t) dt + K_d \frac{de(t)}{dt}$$
* **$P$:** Reacts to current error (Speed).
* **$I$:** Corrects steady-state lag (Accuracy).
* **$D$:** Predicts future error to dampen oscillations (Stability).

### C. Kalman Filtering
A Linear Kalman Filter estimates the true state of the target $[x, y, z, v_x, v_y, v_z]^T$. This allows the drone to **coast** smoothly even if the visual detector misses a few frames or is noisy.

---

## 🔭 Advanced Vision: Translation & Shift Detection

Recent updates include a hybrid engine to calculate the precise pixel shift ($\Delta x, \Delta y$) between a reference image and a live feed. This is critical for **Optical Flow** and **Image Stabilization**.

### 1. Phase Correlation (The Fourier Shift Theorem)
For pure translations, we calculate shift in the frequency domain.
* **Theory:** A shift in space becomes a phase shift in frequency: $\mathcal{F}\{f(x-x_0)\} = F(u)e^{-i2\pi u x_0}$.
* **Spectral Whitening:** We normalize the cross-power spectrum to isolate phase information: $$R = \frac{F_1 \cdot F_2^*}{|F_1 \cdot F_2^*|}$$
* **Robustness:** To handle lighting changes, we pre-process images using **Sobel Edge Detection** and **Otsu Thresholding** before the FFT. This tracks "structure" rather than "brightness."



### 2. Sub-Pixel Refinement
The FFT peak only gives integer accuracy. To get precise floating-point shifts (e.g., $\Delta x = 3.42$), we perform a **3x3 Quadratic Fit** around the peak energy.
$$dx \approx \frac{C(x+1) - C(x-1)}{2(2C(x) - C(x+1) - C(x-1))}$$

### 3. Hybrid Homography Pipeline
Phase correlation fails if the camera rotates or zooms. The system solves this with a two-step "Rectify & Correlate" pipeline:
1.  **Feature Matching:** Detects keypoints (ORB) and matches them between frames.
2.  **RANSAC Homography:** Estimates the transformation matrix $H$.
3.  **Decision Logic:**
    * If $H$ is simple (translation only) $\rightarrow$ Use direct Phase Correlation.
    * If $H$ is complex (rotation/scale) $\rightarrow$ **Warp** (Rectify) the live image using $H^{-1}$, *then* apply Phase Correlation for the final alignment.



---

## 🛠️ Installation & Usage

### Dependencies
```bash
pip install airsim opencv-contrib-python filterpy numpy matplotlib pandas tensorflow
