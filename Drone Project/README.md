# Autonomous Visual Drone Tracking System

An autonomous flight control system capable of tracking moving targets in a 3D environment using Computer Vision. This project implements advanced control theory concepts including **PID Control**, **Kalman Filtering**, and **Visual Servoing** to achieve smooth, robust tracking behavior in the AirSim simulator.



## 🚀 Key capabilities
* **Visual Servoing:** Translates 2D camera data into 3D velocity commands ($v_x, v_y, v_z, \dot{\psi}$).
* **State Estimation:** Uses a **Kalman Filter** to predict target velocity and smooth out noisy detection data.
* **Robust Control:** Custom **Enhanced PID** controller with Anti-Windup and Rate Limiting.
* **Autonomous Recovery:** Implements a spiral search pattern to re-acquire targets if tracking is lost.
* **Performance Analytics:** Logs telemetry data to CSV for post-flight variance and jitter analysis.

---

## 🧠 Physics & Control Theory

### 1. Visual Servoing & Error Normalization
The drone does not know the target's GPS coordinates. Instead, it steers based on the **Visual Error** ($e$) derived from the camera frame. To ensure the control logic works at any resolution, pixel error is normalized to a range of $[-1, 1]$:

$$e_{lateral} = \frac{x_{target} - x_{center}}{width / 2}$$

* **$e < 0$:** Target is left $\rightarrow$ Strafe Left.
* **$e > 0$:** Target is right $\rightarrow$ Strafe Right.
* **$e = 0$:** Target is centered.

### 2. Depth Estimation (Pinhole Model)
Since a mono-camera lacks depth information, distance is estimated using the **Pinhole Camera Model** and the known physical dimensions of the target:

$$d = \frac{W_{known} \cdot f_{focal}}{W_{pixels}}$$

This allows the drone to maintain a fixed "Follow Distance" (e.g., 12m) regardless of the target's orientation.

### 3. The PID Control Loop
To convert the visual error into smooth motor commands, a **Proportional-Integral-Derivative (PID)** controller is used.

$$u(t) = K_p e(t) + K_i \int e(t) dt + K_d \frac{de(t)}{dt}$$

* **$P$ (Proportional):** Reacts to the current error. Drives the drone toward the target.
* **$I$ (Integral):** Accumulates past error. Corrects steady-state lag (e.g., if the drone is constantly 2m too far back).
* **$D$ (Derivative):** Predicts future error. Acts as a "damper" to prevent overshooting and oscillation during sharp turns.

**Implementation Note:** The controller includes **Integral Anti-Windup** (clamping the $I$ term) to prevent runaway acceleration if the target is unreachable.

### 4. Kalman Filtering (State Estimation)
Raw bounding boxes from Computer Vision are noisy and jittery. A Linear Kalman Filter estimates the true state of the target:

**State Vector:** $[x, y, z, v_x, v_y, v_z]^T$

The filter predicts the target's position in the next frame based on its estimated velocity. This allows the drone to:
1.  **Smooth Tracking:** Ignore random pixel jitter.
2.  **Coast:** Continue moving in the last known direction if the target is briefly occluded (e.g., behind a tree).

---

## 📂 Code Structure & Logic

### `Drone Project/track_PID.py`
The core autonomous flight script.
* **`DroneTracker` Class:** Manages the system state machine (`ACQUIRING`, `TRACKING`, `LOST`, `RETURNING`).
* **`TargetKalmanFilter` Class:** Implements the matrix math for the predict/update steps of state estimation.
* **`EnhancedPID` Class:** A custom PID implementation that handles `dt` (delta time) dynamically to ensure consistent flight physics even if the frame rate drops.

### `Drone Project/analyzer.py`
A post-processing tool that parses the flight logs.
* Calculates **Rolling Variance** of the tracking error to quantify system stability (Jitter).
* Plots **Kalman Uncertainty** ($Trace(P)$) to visualize tracking confidence over time.

---

## 🛠️ Usage

1.  **Environment:** Launch Microsoft AirSim (Blocks or City environment).
2.  **Dependencies:**
    ```bash
    pip install airsim opencv-contrib-python filterpy numpy matplotlib pandas
    ```
3.  **Run the Tracker:**
    ```bash
    python "Drone Project/track_PID.py"
    ```
4.  **Controls:**
    * `t`: Draw a bounding box to select a target.
    * `s`: Force manual search mode.
    * `q`: Land and quit.

---

## 📊 Performance Metrics

The system aims to minimize **RMSE (Root Mean Square Error)** and **Variance**.
* **Typical Lateral Variance:** $< 0.005$ (Stable Flight)
* **Re-acquisition Time:** $< 3.0s$ (using Spiral Search)
