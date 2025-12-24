import time
import math
import cv2
import numpy as np
import airsim
from filterpy.kalman import KalmanFilter
from enum import Enum, auto

# Configuration
PORT = 41451
DRONE = "Drone1"
TARGET = "Target1"
CAM = "front_rgb"
INITIAL_ALT = 40.0
HZ = 20.0
DT = 1.0 / HZ
SAFETY_ALT_MIN = 10.0
SAFETY_ALT_MAX = 100.0
GEOFENCE_RADIUS = 100.0  # meters from start position

# PID Gains 
# Horizontal control (x_error: lateral, y_error: forward/backward)
XY_PID = (0.12, 0.001, 0.008)  # P, I, D for lateral/forward motion
# Vertical control (z_error: altitude)
Z_PID = (0.08, 0.0005, 0.005)  # P, I, D for altitude
# Yaw control (center target horizontally)
YAW_PID = (0.06, 0.0005, 0.004)

# Kalman Filter parameters
KF_PROCESS_NOISE = 0.01
KF_MEASUREMENT_NOISE = 5.0

# Target properties (for depth estimation)
KNOWN_TARGET_WIDTH = 2.0  # meters (typical car width)
KNOWN_TARGET_HEIGHT = 1.5  # meters (typical car height)
FOCAL_LENGTH = 320.0  # Approximate focal length in pixels (640x480 camera)

# Camera intrinsic parameters (AirSim default camera)
CAMERA_MATRIX = np.array([[FOCAL_LENGTH, 0, 320],
                          [0, FOCAL_LENGTH, 240],
                          [0, 0, 1]], dtype=np.float32)

# Search pattern parameters
SEARCH_SPIRAL_RADIUS_INIT = 10.0
SEARCH_SPIRAL_GROWTH = 0.5
SEARCH_SPIRAL_HEIGHT_VAR = 5.0

class TrackingState(Enum):
    ACQUIRING = auto()  # Looking for target
    TRACKING = auto()   # Actively tracking
    LOST = auto()       # Lost target, searching
    RETURNING = auto()  # Returning to start

class PID:
    """PID controller with anti-windup and rate limiting"""
    def __init__(self, kP, kI, kD, setpoint=0, output_limits=(-10, 10), i_windup_limit=5.0):
        self.kP, self.kI, self.kD = kP, kI, kD
        self.setpoint = setpoint
        self.integral = 0
        self.prev_error = 0
        self.prev_measurement = None
        self.output_min, self.output_max = output_limits
        self.i_windup_limit = i_windup_limit
        self.last_output = 0
        
    def update(self, measurement, dt, measurement_derivative=None):
        error = self.setpoint - measurement
        
        # Calculate derivative (use provided or estimate)
        if measurement_derivative is not None:
            derivative = -measurement_derivative  # Derivative of error
        elif self.prev_measurement is not None:
            derivative = (measurement - self.prev_measurement) / dt
            derivative = -derivative  # Derivative of error
        else:
            derivative = 0
        
        # Update integral with anti-windup
        self.integral += error * dt
        self.integral = max(min(self.integral, self.i_windup_limit), -self.i_windup_limit)
        
        # Calculate PID output
        output = (self.kP * error) + (self.kI * self.integral) + (self.kD * derivative)
        
        # Rate limiting
        max_change = 2.0 * dt  # Limit rate of change
        if abs(output - self.last_output) > max_change:
            output = self.last_output + np.sign(output - self.last_output) * max_change
        
        # Clamp output
        output = np.clip(output, self.output_min, self.output_max)
        
        self.prev_error = error
        self.prev_measurement = measurement
        self.last_output = output
        
        return output
    
    def reset(self):
        self.integral = 0
        self.prev_error = 0
        self.prev_measurement = None
        self.last_output = 0

class TargetKalmanFilter:
    """Kalman filter for target position and velocity estimation"""
    def __init__(self, dt, initial_position, process_noise=0.01, measurement_noise=5.0):
        self.kf = KalmanFilter(dim_x=6, dim_z=3)  # State: [x, y, z, vx, vy, vz]
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([[1, 0, 0, dt, 0, 0],
                              [0, 1, 0, 0, dt, 0],
                              [0, 0, 1, 0, 0, dt],
                              [0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 1]])
        
        # Measurement matrix (we only measure position)
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0]])
        
        # Process noise covariance
        self.kf.Q = np.eye(6) * process_noise
        self.kf.Q[3:, 3:] *= 10  # Higher uncertainty in velocity
        
        # Measurement noise covariance
        self.kf.R = np.eye(3) * measurement_noise
        
        # Initial state
        self.kf.x = np.array([initial_position[0], initial_position[1], initial_position[2], 0, 0, 0])
        
        # Initial covariance (high uncertainty in velocity)
        self.kf.P = np.eye(6) * 100
        self.kf.P[3:, 3:] = np.eye(3) * 1000
        
    def predict(self):
        self.kf.predict()
        return self.kf.x[:3]  # Return predicted position
        
    def update(self, measurement):
        self.kf.update(measurement)
        return self.kf.x  # Return full state (position and velocity)

class DroneTracker:
    def __init__(self):
        self.client = None
        self.state = TrackingState.ACQUIRING
        self.tracker = None
        self.kalman_filter = None
        self.start_position = None
        self.last_known_position = None
        self.search_step = 0
        self.search_radius = SEARCH_SPIRAL_RADIUS_INIT
        self.target_lost_time = 0
        self.frame_count = 0
        
        # Initialize controllers
        self.x_controller = PID(*XY_PID, setpoint=0, output_limits=(-5, 5))  # Lateral (right/left)
        self.y_controller = PID(*XY_PID, setpoint=12.0, output_limits=(-2, 8))  # Forward/backward (distance)
        self.z_controller = PID(*Z_PID, setpoint=0, output_limits=(-3, 3))  # Vertical
        self.yaw_controller = PID(*YAW_PID, setpoint=0, output_limits=(-30, 30))  # Rotation
        
        # Performance metrics
        self.tracking_history = []
        self.lost_count = 0
        
    def connect(self):
        """Connect to AirSim and initialize drone"""
        self.client = airsim.MultirotorClient(port=PORT)
        self.client.confirmConnection()
        self.client.enableApiControl(True, DRONE)
        self.client.armDisarm(True, DRONE)
        self.client.takeoffAsync(vehicle_name=DRONE).join()
        self.client.moveToZAsync(-INITIAL_ALT, 3.0, vehicle_name=DRONE).join()
        
        # Get start position for geofencing
        start_pose = self.client.simGetVehiclePose(vehicle_name=DRONE)
        self.start_position = np.array([start_pose.position.x_val, 
                                        start_pose.position.y_val, 
                                        start_pose.position.z_val])
        
        print(f"Drone initialized at altitude: {INITIAL_ALT}m")
        
    def estimate_depth(self, bbox_width, bbox_height):
        """Estimate distance to target using known size and camera parameters"""
        # Use both width and height for robustness
        distance_from_width = (KNOWN_TARGET_WIDTH * FOCAL_LENGTH) / bbox_width
        distance_from_height = (KNOWN_TARGET_HEIGHT * FOCAL_LENGTH) / bbox_height
        
        # Average with weighting based on confidence (aspect ratio close to expected)
        expected_aspect = KNOWN_TARGET_WIDTH / KNOWN_TARGET_HEIGHT
        actual_aspect = bbox_width / bbox_height
        aspect_error = abs(actual_aspect - expected_aspect) / expected_aspect
        
        # Weight more confident measurement higher
        if aspect_error < 0.3:  # Good aspect ratio match
            # Both measurements should be similar
            distance = (distance_from_width + distance_from_height) / 2
        else:
            # Use the more stable one (usually width for ground vehicles)
            distance = distance_from_width
            
        return max(2.0, min(50.0, distance))  # Clamp to reasonable range
    
    def check_position(self, current_position):
        """Ensure drone stays within safe operating area"""
        distance_from_start = np.linalg.norm(current_position[:2] - self.start_position[:2])
        
        if distance_from_start > GEOFENCE_RADIUS:
            print(f"Warning: Approaching geofence boundary ({distance_from_start:.1f}m from start)")
            return False
        return True
    def calculate_metrics(self):
        """Calculate and print RMSE and average FPS."""
        if not self.tracking_history:
            print("No tracking data collected.")
            return

        # Convert history to numpy arrays
        # history stores: (timestamp, error_x, error_y, error_distance)
        data = np.array(self.tracking_history)
        
        # Calculate RMSE for lateral error (X) and vertical error (Y)
        rmse_x = np.sqrt(np.mean(data[:, 1]**2))
        rmse_y = np.sqrt(np.mean(data[:, 2]**2))
        
        # Calculate Average FPS
        timestamps = data[:, 0]
        duration = timestamps[-1] - timestamps[0]
        avg_fps = len(timestamps) / duration if duration > 0 else 0

        print("\n" + "="*40)
        print("      PERFORMANCE METRICS       ")
        print("="*40)
        print(f"Lateral RMSE (Centering): {rmse_x:.4f} (normalized -1 to 1)")
        print(f"Vertical RMSE (Centering): {rmse_y:.4f} (normalized -1 to 1)")
        print(f"Average FPS:              {avg_fps:.2f}")
        print("="*40)

        # Save to CSV for plotting
        np.savetxt("flight_metrics.csv", data, delimiter=",", header="timestamp,error_x,error_y,dist", comments="")
        print("Metrics saved to 'flight_metrics.csv'")


    def calculate_3d_control(self, bbox, frame_shape, target_distance):
        """Calculate 3D control commands from bounding box"""
        H, W = frame_shape[:2]
        x, y, w, h = bbox
        
        # Calculate normalized errors (-1 to 1 range)
        target_center_x = x + w/2
        target_center_y = y + h/2
        
        error_x = (target_center_x - W/2) / (W/2)  # Normalized lateral error
        error_y = (target_center_y - H/2) / (H/2)  # Normalized vertical error
        
        # Use Kalman filter for smooth control
        if self.kalman_filter:
            # Predict target position
            predicted_pos = self.kalman_filter.predict()
            
            # Create measurement vector [x_error, y_error, distance]
            measurement = np.array([error_x, error_y, target_distance])
            
            # Update Kalman filter
            state = self.kalman_filter.update(measurement)
            
            # Use filtered values for control
            filtered_error_x = state[0]
            filtered_error_y = state[1]
            filtered_distance = state[2]
            target_velocity = state[3:6]  # Extract velocity from state
        else:
            filtered_error_x = error_x
            filtered_error_y = error_y
            filtered_distance = target_distance
            target_velocity = np.zeros(3)
        
        # Update controllers with filtered measurements
        # Lateral control (move left/right to center horizontally)
        vx = self.x_controller.update(filtered_error_x, DT)
        
        # Forward/backward control (maintain desired distance)
        vy = self.y_controller.update(filtered_distance, DT, target_velocity[1])
        
        # Vertical control (move up/down to center vertically)
        vz = self.z_controller.update(filtered_error_y, DT, target_velocity[2])
        
        # Yaw control (rotate to keep target centered, but less aggressive with lateral control)
        yaw_rate = self.yaw_controller.update(filtered_error_x * 0.5, DT)
        
        #  Reduce forward speed when target is far from center
        center_distance = math.sqrt(filtered_error_x**2 + filtered_error_y**2)
        if center_distance > 0.5:  # Target near edge
            vy *= 0.5  # Slow down when not well centered
            yaw_rate *= 1.5  # More aggressive yaw correction
        # Log the metrics (Time, Error X, Error Y, Filtered Distance)
        self.tracking_history.append([time.time(), filtered_error_x, filtered_error_y, filtered_distance])

        return vx, vy, vz, yaw_rate, filtered_distance
    
    def execute_search_pattern(self):
        """Execute spiral search pattern to reacquire target"""
        if self.state != TrackingState.LOST:
            return 0, 0, 0, 0
        
        # Spiral search pattern
        angle = self.search_step * 0.2
        radius = self.search_radius
        
        # Calculate spiral position
        vx = math.cos(angle) * 2.0  # Lateral movement
        vy = math.sin(angle) * 1.5  # Forward movement
        vz = math.sin(angle * 2) * 0.5  # Vertical oscillation
        
        # Slowly expand search area
        if self.search_step % 50 == 0:
            self.search_radius += SEARCH_SPIRAL_GROWTH
        
        self.search_step += 1
        
        # Check if we should return to start
        if time.time() - self.target_lost_time > 30:  # 30 seconds without target
            self.state = TrackingState.RETURNING
            
        return vx, vy, vz, 15.0  # Slow rotation during search
    
    def return_to_start(self):
        """Return to start position"""
        current_pose = self.client.simGetVehiclePose(vehicle_name=DRONE)
        current_pos = np.array([current_pose.position.x_val, 
                               current_pose.position.y_val, 
                               current_pose.position.z_val])
        
        # Calculate vector to start
        to_start = self.start_position - current_pos
        
        # Normalize for velocity control
        distance = np.linalg.norm(to_start)
        if distance < 5.0:  # Close enough
            print("Returned to start position")
            return 0, 0, 0, 0
        
        direction = to_start / distance
        
        # Limit speed based on distance
        speed = min(3.0, distance * 0.5)
        
        vx = direction[0] * speed
        vy = direction[1] * speed
        vz = direction[2] * speed
        
        # Face direction of movement
        yaw = math.atan2(direction[1], direction[0]) * 180 / math.pi
        
        return vx, vy, vz, yaw
    
    def process_frame(self, frame):
        """Process a single frame and return tracking info"""
        if frame is None:
            return None, None, None
        
        if self.state == TrackingState.TRACKING and self.tracker is not None:
            # Update tracker
            ok, bbox = self.tracker.update(frame)
            
            if ok:
                x, y, w, h = map(int, bbox)
                
                # Check if target is reasonable size
                if w > 10 and h > 10 and w < frame.shape[1] and h < frame.shape[0]:
                    self.last_known_position = (x, y, w, h)
                    
                    # Estimate distance
                    distance = self.estimate_depth(w, h)
                    
                    return bbox, distance, ok
            
            # Tracking failed
            self.lost_count += 1
            if self.lost_count > 10:  # Require multiple failures
                self.state = TrackingState.LOST
                self.target_lost_time = time.time()
                print("Target lost, initiating search pattern")
            return None, None, False
        
        return None, None, False
    
    def initialize_tracker(self, frame, roi):
        """Initialize tracker with ROI"""
        self.tracker = cv2.TrackerCSRT_create()
        if self.tracker.init(frame, roi):
            self.state = TrackingState.TRACKING
            self.lost_count = 0
            
            # Get initial bounding box
            x, y, w, h = roi
            initial_distance = self.estimate_depth(w, h)
            
            # Initialize Kalman filter with initial state
            self.kalman_filter = TargetKalmanFilter(
                DT, 
                initial_position=[0, initial_distance, 0],  # Initial errors: x=0, distance, z=0
                process_noise=KF_PROCESS_NOISE,
                measurement_noise=KF_MEASUREMENT_NOISE
            )
            
            # Set desired distance based on initial
            self.y_controller.setpoint = initial_distance
            
            print(f"Tracker initialized. Initial distance: {initial_distance:.1f}m")
            return True
        return False
    
    def draw_hud(self, frame, bbox, state, control_values):
        """Draw HUD overlay on frame"""
        H, W = frame.shape[:2]
        overlay = frame.copy()
        
        # State indicator
        state_colors = {
            TrackingState.ACQUIRING: (0, 255, 255),  # Yellow
            TrackingState.TRACKING: (0, 255, 0),     # Green
            TrackingState.LOST: (0, 0, 255),         # Red
            TrackingState.RETURNING: (255, 255, 0)   # Cyan
        }
        
        cv2.putText(overlay, f"State: {state.name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_colors[state], 2)
        
        # Draw center crosshair
        cv2.line(overlay, (W//2 - 20, H//2), (W//2 + 20, H//2), (255, 255, 255), 1)
        cv2.line(overlay, (W//2, H//2 - 20), (W//2, H//2 + 20), (255, 255, 255), 1)
        cv2.circle(overlay, (W//2, H//2), 50, (255, 255, 255), 1)
        
        if bbox is not None:
            x, y, w, h = map(int, bbox)
            
            # Draw bounding box
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw target center
            center_x, center_y = x + w//2, y + h//2
            cv2.circle(overlay, (center_x, center_y), 5, (0, 255, 255), -1)
            
            # Draw line from center to target
            cv2.line(overlay, (W//2, H//2), (center_x, center_y), (255, 0, 0), 2)
            
            # Display distance if available
            if control_values and 'distance' in control_values:
                cv2.putText(overlay, f"Dist: {control_values['distance']:.1f}m", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Control values
        if control_values:
            y_pos = 60
            for key, value in control_values.items():
                if key != 'distance':
                    cv2.putText(overlay, f"{key}: {value:.2f}", (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_pos += 20
        
        # Search info
        if state == TrackingState.LOST:
            cv2.putText(overlay, f"Search Radius: {self.search_radius:.1f}m", 
                       (W-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            cv2.putText(overlay, f"Lost for: {time.time() - self.target_lost_time:.0f}s", 
                       (W-200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        return frame
    
    def run(self):
        """Main tracking loop"""
        self.connect()
        
        print("Drone Tracking System Ready")
        print("Controls:")
        print("  't' - Select target ROI")
        print("  's' - Toggle search mode")
        print("  'r' - Return to start")
        print("  'q' - Quit")
        
        while True:
            try:
                # Get frame
                png = self.client.simGetImage(CAM, airsim.ImageType.Scene, vehicle_name=DRONE)
                if not png:
                    time.sleep(0.1)
                    continue
                
                frame = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                
                # Get current drone state for safety checks
                drone_state = self.client.getMultirotorState(vehicle_name=DRONE)
                current_alt = -drone_state.kinematics_estimated.position.z_val
                
                # Process frame
                bbox, distance, tracking_ok = self.process_frame(frame)
                
                # Default control values
                vx, vy, vz, yaw_rate = 0, 0, 0, 0
                control_values = {}
                
                # State machine
                if self.state == TrackingState.TRACKING and tracking_ok:
                    # Calculate 3D control
                    vx, vy, vz, yaw_rate, filtered_dist = self.calculate_3d_control(bbox, frame.shape, distance)
                    control_values = {
                        'vx': vx, 'vy': vy, 'vz': vz, 
                        'yaw': yaw_rate, 'distance': filtered_dist
                    }
                    
                    # Safety altitude adjustment
                    if current_alt < SAFETY_ALT_MIN:
                        vz += 1.0  # Increase altitude
                    elif current_alt > SAFETY_ALT_MAX:
                        vz -= 1.0  # Decrease altitude
                    
                elif self.state == TrackingState.LOST:
                    # Execute search pattern
                    vx, vy, vz, yaw_rate = self.execute_search_pattern()
                    control_values = {'Search': self.search_step}
                    
                elif self.state == TrackingState.RETURNING:
                    # Return to start position
                    vx, vy, vz, yaw_rate = self.return_to_start()
                    control_values = {'Returning': True}
                    
                
                current_pos = np.array([
                    drone_state.kinematics_estimated.position.x_val,
                    drone_state.kinematics_estimated.position.y_val,
                    drone_state.kinematics_estimated.position.z_val
                ])
                
                if not self.check_position(current_pos):
                    # Slow down near boundary
                    vx *= 0.5
                    vy *= 0.5
                
                # Send control command
                self.client.moveByVelocityBodyFrameAsync(
                    vx=float(vy),  # AirSim: vx is forward, vy is right, vz is down
                    vy=float(vx),
                    vz=float(-vz),  # Invert for NED coordinates
                    duration=DT,
                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=float(yaw_rate)),
                    vehicle_name=DRONE
                )
                
                # Draw HUD
                frame = self.draw_hud(frame, bbox, self.state, control_values)
                
                # Display
                cv2.imshow("Drone Tracking", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('t'):
                    # Target selection
                    roi = cv2.selectROI("Select Target", frame, False)
                    if roi != (0, 0, 0, 0):
                        if self.initialize_tracker(frame, roi):
                            print("Target locked")
                elif key == ord('s'):
                    # Toggle search mode
                    if self.state != TrackingState.LOST:
                        self.state = TrackingState.LOST
                        self.target_lost_time = time.time()
                        print("Manual search initiated")
                    else:
                        self.state = TrackingState.ACQUIRING
                        print("Search stopped")
                elif key == ord('r'):
                    # Return to start
                    self.state = TrackingState.RETURNING
                    print("Returning to start position")
                elif key == ord('q'):
                    break
                    
                self.frame_count += 1
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(1.0)
        
        self.calculate_metrics() 
        # Cleanup
        print("Landing...")
        self.client.landAsync(vehicle_name=DRONE).join()
        self.client.armDisarm(False, DRONE)
        cv2.destroyAllWindows()

def main():
    tracker = DroneTracker()
    try:
        tracker.run()
    except KeyboardInterrupt:
        print("\nTracking interrupted by user")
    finally:
        if tracker.client:
            tracker.client.armDisarm(False, DRONE)
            tracker.client.enableApiControl(False, DRONE)

if __name__ == "__main__":
    main()