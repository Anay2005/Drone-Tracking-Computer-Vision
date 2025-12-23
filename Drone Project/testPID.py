import time
import math
import cv2
import numpy as np
import airsim
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from enum import Enum
import json
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Configuration
PORT = 41451
DRONE = "Drone1"
TARGET = "Target1"
CAM = "front_rgb"
TEST_DURATION = 90  # seconds per test
NUM_TESTS = 3  # Number of repetitions per scenario
TEST_SCENARIOS = ["straight_line", "circle", "zigzag", "sudden_stop", "occlusion", "climbing"]

class TrackingState(Enum):
    ACQUIRING = "ACQUIRING"
    TRACKING = "TRACKING"
    LOST = "LOST"
    RETURNING = "RETURNING"

class PerformanceMetrics:
    """Collect and analyze performance metrics for 3D tracker"""
    def __init__(self, test_id, scenario):
        self.test_id = test_id
        self.scenario = scenario
        self.start_time = None
        self.metrics = {
            'timestamp': [],
            'tracking_state': [],
            'position_error': [],      # Euclidean distance to target (m)
            'lateral_error': [],       # Horizontal error (m)
            'vertical_error': [],      # Vertical error (m)
            'forward_error': [],       # Distance error (m)
            'control_effort': [],      # Sum of velocity commands (m/s)
            'yaw_error': [],          # Yaw angle error (degrees)
            'bbox_area': [],          # Bounding box area (pixels)
            'distance_estimate': [],  # Estimated distance (m)
            'true_distance': [],      # Ground truth distance (m)
            'fps': [],                # Frames per second
            'control_vx': [],         # Lateral velocity (m/s)
            'control_vy': [],         # Forward velocity (m/s)
            'control_vz': [],         # Vertical velocity (m/s)
            'control_yaw': [],        # Yaw rate (deg/s)
            'altitude': [],           # Drone altitude (m)
            'target_in_frame': [],    # Whether target is in frame (0/1)
        }
        self.frame_count = 0
        self.lost_frames = 0
        self.tracking_frames = 0
        self.search_start_time = None
        self.total_search_time = 0
        
    def start_recording(self):
        self.start_time = time.time()
        
    def record_frame(self, **kwargs):
        """Record metrics for a single frame"""
        current_time = time.time() - self.start_time
        self.frame_count += 1
        
        # Default values
        frame_data = {
            'timestamp': current_time,
            'tracking_state': kwargs.get('tracking_state', 'UNKNOWN'),
            'position_error': kwargs.get('position_error', 0),
            'lateral_error': kwargs.get('lateral_error', 0),
            'vertical_error': kwargs.get('vertical_error', 0),
            'forward_error': kwargs.get('forward_error', 0),
            'control_effort': kwargs.get('control_effort', 0),
            'yaw_error': kwargs.get('yaw_error', 0),
            'bbox_area': kwargs.get('bbox_area', 0),
            'distance_estimate': kwargs.get('distance_estimate', 0),
            'true_distance': kwargs.get('true_distance', 0),
            'fps': kwargs.get('fps', 0),
            'control_vx': kwargs.get('control_vx', 0),
            'control_vy': kwargs.get('control_vy', 0),
            'control_vz': kwargs.get('control_vz', 0),
            'control_yaw': kwargs.get('control_yaw', 0),
            'altitude': kwargs.get('altitude', 0),
            'target_in_frame': kwargs.get('target_in_frame', 0),
        }
        
        # Update counters
        if frame_data['tracking_state'] == 'TRACKING':
            self.tracking_frames += 1
        elif frame_data['tracking_state'] == 'LOST':
            self.lost_frames += 1
            if self.search_start_time is None:
                self.search_start_time = current_time
        elif self.search_start_time is not None:
            search_duration = current_time - self.search_start_time
            self.total_search_time += search_duration
            self.search_start_time = None
        
        # Store metrics
        for key, value in frame_data.items():
            self.metrics[key].append(value)
            
    def calculate_performance_summary(self):
        """Calculate comprehensive performance summary"""
        if not self.metrics['position_error']:
            return {}
        
        errors = np.array(self.metrics['position_error'])
        tracking_ratio = self.tracking_frames / max(1, self.frame_count)
        
        summary = {
            'test_id': self.test_id,
            'scenario': self.scenario,
            'test_duration': self.metrics['timestamp'][-1] if self.metrics['timestamp'] else 0,
            'total_frames': self.frame_count,
            'tracking_frames': self.tracking_frames,
            'lost_frames': self.lost_frames,
            'tracking_ratio': tracking_ratio,
            'mean_position_error': float(np.mean(errors)),
            'median_position_error': float(np.median(errors)),
            'std_position_error': float(np.std(errors)),
            'max_position_error': float(np.max(errors)),
            'rmse': float(np.sqrt(np.mean(errors**2))),
            'mean_lateral_error': float(np.mean(np.abs(self.metrics['lateral_error']))),
            'mean_vertical_error': float(np.mean(np.abs(self.metrics['vertical_error']))),
            'mean_forward_error': float(np.mean(np.abs(self.metrics['forward_error']))),
            'mean_control_effort': float(np.mean(self.metrics['control_effort'])),
            'total_control_effort': float(np.sum(np.abs(self.metrics['control_effort']))),
            'mean_fps': float(np.mean(self.metrics['fps'][1:]) if len(self.metrics['fps']) > 1 else 0),
            'min_fps': float(np.min(self.metrics['fps'][1:]) if len(self.metrics['fps']) > 1 else 0),
            'total_search_time': self.total_search_time,
            'search_ratio': self.total_search_time / max(1, self.metrics['timestamp'][-1]),
            'distance_estimation_error': self._calculate_distance_estimation_error(),
            'control_smoothness': self._calculate_control_smoothness(),
            'response_time': self._calculate_response_time(),
            'stability_index': self._calculate_stability_index(),
        }
        return summary
    
    def _calculate_distance_estimation_error(self):
        """Calculate accuracy of distance estimation"""
        if not self.metrics['distance_estimate'] or not self.metrics['true_distance']:
            return 0
        
        estimates = np.array(self.metrics['distance_estimate'])
        truths = np.array(self.metrics['true_distance'])
        
        # Only calculate where we have both values
        mask = (estimates > 0) & (truths > 0)
        if np.sum(mask) == 0:
            return 0
        
        relative_errors = np.abs(estimates[mask] - truths[mask]) / truths[mask]
        return float(np.mean(relative_errors)) * 100  # Percentage error
    
    def _calculate_control_smoothness(self):
        """Calculate how smooth the control inputs are"""
        if len(self.metrics['control_vx']) < 2:
            return 0
        
        # Calculate jerk (derivative of acceleration)
        vx = np.array(self.metrics['control_vx'])
        vy = np.array(self.metrics['control_vy'])
        vz = np.array(self.metrics['control_vz'])
        
        # Approximate derivatives
        dt = np.diff(self.metrics['timestamp'])
        dt = np.where(dt > 0, dt, 0.05)  # Avoid division by zero
        
        jerk_x = np.diff(vx) / dt if len(vx) > 1 else 0
        jerk_y = np.diff(vy) / dt if len(vy) > 1 else 0
        jerk_z = np.diff(vz) / dt if len(vz) > 1 else 0
        
        # Lower jerk means smoother control
        total_jerk = np.mean(jerk_x**2 + jerk_y**2 + jerk_z**2)
        smoothness = 1.0 / (1.0 + total_jerk)  # Higher is smoother
        return float(smoothness)
    
    def _calculate_response_time(self):
        """Calculate average response time to errors"""
        errors = np.array(self.metrics['position_error'])
        if len(errors) < 100:
            return 0
        
        # Find when error crosses threshold
        threshold = np.mean(errors) + np.std(errors)
        crossings = np.where(errors > threshold)[0]
        
        if len(crossings) < 2:
            return 0
        
        # Calculate decay time
        response_times = []
        for crossing in crossings[:5]:  # First 5 major errors
            if crossing + 50 < len(errors):
                # Find when error drops to 50% of peak
                peak = errors[crossing]
                target = peak * 0.5
                future_errors = errors[crossing:crossing+50]
                below_target = np.where(future_errors < target)[0]
                if len(below_target) > 0:
                    response_time = below_target[0] * (1/self.mean_fps if hasattr(self, 'mean_fps') and self.mean_fps > 0 else 0.05)
                    response_times.append(response_time)
        
        return float(np.mean(response_times)) if response_times else 0
    
    def _calculate_stability_index(self):
        """Calculate overall tracking stability"""
        errors = np.array(self.metrics['position_error'])
        if len(errors) < 2:
            return 0
        
        # Combine multiple stability metrics
        error_std = np.std(errors)
        error_mean = np.mean(errors)
        tracking_consistency = self.tracking_frames / max(1, self.frame_count)
        
        # Lower error variance and higher tracking consistency = better stability
        stability = tracking_consistency / (1 + error_std/error_mean if error_mean > 0 else 1)
        return float(stability)
    
    def save_raw_data(self, output_dir):
        """Save raw metrics data to CSV"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = output_dir / f"raw_metrics_{self.test_id}_{self.scenario}.csv"
        df = pd.DataFrame(self.metrics)
        df.to_csv(filename, index=False)
        return filename
    
    def plot_performance(self, output_dir):
        """Generate comprehensive performance plots"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Position Tracking
        ax1 = plt.subplot(4, 3, 1)
        ax1.plot(self.metrics['timestamp'], self.metrics['position_error'], 'b-', linewidth=1)
        ax1.set_title('Position Error Over Time')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Error (m)')
        ax1.grid(True, alpha=0.3)
        ax1.fill_between(self.metrics['timestamp'], 0, self.metrics['position_error'], alpha=0.3)
        
        # 2. 3D Error Components
        ax2 = plt.subplot(4, 3, 2)
        ax2.plot(self.metrics['timestamp'], self.metrics['lateral_error'], 'r-', label='Lateral', linewidth=1)
        ax2.plot(self.metrics['timestamp'], self.metrics['vertical_error'], 'g-', label='Vertical', linewidth=1)
        ax2.plot(self.metrics['timestamp'], self.metrics['forward_error'], 'b-', label='Forward', linewidth=1)
        ax2.set_title('3D Error Components')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Error (m)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Control Effort
        ax3 = plt.subplot(4, 3, 3)
        ax3.plot(self.metrics['timestamp'], self.metrics['control_effort'], 'purple', linewidth=1)
        ax3.set_title('Control Effort Over Time')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Effort (m/s)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Velocity Commands
        ax4 = plt.subplot(4, 3, 4)
        ax4.plot(self.metrics['timestamp'], self.metrics['control_vx'], 'r-', label='Vx (Lateral)', linewidth=1)
        ax4.plot(self.metrics['timestamp'], self.metrics['control_vy'], 'g-', label='Vy (Forward)', linewidth=1)
        ax4.plot(self.metrics['timestamp'], self.metrics['control_vz'], 'b-', label='Vz (Vertical)', linewidth=1)
        ax4.set_title('Velocity Commands')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Velocity (m/s)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Distance Estimation
        ax5 = plt.subplot(4, 3, 5)
        if any(d > 0 for d in self.metrics['distance_estimate']):
            ax5.plot(self.metrics['timestamp'], self.metrics['distance_estimate'], 'orange', 
                    label='Estimated', linewidth=1, alpha=0.7)
        if any(d > 0 for d in self.metrics['true_distance']):
            ax5.plot(self.metrics['timestamp'], self.metrics['true_distance'], 'black', 
                    label='True', linewidth=2, alpha=0.5)
        ax5.set_title('Distance Estimation')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Distance (m)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. FPS
        ax6 = plt.subplot(4, 3, 6)
        if len(self.metrics['fps']) > 1:
            ax6.plot(self.metrics['timestamp'][1:], self.metrics['fps'][1:], 'brown', linewidth=1)
            ax6.axhline(y=np.mean(self.metrics['fps'][1:]), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(self.metrics["fps"][1:]):.1f}')
        ax6.set_title('Frames Per Second')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('FPS')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Bounding Box Area
        ax7 = plt.subplot(4, 3, 7)
        ax7.plot(self.metrics['timestamp'], self.metrics['bbox_area'], 'teal', linewidth=1)
        ax7.set_title('Bounding Box Area')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Area (pixels)')
        ax7.grid(True, alpha=0.3)
        
        # 8. Altitude
        ax8 = plt.subplot(4, 3, 8)
        ax8.plot(self.metrics['timestamp'], self.metrics['altitude'], 'darkblue', linewidth=1)
        ax8.set_title('Drone Altitude')
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Altitude (m)')
        ax8.grid(True, alpha=0.3)
        
        # 9. Yaw Control
        ax9 = plt.subplot(4, 3, 9)
        ax9.plot(self.metrics['timestamp'], self.metrics['control_yaw'], 'darkgreen', linewidth=1)
        ax9.set_title('Yaw Rate Control')
        ax9.set_xlabel('Time (s)')
        ax9.set_ylabel('Yaw Rate (deg/s)')
        ax9.grid(True, alpha=0.3)
        
        # 10. Tracking State Timeline
        ax10 = plt.subplot(4, 3, 10)
        state_map = {'ACQUIRING': 0, 'TRACKING': 1, 'LOST': 2, 'RETURNING': 3}
        state_numeric = [state_map.get(s, 0) for s in self.metrics['tracking_state']]
        ax10.fill_between(self.metrics['timestamp'], 0, state_numeric, alpha=0.3)
        ax10.plot(self.metrics['timestamp'], state_numeric, 'k-', linewidth=0.5)
        ax10.set_title('Tracking State Timeline')
        ax10.set_xlabel('Time (s)')
        ax10.set_ylabel('State')
        ax10.set_yticks([0, 1, 2, 3])
        ax10.set_yticklabels(['Acquiring', 'Tracking', 'Lost', 'Returning'])
        ax10.grid(True, alpha=0.3)
        
        # 11. Error Histogram
        ax11 = plt.subplot(4, 3, 11)
        ax11.hist(self.metrics['position_error'], bins=50, alpha=0.7, edgecolor='black')
        ax11.axvline(x=np.mean(self.metrics['position_error']), color='r', 
                    linestyle='--', label=f'Mean: {np.mean(self.metrics["position_error"]):.2f}')
        ax11.set_title('Position Error Distribution')
        ax11.set_xlabel('Error (m)')
        ax11.set_ylabel('Frequency')
        ax11.legend()
        ax11.grid(True, alpha=0.3)
        
        # 12. Cumulative Error
        ax12 = plt.subplot(4, 3, 12)
        cumulative_error = np.cumsum(np.abs(self.metrics['position_error']))
        ax12.plot(self.metrics['timestamp'], cumulative_error, 'darkred', linewidth=2)
        ax12.set_title('Cumulative Position Error')
        ax12.set_xlabel('Time (s)')
        ax12.set_ylabel('Cumulative Error (m)')
        ax12.grid(True, alpha=0.3)
        
        plt.suptitle(f'3D Tracker Performance - {self.scenario} (Test: {self.test_id})', fontsize=16)
        plt.tight_layout()
        
        filename = output_dir / f"performance_plot_{self.test_id}_{self.scenario}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return filename

class EnhancedTrackerTester:
    """Test framework for 3D enhanced tracker"""
    def __init__(self):
        self.client = None
        self.results = []
        self.output_dir = Path("test_results") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def connect(self):
        """Connect to AirSim"""
        print("Connecting to AirSim...")
        self.client = airsim.MultirotorClient(port=PORT)
        self.client.confirmConnection()
        self.client.enableApiControl(True, DRONE)
        self.client.armDisarm(True, DRONE)
        print("Connected successfully!")
        
    def setup_scenario(self, scenario):
        """Setup test scenario"""
        print(f"\nSetting up scenario: {scenario}")
        
        # Reset simulation
        self.client.reset()
        self.client.enableApiControl(True, DRONE)
        self.client.armDisarm(True, DRONE)
        time.sleep(2)
        
        # Takeoff
        self.client.takeoffAsync(vehicle_name=DRONE).join()
        self.client.moveToZAsync(-40, 2.0, vehicle_name=DRONE).join()
        
        # Place target based on scenario
        if scenario == "straight_line":
            start_pose = airsim.Pose(airsim.Vector3r(20, 0, 0), airsim.to_quaternion(0, 0, 0))
            self.client.simSetVehiclePose(start_pose, True, TARGET)
            
        elif scenario == "circle":
            start_pose = airsim.Pose(airsim.Vector3r(15, 0, 0), airsim.to_quaternion(0, 0, 0))
            self.client.simSetVehiclePose(start_pose, True, TARGET)
            
        elif scenario == "zigzag":
            start_pose = airsim.Pose(airsim.Vector3r(15, 5, 0), airsim.to_quaternion(0, 0, 0))
            self.client.simSetVehiclePose(start_pose, True, TARGET)
            
        elif scenario == "sudden_stop":
            start_pose = airsim.Pose(airsim.Vector3r(20, 0, 0), airsim.to_quaternion(0, 0, 0))
            self.client.simSetVehiclePose(start_pose, True, TARGET)
            
        elif scenario == "occlusion":
            start_pose = airsim.Pose(airsim.Vector3r(15, 0, 0), airsim.to_quaternion(0, 0, 0))
            self.client.simSetVehiclePose(start_pose, True, TARGET)
            
        elif scenario == "climbing":
            start_pose = airsim.Pose(airsim.Vector3r(15, 0, 5), airsim.to_quaternion(0, 0, 0))
            self.client.simSetVehiclePose(start_pose, True, TARGET)
        
        time.sleep(1)
        print(f"Scenario '{scenario}' setup complete")
        
    def get_frame(self):
        """Get frame from drone camera"""
        png = self.client.simGetImage(CAM, airsim.ImageType.Scene, vehicle_name=DRONE)
        if not png:
            return None
        return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    
    def start_target_movement(self, scenario):
        """Start target movement based on scenario"""
        print(f"Starting target movement: {scenario}")
        
        if scenario == "straight_line":
            self.client.moveByVelocityAsync(vx=4, vy=0, vz=0, duration=TEST_DURATION, vehicle_name=TARGET)
            
        elif scenario == "circle":
            # Circular motion - we'll handle this in the test loop
            pass
            
        elif scenario == "zigzag":
            # Zigzag pattern
            pass
            
        elif scenario == "sudden_stop":
            # Move then stop suddenly
            self.client.moveByVelocityAsync(vx=5, vy=0, vz=0, duration=TEST_DURATION/3, vehicle_name=TARGET)
            
        elif scenario == "occlusion":
            self.client.moveByVelocityAsync(vx=3, vy=2, vz=0, duration=TEST_DURATION, vehicle_name=TARGET)
            
        elif scenario == "climbing":
            self.client.moveByVelocityAsync(vx=3, vy=0, vz=1, duration=TEST_DURATION, vehicle_name=TARGET)
    
    def run_single_test(self, test_id, scenario):
        """Run a single test with the enhanced tracker"""
        print(f"\n{'='*60}")
        print(f"Starting Test {test_id}: {scenario}")
        print(f"{'='*60}")
        
        # Import enhanced tracker here to avoid circular imports
        try:
            # Assuming enhanced tracker is in a separate file
            from enhanced_tracker import DroneTracker
        except ImportError:
            print("Warning: Enhanced tracker not found. Using mock tracker for testing.")
            # Create a mock tracker for demonstration
            class MockTracker:
                def __init__(self):
                    self.state = TrackingState.ACQUIRING
                    self.bbox = None
                    self.distance_estimate = 0
                    
                def connect(self):
                    print("Mock tracker connected")
                    
                def initialize_tracker(self, frame, roi):
                    self.state = TrackingState.TRACKING
                    self.bbox = roi
                    return True
                    
                def process_frame(self, frame):
                    if self.state == TrackingState.TRACKING:
                        # Simulate some tracking
                        return self.bbox, 15.0, True
                    return None, None, False
                    
                def calculate_3d_control(self, bbox, frame_shape, distance):
                    # Simulate control outputs
                    return 0.5, 2.0, 0.1, 5.0, distance
            
            DroneTracker = MockTracker
        
        # Initialize tracker
        tracker = DroneTracker()
        tracker.connect()
        
        # Initialize metrics
        metrics = PerformanceMetrics(test_id, scenario)
        metrics.start_recording()
        
        # Get initial frame and select ROI
        print("Select target ROI for tracking...")
        frame = self.get_frame()
        if frame is None:
            print("Failed to get initial frame")
            return None
            
        cv2.imshow("Select Target", frame)
        roi = cv2.selectROI("Select Target", frame, False)
        cv2.destroyAllWindows()
        
        if roi == (0, 0, 0, 0):
            print("No ROI selected, skipping test")
            return None
            
        # Initialize tracker
        if not tracker.initialize_tracker(frame, roi):
            print("Failed to initialize tracker")
            return None
            
        # Start target movement
        self.start_target_movement(scenario)
        
        # Main test loop
        start_time = time.time()
        last_frame_time = start_time
        frame_times = []
        
        print(f"\nStarting {TEST_DURATION} second tracking test...")
        print("Press 'q' in the OpenCV window to abort early")
        
        while time.time() - start_time < TEST_DURATION:
            frame_start = time.time()
            
            # Get current frame
            frame = self.get_frame()
            if frame is None:
                continue
                
            # Calculate FPS
            current_time = time.time()
            frame_duration = current_time - last_frame_time
            fps = 1.0 / frame_duration if frame_duration > 0 else 0
            last_frame_time = current_time
            frame_times.append(frame_duration)
            
            # Get ground truth positions
            drone_pose = self.client.simGetVehiclePose(DRONE)
            target_pose = self.client.simGetVehiclePose(TARGET)
            
            # Calculate ground truth metrics
            true_distance = math.sqrt(
                (drone_pose.position.x_val - target_pose.position.x_val)**2 +
                (drone_pose.position.y_val - target_pose.position.y_val)**2 +
                (drone_pose.position.z_val - target_pose.position.z_val)**2
            )
            
            lateral_error = drone_pose.position.y_val - target_pose.position.y_val
            vertical_error = -drone_pose.position.z_val - target_pose.position.z_val
            forward_error = drone_pose.position.x_val - target_pose.position.x_val
            
            # Process frame with tracker
            bbox, distance_estimate, tracking_ok = tracker.process_frame(frame)
            
            # Get control outputs
            control_vx, control_vy, control_vz, control_yaw = 0, 0, 0, 0
            if tracking_ok and bbox is not None:
                # Get 3D control from tracker
                vx, vy, vz, yaw_rate, filtered_dist = tracker.calculate_3d_control(bbox, frame.shape, distance_estimate)
                control_vx, control_vy, control_vz, control_yaw = vy, vx, -vz, yaw_rate  # Convert to AirSim frame
                tracking_state = "TRACKING"
                bbox_area = bbox[2] * bbox[3] if len(bbox) >= 4 else 0
                target_in_frame = 1
            else:
                tracking_state = tracker.state.name if hasattr(tracker, 'state') else "LOST"
                bbox_area = 0
                distance_estimate = 0
                target_in_frame = 0
            
            # Calculate control effort
            control_effort = abs(control_vx) + abs(control_vy) + abs(control_vz) + abs(control_yaw) * 0.1
            
            # Calculate yaw error (simplified)
            yaw_error = abs(control_yaw) * 0.1  # Simplified calculation
            
            # Record metrics
            metrics.record_frame(
                tracking_state=tracking_state,
                position_error=true_distance,
                lateral_error=lateral_error,
                vertical_error=vertical_error,
                forward_error=forward_error,
                control_effort=control_effort,
                yaw_error=yaw_error,
                bbox_area=bbox_area,
                distance_estimate=distance_estimate,
                true_distance=true_distance,
                fps=fps,
                control_vx=control_vx,
                control_vy=control_vy,
                control_vz=control_vz,
                control_yaw=control_yaw,
                altitude=-drone_pose.position.z_val,  # Convert NED to altitude
                target_in_frame=target_in_frame,
            )
            
            # Send control command if tracking
            if tracking_ok:
                self.client.moveByVelocityBodyFrameAsync(
                    vx=float(control_vy),
                    vy=float(control_vx),
                    vz=float(control_vz),
                    duration=0.05,
                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=float(control_yaw)),
                    vehicle_name=DRONE
                )
            
            # Display
            if frame is not None:
                # Add HUD overlay
                overlay = frame.copy()
                cv2.putText(overlay, f"State: {tracking_state}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(overlay, f"Error: {true_distance:.1f}m", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if bbox is not None:
                    x, y, w, h = map(int, bbox)
                    cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                cv2.imshow("Tracking Test", overlay)
                
                # Check for early exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nTest aborted by user")
                    break
            
            # Maintain frame rate
            elapsed = time.time() - frame_start
            if elapsed < 0.05:  # ~20 FPS
                time.sleep(0.05 - elapsed)
        
        # Stop target and clean up
        self.client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=1, vehicle_name=TARGET)
        cv2.destroyAllWindows()
        
        # Calculate and store results
        summary = metrics.calculate_performance_summary()
        if summary:
            summary['mean_frame_time'] = float(np.mean(frame_times)) if frame_times else 0
            summary['std_frame_time'] = float(np.std(frame_times)) if frame_times else 0
            self.results.append(summary)
            
            # Save data
            metrics.save_raw_data(self.output_dir)
            metrics.plot_performance(self.output_dir)
            
            print(f"\nTest {test_id} completed!")
            print(f"Tracking Ratio: {summary['tracking_ratio']:.2%}")
            print(f"Mean Position Error: {summary['mean_position_error']:.2f}m")
            print(f"Mean FPS: {summary['mean_fps']:.1f}")
        
        return summary
    
    def run_all_tests(self):
        """Run all test scenarios"""
        print("\n" + "="*60)
        print("STARTING 3D TRACKER PERFORMANCE TEST SUITE")
        print("="*60)
        
        self.connect()
        
        test_id = 1
        for scenario in TEST_SCENARIOS:
            for repetition in range(NUM_TESTS):
                test_name = f"{scenario}_run{repetition+1}"
                
                # Setup and run test
                self.setup_scenario(scenario)
                result = self.run_single_test(test_name, scenario)
                
                if result:
                    print(f"\nTest {test_name} Summary:")
                    print(f"  Tracking Ratio: {result['tracking_ratio']:.2%}")
                    print(f"  Mean Position Error: {result['mean_position_error']:.2f}m")
                    print(f"  RMSE: {result['rmse']:.2f}m")
                    print(f"  Mean FPS: {result['mean_fps']:.1f}")
                    print(f"  Control Smoothness: {result['control_smoothness']:.3f}")
                    print(f"  Stability Index: {result['stability_index']:.3f}")
                
                test_id += 1
        
        # Analyze and report results
        self.generate_performance_report()
        
        # Cleanup
        self.client.armDisarm(False, DRONE)
        self.client.enableApiControl(False, DRONE)
        
        print("\n" + "="*60)
        print("TEST SUITE COMPLETED")
        print(f"Results saved to: {self.output_dir}")
        print("="*60)
        
        return self.results
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        if not self.results:
            print("No results to analyze")
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        # Save raw results
        results_file = self.output_dir / "test_results_summary.csv"
        df.to_csv(results_file, index=False)
        
        # Calculate overall statistics
        overall_stats = {
            'total_tests': len(df),
            'avg_tracking_ratio': df['tracking_ratio'].mean() * 100,
            'avg_position_error': df['mean_position_error'].mean(),
            'avg_rmse': df['rmse'].mean(),
            'avg_fps': df['mean_fps'].mean(),
            'avg_control_smoothness': df['control_smoothness'].mean(),
            'avg_stability_index': df['stability_index'].mean(),
            'worst_tracking_ratio': df['tracking_ratio'].min() * 100,
            'best_tracking_ratio': df['tracking_ratio'].max() * 100,
            'worst_position_error': df['mean_position_error'].max(),
            'best_position_error': df['mean_position_error'].min(),
        }
        
        # Scenario analysis
        scenario_stats = {}
        for scenario in TEST_SCENARIOS:
            scenario_data = df[df['scenario'] == scenario]
            if not scenario_data.empty:
                scenario_stats[scenario] = {
                    'avg_tracking_ratio': scenario_data['tracking_ratio'].mean() * 100,
                    'avg_position_error': scenario_data['mean_position_error'].mean(),
                    'avg_rmse': scenario_data['rmse'].mean(),
                    'tests_run': len(scenario_data),
                }
        
        # Generate report
        report = {
            'test_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'test_configuration': {
                'test_duration': TEST_DURATION,
                'num_tests_per_scenario': NUM_TESTS,
                'scenarios_tested': TEST_SCENARIOS,
            },
            'overall_performance': overall_stats,
            'scenario_performance': scenario_stats,
            'key_insights': self._generate_insights(df),
            'recommendations': self._generate_recommendations(df),
        }
        
        # Save report
        report_file = self.output_dir / "performance_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate summary plots
        self._generate_summary_plots(df)
        
        # Print summary
        print("\n" + "="*60)
        print("PERFORMANCE REPORT SUMMARY")
        print("="*60)
        print(f"\nOverall Performance:")
        print(f"  Average Tracking Ratio: {overall_stats['avg_tracking_ratio']:.1f}%")
        print(f"  Average Position Error: {overall_stats['avg_position_error']:.2f}m")
        print(f"  Average RMSE: {overall_stats['avg_rmse']:.2f}m")
        print(f"  Average FPS: {overall_stats['avg_fps']:.1f}")
        
        print(f"\nBest/Worst Performance:")
        print(f"  Best Tracking: {overall_stats['best_tracking_ratio']:.1f}%")
        print(f"  Worst Tracking: {overall_stats['worst_tracking_ratio']:.1f}%")
        print(f"  Best Position Error: {overall_stats['best_position_error']:.2f}m")
        print(f"  Worst Position Error: {overall_stats['worst_position_error']:.2f}m")
        
        print(f"\nScenario Performance:")
        for scenario, stats in scenario_stats.items():
            print(f"  {scenario}: {stats['avg_tracking_ratio']:.1f}% tracking, {stats['avg_position_error']:.2f}m error")
        
        print(f"\nDetailed report saved to: {report_file}")
    
    def _generate_insights(self, df):
        """Generate insights from test results"""
        insights = []
        
        # Tracking ratio insights
        avg_tracking = df['tracking_ratio'].mean()
        if avg_tracking > 0.9:
            insights.append("Excellent tracking performance with over 90% success rate")
        elif avg_tracking > 0.7:
            insights.append("Good tracking performance with room for improvement")
        else:
            insights.append("Tracking performance needs significant improvement")
        
        # Error analysis
        avg_error = df['mean_position_error'].mean()
        if avg_error < 2.0:
            insights.append("Very accurate positioning with low error")
        elif avg_error < 5.0:
            insights.append("Reasonable positioning accuracy")
        else:
            insights.append("Positioning accuracy needs improvement")
        
        # FPS analysis
        avg_fps = df['mean_fps'].mean()
        if avg_fps > 15:
            insights.append("Good real-time performance")
        elif avg_fps > 10:
            insights.append("Acceptable real-time performance")
        else:
            insights.append("Frame rate is below desired real-time performance")
        
        # Scenario-specific insights
        for scenario in TEST_SCENARIOS:
            scenario_data = df[df['scenario'] == scenario]
            if not scenario_data.empty:
                scenario_tracking = scenario_data['tracking_ratio'].mean()
                if scenario_tracking < 0.5:
                    insights.append(f"Struggles with {scenario} scenario ({(scenario_tracking*100):.0f}% tracking)")
        
        return insights
    
    def _generate_recommendations(self, df):
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Based on tracking ratio
        avg_tracking = df['tracking_ratio'].mean()
        if avg_tracking < 0.8:
            recommendations.append("Improve tracker robustness for occlusions and fast movements")
            recommendations.append("Implement better reacquisition algorithms for lost targets")
        
        # Based on position error
        avg_error = df['mean_position_error'].mean()
        if avg_error > 3.0:
            recommendations.append("Tune PID controllers for better position accuracy")
            recommendations.append("Improve depth estimation accuracy")
        
        # Based on frame rate
        avg_fps = df['mean_fps'].mean()
        if avg_fps < 15:
            recommendations.append("Optimize image processing pipeline for higher FPS")
            recommendations.append("Consider using a lighter tracker algorithm")
        
        # Based on control smoothness
        avg_smoothness = df['control_smoothness'].mean()
        if avg_smoothness < 0.5:
            recommendations.append("Add more filtering to control outputs for smoother movements")
            recommendations.append("Implement rate limiting on control commands")
        
        return recommendations
    
    def _generate_summary_plots(self, df):
        """Generate summary plots"""
        # 1. Tracking Performance by Scenario
        plt.figure(figsize=(12, 8))
        
        # Tracking Ratio by Scenario
        plt.subplot(2, 2, 1)
        scenario_groups = df.groupby('scenario')['tracking_ratio'].mean()
        scenario_groups.plot(kind='bar', color='skyblue')
        plt.title('Average Tracking Ratio by Scenario')
        plt.xlabel('Scenario')
        plt.ylabel('Tracking Ratio')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Position Error by Scenario
        plt.subplot(2, 2, 2)
        error_by_scenario = df.groupby('scenario')['mean_position_error'].mean()
        error_by_scenario.plot(kind='bar', color='lightcoral')
        plt.title('Average Position Error by Scenario')
        plt.xlabel('Scenario')
        plt.ylabel('Error (m)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # FPS Distribution
        plt.subplot(2, 2, 3)
        plt.hist(df['mean_fps'], bins=20, alpha=0.7, edgecolor='black', color='lightgreen')
        plt.axvline(x=df['mean_fps'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["mean_fps"].mean():.1f}')
        plt.title('FPS Distribution Across Tests')
        plt.xlabel('Frames Per Second')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Control Smoothness vs Tracking Ratio
        plt.subplot(2, 2, 4)
        plt.scatter(df['control_smoothness'], df['tracking_ratio'], alpha=0.6, c=df['mean_position_error'], 
                   cmap='viridis', s=100)
        plt.colorbar(label='Position Error (m)')
        plt.title('Control Smoothness vs Tracking Performance')
        plt.xlabel('Control Smoothness (higher = smoother)')
        plt.ylabel('Tracking Ratio')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle('3D Tracker Performance Summary', fontsize=16)
        plt.tight_layout()
        
        summary_plot = self.output_dir / "performance_summary.png"
        plt.savefig(summary_plot, dpi=150, bbox_inches='tight')
        plt.close()

def main():
    """Main function to run the test suite"""
    print("3D Enhanced Tracker Performance Testing Framework")
    print("-" * 50)
    
    # Create tester instance
    tester = EnhancedTrackerTester()
    
    try:
        # Run all tests
        results = tester.run_all_tests()
        
        if results:
            print(f"\nTesting completed successfully!")
            print(f"Total tests run: {len(results)}")
            print(f"Results directory: {tester.output_dir}")
            
            # Quick summary
            df = pd.DataFrame(results)
            print(f"\nQuick Summary:")
            print(f"Average Tracking Success: {df['tracking_ratio'].mean()*100:.1f}%")
            print(f"Average Position Error: {df['mean_position_error'].mean():.2f}m")
            print(f"Average FPS: {df['mean_fps'].mean():.1f}")
            
        else:
            print("No tests were completed successfully")
            
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user")
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure cleanup
        try:
            if tester.client:
                tester.client.armDisarm(False, DRONE)
                tester.client.enableApiControl(False, DRONE)
                print("\nCleaned up AirSim connection")
        except:
            pass
        
        print("\nTest framework shutdown complete")

if __name__ == "__main__":
    main()