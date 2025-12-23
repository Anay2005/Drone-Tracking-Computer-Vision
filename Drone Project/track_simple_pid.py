import time
import math
import cv2
import numpy as np
import airsim
from airsim import Pose, Vector3r, to_quaternion



PORT = 41451
DRONE = "Drone1"
TARGET = "Target1"
CAM = "front_rgb"

# Altitude and initial gap ahead of the tracker
ALT = 50.0       #  Both tracker and target stay at this altitude
GAP = 12.0       # target spawns this far ahead

# 0° is +X, positive angles rotate counter‑clockwise
HEADING_DEG = 0.0


TARGET_SPEED = 4.0  # m/s
DES_SEP = 6.0       # desired separation behind target (m)


HZ = 20.0
KP_POS = 0.8
# vertical gain
KP_Z = 0.8  

# yaw control
K_YAW = 3.5
MAX_SPEED = 7.0
MAX_VZ = 2.0
MAX_YAWRATE = 100.0
MAX_DISTANCE_METERS = 370.0

# Rate at which the target rotates during a 90‑degree turn (degrees per second)
# A lower value makes a slower but smoother turn The timing of the turn for a 90°
# change is 90 / TURN_RATE_DEG seconds  
# change this if the tracker
# struggles to keep the target within view
TURN_RATE_DEG = 20.0


def clamp(v: float, a: float, b: float) -> float:
    # clamp v between a and b
    return max(a, min(b, v))


def yaw_from_q(q) -> float:
    # get yaw angle in degrees from a quaternion
    siny_cosp = 2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
    cosy_cosp = 1.0 - 2.0 * (q.y_val * q.y_val + q.z_val * q.z_val)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def wrap_deg(theta: float) -> float:
    # wrap the angle between (-180,180)
    while theta > 180:
        theta -= 360
    while theta <= -180:
        theta += 360
    return theta


def get_png(client: airsim.MultirotorClient, cam: str, veh: str):
    # get image from the specified camera on the drone
    png = client.simGetImage(cam, airsim.ImageType.Scene, vehicle_name=veh)
    return None if not png else cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)


def csrt() -> cv2.Tracker:
    # create a CSRT tracker
    # Depending on OpenCV version, the CSRT tracker resides in different modules
    if hasattr(cv2, "legacy"):
        return cv2.legacy.TrackerCSRT_create()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    raise RuntimeError("CSRT tracker is unavailable; install opencv-contrib-python")


def main() -> None:
    
    dt = 1.0 / HZ

    # calculate the initital unit vector from the heading
    heading_rad = math.radians(HEADING_DEG)
    ux, uy = math.cos(heading_rad), math.sin(heading_rad)

    # Build the target path: list of (segment_length, turn_direction)
    # The sequence: move 30 m, turn left; move 10 m, turn right
    # move 40 m, turn right move 10 m, turn left then move straight for the final segment.
    base_segments = [
        (30.0, "right"),
        (5.0,  "left"),
        (5.0,  "left"),
        (5.0,  "right"),
    ]
    total_base = sum(s[0] for s in base_segments)
    final_len = max(0.0, MAX_DISTANCE_METERS - total_base)
    # Append final segment with no turn (None) if positive length
    turn_segments = list(base_segments)
    if final_len > 1e-3:
        turn_segments.append((final_len, None))

    # Track current segment index and distance travelled by target
    segment_idx = 0
    segment_start_distance = 0.0
    distance_target_travelled = 0.0
    prev_target_pos = None 

    # This will be updated as the target turns
    target_heading_deg = HEADING_DEG

    # whether the target is currently performing a smooth turn
    turning = False
    turn_direction = None  # left or right when turning
    turn_remaining_deg = 0.0
    target_heading_end = target_heading_deg  # final heading after turn completes

    # connect to AirSim client
    c = airsim.MultirotorClient(port=PORT)
    c.confirmConnection()

    # Tracker drone setup
    c.enableApiControl(True, DRONE)
    c.armDisarm(True, DRONE)
    c.takeoffAsync(vehicle_name=DRONE).join()
    c.moveToZAsync(-ALT, 2.0, vehicle_name=DRONE).join()  # NED: -ALT is up

    # Target drone setup
    p0 = c.simGetVehiclePose(DRONE).position
    x0, y0 = p0.x_val, p0.y_val
    tx0, ty0, tz0 = x0 + ux * GAP, y0 + uy * GAP, -ALT  # same altitude as tracker
    c.simAddVehicle(TARGET, "SimpleFlight",
                    Pose(Vector3r(tx0, ty0, tz0), to_quaternion(0, 0, heading_rad)), "")

    c.enableApiControl(True, TARGET)
    c.armDisarm(True, TARGET)
    c.takeoffAsync(vehicle_name=TARGET).join()
    c.hoverAsync(vehicle_name=TARGET).join()

    # Face the target to help draw ROI for visual tracker
    yaw_des = math.degrees(math.atan2(ty0 - y0, tx0 - x0))
    c.moveByVelocityBodyFrameAsync(0, 0, 0, 0.5,
        airsim.YawMode(is_rate=False, yaw_or_rate=yaw_des),
        vehicle_name=DRONE).join()

   
    cv2.namedWindow("live", cv2.WINDOW_AUTOSIZE)
    print("Controls: press 't' to draw ROI on the target, then ENTER; 'q' quits.")
    tracker = None
    have_roi = False
    total_distance = 0.0  # distance travelled by tracker
    last_tracker_pos = None  

    # Pause both drones until ROI is selected
    c.hoverAsync(vehicle_name=DRONE).join()
    c.hoverAsync(vehicle_name=TARGET).join()

    while True:
        # Acquire frame from tracker camera
        frame = get_png(c, CAM, DRONE)
        if frame is None:
            time.sleep(0.02)
            continue
        H, W = frame.shape[:2]

        # Get states of both drones
        ds = c.getMultirotorState(DRONE)
        ts = c.getMultirotorState(TARGET)

        # Positions in world frame
        px, py, pz = (ds.kinematics_estimated.position.x_val,
                      ds.kinematics_estimated.position.y_val,
                      ds.kinematics_estimated.position.z_val)
        tx, ty = (ts.kinematics_estimated.position.x_val,
                  ts.kinematics_estimated.position.y_val)

        # Start target motion only after ROI is not none 
        if have_roi:
            # Update distance travelled by target 
            if prev_target_pos is not None:
                dx_t = tx - prev_target_pos.x_val
                dy_t = ty - prev_target_pos.y_val
                distance_target_travelled += math.hypot(dx_t, dy_t)
            prev_target_pos = ts.kinematics_estimated.position

            # Turning for the target
            if turning:
                # Gradually change heading at a constant rate
                delta_yaw = TURN_RATE_DEG * dt
                if turn_direction and turn_direction.lower() == 'right':
                    delta_yaw = -delta_yaw
                target_heading_deg += delta_yaw
                turn_remaining_deg -= abs(delta_yaw)
                # Update unit direction vector
                heading_rad = math.radians(target_heading_deg)
                ux, uy = math.cos(heading_rad), math.sin(heading_rad)
                # Finish the turn when the remaining angle is 0
                if turn_remaining_deg <= 0.0:
                    turning = False
                    # clamp to the final heading
                    target_heading_deg = wrap_deg(target_heading_end)
                    heading_rad = math.radians(target_heading_deg)
                    ux, uy = math.cos(heading_rad), math.sin(heading_rad)
                    # Move on to the next segment and reset the distance 
                    segment_idx += 1
                    segment_start_distance = distance_target_travelled
            else:
                # Check whether we have reached the end of the current straight segment
                if segment_idx < len(turn_segments):
                    seg_len, seg_turn_dir = turn_segments[segment_idx]
                    if distance_target_travelled - segment_start_distance >= seg_len:
                        if seg_turn_dir is not None:
                            # Begin a new smooth turn
                            turning = True
                            turn_direction = seg_turn_dir
                            turn_remaining_deg = 90.0
                            # Determine the target's final heading after the turn
                            if seg_turn_dir.lower() == 'left':
                                target_heading_end = wrap_deg(target_heading_deg + 90.0)
                            else:
                                target_heading_end = wrap_deg(target_heading_deg - 90.0)
                        else:
                            # No turn at this final straight segment
                            segment_idx += 1
                            segment_start_distance = distance_target_travelled

            # make the target to move along its heading If turning
            # specify yaw rate otherwise fixed yaw
            if turning:
                yaw_rate_cmd = TURN_RATE_DEG if (turn_direction and turn_direction.lower() == 'left') else -TURN_RATE_DEG
                c.moveByVelocityAsync(
                    vx=TARGET_SPEED * ux,
                    vy=TARGET_SPEED * uy,
                    vz=0,
                    duration=1.0,
                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate_cmd),
                    vehicle_name=TARGET
                )
            else:
                c.moveByVelocityAsync(
                    vx=TARGET_SPEED * ux,
                    vy=TARGET_SPEED * uy,
                    vz=0,
                    duration=1.0,
                    yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=target_heading_deg),
                    vehicle_name=TARGET
                )

            # printing distance travelled by tracker on console
            if last_tracker_pos is not None:
                dx_p = px - last_tracker_pos.x_val
                dy_p = py - last_tracker_pos.y_val
                total_distance += math.hypot(dx_p, dy_p)
            last_tracker_pos = ds.kinematics_estimated.position
            if total_distance >= MAX_DISTANCE_METERS:
                print(f"Tracker reached {total_distance:.1f} m. Stopping.")
                break

        # calculate position behind target
        # world coordinates a fixed distance behind the target along its current heading
        des_x, des_y = tx - ux * DES_SEP, ty - uy * DES_SEP
        ex, ey = des_x - px, des_y - py

        vx_w = clamp(KP_POS * ex, -MAX_SPEED, MAX_SPEED)
        vy_w = clamp(KP_POS * ey, -MAX_SPEED, MAX_SPEED)
        # Compute yaw error between drone and target to change camera direction
        yaw_dr = yaw_from_q(ds.kinematics_estimated.orientation)
        yaw_tg = math.degrees(math.atan2(ty - py, tx - px))
        yaw_err = wrap_deg(yaw_tg - yaw_dr)
        yaw_rate = clamp(K_YAW * yaw_err, -MAX_YAWRATE, MAX_YAWRATE)

        c.moveByVelocityZAsync(
            vx=vx_w,
            vy=vy_w,
            z=-ALT,
            duration=1.0,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate),
            vehicle_name=DRONE
        )

        # Update Tracker
        if have_roi and tracker is not None:
            ok, bbox = tracker.update(frame)
            if ok:
                x, y, w_, h_ = map(int, bbox)
                cv2.rectangle(frame, (x, y), (x + w_, y + h_), (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Tracking lost (press t to reselect)",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw travel distance on screen
        cv2.putText(frame, f"Dist {total_distance:.1f} m", (W - 180, H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("live", frame)

        # Handle keyboard input
        k = cv2.waitKey(int(dt * 1000)) & 0xFF
        if k == ord('q') or k == 27:
            break
        elif k == ord('t'):
            # Freeze both drones while selecting ROI
            c.hoverAsync(vehicle_name=DRONE).join()
            c.hoverAsync(vehicle_name=TARGET).join()
            time.sleep(0.05)  # small settle time
            frame2 = get_png(c, CAM, DRONE)
            roi = cv2.selectROI("live", frame2, fromCenter=False, showCrosshair=True)
            if roi != (0, 0, 0, 0):
                tracker = csrt()
                have_roi = tracker.init(frame2, roi)
                if have_roi:
                    
                    print("ROI accepted; starting target motion.")

                else:
                    print("Tracker init failed")
            # Reset counters and path state when starting new tracking instance
            last_tracker_pos = None
            prev_target_pos = None
            total_distance = 0.0
            distance_target_travelled = 0.0
            segment_idx = 0
            segment_start_distance = 0.0
            # Reset target heading and turning state
            target_heading_deg = HEADING_DEG
            heading_rad = math.radians(target_heading_deg)
            ux, uy = math.cos(heading_rad), math.sin(heading_rad)
            turning = False
            turn_direction = None
            turn_remaining_deg = 0.0
            target_heading_end = target_heading_deg

    # disarm 
  
    c.hoverAsync(vehicle_name=DRONE).join()
    c.hoverAsync(vehicle_name=TARGET).join()
    c.armDisarm(False, DRONE)
    c.enableApiControl(False, DRONE)
    c.armDisarm(False, TARGET)
    c.enableApiControl(False, TARGET)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()