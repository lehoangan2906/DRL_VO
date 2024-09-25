#!/usr/bin/python3

# Usage: This script is modified to be compatible with ROS2 and handle Oradar MS200 lidar data.
# Informations published:
# - pedestrian position costmap (ped_pos_map):
#      + Format: float32[] (flattened 2-channel, 80x80 grid).
#      + Description: Represents a Cartesian velocity map of pedestrians in the robot's FOV.
#                     This is a 2-channel map (80x80), where one channel holds the pedestrians' position
#                     The other channel holds the pedestrians' velocity in the x and z directions.
#
# - Front-Facing Lidar Scan History (scan):
#      + Format: float32[] (a list of historical front-facing lidar scans, each 240 points).
#      + Description: The historical list of lidar scans within the front-facing 192-degree 
#                     field of view (-96 to 96 degrees), which is stored in self.lidar_history 
#                     and published as a flattened array.
#  
# - Full 360-Degree lidar scan data (scan_all):
#      + Format: float32[] (450 lidar points for the full 360-degree scan).
#      + Description: The full 360-degree lidar scan data, which includes all 450 lidar points from
#                     the scan_all array.
#
# - Current Goal Position in Cartesian Coordinates (goal_cart):
#      + Format: float32[] (size 2).
#      + Description: The final goal of the robot in polar coordinates (distance and angle from the
#                     robot to the goal).
#
# - Current Velocity of the robot (vel):
#      + Format: float32[] (size 2).
#      + Description: The current velocity of the robot, where the first element is linear velocity
#                     and the second element is angular velocity.
#
# - Depth Image Data (depth):
#      + Format: float32[] (flatttened depth image data).
#      + Description: The depth data from the Intel RealSense camera, flattened into a 1D array.
#
# - Grayscale Image Data (image_gray):
#      + Format: float32[] (flattened grayscale image data).
#      + Description: The grayscale image data from the Intel RealSense camera, also flattened
#                     into a 1D array.


import cv2
import rclpy
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge
from cnn_msgs.msg import CnnData
from scipy.interpolate import interp1d
from geometry_msgs.msg import Point, Twist
from sensor_msgs.msg import LaserScan, Image
from track_ped_msgs.msg import TrackedPersons

# Lidar specifications
LIDAR_FOV = 360
LIDAR_RESOLUTION = 0.8
LIDAR_MIN_RANGE = 0.03
LIDAR_MAX_RANGE = 12.0
LIDAR_FRAME_RATE = 10  # Frame rate of lidar scans in Hz
NUM_LIDAR_BEAMS = int(LIDAR_FOV / LIDAR_RESOLUTION)  # 450 scans per frame

# Time steps for collecting historical data
NUM_TP = 10  # Number of historical lidar scans
# WINDOW_SIZE = 3  # Sliding window size for pooling methods
NUM_FRONT_FACING_POINTS = (
    240  # Number of front-facing lidar scans (-96 degree to 96 degree FOV)
)

# Front-facing lidar scan indices for 192-degree FOV (wrapping around 0 degrees)
FRONT_FACING_PART_1_START = 330  # index corresponding to -96 degrees
FRONT_FACING_PART_1_END = 449  # end of the lidar array (360 degrees)
FRONT_FACING_PART_2_START = 0  # start of lidar array (1 degrees)
FRONT_FACING_PART_2_END = 120  # index corresponding to +96 degrees


# Number of total pedestrians (including the robot)
NUM_PEDS = 3 + 1


class CnnDataNode(Node):
    def __init__(self):
        super().__init__("cnn_data_pub_node")

        # Initialize the data for publishing
        self.ped_pos_map_tmp = np.zeros(
            (2, 80, 80)
        )  # Cartesian velocity map for pedestrians
        self.scan_front = np.zeros(
            NUM_FRONT_FACING_POINTS
        )  # Store processed lidar data for the front-facing 192 degree FOV
        self.scan_all = np.zeros(
            NUM_LIDAR_BEAMS
        )  # Store the full 360 degree lidar scan data
        self.goal_cart = np.zeros(2)
        self.vel = np.zeros(2)
        self.image_gray = None
        self.depth_image = None
        self.goal_final_polar = np.zeros(2)
        self.lidar_history = []  # Store the 10 most recent lidar scans

        # Bridge for converting ROS messages to OpenCV images
        self.bridge = CvBridge()

        # ROS2 subscriptions and publishers
        self.ped_sub = self.create_subscription(
            TrackedPersons, "/track_ped", self.ped_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )
        self.goal_sub = self.create_subscription(
            Point, "/cnn_goal", self.goal_callback, 10
        )
        self.vel_sub = self.create_subscription(
            Twist, "/mobile_base/commands/velocity", self.vel_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, "intel_realsense_d415_depth/depth/image_raw", self.depth_callback, 10
        )
        self.image_sub = self.create_subscription(
            Image, "/intel_realsense_d415_depth/image_raw", self.image_callback, 10
        )

        self.cnn_data_pub = self.create_publisher(CnnData, "/cnn_data", 10)

        # Timer for controlling data publishing rate
        self.rate = LIDAR_FRAME_RATE  # 10 Hz
        self.timer = self.create_timer(1.0 / self.rate, self.timer_callback)

    # Callback to process pedestrian data and generate the pedestrian position map
    def ped_callback(self, trackPed_msg):
        # Get the pedestrian's position
        self.ped_pos_map_tmp = np.zeros((2, 80, 80))  ## Reset to clear old data

        if len(trackPed_msg.tracks) > 0:
            for ped in trackPed_msg.tracks:
                x = ped.pose.pose.position.x
                z = ped.pose.pose.position.z
                vx = ped.twist.twist.linear.x
                vz = ped.twist.twist.linear.z

                # 3m x 3m occupancy grid map:
                if x >= 0 and x <= 3 and np.abs(z) <= 10:
                    col = int(np.floor(-(z - 10) / 0.25))
                    row = int(np.floor(x / 0.25))

                    # ensure bins are within valid range
                    row = min(row, 79)
                    col = min(col, 79)

                    # Update the cartesian velocity map with pedestrian's velocity
                    self.ped_pos_map_tmp[0, row, col] = vx  # velocity in x - direction
                    self.ped_pos_map_tmp[1, row, col] = vz  # velocity in z - direction

    # Function to update the lidar history with new scans
    def update_lidar_history(self, new_scan):
        if len(self.lidar_history) >= NUM_TP:
            # Remove the oldest scan if history already has 10 scans
            self.lidar_history.pop(0)

        # Append the new scan
        self.lidar_history.append(new_scan)

    # Callback function for lidar scan data
    def scan_callback(self, laserScan_msg):
        # Extract lidar ranges data from the ROS topic message as a numpy array
        scan_data = np.array(laserScan_msg.ranges, dtype=np.float32)

        # Filter for valid points by replacing invalid data (NaN and Inf) with 0.0
        scan_data[np.isnan(scan_data)] = 0.0
        scan_data[np.isinf(scan_data)] = 0.0

        # Create a mask to identify which points fall within a valid range
        # e.g., valid_mask ==[True, True, False, ...]
        valid_mask = (scan_data > LIDAR_MIN_RANGE) & (scan_data < LIDAR_MAX_RANGE)

        # Check if there are any invalid points (0.0 or out of range values)
        if np.any(~valid_mask):
            # Get the indices where the data is valid (within the lidar's range)
            valid_indices = np.where(valid_mask)[0]

            # Extract only the valid lidar scan data points
            valid_scan_data = scan_data[valid_mask]

            # Interpolate the invalid (missing) points using linear interpolation
            # `interp1d` creates a function that will interpolate values based on the valid points
            # valid_indices: indices of valid points
            # valid_scan_data: corresponding valid distance values at those indices
            # kind="linear": linear interpolation method
            # bounds_error=False: allows interpolation for out-of-bound indices without raising an error
            # fill_value="extrapolate": fills missing values by extrapolating where data points are missing
            interpolation = interp1d(
                valid_indices,
                valid_scan_data,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )

            # Use the interpolation function to fill in missing points for all indices
            # np.arange(len(scan_data)) creates an array with all indices from 0 to len(scan_data) - 1
            interpolated_scan = interpolation(np.arange(len(scan_data)))

            # Clip the interpolated scan data to ensure no values exceed the lidar's valid range
            # Anything below 0 or above LIDAR_MAX_RANGE will be limited within that range
            scan_data = np.clip(interpolated_scan, 0, LIDAR_MAX_RANGE)

        # Store the full interpolated scan data (including the front, back, and sides)
        self.scan_all = scan_data

        # Limit the data to the front-facing 192 degrees (for 240 lidar points as mentioned before)
        front_facing_part_1 = scan_data[
            FRONT_FACING_PART_1_START : FRONT_FACING_PART_1_END + 1
        ]
        front_facing_part_2 = scan_data[
            FRONT_FACING_PART_2_START : FRONT_FACING_PART_2_END + 1
        ]

        # Combine both front-facing sections to form the complete front-facing scan
        front_facing_scan = np.concatenate(
            (front_facing_part_1, front_facing_part_2), axis=0
        )

        # Store the front-facing scan data
        self.scan_front = front_facing_scan

        # Add the new scan to the lidar history
        self.update_lidar_history(self.scan_front)

    # Callback function for goal position in polar coordinate
    def goal_callback(self, goal_msg):
        self.goal_cart[0] = goal_msg.x
        self.goal_cart[1] = goal_msg.y
        self.goal_final_polar[0] = np.sqrt(goal_msg.x**2 + goal_msg.y**2)  # Distance to goal
        self.goal_final_polar[1] = np.arctan2(goal_msg.y, goal_msg.x)  # Angle to goal

    def vel_callback(self, vel_msg):
        self.vel[0] = vel_msg.linear.x
        self.vel[1] = vel_msg.angular.z

    def timer_callback(self):
        # Check if lidar history has the required number of scans (10) and if scan_front is valid
        if len(self.lidar_history) < NUM_TP or self.scan_front is None:
            self.get_logger().info("Not enough lidar data to publish")
            return  # Skip publishing if required data is missing

        # Check if pedestrian data is valid
        if self.ped_pos_map_tmp is None:
            self.get_logger().info("Pedestrian data is missing, skipping publishing")
            return  # Skip publishing if pedestrian data is missing

        # Initialize the custom message
        cnn_data = CnnData()

        # Publish the pedestrian position map
        cnn_data.ped_pos_map = [
            float(val)
            for sublist in self.ped_pos_map_tmp
            for subb in sublist
            for val in subb
        ]

        # Publish the historical list of front-facing 240 points lidar scans
        cnn_data.scan = [float(val) for scan in self.scan_history for val in scan]

        # Publish the full 360 degree scan (450 points)
        cnn_data.scan_all = self.scan_all.tolist()

        # Publish the goal position
        cnn_data.goal_cart = self.goal_cart.tolist()  # Goal in cartesian coordinates
        cnn_data.goal_final_polar = (self.goal_final_polar.tolist())  # Goal in polar coordinates

        # Publish the robot's velocity
        cnn_data.vel = self.vel.tolist()

        # Publish the depth image data
        cnn_data.depth = (
            self.depth_image.flatten().tolist() if self.depth_image is not None else []
        )
        cnn_data.image_gray = (
            self.image_gray.flatten().tolist() if self.image_gray is not None else []
        )

        # Publish the CNN data
        self.cnn_data_pub.publish(cnn_data)


def main(args=None):
    rclpy.init(args=args)
    node = CnnDataNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
