#!/usr/bin/python3

import sys
import os
import numpy as np

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

# ROS2 message types
from sensor_msgs.msg import CnnData
from geometry_msgs.msg import Twist, Point

# DRL and custom modules
from stable_baselines3 import PPO
from custom_cnn_full import CustomCNN


# Global variables and configurations

# Define policy settings for loading the trained model
policy_kwargs = dict(
    features_extractor_class=CustomCNN, # Custom CNN model for feature extraction
    features_extractor_kwargs=dict(features_dim=256), # Output dimension from the CNN
)


# Main inference class definition

class DrlInference(Node):
    """
    ROS2 Node class for performing inference using the trained DRL-VO policy.
    This class subscribes to sensor data, processes it, and uses the DRL model
    to predict the robot's next velocity commands.

    Attributes:
        ped_pos (list): Array to store pedestrian position data.
        scan (list): Array to store LiDAR scan data.
        goal (list): Array to store goal position data.
        model (PPO): The trained DRL model used for inference.
        cnn_data_sub (Subscription): Subscriber for receiving CNN data.
        cmd_vel_pub (Publisher): Publisher for sending velocity commands.
    """

    def __init__(self):
        super().__init__('drl_inference')   # Initialize the ROS2 node with the name 'drl_inference'

        # Initialize data attributes
        self.ped_pos = []   # Stores pedestrian positions   # np.ones((3, 20))*20
        self.scan = []      # Stores LiDAR scan data        # np.zeros((3, 720))
        self.goal = []      # Stores goal position data     # np.zeros((3, 2))
        self.vx = 0
        self.wz = 0
        self.model = None

        # Load the trained model:
        model_file = self.declare_parameter('model_file', './model/drl_vo.zip').get_parameter_value().string_value
        self.model = PPO.load(model_file)   # Load the PPO model from the specified file
        self.get_logger().info("Finish loading model.")

        # Define QoS settings for ROS2 communication
        qos_profile = QoSProfile(depth=10)

        # Initialize ROS2 subscribers and publishers
        self.cnn_data_sub = self.create_subscription(
            CnnData, "/cnn_data", self.cnn_data_callback, qos_profile
        )   # Subscribe to the CNN data topic

        self.cmd_vel_pub = self.create_publisher(
            Twist, '/drl_cmd_vel', qos_profile  # Publish the velocity commands
        )

    def cnn_data_callback(self, cnn_data_msg):
        """
        Callback function that processes the incoming CNN data and performs
        inference to determine the robot's next movement commands.

        Args:
            cnn_data_msg (CNN_data): The incoming CNN data message containing
            pedestrian positions, LiDAR scan data,
            and goal position.
        """

        # Update class attributes with the incoming data
        self.ped_pos = cnn_data_msg.ped_pos_map
        self.scan = cnn_data_msg.scan
        self.goal = cnn_data_msg.goal_cart
        cmd_vel = Twist()   # Initialize the Twist message for velocity commands

        # Process the LiDAR scan data to determine the minimum distance to obstacle
        scan = np.array(self.scan[-540:-180])   # Select the relevant range of scan
        scan = scan[scan != 0]  # Filter out zero values (no obstacle detected)
        min_scan_dist = np.amin(scan) if scan.size != 0 else 10.0    # Calculate the minimum distance


        # Decision logic based on goal proximity and obstacle detection
        if np.linalg.norm(self.goal) <= 0.9:     # check if the robot is close to the foal
            cmd_vel.linear.x = 0    # Stop the linear movement
            cmd_vel.angular.z = 0   # Stop the angular movement
        elif min_scan_dist <= 0.4:  # Check if an obstacle is too close
            cmd_vel.linear.x = 0    # Stop linear movement
            cmd_vel.angular.z = 0.7 # Rotate to avoid the obstacle
        else:
            # Process and scale the sensor data
            self.process_data()

            # Concatenate the processed data into a single observation vector
            self.observation = np.concatenate((self.ped_pos, self.scan, self.goal), axis = None)

            # Use the model to predict the next action based on the observation
            action, _states = self.model.predict(self.observation)

            # Scale the predicted action into velocity commands
            cmd_vel.linear.x = (action[0] + 1) * (0.5 - 0) / 2 + 0  # Scale the linear velocity
            cmd_vel.angular.z = (action[1] + 1) * (2 - (-2)) / 2 - 2 # Scale the angular velocity

        # Publish the velocity commands if they are valid (i.e., not NaN)
        if not np.isnan(cmd_vel.linear.x) and not np.isnan(cmd_vel.angular.z):
            self.cmd_vel_pub.publish(cmd_vel)

    def process_data(self):
        """
        Processes and scales the sensor data (pedestrian positions, LiDAR scan, and goal)
        for input into the DRL model. This function handles the necessary preprocessing 
        steps to transform raw data into the format expected by the model.
        """

        # Scale the pedestrian positions using MaxAbsScaler
        v_min, v_max = -2, 2
        self.ped_pos = np.array(self.ped_pos, dtype=np.float32)  # Convert to float32 for scaling
        self.ped_pos = 2 * (self.ped_pos - v_min) / (v_max - v_min) + (-1) # Scale the pedestrian positions

        # Process and scale LiDAR scan data
        temp = np.array(self.scan, dtype=np.float32)
        scan_avg = np.zeros((20, 80))
        for n in range(10):
            scan_tmp = temp[n * 720 : (n + 1) * 720]
            for i in range(80):
                scan_avg[2 * n, i] = np.min(scan_tmp[i * 9 : (i + 1) * 9]) # Minimum value in each segment
                scan_avg[2 * n + 1, i] = np.mean(scan_tmp[i * 9 : (i + 1) * 9]) # Average value in each segment
        
        scan_avg = scan_avg.reshape(1600)
        scan_avg_map = np.matlib.repmat(scan_avg, 1, 4)
        self.scan = scan_avg_map.reshape(6400)
        s_min, s_max = 0, 30
        self.scan = 2 * (self.scan - s_min) / (s_max - s_min) + (-1)

        # Scale the goal position 
        g_min, g_max = -2, 2
        goal_original = np.array(self.goal, dtype=np.float32)
        self.goal = 2 * (goal_original - g_min) / (g_max - g_min) + (-1)


# --- Main program execution ---
def main (args=None):
    """
    Main entry point for the ROS2 node. Initializes ROS2, creates the DrlInference
    node, and keeps it running to process incoming data and control the robot.
    """
    rclpy.init(args=args)   # Initialize the ROS2 communication Node
    drl_infe = DrlInference()   # Create an instance of the DrlInference node
    rclpy.spin(drl_infe)    # Keep the node running until shutdown
    drl_infe.destroy_node() # Destroy the node explicitly (optional)
    rclpy.shutdown()    # Shutdown the node

if __name__ == '__main__':
    main()  # Run the main function if this script is executed directly



