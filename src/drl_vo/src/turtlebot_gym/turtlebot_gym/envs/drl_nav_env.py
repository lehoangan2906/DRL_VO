#!/usr/bin/python3

"""
This module defines the DRLNavEnv class, which integrates ROS 2 and OpenAI Gym
to create a reinforcement learning environment for navigation tasks using a
TurtleBot.
"""


import gym
import math
import time
import rclpy
import random
import numpy as np
from gym import spaces
from rclpy.node import Node
from rclpy.clock import Clock
from gym.utils import seeding
from cnn_msgs.msg import CnnData
from gazebo_msgs.msg import ModelState
from action_msgs.msg import GoalStatusArray
from track_ped_msgs.msg import TrackedPersons
from nav_msgs.msg import Odometry, OccupancyGrid
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped
from gazebo_msgs.srv import GetEntityState, SetEntityState
from geometry_msgs.msg import Pose, Twist, Point, PoseStamped


class DRLNavEnv(Node, gym.Env):
    """
    Gazebo environment that converts standard OpenAI Gym methods into Gazebo commands.


    To check any topic we need to have the simulations running, we need to do
    two things:

        1) Unpause the simulation: without that the stream of data doesn't
        flow. This is for simulations that are paused for whatever reason.

        2) If the simulation was running already for some reason, we need to
        reset the controllers. This has to do with the fact that some plugins
        with tf, don't understand the reset of the simulation and need to be
        reset to work properly.
    """
    
    # Set up the environment 
    def __init__(self):

        # Initialize the ROS 2 system
        super().__init__('drl_nav_env')

        # Initialize the OpenAI Gym environment
        gym.Env.__init__(self)

        # Set a random seed for reproducibility
        if hasattr(self, 'seed'):
            self.seed()

        # Robot parameters:
        self.ROBOT_RADIUS = 0.3  # for collision checking
        self.GOAL_RADIUS = (0.3) # How close the robot must get to the goal to consider it reached
        self.DIST_NUM = 10  # Number of distance measurements to track the robot's progress to the goal
        self.pos_valid_flag = True  # Flag to check if the robot's position is valid (used for initialization)

        # Define bumper-related flags
        self.bump_flag = False  # if the robot bumped into something
        self.bump_num = 0  # track the number of bumps

        # Reward related variables for RL:
        self.dist_to_goal_reg = np.zeros(self.DIST_NUM)  # store distances to the goal
        self.num_iterations = 0  # number of iterations (steps) in each episode

        # Action limits for the robot
        self.max_linear_speed = 0.4  # Maximum linear speed (m/sec)
        self.max_angular_speed = 0.6  # Maximum angular speed (rad/sec)

        # Action space: Normalized to [-1, 1]
        self.high_action = np.array(
            [1, 1]
        )  # Upper bounds of the action space (linear and angular velocity)
        self.low_action = np.array([-1, -1])  # Lower bounds of the action space
        self.action_space = spaces.Box(
            low=self.low_action, high=self.high_action, dtype=np.float32
        )  # possible actions that the agent can take. .

        # Observation space
        self.cnn_data = CnnData()  # Placeholder for sensor data
        self.ped_pos = []  # Placeholder for pedestrian positions
        self.scan = []  # Placeholder for lidar data
        self.goal = []  # Placeholder for goal position
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(19202,), dtype=np.float32
        )  # Observation space size and range

        # initialize robot and environment information
        self.init_pose = Pose()  # Initial pose of the robot
        self.curr_pose = Pose()  # Current pose of the robot
        self.curr_vel = Twist()  # Current velocity of the robot
        self.goal_position = Point()  # Position of the goal
        self.info = {}  # Additional information about the environment

        # Episode control flags:
        self._episode_done = False  # if the episode is done
        self._goal_reached = False  # if the goal has been reached
        self._reset = True  # reset the simulation at the beginning of an episode

        # Pedestrian tracking related variable
        self.peds = TrackedPersons()  # Tracks pedestrians in the environment

        # ROS 2 service clients for interacting with Gazebo
        self.gazebo_client = self.create_client(
            SetEntityState,
            "/gazebo/set_entity_state",  # Set the robot's pose in the Gazebo simulation
        )
        self.get_model_state = self.create_client(
            GetEntityState,
            "/gazebo/get_entity_state",  # Get the robot's current pose from Gazebo
        )

        # quality of service settings for subscribers and publishers
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE)

        # ROS 2 subsriptions and receiving data from topics
        self._map_sub = self.create_subscription(
            OccupancyGrid,
            "/map",
            self._map_callback,
            qos,  # Occupancy grid map of the environment
        )
        self._cnn_data_sub = self.create_subscription(
            CnnData,
            "/cnn_data",
            self._cnn_data_callback,
            qos,  # sensor data and goal information
        )
        self._robot_pos_sub = self.create_subscription(
            PoseStamped,
            "/amcl_pose",
            self._robot_pose_callback,
            qos,  # Robot pose from the AMCL node (for localization)
        )
        self._robot_vel_sub = self.create_subscription(
            Odometry, "/odom", self._robot_vel_callback, qos  # Odometry data
        )
        self._final_goal_sub = self.create_subscription(
            PoseStamped,
            "/move_base/current_goal",
            self._final_goal_callback,
            qos,  # Goal position from the navigation stack
        )
        self._goal_status_sub = self.create_subscription(
            GoalStatusArray,
            "/move_base/status",
            self._goal_status_callback,
            qos,  # Goal reached status (success/failure) from the navigation stack
        )
        self._ped_sub = self.create_subscription(
            TrackedPersons,
            "/track_ped",
            self._ped_callback,
            qos,  # Pedestrian tracking data
        )

        # ROS 2 Publishers
        self._cmd_vel_pub = self.create_publisher(
            Twist, "/drl_cmd_vel", qos
        )  # Publish the robot's velocity (for controlling its movement)

        self._initial_goal_pub = self.create_publisher(
            PoseStamped,
            "/move_base_simple/goal",
            qos,  # Publish the initial goal position for the navigation stack
        )

        self._initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            "initialpose",
            qos,  # Publish the robot's initial pose to AMCL for localization
        )


    # ========================================================================= #
    # Env methods

    # Set the seed for the environment's random number generator
    # Ensuring reproducibility of the environment
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Interacting with the Gazebo simulation
    def step(self, action):  # OpenAI Gym method
        """
        Interacting with the Gazebo simulation
        - Taking action
        - Updating the environment
        - Return new state
        - Return reward
        - Return completion status

        Gives env an action to enter the next state,
        obs, reward, done, info = env.step(action)
        """
        self._take_action(action)
        obs = self._get_observation()
        reward = self._compute_reward()
        done = self._is_done(reward)
        info = self._post_information()
        return obs, reward, done, info

    # Resets the robot and the environment to a consistent initial state
    def reset(self):  # OpenAI Gym method
        """
        Resets the robot and the environment to a consistent initial state.
        - Resetting the robot's pose and goal position
        - Resetting internal variables related to the state of the environment (flags for collision or goals)
        - Return the initial observation

        obs, info = env.reset()
        """
        self._reset_sim()
        obs = self._get_observation()
        info = self._post_information()
        return obs

    # Safely shutdown the environment when it's no longer needed
    def close(self):  # OpenAI Gym method
        """
        Function executed when closing the environment.
        Use it for closing GUIs and other systems that need closing

        :return:
        """
        rclpy.shutdown()

    # Reset the simulation to initial conditions
    def _reset_sim(self):
        """
        Resets the simulation to initial conditions
        """
        self._set_init()
        return True

    # Initialize the robot and the environment at the start of an episode
    def _set_init(self):
        """
        Set initial condition for simulation
        - Resets the robot's velocity
        - Checks if the robot's initial pose is valid (in a free space)
        - Publishes a random goal position.
        - Initializes internal variables that track the robot's state
            + pose validity
            + bumping status
            + distance to the goal
        """

        # Stop the robot by sending 0 velocity
        self._cmd_vel_pub.publish(Twist())

        if self._reset:  # Check if the simulation needs to be reset
            self._reset = False
            self._check_all_systems_ready()  # Ensure all systems are ready

            self.pos_valid_flag = False
            occ_map = self.occ_map  # Get the map data

            # Loop until a valid initial pose is found
            while not self.pos_valid_flag:
                # Randomly select an initial pose for the robot
                seed_initial_pose = random.randint(0, 18)
                # Set the initial pose of the robot
                self._set_initial_pose(seed_initial_pose)

                time.sleep(4)  # Wait for the robot to stabilize

                # Get the current position of the robot
                x = self.curr_pose.position.x
                y = self.curr_pose.position.y

                radius = self.ROBOT_RADIUS  # Get the robot's radius

                self.pos_valid_flag = self._is_pos_valid(
                    x, y, radius, occ_map
                )  # Check if the initial pose is valid

        # Publish a random goal position for the robot to navigate to
        goal_x, goal_y, goal_yaw = self._publish_random_goal()

        time.sleep(1)  # Wait for the goal to be published
        self._check_all_systems_ready()  # Ensure all systems are ready

        # Initialize internal variables for tracking the robot's state
        self.init_pose = self.curr_pose  # Set the initial pose of the robot
        self.curr_pose = self.curr_pose  # Set the current pose of the robot
        self.goal_position.x = goal_x  # Set the goal position (x-coordinate)
        self.goal_position.y = goal_y  # Set the goal position (y-coordinate)

        self.pos_valid_flag = True  # Reset the position validity flag
        self.bump_flag = False  # Reset the bumper flag
        self.num_iterations = 0  # Reset the iteration counter
        # Reset the distance to goal array
        self.dist_to_goal_reg = np.zeros(self.DIST_NUM)
        self._episode_done = False  # Reset the episode done flag

        return self.init_pose, self.goal_position

    # Chooses one of several predefined initial positions based on a seed_initial_pose value
    # Which determines where in the environment the robot will start
    def _set_initial_pose(self, seed_initial_pose):
        # Set initial pose of the robot

        poses = [
            [1, 1, 0],
            [14, 7, 1.5705],
            [1, 16, 0],
            [14, 22.5, -1.3113],
            [4, 4, 1.5705],
            [2, 9, 0],
            [30, 9, 3.14],
            [25, 17, 3.14],
            [5, 8, 0],
            [10, 12, 0],
            [14, 15, 1.576],
            [18.5, 15.7, 3.14],
            [18.5, 11.3, 3.14],
            [14, 11.3, 3.14],
            [12.5, 13.2, 0.78],
            [12.07, 16.06, 0],
            [21, 14, -1.576],
            [14, 22.5, 1.576],
            [18, 8.5, -1.576],
        ]

        pose = poses[seed_initial_pose]  # Randomly select initial pose
        self._pub_initial_model_state(*pose)  # Publish initial pose
        time.sleep(1)
        self._pub_initial_position(*pose)

    # check if all the subscribers, publishers, and services are ready
    def _check_all_systems_ready(self):
        self._check_all_subscribers_ready()  # Check if all subscribers are ready
        self._check_all_publishers_ready()  # Check if all publishers are ready
        self._check_service_ready("/gazebo/set_model_state")
        return True

    # check if /map, /cnn_data, /amcl_pose, /move_base/status topic subscribers are ready
    def _check_all_subscribers_ready(self):
        """
        Check if all subscribers are ready
        """

        self.get_logger().debug("START TO CHECK ALL SUBSCRIBERS READY")

        self._check_subscriber_ready(
            "/map", OccupancyGrid
        )  # The occupancy grid map of the environment, which is used to determine free and occupied spaces.
        # The data from the CNN module, which includes pedestrian positions, lidar scan data, and goal position.
        self._check_subscriber_ready("/cnn_data", CnnData)
        # The current pose of the robot in the environment. Generated by the AMCL localization system.
        self._check_subscriber_ready("/amcl_pose", PoseStamped)
        # The status of the robot's goal, which indicates if the goal has been reached or not. Generated by the ROS navigation stack.
        self._check_subscriber_ready("/move_base/status", GoalStatusArray)

        self.get_logger().debug("ALL SUBSCRIBERS READY")

    # check if a subscriber is ready
    def _check_subscriber_ready(self, name, type, timeout=5.0):
        """
        Waits for a sensor topic to get ready for connection.

        :param name: The name of the topic (e.g., "/map", "/robot_pose", etc.)
        :param type: The message type that the topic publishes (e.g., OccupancyGrid, PoseStamped).
        :param timeout: The time (in seconds) to wait before retrying (default is 5 seconds).

        :return: None
        """

        var = None

        self.get_logger().debug(f"Waiting for '{name}' to be READY...")

        # This loop keeps running until the topic becomes available (i.e., it starts publishing data)
        while var is None:
            try:
                # This line waits for a message to be published on the specified topic
                var = rclpy.wait_for_message(name, type, timeout)
                self.get_logger().debug(f"Current '{name}' READY=>")

            except Exception:
                # If the topic is not available, log a message and keep waiting
                self.get_logger().debug(
                    f"Sensor topic '{name}' is not available, retrying..."
                )

        # Return the variable (message) received from the topic
        return var

    # check if /cmd_vel, /move_base_simple/goal, /initialpose topic publishers are ready and if /gazebo/set_model_state service is ready
    def _check_all_publishers_ready(self):
        """
        Check if necessary publishers and services are ready
        """

        self.get_logger().debug("START TO CHECK ALL PUBLISHERS READY")

        # Check if the publishers have active subscribers
        # Send velocity commands to the robot
        self._check_publisher_ready("/cmd_vel", self._cmd_vel_pub)
        # Publish the initial goal position for the robot to the navigation stack
        self._check_publisher_ready("/move_base_simple/goal", self._initial_goal_pub)
        # Publish the initial pose of the robot to the localization system (amcl)
        self._check_publisher_ready("/initialpose", self._initial_pose_pub)

        # Check if the Gazebo service is ready
        # Set the initial state of the robot in the Gazebo simulation
        self._check_service_ready("/gazebo/set_model_state")

        self.get_logger().debug("ALL PUBLISHERS READY")

    # check if a publisher is ready
    def _check_publisher_ready(self, name, obj, timeout=5.0):
        """
        Waits for a publisher to get response (i.e., have active subscribers).
        :param name: The name of the ROS topic being published to.
        :param obj: The ROS publisher object that needs to be checked.
        :param timeout: The time in seconds to wait before logging a fatal message.
        """

        self.get_logger().debug(f"Waiting for '{name}' to be READY...")
        start_time = Clock().now().to_msg()

        while obj.get_subscription_count() == 0 and rclpy.ok():
            # Log a warning if no subscribers are connected
            self.get_logger().warn(
                f"No subscribers found for publisher {name}. Retrying..."
            )

            # Sleep for a short duration to avoid a busy loop
            rclpy.sleep(1.0)

            # Check if the timeout has been exceeded
            elapsed_time = Clock().now().to_msg() - start_time
            if elapsed_time.nanoseconds / 1e9 > timeout:
                self.get_logger().fatal(
                    f"Publisher '{name}' is not available after waiting for {timeout} seconds."
                )
                break

        self.get_logger().debug(f"Publisher '{name}' is READY.")

    # Check if a service is ready
    def _check_service_ready(self, service_name, timeout=5.0):
        """
        Waits for a service to be available
        """
        self.get_logger().debug(f"Waiting for '{service_name}' to be READY...")

        start_time = Clock().now().to_msg()

        # Try waiting for the service to become available
        try:
            if ready := self.wait_for_service(service_name, timeout_sec=timeout):
                self.get_logger().debug(f"Service '{service_name}' is READY.")
            else:
                self.get_logger().fatal(
                    f"Service '{service_name}' is not available after waiting for {timeout} seconds."
                )
        except Exception as e:
            self.get_logger().fatal(f"Service '{service_name}' is not available: {e}")

    # callback functions for ROS2
    # ========================================================================= #

    # callback function to get the occupancy grid map
    def _map_callback(self, map_msg):
        self.occ_map = map_msg

    # callback function to get the data of lidar, pedestrian, and goal position from the CNN module
    def _cnn_data_callback(self, cnn_data_msg):
        self.cnn_data = cnn_data_msg

    # callback function to get the robot current pose from the AMCL localization system
    def _robot_pose_callback(self, robot_pose_msg):
        self.curr_pose = robot_pose_msg.pose

    # callback function to get the robot velocity from the odometry data
    def _robot_vel_callback(self, robot_vel_msg):
        self.curr_vel = robot_vel_msg.twist.twist

    # callback function to get the goal position from the move_base node of the ROS navigation stack
    def _final_goal_callback(self, final_goal_msg):
        self.goal_position = final_goal_msg.pose.position

    # Check if the robot has reached the goal or still moving
    def _goal_status_callback(self, goal_status_msg):
        """
        Callback function for receiving the goal status from the /move_base/status
        It checks if the robot has reached the goal or is still moving.
        """

        # Check if there are any status messages in the status list
        if len(goal_status_msg.status_list) > 0:
            # Get the most recent goal status message
            last_element = goal_status_msg.status_list[-1]

            # Log the status message text
            rclpy.logging.get_logger().info(last_element.text)

            # Check if the last status indicates that the goal has been reached
            # Status 3 means 'SUCCEEDED' (default in actionlib_msgs/GoalStatus in ROS)
            if last_element.status == 3:
                self._goal_reached = True
            else:
                self._goal_reached = False
        else:
            # No status means the goal hasn't been reached yet
            self._goal_reached = False

    # Callback function to get the pedestrian data from the detection and tracking module
    def _ped_callback(self, trackPed_msg):
        """
        peds.header contains:
        - stamp: A timestamp indicating when the data was recorded
        - frame_id: The coordinate frame in which the tracked persons' positions are specified (e.g., camera_frame, base_link, map)

        peds.tracks contains:
        = A list of TrackedPerson messages representing all the tracked objects in the current frame.
        = Each TrackedPerson message contains the following information:
            + id: A unique identifier for the tracked person
            + bbox_upper_left_x: The x-coordinate of the upper-left corner of the bounding box
            + bbox_upper_left_y: The y-coordinate of the upper-left corner of the bounding box
            + bbox_lower_right_x: The x-coordinate of the lower-right corner of the bounding box
            + bbox_lower_right_y: The y-coordinate of the lower-right corner of the bounding box
            + depth: The estimated depth (distance) from the camera to the tracked object.
            + angle: The angle of the object relative to the camera's perpendicular bisector
            + velocity_x: The x-component of the object's velocity
            + velocity_z: The z-component of the object's velocity
            + x: The x-coordinate of the object's position in the camera frame
            + z: The z-coordinate of the object's position in the camera frame
        """
        self.peds = trackPed_msg

    # ========================================================================= #
    # Publisher functions for publishing information about the robot and goal position

    # Publish the new initial position (x, y, theta) of the robot in the Gazebo environment
    def _pub_initial_model_state(self, x, y, theta):
        """
        Publishes a new initial position (x, y, theta) for the robot in Gazebo

        :param x: x-position of the robot in the world frame.
        :param y: y-position of the robot in the world frame.
        :param theta: Orientation of the robot (yaw angle) in radians.
        """

        # Create a ModelState message to represent the robot's position and orientation
        robot_state = ModelState()

        # Specify the model name of the robot to be controlled (in this case: mobile_base)
        # Get the pose/twist relative to the frame of the model_base
        robot_state.model_name = "mobile_base"

        # Set the robot's initial position in the Gazebo world (x, y, z coordinates)
        robot_state.pose.position.x = x
        robot_state.pose.position.y = y
        robot_state.pose.position.z = 0  # The robot is assumed to be on the ground

        # Convert the yaw angle (theta) to quaternion for setting the robot's orientation
        # Gazebo uses quaternions to represent orientation in 3D space
        robot_state.pose.orientation.x = 0  # No roll
        robot_state.pose.orientation.y = 0  # No pitch
        robot_state.pose.orientation.z = np.sin(theta / 2)  # Yaw converted to quaternion
        robot_state.pose.orientation.w = np.cos(theta / 2)  # Quaternion w component

        # Set the reference frame for the robot, which is "world" in Gazebo
        robot_state.reference_frame = "world"

        # Wait for the service to be available
        if not self.gazebo_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('/gazebo/set_entity_state service is not available')
            return
        
        request = SetEntityState.Request()
        request.state = robot_state

        # Call the service
        future = self.gazebo_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            self.get_logger().info('Set the model state successfully')
        else:
            self.get_logger().warn("/gazebo/set_entity_state service call failed")

    # Function to publish the initial position of the robot (x, y, theta) in the environment --> for localization
    def _pub_initial_position(self, x, y, theta):
        """
        Publishing new initial position (x, y, theta) --> for localization
        :param x: x-position of the robot
        :param y: y-position of the robot
        :param theta: theta-position (orientation) of the robot
        """

        # Create a PoseWithCovarianceStamped message to represent the robot's pose with uncertainty (covariance)
        initial_pose = PoseWithCovarianceStamped()

        # Set the frame of reference for this pose to "map"
        initial_pose.header.frame_id = "map"

        # Get the current time and add it to the header (important for time synchronization)
        initial_pose.header.stamp = Clock().now().to_msg()

        # Set the robot's position (x, y, z coordinates)
        initial_pose.pose.pose.position.x = x
        initial_pose.pose.pose.position.y = y
        # Assuming the robot is on the ground (z = 0)
        initial_pose.pose.pose.position.z = 0

        # Convert the yaw angle (theta) to quaternion for setting the robot's orientation
        initial_pose.pose.pose.orientation.x = 0  # No roll
        initial_pose.pose.pose.orientation.y = 0  # No pitch
        # Yaw converted to quaternion (z-axis rotation)
        initial_pose.pose.pose.orientation.z = np.sin(theta / 2)
        initial_pose.pose.pose.orientation.w = np.cos(theta / 2)  # Quaternion w component

        # Publish the initial pose message to the ROS topic that handles localization (e,g., for AMCL)
        self._initial_pose_pub.publish(initial_pose)

    # Function to randomly publish the goal position (x, y, theta) for the robot in the environment --> for global planner
    def _publish_random_goal(self):
        """
        The function randomly selects a goal position (x, y, theta) on the map for the robot,
        ensuring that the goal is within a certain distance range from the robot's current position.
        It then publishes this goal to be used by the robot's navigation system.

        Publishing new random goal [x, y, theta] for global planner
        :return: goal position [x, y, theta]
        """

        dis_diff = 21  # Initialize the distance difference between the robot's current position and the randomly generated goal position

        # Ensure that the goal is at least 2 meters away from the robot but not more than 7 meters away
        while dis_diff >= 7 or dis_diff < 2:
            # if the distance is too short or too long, the loop continues generating new positions
            # Get a random position (x, y, theta) on the map
            x, y, theta = self._get_random_pos_on_map(self.occ_map)

            dis_diff = np.linalg_norm(
                np.array([self.curr_pose.position.x - x, self.curr_pose.position.y - y])
            )  # Calculate the Euclidean distance between the robot's current position and the randomly generated goal position

        # Publish the goal position for the robot to navigate to
        self._publish_goal(x, y, theta)

        return x, y, theta

    # Function to publish a goal position and orientation (x, y, theta) for the robot to navigate to
    def _publish_goal(self, x, y, theta):
        """
        Publishes a goal position and orientation for the robot to navigate to.

        Args:
            x (float): The x-coordinate of the goal position in the map frame.
            y (float): The y-coordinate of the goal position in the map frame.
            theta (float): The orientation (yaw) angle of the robot at the goal position, in radians.

        This function creates a PoseStamped message with the given position and orientation,
        and publishes it to the relevant topic, which is typically used by the robot's navigation system
        to plan a path to the goal.
        """

        # Convert the yaw angle (theta) to a format usable for quaternion calculations
        goal_yaw = theta

        # Create a PoseStamped message, which includes position and orientation in a specific frame
        goal = PoseStamped()

        # Set the timestamp for the message to the current time, ensuring synchronization with other nodes
        goal.header.stamp = Clock().now().to_msg()

        # Specify that the goal coordinates are relative to the "map" frame, which is the global reference frame.
        goal.header.frame_id = "map"

        # Set the x, y, and z coordinates for the goal position
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0  # Assuming a 2D plane, so z is set to 0

        # Convert the yaw angle (theta) into a quaternion for the robot's orientation at the goal
        # Quaternions are used in 3D space to avoid issues like gimbal lock
        goal.pose.orientation.x = 0  # No rotation around the x-axis
        goal.pose.orientation.y = 0  # No rotation around the y-axis
        
        # Rotation around the z-axis (yaw)
        goal.pose.orientation.z = np.sin(goal_yaw / 2)
        # Rotation component to complete the quaternion
        goal.pose.orientation.w = np.cos(goal_yaw / 2)

        # Publish the goal to the topic that the robot's navigation stack listens to
        # This triggers the navigation system to start planning a path to the specified goal
        self._initial_goal_pub.publish(goal)

    # Function to find a valid (free) random position (x, y, theta) on the map
    # that the robot can potentially navigates to
    def _get_random_pos_on_map(self, occ_map):
        """
        Generates a random valid position (x, y) and orientation (theta) on the map.

        Args:
            occ_map (OccupancyGrid): The map data, which includes information about the map's dimentions, resolution,
            and any obstacles.

        Returns:
            tuple: A tuple (x, y, theta) representing a random valid position (x, y) and orientation (theta) on the map.
        """

        # Calculate the total width of the map in the global coordinate system
        map_width = occ_map.info.width * occ_map.info.resolution + occ_map.info.origin.position.x

        # Calculate the total height of the map in the global coordinate system
        map_height = occ_map.info.height * occ_map.info.resolution + occ_map.info.origin.position.y

        # Generate a random x, y-coordinates within the map's width
        x = random.uniform(0.0, map_width)
        y = random.uniform(0.0, map_height)

        # Define a safe radius around the robot, combining the robot's own radius with and additional safety margin
        radius = self.ROBOT_RADIUS + 0.5

        # Loop until a valid position is found (i.e., the position is free of obstacles)
        while not self._is_pos_valid(x, y, radius, occ_map):
            # If the position is not valid, generate new random coordinates
            x = random.uniform(0.0, map_width)
            y = random.uniform(0.0, map_height)

        # Generate a random orientation angle (theta) between -pi and pi
        theta = random.uniform(-math.pi, math.pi)

        # Return the valid random position and orientation as a tuple
        return x, y, theta

    # Function to check if a position (x, y) is valid (free of obstacles) on the map
    def _is_pos_valid(self, x, y, radius, occ_map):
        """
        Checks if the position (x, y) is valid on the map, meaning it is free of obstacles and within bounds.

        Args:
            x (float): The x-coordinate of the position to check.
            y (float): The y-coordinate of the position to check.
            radius (float): The safe radius of the robot to check for obstacles.
            occ_map (OccupancyGrid): The map data, which includes information about obstacles.

        Returns:
            bool: True if the position is valid (free of obstacles and within bounds), False otherwise
        """

        # Convert the safe radius into map grid cells
        cell_radius = int(radius / occ_map.info.resolution)

        # Convert the global x, y coordinates to map grid indices
        y_index = int((y - occ_map.info.origin.position.y) / occ_map.info.resolution)
        x_index = int((x - occ_map.info.origin.position.x) / occ_map.info.resolution)

        # Check the cells within the safe radius around the robot
        for i in range(x_index - cell_radius, x_index + cell_radius, 1):
            for j in range(y_index - cell_radius, y_index + cell_radius, 1):
                # Calculate the linear index of the cell in the map's data array
                index = j * occ_map.info.width + i

                # Check if the index is out of bounds
                if index >= len(occ_map.data):
                    return False

                try:
                    # Get the value of the cell from the map's data
                    val = occ_map.data[index]
                except IndexError:
                    self.get_logger().error(f"IndexError: index: {index}, map_length: {len(map.data)}")
                    return False # The position is invalid because the index is out of bounds
                
                # If any cell is not free (non-zero), the position is invalid
                if val != 0:
                    return False

        # If all cells in the checked area are free and within bounds, the position is valid
        return True

    # ========================================================================= #
    # Observation and action functions

    # Return the observation data as a vector
    def _get_observation(self):
        """
        Processes and normalizes sensor data (pedestrian positions, scan data, and goal position)
        into a single observation vector, with modifications fro the Oradar MS200 lidar specification.

        - The Lidar data is split into 20-22 segments.
        - Interpolation is applied to handle non-uniform lidar data points.

        Returns:
            np.ndarray: A combined and normalized observation vector.
        """

        # Extract the pedestrian positions, lidar scan, goal position, robot velocity from the cnn_data structure
        self.ped_pos = self.cnn_data.ped_pos_map
        self.scan = self.cnn_data.scan
        self.goal = self.cnn_data.goal_cart
        self.vel = self.cnn_data.vel

        # Pedestrian positions
        # MaxAbsScaler: Normalize the pedestrian position map to the range [-1, 1]
        v_min = -2
        v_max = 2
        self.ped_pos = np.array(self.ped_pos, dtype=np.float32)
        self.ped_pos = 2 * (self.ped_pos - v_min) / (v_max - v_min) + (-1)


        # Laser scan data:
        temp = np.array(scan, dtype=np.float32)   # Convert the flattened list into a numpy array
        scan_avg = np.zeros((20, 80))             # Initialize the scan_avg array (20 rows x 80 columns)

        # Apply min and average pooling with window size = 3 to each scan data
        for n in range(10):
            scan_tmp = temp[n * 240 : (n + 1) * 240]    # Get 240 points per scan
            for i in range(80):                         # 240 points / 3 = 80 (for min and avg pooling)
                scan_avg[2 * n, i] = np.min(scan_tmp[i * 3 : (i + 1) * 3])          # Min pooling
                scan_avg[2 * n + 1, i] = np.mean(scan_tmp[i * 3 : (i + 1) * 3])     # Average pooling

        # Reshape and normalize 
        scan_avg = scan_avg.reshape(1600)
        scan_avg_map = np.matlib.repmat(scan_avg, 1, 4)
        scan = scan_avg_map.reshape(6400)

        # Normalize between -1 and 1
        s_min = 0
        s_max = 30
        scan = 2 * (scan - s_min) / (s_max - s_min) + (-1)

        # Goal positions:
        # MaxAbsScaler: Normalize the goal position to the range [-1, 1]
        g_min = -2
        g_max = 2
        self.goal = np.array(self.goal, dtype=np.float32)
        self.goal = 2 * (self.goal - g_min) / (g_max - g_min) + (-1)

        # Velocity normalization (currently commented out)
        '''
        vx_min = 0
        vx_max = 0.5
        wz_min = -2
        wz_max = 2
        self.vel = np.array(self.vel, dtype=np.float32)
        self.vel[0] = 2 * (self.vel[0] - vx_min) / (vx_max - vx_min) + (-1)
        self.vel[1] = 2 * (self.vel[1] - wz_min) / (wz_max - wz_min) + (-1)
        '''

        # Combine the normalized pedestrian positions, scan data, and goal position into a single observation vector
        self.observation = np.concatenate((self.ped_pos, self.scan, self.goal), axis=None)

        # Logging
        self.get_logger().debug(f"Observation ==>: {self.observation}")

        # Return the combined observation vector
        return self.observation

    # collect and return key pieces of information about the robot's state
    # including initial pose, current pose, and goal position
    def _post_information(self):
        """
        Return:
        info: {"initial_pose", "goal_position", "current_pose"}
        """

        # Create a dictionary with initial_pose, goal_position, and current_pose
        self.info = {
            "initial_pose": self.init_pose,
            "goal_position": self.goal_position,
            "current_pose": self.curr_pose,
        }

        return self.info

    # Translate a given action into velocity commands, then publishing these commands to the robot
    def _take_action(self, action):
        """
        Set linear and angular speed for TurtleBot and execute.

        Args:
            action (list): A list where action[0] is the linear velocity control input
                        and action[1] is the angular velocity control input.
        """

        # Use ROS2 logging instead of rospy.logdebug
        self.get_logger().debug(f"TurtleBot2 Base Twist Cmd>>\nLinear: {action[0]}\nangular: {action[1]}")

        cmd_vel = Twist()  # Create a Twist message to store the velocity commands

        # Set the new linear velocity range [-0.0, 0.4] m/s (no backward moving)
        vx_min = 0.0
        vx_max = 0.4

        # Set the new angular velocity range [-0.6, 0.6] rad/s
        vz_min = -0.6
        vz_max = 0.6

        # Scale the action[0] (assumed to be in the range [-1, 1]) to the robot's actual linear velocity range
        cmd_vel.linear.x = (action[0] + 1) * (vx_max - vx_min) / 2 + vx_min

        # Scale the action[1] (assumed to be in the range [-1, 1]) to the robot's actual angular velocity range
        cmd_vel.angular.z = (action[1] + 1) * (vz_max - vz_min) / 2 + vz_min

        # Publish the velocity command
        for _ in range(1):
            self._cmd_vel_pub.publish(cmd_vel)

            # Sleep to maintain a publishing rate of 20 Hz 
            rate = self.create_rate(20)
            rate.sleep()

        # logging
        self.get_logger().warn(f"cmd_vel: \nLinear: {cmd_vel.linear.x}\n angular: {cmd_vel.angular.z}")


    # ========================================================================= #
    # Reward computation

    # compute goal reaching reward
    def _goal_reached_reward(self, r_arrival, r_waypoint):
        """
        Computes the reward based on how close the robot is to the goal.
        It rewards the robot for reaching the goal or making progress towards it
        and penalizes the robot if it fails to reach the goal within the allowed iterations.

        Args:
            r_arrival (float): Reward for reaching the goal.
            r_waypoint (float): Reward for making progress towards the goal.

        Returns:
            reward based on proximity to the goal
        """

        # Calculate the Euclidean distance between the robot's current position and the goal position
        dist_to_goal = np.linalg.norm(
            np.array(
                [
                    self.curr_pose.position.x - self.goal_position.x,  # X-distance to goal
                    self.curr_pose.position.y - self.goal_position.y,  # Y-distance to goal
                    self.curr_pose.position.z - self.goal_position.z,  # Z-distance to goal
                ]
            )
        )

        # Use modulo operation to track progress every DIST_NUM iterations
        t_1 = self.num_iterations % self.DIST_NUM

        # Initialize the distance tracking array on the first iteration
        if self.num_iterations == 0:
            # Initialize the array with the current distance
            self.dist_to_goal_reg = np.ones(self.DIST_NUM) * dist_to_goal

        # logging
        self.get_logger().warn(f"distance_to_goal_reg = {self.dist_to_goal_reg[t_1]}")
        self.get_logger().warn(f"distance_to_goal = {dist_to_goal}")

        # Define the maximum number of iterations allowed for reaching the goal
        max_iteration = 512

        # Case 1: Robot reaches the goal (distance to goal is within the specified radius)
        if dist_to_goal <= self.GOAL_RADIUS:
            reward = r_arrival  # Assign maximum reward for reaching the goal

        # Case 2: Robot fails to reach teh goal within the maximum allowed iterations
        elif self.num_iterations >= max_iteration:
            reward = -r_arrival  # Penalize the robot for not reaching the goal in time

        # Case 3: Robot is making progress but hasn't reached the goal_yet
        else:
            # Reward is proportional to the reduction in distance to the goal since the last check
            reward = r_waypoint * (self.dist_to_goal_reg[t_1] - dist_to_goal)

        # Update the distance reward for the current iteration (used for future comparisons)
        self.dist_to_goal_reg[t_1] = dist_to_goal

        # Log the reward value
        self.get_logger().warn(f"Goal reached reward: {reward}")

        # Return the calculated reward (positive for progress or goal reached, negative for failure)
        return reward

    # Punishment after collision with obstacles
    def _obstacle_collision_punish(self, scan, r_scan, r_collision):
        """
        Computes the penalty for being too close to or colliding with obstacles based on lidar scan data.

        Args:
            scan (np.array): Lidar scan data (distances to obstacles).
            r_scan (float): Penalty factor for proximity to obstacles.
            r_collision (float): Penalty factor for colliding with an obstacle (i.e. getting too close)

        Returns:
            negative reward if the robot collides with obstacles.
        """

        # Find the minimum distance from the lidar scan
        min_scan_dist = np.amin(scan[scan != 0])

        # Case 1: The robot is very close to an obstacle, potentially colliding (distance less than or equal to its radius)
        if min_scan_dist <= self.ROBOT_RADIUS and min_scan_dist >= 0.03:
            reward = r_collision

        # Case 2: The robot is near an obstacle but not colliding (distance within 3 times its radius)
        elif min_scan_dist < 3 * self.ROBOT_RADIUS:
            # Apply a smaller penalty based on proximity
            reward = r_scan * (3 * self.ROBOT_RADIUS - min_scan_dist)

        # Case 3: The robot is far enough from obstacles (no penalty)
        else:
            reward = 0.0
        
        # logging
        self.get_logger().warn(f"Obstacle collision reward: {reward}")

        return reward

    # Punishment for high angular velocity (fast turns)
    def _angular_velocity_punish(self, w_z, r_rotation, w_thresh):
        """
        Computes the penalty for excessive angular velocity (rotational speed).

        Args:
            w_z (float): The robot's current angular velocity (rotational speed around the z-axis).
            r_rotation (float): Penalty factor for angular velocity.
            w_thresh (float): Threshold for penalizing excessive angular velocity.

        Returns:
            Negative reward if the robot turns too much
        """

        # Apply punishment if angular velocity exceeds the threshold
        if abs(w_z) > w_thresh:
            reward = abs(w_z) * r_rotation
        else:
            reward = 0.0

        # logging
        self.get_logger().warn(f"Angular velocity punish reward: {reward}")

        return reward

    # Reward for aligning with the goal
    def _theta_reward(self, goal, peds, v_x, r_angle, angle_thresh):
        """
        Computes a reward for aligning the robot's heading with the goal direction,
        while considering the presence of pedestrians to avoid collisions.

        Args:
            goal (np.array): The goal's position relative to the robot.
            peds (object): Pedestrian tracking data (positions and velocities of multiple pedestrians in a frame).
            v_x (float): The robot's current linear velocity (forward motion).
            r_angle (float): Reward factor for maintaining alignment with the goal.
            angle_thresh (float): Threshold for the angular reward (maximum allowable deviation).

        Returns:
            float: The computed reward based on heading alignment and pedestrian avoidance.
        """

        # Preferred goal theta (angle to the goal in counterclockwise direction)
        theta_pre = np.arctan2(goal[1], goal[0])
        d_theta = theta_pre  # Initialize the heading difference as the preferred angle


        # Case 1: If there are pedestrians in the scene
        if len(peds.tracks) != 0:
            # Set initial delta theta to 90 degrees (default avoidance angle)
            d_theta = np.pi / 2
            N = 60  # Number of random angle samples to check for safe movement
            theta_min = 1000  # Large initial value for minimum angle difference

            # Sample random angles to find a safe direction to move
            for _ in range(N):
                # Random angle in range [-pi, pi]
                theta = random.uniform(-np.pi, np.pi)
                free = True  # Assume the angle is free from obstacles/pedestrians

                # Loop through the tracked pedestrians to check for potential collisions
                for ped in peds.tracks:
                    p_x = ped.pose.pose.position.x  # Pedestrian's x-position
                    p_z = ped.pose.pose.position.z
                    p_vx = ped.twist.twist.linear.x  # Pedestrian's x-velocity
                    p_vz = ped.twist.twist.linear.z

                    # Calculate the distance to the pedestrian
                    ped_dis = np.linalg.norm([p_x, p_z])

                    # If the pedestrian is within a 3m radius (maximum depth cam range), check for collision potential
                    if ped_dis <= 3:
                        # Pedestrian's angle relative to the robot
                        ped_theta = np.arctan2(p_z, p_x)
                        vo_theta = np.arctan2(self.ROBOT_RADIUS, np.sqrt(ped_dis**2 - self.ROBOT_RADIUS ** 2),)

                        # Calculate the relative angle between the robot and pedestrian to avoid collision
                        theta_rp = np.arctan2(v_x * np.sin(theta) - p_vz, v_x * np.cos(theta) - p_vx)

                        # If the angle overlaps with the pedestrian's path (collision cone), mark as unsafe
                        if ped_theta - vo_theta <= theta_rp <= ped_theta + vo_theta:
                            free = False
                            break
                
                # Choosing the safest turning angle (minimizing deviation from the preferred goal angle)
                if free:
                    theta_diff = (theta - theta_pre) ** 2
                    if theta_diff < theta_min:
                        theta_min = theta_diff
                        d_theta = theta

        # Case 2: If there are no pedestrians, simply align with the goal
        else:
            d_theta = theta_pre

        # Calculate the reward based on the angular difference from the goal direction
        reward = r_angle * (angle_thresh - abs(d_theta))

        # logging
        self.get_logger().warn(f"Theta reward: {reward}")

        return reward

    # Aggregate reward computation
    def _compute_reward(self):
        """
        Computes the total reward for the robot's action based on various criteria such as:
        - Reaching the goal
        - Progressing toward the goal
        - Avoiding collisions
        - Proximity to obstacles
        - Minimizing angular velocity (smooth turning)
        - Goal alignment

        Returns:
            float: The total reward for the current action.
        """

        # Reward for reaching the goal
        r_arrival = 20  # Large positive reward for reaching the goal.
        r_waypoint = 3.2  # Smaller reward for moving closer to the goal.

        # Penalty for colliding with obstacles
        r_collision = -20  # Large negative penalty for collisions
        # Penalty for being too close to obstacles (based on scan data)
        r_scan = -0.2

        # Rewards related to the robot's angle and smooth turning
        r_angle = (0.6)  # Reward for goal alignment
        r_rotation = -0.1  # Penalty for excessive rotation (fast turns)

        # Thresholds for angular velocity and angle deviation
        angle_thresh = np.pi / 6  # Angle threshold (30 degrees)
        w_thresh = 1  # Angular velocity threshold

        # 1. Compute the reward for reaching or moving toward the goal
        r_g = self._goal_reached_reward(r_arrival, r_waypoint)

        # 2. Compute the penalty for collisions or being too close to obstacles
        r_c = self._obstacle_collision_punish(self.cnn_data.scan[-240:], r_scan, r_collision)

        # 3. Compute the penalty for high angular velocity (fast turns)
        r_w = self._angular_velocity_punish(self.curr_vel.angular.z, r_rotation, w_thresh)

        # 3. Compute the reward for aligning with the goal.
        r_t = self._theta_reward(self.goal, self.peds, self.curr_vel.linear.x, r_angle, angle_thresh)

        # Total reward is a sum of all components.
        reward = r_g + r_c + r_t + r_w

        # logging
        self.get_logger().warn(f"Compute reward done. \nreward = {reward}")

        return reward

    # ========================================================================= #
    # Done flag computation

    # Determine whether the current episode (i.e., the task or goal the robot is trying to accomplish) has ended.
    def _is_done(self, reward):
        """
        Determines whether the current episode should end based on several conditions:
        - The robot reaches the goal.
        - The robot collides with obstacles.
        - The reward function returns a high negative value.
        - The maximum number of iterations is exceeded.

        Args:
            reward (float): Current reward (not used directly in this function).

        Returns:
            bool: True if the episode is done, False otherwise.
        """

        # Increment the number of iterations (i.e., time steps taken in the current episode)
        self.num_iterations += 1
        self.get_logger().warn(f"\nnum_iterations = {self.num_iterations}")


        # 1) Goal reached?
        # Calculate the Euclidean distance between the robot's current position and the goal position.
        dist_to_goal = np.linalg.norm(
            np.array(
                [
                    self.curr_pose.position.x - self.goal_position.x,  # Difference in x-coordinates
                    self.curr_pose.position.y - self.goal_position.y,  # Difference in y-coordinates
                    self.curr_pose.position.z - self.goal_position.z,  # Difference in z-coordinates
                ]
            )
        )

        if dist_to_goal <= self.GOAL_RADIUS:
            # Stop the robot by sending a zero velocity command
            self._cmd_vel_pub.publish(Twist())
            self._episode_done = True  # Mark the episode as done
            self.get_logger().warn("\n!!!\nTurtlebot reached the goal\n!!!")
            return True  # Return True indicating that the episode is finished


        # 2) Obstacle collision more than 1 times?
        # Fetch the latest scan data (e.g., from the lidar) to check the proximity of obstacles
        scan = self.cnn_data.scan[-240:]  # Use the last 240 scan points

        # Find the closest obstacle distance
        min_scan_dist = np.amin(scan[scan != 0])

        # Ensure min_scan_dist falls within the lidar's valid range of [0.03, 12] meters
        min_scan_dist = np.clip(min_scan_dist, 0.03, 12.0)

        # If the robot is too close to an obstacle (distance <= robot_radius), increment the bump counter
        if min_scan_dist <= self.ROBOT_RADIUS and min_scan_dist >= 0.03:
            self.bump_num += 1  # Increment the bump counter if the robot is too close

        if self.bump_num >= 1:
            # Stop the robot by sending a zero velocity command
            self._cmd_vel_pub.publish(Twist())
            self.bump_num = 0  # Reset the bump counter for the next episode
            self._episode_done = True  # Mark the episode as done
            self._reset = True  # Flag the environment for a reset after the episode end
            return True  # Return True indicating that the episode is finished

        # 4) Check if the robot has exceeded the maximum number of allowed iteration
        max_iteration = (512)   # Maximum number of iterations allowed for reaching the goal
        if self.num_iterations > max_iteration:
            # Stop the robot by sending a zero velocity command
            self._cmd_vel_pub.publish(Twist())
            self._episode_done = True  # Mark the episode as done
            self._reset = True  # Flag the environment for a reset after the episode end
            self.get_logger().warn(f"Turtlebot exceeded the maximum number of iterations before reaching the goal @ {self.goal_position}")
            return True

        # 5) Check for excessive negative reward (penalizing bad behavior)
        negative_reward_threshold = -50
        if reward <= negative_reward_threshold:
            # Stop the robot by sending a zero velocity command
            self._cmd_vel_pub.publish(Twist())
            self._episode_done = True
            self._reset = True
            self.get_logger().warn(f"Turtlebot performed poorly with a negative reward of {reward}, resetting the episode.")
            return True

        # If none of the termination conditions are met, the episode continues
        return False  # Return False indicating that the episode is still ongoing


def main(args=None):
    # Initialize ROS 2
    rclpy.init(args=args)

    # Create the environment object
    environment = DRLNavEnv()

    # Ensure all system (publishers, subscribers, services) are ready
    environment._check_all_systems_ready()

    try:
        # Spin the node so callbacks get processed
        rclpy.spin(environment)
    except KeyboardInterrupt:
        print("Shutting down due to user interruption")
    finally:
        # Cleanup and shutdown
        environment.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()