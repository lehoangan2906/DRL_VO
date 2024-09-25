#!/usr/bin/env python3
#
# usage:
#
# This script is to publish the sub-goal point using the pure pursuit algorithm.
#------------------------------------------------------------------------------

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, PoseStamped, Point
import tf_transformations  # For quaternion operations in ROS2
import numpy as np
import threading
from tf2_ros import Buffer, TransformListener
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class PurePursuit(Node):
    """
    Implements the Pure Pursuit algorithm to calculate and publish sub-goal points for TurtleBot3.
    This node subscribes to a global path and transforms it into local sub-goals based on the robot's current position.
    """
    def __init__(self):
        super().__init__('pure_pursuit')

        # Parameters of the controller
        self.lookahead = 2.0  # Lookahead distance [m]
        self.rate = self.declare_parameter('rate', 20.0).value  # Rate to run the controller [Hz]
        self.goal_margin = 0.9  # Maximum distance to goal before stopping [m]

        # Parameters of the robot (specific to TurtleBot3)
        self.wheel_base = 0.23  # Distance between wheels
        self.wheel_radius = 0.025  # Wheel radius
        self.v_max = 0.5  # Maximum linear velocity [m/s]
        self.w_max = 5.0  # Maximum angular velocity [rad/s]

        # TF2 for managing transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ROS2 topics: subscribing to the global path and publishing sub-goals
        qos_profile = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE, history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        self.path_sub = self.create_subscription(Path, 'path', self.path_callback, qos_profile)
        self.cnn_goal_pub = self.create_publisher(Point, 'cnn_goal', 10)
        self.final_goal_pub = self.create_publisher(Point, 'final_goal', 10)

        # Lock to manage thread safety for path data
        self.lock = threading.Lock()

        # Variables to store path and timer
        self.path = None
        self.timer = None

        self.get_logger().info("Pure Pursuit initialized. Waiting for path...")

    def path_callback(self, msg):
        """
        Callback for receiving the global path. Starts the control loop once the first path is received.
        """
        self.get_logger().debug('Received global path.')
        with self.lock:
            self.path = msg  # Store the received path
        if self.timer is None:
            self.start_timer()  # Start the control loop

    def start_timer(self):
        """
        Starts the timer to run the control loop at the specified rate.
        """
        self.timer = self.create_timer(1.0 / self.rate, self.timer_callback)

    def get_current_pose(self):
        """
        Fetches the robot's current pose from the TF tree.
        """
        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
            x = np.array([trans.transform.translation.x, trans.transform.translation.y])
            theta = tf_transformations.euler_from_quaternion(
                [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])[2]
            return x, theta
        except Exception as e:
            self.get_logger().warn(f"Failed to get current pose: {e}")
            return np.array([np.nan, np.nan]), np.nan

    def find_closest_point(self, x):
        """
        Finds the closest point on the global path to the current robot position.
        """
        pt_min = np.array([np.nan, np.nan])
        dist_min = np.inf
        seg_min = -1

        if self.path is None:
            self.get_logger().warn("No path received yet.")
            return pt_min, dist_min, seg_min

        # Iterate through all path segments to find the closest point
        for i in range(len(self.path.poses) - 1):
            p_start = np.array([self.path.poses[i].pose.position.x, self.path.poses[i].pose.position.y])
            p_end = np.array([self.path.poses[i+1].pose.position.x, self.path.poses[i+1].pose.position.y])

            # Compute closest point on segment
            v = p_end - p_start
            v /= np.linalg.norm(v)
            dist_projected = np.dot(x - p_start, v)
            pt = p_start + dist_projected * v if 0 <= dist_projected <= np.linalg.norm(v) else p_end

            dist = np.linalg.norm(pt - x)
            if dist < dist_min:
                pt_min, dist_min, seg_min = pt, dist, i

        return pt_min, dist_min, seg_min

    def find_goal(self, x, pt, dist, seg):
        """
        Finds the goal point on the path based on the lookahead distance.
        """
        goal = pt if dist > self.lookahead else None
        while goal is None and seg < len(self.path.poses) - 1:
            seg += 1
            p_end = np.array([self.path.poses[seg].pose.position.x, self.path.poses[seg].pose.position.y])
            dist_to_end = np.linalg.norm(x - p_end)
            if dist_to_end > self.lookahead:
                goal = p_end

        end_goal_pos = [self.path.poses[-1].pose.position.x, self.path.poses[-1].pose.position.y]
        end_goal_rot = [
            self.path.poses[-1].pose.orientation.x,
            self.path.poses[-1].pose.orientation.y,
            self.path.poses[-1].pose.orientation.z,
            self.path.poses[-1].pose.orientation.w
        ]
        return goal, end_goal_pos, end_goal_rot

    def timer_callback(self):
        """
        Timer callback to continuously compute and publish sub-goals based on the robot's current position and the global path.
        """
        with self.lock:
            if self.path is None:
                return

        # Get the robot's current position
        x, theta = self.get_current_pose()
        if np.isnan(x[0]):
            return

        # Find the closest point and compute the sub-goal
        pt, dist, seg = self.find_closest_point(x)
        if np.isnan(pt).any():
            return

        # Compute the goal based on lookahead distance
        goal, end_goal_pos, end_goal_rot = self.find_goal(x, pt, dist, seg)
        if goal is None or end_goal_pos is None:
            return

        # Transform the goal to local coordinates
        map_T_robot = np.array([[np.cos(theta), -np.sin(theta), x[0]],
                                [np.sin(theta), np.cos(theta), x[1]],
                                [0, 0, 1]])

        local_goal = np.matmul(np.linalg.inv(map_T_robot), np.array([goal[0], goal[1], 1]))[:2]
        final_goal = np.matmul(np.linalg.inv(map_T_robot), np.array([end_goal_pos[0], end_goal_pos[1], 1]))[:2]
        yaw = tf_transformations.euler_from_quaternion(end_goal_rot)[2]

        # Publish the CNN goal
        cnn_goal = Point()
        cnn_goal.x, cnn_goal.y, cnn_goal.z = local_goal[0], local_goal[1], 0
        self.cnn_goal_pub.publish(cnn_goal)

        # Publish the final goal
        final_goal_msg = Point()
        final_goal_msg.x, final_goal_msg.y, final_goal_msg.z = final_goal[0], final_goal[1], yaw
        self.final_goal_pub.publish(final_goal_msg)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = PurePursuit()
        rclpy.spin(node)
    except Exception as e:
        print(f"Exception caught: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
