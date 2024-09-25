#!/usr/bin/env python3
# usage:
#
# This script is to publish the goal points and evaluate the navigation performance.
#------------------------------------------------------------------------------

import rclpy
from rclpy.node import Node
import math
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from tf_transformations import quaternion_from_euler
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan  # For collision detection

from math import hypot

GOAL_NUM = 3

class MoveBaseSeq(Node):

    def __init__(self):
        super().__init__('move_base_sequence')

        # Success and timing metrics
        self.success_num = 0
        self.total_time = 0
        self.start_time = 0
        self.end_time = 0

        # Distance traveled tracking
        self.total_distance = 0.0
        self.previous_x = 0.0
        self.previous_y = 0.0
        self.odom_start = True

        # Load parameters from the ROS2 parameter server (e.g., goal points and yaw angles)
        self.declare_parameter('p_seq', [])
        self.declare_parameter('yaw_seq', [])
        points_seq = self.get_parameter('p_seq').value
        yaweulerangles_seq = self.get_parameter('yaw_seq').value

        # Replace bumper with laser scan for obstacle detection
        self.collision_flag = False
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)

        # Odometry for tracking distance traveled
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Initialize action client for sending goals to the Nav2 action server
        self.client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Convert goal points to PoseStamped messages
        self.goal_cnt = 0
        quat_seq = [Quaternion(*quaternion_from_euler(0, 0, yaw)) for yaw in yaweulerangles_seq]
        self.pose_seq = [Pose(Point(*points_seq[i:i+3]), quat_seq[i]) for i in range(len(points_seq)//3)]

        # Wait for the action server
        self.get_logger().info("Waiting for Nav2 action server...")
        while not self.client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn("Action server not available, waiting...")

        self.get_logger().info("Connected to Nav2 action server. Starting goal achievements...")
        self.send_goal()

    def laser_callback(self, scan_msg):
        # Check if any points are closer than a threshold (e.g., 0.3 meters)
        if scan_msg.ranges and min(scan_msg.ranges) < 0.3:
            self.collision_flag = True
            self.get_logger().warn("Obstacle detected, stopping...")

    def odom_callback(self, odom_msg):
        if self.odom_start:
            self.previous_x = odom_msg.pose.pose.position.x
            self.previous_y = odom_msg.pose.pose.position.y
            self.odom_start = False

        x = odom_msg.pose.pose.position.x
        y = odom_msg.pose.pose.position.y
        d_increment = hypot(x - self.previous_x, y - self.previous_y)
        self.total_distance += d_increment

        self.previous_x = x
        self.previous_y = y

        # Log the total distance traveled so far
        self.get_logger().info(f"Total distance traveled: {self.total_distance:.4f} meters")

    def send_goal(self):
        if self.goal_cnt < len(self.pose_seq):
            goal_msg = NavigateToPose.Goal()
            goal_msg.pose.header.frame_id = "map"
            goal_msg.pose.header.stamp = self.get_clock().now().to_msg()  # Use ROS2 time
            goal_msg.pose.pose = self.pose_seq[self.goal_cnt]

            self.get_logger().info(f"Sending goal pose {self.goal_cnt+1}")
            self.client.send_goal_async(
                goal_msg, feedback_callback=self.feedback_cb
            ).add_done_callback(self.done_cb)
            self.start_time = self.get_clock().now().seconds_nanoseconds()[0]  # Track start time
        else:
            self.get_logger().info("All goals have been reached.")
            self.shutdown()

    def feedback_cb(self, feedback_msg):
        self.get_logger().info(f"Feedback for goal pose {self.goal_cnt+1}")

    def done_cb(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error(f"Goal {self.goal_cnt+1} was rejected")
            self.shutdown()
            return

        self.goal_cnt += 1

        # Check the result of the goal execution
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_cb)

    def result_cb(self, future):
        result = future.result().result

        if result is None:
            self.get_logger().error(f"Failed to get the result for goal {self.goal_cnt}")
            self.shutdown()
            return

        status = result.status
        if status == 3:  # SUCCEEDED
            self.get_logger().info(f"Goal pose {self.goal_cnt} reached successfully")

            # Count success if no collision occurred
            if not self.collision_flag:
                self.success_num += 1
            self.collision_flag = False  # Reset collision flag

            # Calculate total time for this goal
            self.end_time = self.get_clock().now().seconds_nanoseconds()[0]
            time_diff = self.end_time - self.start_time
            self.total_time += time_diff
            self.get_logger().info(f"Time for goal: {time_diff:.2f} seconds")

            # Send the next goal
            self.send_goal()

        elif status == 4:  # ABORTED
            self.get_logger().error(f"Goal pose {self.goal_cnt} was aborted by the Action Server")
            self.shutdown()

        elif status == 5:  # REJECTED
            self.get_logger().error(f"Goal pose {self.goal_cnt} was rejected by the Action Server")
            self.shutdown()

        else:
            self.get_logger().error(f"Goal pose {self.goal_cnt} ended with status: {status}")
            self.shutdown()

    def shutdown(self):
        self.get_logger().info(f"Final Success Count: {self.success_num}")
        self.get_logger().info(f"Total Distance Traveled: {self.total_distance:.2f} meters")
        self.get_logger().info(f"Total Time Taken: {self.total_time:.2f} seconds")
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    try:
        move_base_sequence = MoveBaseSeq()
        rclpy.spin(move_base_sequence)
    except Exception as e:
        print(f"Exception caught: {e}")
    finally:
        if 'move_base_sequence' in locals():
            move_base_sequence.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
