#!/usr/bin/env python3
#
# This script visualizes the goal point in RViz using ROS2.
# It subscribes to the goal topic and publishes a marker to visualize the goal in RViz.
#------------------------------------------------------------------------------

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Header
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy


class GoalVisualizer(Node):
    """
    This class subscribes to the /goal_pose topic and visualizes the goal in RViz by publishing a Marker.
    """
    def __init__(self):
        super().__init__('goal_visualizer')

        # Define QoS profile for reliable communication
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Create a subscriber to the goal topic (adapt for ROS2, assuming '/goal_pose' or '/navigate_to_pose')
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',  # Update if necessary depending on your goal topic
            self.goal_callback,
            qos_profile
        )

        # Create a publisher to publish goal markers for RViz visualization
        self.goal_vis_pub = self.create_publisher(Marker, 'goal_markers', 10)

        self.get_logger().info("Goal visualizer node started. Waiting for goals...")

    def goal_callback(self, goal_msg):
        """
        Callback to visualize the goal as a marker in RViz.
        """
        self.get_logger().info(f"Received goal at ({goal_msg.pose.position.x}, {goal_msg.pose.position.y})")

        # Initialize header and color for the marker
        h = Header()
        h.frame_id = "map"  # Assumes the goal is in the 'map' frame
        h.stamp = self.get_clock().now().to_msg()

        # Create and configure the marker
        goal_marker = Marker()
        goal_marker.header = h
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        goal_marker.pose = goal_msg.pose  # Use the goal's pose for visualization
        goal_marker.scale.x = 1.8
        goal_marker.scale.y = 1.8
        goal_marker.scale.z = 1.8
        goal_marker.color.r = 1.0
        goal_marker.color.g = 0.0
        goal_marker.color.b = 0.0
        goal_marker.color.a = 0.5  # Set transparency to 50%

        # Publish the marker to RViz
        self.goal_vis_pub.publish(goal_marker)


def main(args=None):
    rclpy.init(args=args)

    try:
        goal_visualizer = GoalVisualizer()
        rclpy.spin(goal_visualizer)  # Keep the node alive to listen for goals and visualize them
    except KeyboardInterrupt:
        pass
    finally:
        goal_visualizer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
