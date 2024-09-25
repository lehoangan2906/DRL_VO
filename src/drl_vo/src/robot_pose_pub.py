#!/usr/bin/env python3

# This script is to publish the robot pose on the /robot_pose topic at a rate of 30 Hz

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import tf_transformations
from tf2_ros import TransformListener, Buffer
from rclpy.time import Time

class RobotPosePublisher(Node):
    def __init__(self):
        super().__init__('robot_pose')
        self.publisher_ = self.create_publisher(PoseStamped, '/robot_pose', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(1.0/30.0, self.timer_callback)  # 30 Hz

    def timer_callback(self):
        try:
            now = self.get_clock().now().to_msg()
            trans = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
            
            # Create and populate the PoseStamped message
            rob_pos = PoseStamped()
            rob_pos.header.stamp = now
            rob_pos.header.frame_id = 'map'
            rob_pos.pose.position.x = trans.transform.translation.x
            rob_pos.pose.position.y = trans.transform.translation.y
            rob_pos.pose.position.z = trans.transform.translation.z
            rob_pos.pose.orientation.x = trans.transform.rotation.x
            rob_pos.pose.orientation.y = trans.transform.rotation.y
            rob_pos.pose.orientation.z = trans.transform.rotation.z
            rob_pos.pose.orientation.w = trans.transform.rotation.w

            # Publish the message
            self.publisher_.publish(rob_pos)

        except Exception as e:
            self.get_logger().warn('Could not get robot pose: ' + str(e))


def main(args=None):
    rclpy.init(args=args)
    node = RobotPosePublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
