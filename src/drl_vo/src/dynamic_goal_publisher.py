#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
import cv2
import numpy as np
from ultralytics import YOLO
from cv_bridge import CvBridge # for converting image messages to OpenCV images

class DynamicGoalPublisher(Node):
    def __init__(self):
        super().__init__('dynamic_goal_publisher')

        # YOLOv8 model initialization
        self.yolo_model = YOLO("yolov5n.pt")

        # ROS2 publishers and subscribers
        self.goal_pub = self.create_publisher(PointStamped, 'dynamic_goal', 10)
        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 10)

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Variables for storing data
        self.depth_image = None
        self.camera_info = None
        self.focal_length = None

    def camera_info_callback(self, msg):
        # Extract camera intrinsic parameters
        self.focal_length = msg.k[0]    # Assuming focal length is in K matrix at 