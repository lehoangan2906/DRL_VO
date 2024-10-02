#!/usr/bin/python3

import rclpy
import numpy as np
from pathlib import Path
from track_func import run
from rclpy.node import Node
from std_msgs.msg import Header
from geometry_msgs.msg import Twist, Pose 
from track_ped_msgs.msg import TrackedPerson, TrackedPersons

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] # YOLOv8 strongsort root directory
WEIGHTS = ROOT /  'weights'

class TrackDataPublisher(Node):
    def __init__(self):
        super().__init__('track_data_publisher')

        # Create ROS2 publishers
        self.tracked_person_pub = self.create_publisher(TrackedPerson, '/track_ped', 10)
        self.tracked_persons_pub = self.create_publisher(TrackedPersons, '/track_peds', 10)

        # Call the tracking function and get the detected objects
        self.publish_tracking_data()

    def publish_tracking_data(self):
        twist = Twist()

        # Get the list of tracked persons, rs_dicts and pre_velocity_cal from track_func's run method
        rs_dicts, pre_velocity_cal = run(
            camera_type = 'simulated',  
            source = '0',
            yolo_weights = WEIGHTS / 'yolov8m-seg.engine',  # Path to the YOLO model weights
            reid_weights = WEIGHTS / 'osnet_x0_25_msmt17.pt',  # Path to ReID model weights
            tracking_method = 'bytetrack',  # Tracking method
            tracking_config = ROOT / 'track_utils' / 'Tracking_Face' / 'yolov8_tracking' / 'trackers' / 'bytetrack' / 'configs' / ('bytetrack' + '.yaml'),
            # Optional tracking configuration
            imgsz = (640, 640),  # Image size
            conf_thres = 0.25,  # Confidence threshold
            iou_thres = 0.45,  # IOU threshold
            max_det = 1000,  # Maximum number of detections per image
            device = '0',  # CUDA device (GPU), can also be 'cpu'
            show_vid = True,  # Whether to show the video results in a window
            save_txt = False,  # Whether to save results in a text file
            save_conf = False,  # Whether to save confidence scores
            save_crop = False,  # Whether to save cropped bounding boxes
            save_trajectories = False,  # Whether to save the trajectories
            save_vid = False,  # Whether to save video output
            nosave = False,  # If set to True, will not save images/videos
            classes = None,  # Filter by class (0 = person)
            agnostic_nms = False,  # Class-agnostic non-max suppression
            augment = False,  # Whether to use augmented inference
            visualize = False,  # Whether to visualize features
            update = False,  # Update all models
            project = ROOT / 'runs' / 'track',  # Directory to save results
            name = 'exp',  # Experiment name
            exist_ok = False,  # Whether it's okay if the save directory exists
            line_thickness = 2,  # Line thickness for bounding boxes
            hide_labels = False,  # Whether to hide labels
            hide_conf = False,  # Whether to hide confidence scores
            hide_class = False,  # Whether to hide IDs
            half = False,  # Use FP16 half-precision inference
            dnn = False,  # Use OpenCV DNN for ONNX inference
            vid_stride = 1,  # Video frame-rate stride
            retina_masks = False,  # Whether to use retina masks
            )
        
        # Create TrackedPersons message (with a list of TrackedPerson)
        tracked_persons_msg = TrackedPersons()
        tracked_persons_msg.header = Header()
        tracked_persons_msg.header.stamp = self.get_clock().now().to_msg()
        tracked_persons_msg.header.frame_id = "camera_frame"

        # Populate the tracked_person_list into the message
        for idx, rs_dict in enumerate(rs_dicts):
            # Create a new TrackedPerson message for each object
            tracked_person_msg = TrackedPerson()
            tracked_person_msg.id = int(rs_dict['id'])
            tracked_person_msg.depth = rs_dict['depth']
            tracked_person_msg.angle = rs_dict['angle']
            tracked_person_msg.bbox_upper_left_x = rs_dict['bbox'][0]
            tracked_person_msg.bbox_upper_left_y = rs_dict['bbox'][1]
            tracked_person_msg.bbox_bottom_right_x = rs_dict['bbox'][2]
            tracked_person_msg.bbox_bottom_right_y = rs_dict['bbox'][3]
            
            # Convert velocity from rs_dict (if available)
            if rs_dict['velocity'] is not None:
                twist.linear.x = rs_dict['velocity'][0] # velocity in x direction
                twist.linear.z = rs_dict['velocity'][1] # velocity in z direction
            else:
                twist.linear.x = 0.0
                twist.linear.z = 0.0

            tracked_person_msg.twist = twist 

            # Calculate x and z coordinates of the pedestrian in the camera frame
            real_x = rs_dict["depth"] * np.tan(rs_dict["angle"])  # Calculate x using trigonometry
            real_z = rs_dict["depth"]                             # z is the depth directly

            # Fill in the pose with real-world coordinates
            tracked_person_msg.pose = Pose()
            tracked_person_msg.pose.position.x = real_x
            tracked_person_msg.pose.position.y = 0.0
            tracked_person_msg.pose.position.z = real_z

            # Orientation can be set to default
            tracked_person_msg.pose.orientation.x = 0.0
            tracked_person_msg.pose.orientation.y = 0.0
            tracked_person_msg.pose.orientation.z = 0.0
            tracked_person_msg.pose.orientation.w = 1.0

            # Add the tracked person message to the TrackedPersons list
            tracked_persons_msg.tracks.append(tracked_person_msg)

            # Publish each tracked person individually
            self.tracked_person_pub.publish(tracked_person_msg)
        
        # Publish the TrackedPersons message (all detected objects in each frame)
        self.tracked_persons_pub.publish(tracked_persons_msg)

def main(args=None):
    rclpy.init(args=args)
    track_data_publisher = TrackDataPublisher()
    rclpy.spin(track_data_publisher)
    track_data_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()