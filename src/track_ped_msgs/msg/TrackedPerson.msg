# Bounding box coordinates (in pixel coordinates)
int32 bbox_upper_left_x
int32 bbox_upper_left_y
int32 bbox_bottom_right_x
int32 bbox_bottom_right_y

# Unique ID for the pedestrian
int32 id 

# Distance from the camera to the pedestrian (depth in meters)
float32 depth

# Angle relative to the camera's perpendicular bisector (in radians)
float32 angle 

# Pedestrian velocity
geometry_msgs/Twist twist # Twist contains linear and angular velocity

# Real life coordinate of the pedestrian relative to the camera
geometry_msgs/Pose pose 
