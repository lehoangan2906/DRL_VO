cmake_minimum_required(VERSION 3.5)
project(drl_vo_nav)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Ensure track_utils is in the Python path
ament_python_install_package(src)

# Install Python scripts
install(PROGRAMS
  src/cmd_vel_pub.py
  src/cnn_data_pub.py
  src/custom_cnn_full.py
  src/drl_vo_inference.py
  src/drl_vo_train.py
  src/dynamic_goal_publisher.py
  src/track_data_pub.py
  src/track_func.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install any launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
