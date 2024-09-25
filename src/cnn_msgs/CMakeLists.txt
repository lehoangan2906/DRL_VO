# This file is used to define the build process of the cnn_msgs package. Its main purposes are:
# - Define the build configuration.
# - Specify the dependencies of the package.
# - Generate ROS 2 interface code from message files.
# - Ensure that the project can be build and integrated correctly within the ROS2 ecosystem.

# Specify the minimum version of CMake required to build this project
cmake_minimum_required(VERSION 3.5)

# Define the project name
project(cnn_msgs)

# Find and include the ament_cmake package, which is required for building ROS 2 packages
find_package(ament_cmake REQUIRED)

# Find and include the rosidl_default_generators package, which is required for generating ROS 2 interface code
find_package(rosidl_default_generators REQUIRED)

# Find and include the std_msgs package, which provides standard message definitions for ROS 2
find_package(std_msgs REQUIRED)

# Add message files and specify dependencies
# rosidl_generate_interfaces is a macro that generates ROS 2 interface code from the specified message files
rosidl_generate_interfaces(${PROJECT_NAME}
    "msg/CnnData.msg"  # Path to the message file
    DEPENDENCIES std_msgs  # Specify dependencies for the message file
)

# Export package dependencies
# ament_export_dependencies makes the specified dependencies available to other packages that depend on this package
ament_export_dependencies(rosidl_default_runtime)

# Define this package as an ament package
# ament_package is a macro that sets up the package for use with the ament build system
ament_package()