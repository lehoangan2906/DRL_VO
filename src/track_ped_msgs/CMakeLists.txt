cmake_minimum_required(VERSION 3.5)
project(track_ped_msgs)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)  # Add this line to find geometry_msgs

# Include directories
include_directories(include)

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TrackedPerson.msg"
  "msg/TrackedPersons.msg"
  DEPENDENCIES std_msgs geometry_msgs  # Add geometry_msgs as a dependency
)

ament_package()
