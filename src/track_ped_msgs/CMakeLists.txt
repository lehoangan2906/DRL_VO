cmake_minimum_required(VERSION 3.5)
project(track_ped_msgs)

# Check compiler flags
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Specify the message files to be generated
set(msg_files
  "msg/TrackedPerson.msg"
  "msg/TrackedPersons.msg"
)

# Generate the ROS interfaces (messages, services, etc.)
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES std_msgs geometry_msgs
)

# Enable testing if required
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # Skip copyright check
  set(ament_cmake_copyright_FOUND TRUE)
  # Skip cpplint (useful when not in a git repo)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Finalize the package
ament_package()
