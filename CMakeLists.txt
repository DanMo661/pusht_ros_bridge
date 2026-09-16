cmake_minimum_required(VERSION 3.8)
project(pusht_ros_bridge)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PushtObservation.msg"
  DEPENDENCIES sensor_msgs std_msgs
)

# Python nodes are NOT pip/ament-installed here: they depend on a lerobot venv
# interpreter and are executed directly by scripts/09-bridge-run.sh.
install(PROGRAMS
  pusht_ros_bridge/env_node.py
  pusht_ros_bridge/policy_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
