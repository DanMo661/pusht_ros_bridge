#!/bin/bash
# 构建 pusht_ros_bridge（系统 python3，包本身不依赖 lerobot）
set -e
source /opt/ros/jazzy/setup.bash
cd /root/ros2_ws
colcon build --packages-select pusht_ros_bridge 2>&1 | tail -3
ls install/pusht_ros_bridge/lib/pusht_ros_bridge/
echo BUILD_OK
